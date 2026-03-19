"""
differ.py – pixel-level diff computation and image reconstruction.

Two public functions:
  compute_patches(ref_img, new_img, threshold, tile_size, merge_gap)
      → list of (x, y, w, h, compressed_bytes)

  reconstruct(base_img, patches)
      → Pillow Image
"""

from __future__ import annotations

import io
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

# (x, y, w, h, compressed_pixel_data)
Patch = Tuple[int, int, int, int, bytes]

def _compress(raw_bytes: bytes) -> bytes:
    """zlib level-6 — used only to wrap PNG blobs."""
    return zlib.compress(raw_bytes, level=6)


def _decompress(data: bytes) -> bytes:
    return zlib.decompress(data)


def _encode_crop(
    img: Image.Image,
    x: int,
    y: int,
    w: int,
    h: int,
    codec: str = "png",
    quality: int = 100,
) -> bytes:
    """
    Crop *img* at (x, y, w, h) and return compressed bytes ready to store.

    codec  : "png"  – lossless; quality ignored; blob is zlib-wrapped after save.
             "webp" – quality=100 → lossless WebP; quality<100 → lossy WebP at
                      that quality level (0-99). WebP carries its own entropy
                      coding so we skip the extra zlib pass.
    """
    crop = img.crop((x, y, x + w, y + h))
    buf = io.BytesIO()
    if codec == "webp":
        lossless = quality == 100
        if lossless:
            crop.save(buf, format="WEBP", lossless=True)
        else:
            crop.save(buf, format="WEBP", lossless=False, quality=quality, method=4)
        return buf.getvalue()          # WebP is self-compressed; no extra zlib
    else:
        crop.save(buf, format="PNG", compress_level=1)
        return _compress(buf.getvalue())


def _decode_patch(data: bytes, codec: str = "png") -> Image.Image:
    """Inverse of _encode_crop — returns a Pillow Image."""
    if codec == "webp":
        return Image.open(io.BytesIO(data))
    else:
        raw = _decompress(data)
        return Image.open(io.BytesIO(raw))


def compute_patches(
    ref_img: Image.Image,
    new_img: Image.Image,
    threshold: int = 4,
    tile_size: int = 32,
    merge_gap: int = 8,
    codec: str = "png",
    quality: int = 100,
    workers: Optional[int] = None,
) -> List[Patch]:
    """
    Compare *new_img* against *ref_img* using a tile grid.

    Parameters
    ----------
    ref_img    : PIL Image – reference (what we're diffing against)
    new_img    : PIL Image – the new frame
    threshold  : per-channel absolute pixel delta that counts as "changed"
                 (use 0 for perfectly lossless; 3-6 absorbs JPEG noise)
    tile_size  : grid cell size in pixels (smaller = finer granularity but
                 more patch overhead; 16–64 is a good range)
    merge_gap  : changed tiles within this many pixels of each other on the
                 same row are merged into one patch (reduces patch count)
    workers    : thread-pool size for parallel patch compression.
                 None = use ThreadPoolExecutor default (cpu_count × 4 or so).
                 Set to 1 to disable parallelism entirely.

    Returns
    -------
    List of (x, y, w, h, compressed_bytes) tuples.
    Empty list if the images are identical within threshold.
    """
    ref = np.array(ref_img.convert("RGBA"), dtype=np.int16)
    new = np.array(new_img.convert("RGBA"), dtype=np.int16)

    if ref.shape != new.shape:
        raise ValueError(
            f"Image dimensions differ: ref={ref.shape[:2]} new={new.shape[:2]}"
        )

    h, w = ref.shape[:2]
    diff = np.abs(new - ref).max(axis=2)

    cols = (w + tile_size - 1) // tile_size
    rows = (h + tile_size - 1) // tile_size

    pad_h = rows * tile_size - h
    pad_w = cols * tile_size - w
    if pad_h or pad_w:
        diff_padded = np.pad(diff, ((0, pad_h), (0, pad_w)), constant_values=0)
    else:
        diff_padded = diff

    tile_max = (
        diff_padded
        .reshape(rows, tile_size, cols, tile_size)
        .max(axis=(1, 3))
    )
    changed_mask = tile_max > threshold
    tile_rows, tile_cols = np.where(changed_mask)
    changed_tiles = list(zip(tile_rows.tolist(), tile_cols.tolist()))

    if not changed_tiles:
        return []

    rects = _tiles_to_rects(changed_tiles, tile_size, w, h)
    merged = _merge_rects(rects, merge_gap, tile_size, w, h)

    if len(merged) <= 2 or workers == 1:
        patches: List[Patch] = [
            (rx, ry, rw, rh, _encode_crop(new_img, rx, ry, rw, rh, codec=codec, quality=quality))
            for (rx, ry, rw, rh) in merged
        ]
    else:
        def _compress_rect(rect: tuple) -> Patch:
            rx, ry, rw, rh = rect
            return rx, ry, rw, rh, _encode_crop(new_img, rx, ry, rw, rh, codec=codec, quality=quality)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            patches = list(pool.map(_compress_rect, merged))

    return patches


def _tiles_to_rects(tiles, tile_size, img_w, img_h):
    """Convert (tile_row, tile_col) list to pixel (x, y, w, h) tuples."""
    rects = []
    for tr, tc in tiles:
        x = tc * tile_size
        y = tr * tile_size
        w = min(tile_size, img_w - x)
        h = min(tile_size, img_h - y)
        rects.append((x, y, w, h))
    return rects


def _merge_rects(rects, merge_gap, tile_size, img_w, img_h):
    """
    Merge horizontally adjacent/nearby rects on the same tile row.
    Then optionally merge vertically if rows are adjacent (simple sweep).
    """
    if not rects:
        return []

    by_row: dict[int, list] = {}
    for (x, y, w, h) in rects:
        by_row.setdefault(y, []).append((x, y, w, h))

    merged_rows = []
    for y_key in sorted(by_row):
        row = sorted(by_row[y_key], key=lambda r: r[0])
        current = list(row[0])
        for rect in row[1:]:
            rx, ry, rw, rh = rect
            gap = rx - (current[0] + current[2])
            if gap <= merge_gap:
                # extend current rect rightward
                new_right = max(current[0] + current[2], rx + rw)
                current[2] = new_right - current[0]
                current[3] = max(current[3], rh)
            else:
                merged_rows.append(tuple(current))
                current = [rx, ry, rw, rh]
        merged_rows.append(tuple(current))

    merged_rows.sort(key=lambda r: (r[0], r[1]))
    final = []
    used = [False] * len(merged_rows)
    for i, ra in enumerate(merged_rows):
        if used[i]:
            continue
        ax, ay, aw, ah = ra
        for j, rb in enumerate(merged_rows[i + 1:], i + 1):
            if used[j]:
                continue
            bx, by, bw, bh = rb
            if bx == ax and bw == aw:
                gap = by - (ay + ah)
                if gap <= merge_gap:
                    ah = by + bh - ay
                    used[j] = True
        final.append((ax, ay, aw, ah))
        used[i] = True

    clamped = []
    for (x, y, w, h) in final:
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        if w > 0 and h > 0:
            clamped.append((x, y, w, h))
    return clamped


def reconstruct(base_img: Image.Image, patches: List[Patch], codec: str = "png") -> Image.Image:
    """
    Apply a list of patches onto *base_img* and return a new Image.

    *base_img* is not modified in place.
    *codec* must match the codec used when the patches were encoded.
    """
    result = base_img.copy().convert("RGBA")
    for (x, y, w, h, compressed) in patches:
        patch_img = _decode_patch(compressed, codec=codec).convert("RGBA")
        result.paste(patch_img, (x, y))
    return result


def images_are_identical(img_a: Image.Image, img_b: Image.Image, threshold: int = 0) -> bool:
    """Quick check – True if max per-pixel delta ≤ threshold."""
    a = np.array(img_a.convert("RGBA"), dtype=np.int16)
    b = np.array(img_b.convert("RGBA"), dtype=np.int16)
    if a.shape != b.shape:
        return False
    return int(np.abs(a - b).max()) <= threshold


def similarity_score(img_a: Image.Image, img_b: Image.Image) -> float:
    """
    Return fraction of pixels that are identical (0.0 = completely different,
    1.0 = identical).  Useful for deciding whether to force a keyframe.
    """
    a = np.array(img_a.convert("RGBA"), dtype=np.int16)
    b = np.array(img_b.convert("RGBA"), dtype=np.int16)
    if a.shape != b.shape:
        return 0.0
    diff = np.abs(a - b).max(axis=2)
    identical_pixels = int((diff == 0).sum())
    total = a.shape[0] * a.shape[1]
    return identical_pixels / total

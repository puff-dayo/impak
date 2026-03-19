"""
encoder.py – write .impak files.

Usage (context-manager style):

    with ImpakWriter("out.impak", mode="vs_first") as w:
        w.add(PIL.Image.open("a.png"), name="frame_00")
        w.add(PIL.Image.open("b.png"), name="frame_01")

Usage (manual):

    w = ImpakWriter("out.impak", mode="keyframe", keyframe_interval=10)
    w.add(img)
    w.close()          # MUST call close() to finalise the file

Manual mode (pinned baselines + fallback):

    with ImpakWriter(
        "out.impak",
        mode="manual",
        baselines=["bg_day.png", "bg_night.png"],
        fallback_mode="lto",
    ) as w:
        for path in content_frames:
            w.add(path)

    # The baselines are stored as hidden reference keyframes at the front of
    # the file.  ImpakReader skips them in iteration / __len__ / __getitem__
    # so only content frames are visible to callers.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image

from .differ import compute_patches, _encode_crop
from .formats import (
    FILE_HEADER_SIZE,
    FRAME_INDEX_ENTRY_SIZE,
    FRAME_DELTA,
    FRAME_KEYFRAME,
    MODE_FROM_NAME,
    MODE_KEYFRAME,
    MODE_LTO,
    MODE_MANUAL,
    MODE_VS_FIRST,
    MODE_VS_PRIOR,
    CODEC_FROM_NAME,
    pack_file_header,
    pack_index_entry,
    pack_patch_header,
)


class ImpakWriter:
    """
    Sequential writer for .impak collections.

    Parameters
    ----------
    path              : output file path
    mode              : "vs_first" | "vs_prior" | "keyframe" | "lto" | "manual"
    keyframe_interval : (keyframe mode only) store a full image every N frames
    threshold         : pixel delta considered "changed" (0 = perfectly lossless)
    tile_size         : diff grid cell size in pixels
    merge_gap         : pixels gap between changed tiles before they are merged
    auto_keyframe_sim : if similarity drops below this fraction a keyframe is
                        forced regardless of mode (0.0 = never force)
    workers           : thread-pool size used for parallel patch compression and
                        LTO candidate probing.  None = os.cpu_count().
                        Set to 1 to disable parallelism entirely.
    baselines         : (manual mode) list of PIL Images or file paths used as
                        pinned reference anchors.  They are stored as hidden
                        leading keyframes so the delta chain is self-contained,
                        but ImpakReader hides them from callers — only content
                        frames are visible when iterating or indexing.
                        Baselines are automatically resized to match the canvas
                        size established by the first content frame (or the
                        first baseline if all are the same size).
    fallback_mode     : (manual mode) diff strategy applied when no baseline
                        yields a patch set smaller than this alternative.
                        Accepts "vs_first", "vs_prior", "keyframe", or "lto".
                        Defaults to "lto".
    """

    def __init__(
        self,
        path: Union[str, Path],
        mode: str = "vs_first",
        keyframe_interval: int = 10,
        threshold: int = 4,
        tile_size: int = 64,
        merge_gap: int = 8,
        auto_keyframe_sim: float = 0.5,
        codec: str = "webp",
        quality: int = 100,
        lto_candidates: int = 6,
        max_ref_depth: int = 8,
        workers: Optional[int] = None,
        baselines: Optional[List[Union[Image.Image, str, Path]]] = None,
        fallback_mode: str = "lto",
    ):
        if mode not in MODE_FROM_NAME:
            raise ValueError(f"mode must be one of {list(MODE_FROM_NAME)}")
        if codec not in ("png", "webp"):
            raise ValueError("codec must be 'png' or 'webp'")
        if not (0 <= quality <= 100):
            raise ValueError("quality must be 0-100")
        if lto_candidates < 1:
            raise ValueError("lto_candidates must be >= 1")
        if max_ref_depth < 1:
            raise ValueError("max_ref_depth must be >= 1")

        if mode == "manual":
            _valid_fallbacks = [m for m in MODE_FROM_NAME if m != "manual"]
            if fallback_mode not in _valid_fallbacks:
                raise ValueError(
                    f"fallback_mode must be one of {_valid_fallbacks}"
                )
            if not baselines:
                raise ValueError(
                    "manual mode requires at least one entry in 'baselines'"
                )

        self.path = Path(path)
        self.mode = mode
        self.mode_id = MODE_FROM_NAME[mode]
        self.keyframe_interval = keyframe_interval
        self.threshold = threshold
        self.tile_size = tile_size
        self.merge_gap = merge_gap
        self.auto_keyframe_sim = auto_keyframe_sim
        self.codec = codec
        self.quality = quality
        self.lto_candidates = lto_candidates
        self.max_ref_depth = max_ref_depth
        self.workers = workers
        self.fallback_mode = fallback_mode
        self.fallback_mode_id = MODE_FROM_NAME.get(fallback_mode, MODE_LTO)

        self._frames: list[dict] = []
        self._frame_data: list[bytes] = []
        self._ref_images: list[Image.Image] = []
        self._ref_arrays: list[np.ndarray] = []
        self._depths: list[int] = []

        self._canvas: Optional[tuple[int, int]] = None
        self._closed = False

        self._baseline_ids: list[int] = []
        self._baseline_count: int = 0
        self._pending_baselines: list = list(baselines) if baselines else []

        _pool_size = workers if workers is not None else (os.cpu_count() or 4)
        self._pool = ThreadPoolExecutor(max_workers=max(1, _pool_size))

    def add(
        self,
        image: Union[Image.Image, str, Path],
        name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Add one content image to the collection.  Returns the 0-based frame
        index as seen by readers (i.e. not counting hidden baseline frames).

        *image* may be a Pillow Image or a file path.
        *name* is stored in per-frame metadata (useful for later retrieval).
        *metadata* is an arbitrary JSON-serialisable dict merged with name.
        """
        if self._closed:
            raise RuntimeError("Writer is already closed")

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        image = image.convert("RGBA")

        if self._canvas is None:
            self._canvas = (image.width, image.height)
        elif (image.width, image.height) != self._canvas:
            raise ValueError(
                f"Frame {len(self._frames)}: image size {image.size} does not "
                f"match canvas {self._canvas}"
            )

        if self._pending_baselines:
            self._inject_baselines(self._pending_baselines)
            self._pending_baselines = []

        frame_id = len(self._frames)

        frame_type, ref_id, patches = self._encode_frame(image, frame_id)

        meta: dict = {}
        if name:
            meta["name"] = name
        if metadata:
            meta.update(metadata)
        meta_bytes = json.dumps(meta, separators=(",", ":")).encode() if meta else b""

        parts: list[bytes] = []
        for (x, y, w, h, data) in patches:
            parts.append(pack_patch_header(x, y, w, h, len(data)))
            parts.append(data)
        parts.append(meta_bytes)
        frame_bytes = b"".join(parts)

        if frame_type == FRAME_KEYFRAME:
            self._depths.append(0)
        else:
            self._depths.append(self._depths[ref_id] + 1)

        self._frames.append({
            "patch_count": len(patches),
            "ref_frame_id": ref_id,
            "metadata_len": len(meta_bytes),
            "frame_type": frame_type,
        })
        self._frame_data.append(frame_bytes)
        self._ref_images.append(image)
        self._ref_arrays.append(np.array(image, dtype=np.int16))

        return frame_id - self._baseline_count

    def close(self):
        """Finalise and write the file.  Called automatically by __exit__."""
        if self._closed:
            return
        if not self._frames:
            raise RuntimeError("No frames added — nothing to write")

        self._pool.shutdown(wait=False)

        w, h = self._canvas
        frame_count = len(self._frames)

        index_offset = FILE_HEADER_SIZE
        data_start = index_offset + frame_count * FRAME_INDEX_ENTRY_SIZE
        offsets: list[int] = []
        cursor = data_start
        for fd in self._frame_data:
            offsets.append(cursor)
            cursor += len(fd)

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.path, "wb") as f:
            f.write(pack_file_header(
                self.mode_id, frame_count, index_offset, w, h,
                codec=CODEC_FROM_NAME[self.codec],
                quality=self.quality,
            ))
            for i, fm in enumerate(self._frames):
                f.write(pack_index_entry(
                    data_offset=offsets[i],
                    patch_count=fm["patch_count"],
                    ref_frame_id=fm["ref_frame_id"],
                    metadata_len=fm["metadata_len"],
                    frame_type=fm["frame_type"],
                ))
            for fd in self._frame_data:
                f.write(fd)

        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()
        else:
            self._pool.shutdown(wait=False)
            self._closed = True
        return False

    @property
    def frame_count(self) -> int:
        """Total frames including hidden baselines."""
        return len(self._frames)

    @property
    def content_frame_count(self) -> int:
        """Content frames only (what callers see)."""
        return len(self._frames) - self._baseline_count

    @property
    def baseline_count(self) -> int:
        return self._baseline_count

    @property
    def stats(self) -> list[dict]:
        """Per-frame stats including hidden baseline frames."""
        result = []
        for i, (fm, fd) in enumerate(zip(self._frames, self._frame_data)):
            is_baseline = i in self._baseline_ids
            result.append({
                "frame_id": i,
                "content_id": None if is_baseline else i - self._baseline_count,
                "frame_type": "keyframe" if fm["frame_type"] == FRAME_KEYFRAME else "delta",
                "ref_frame_id": fm["ref_frame_id"],
                "patch_count": fm["patch_count"],
                "data_bytes": len(fd),
                "is_baseline": is_baseline,
            })
        return result

    def _inject_baselines(self, baselines: list):
        """
        Store each baseline as a hidden leading FRAME_KEYFRAME.

        Called from the first add() so self._canvas is already set.
        Baselines are resized to the canvas if their native size differs.
        """
        for idx, raw in enumerate(baselines):
            if isinstance(raw, (str, Path)):
                img = Image.open(raw).convert("RGBA")
            else:
                img = raw.convert("RGBA")

            # LANCZOS? not sure..
            if img.size != self._canvas:
                img = img.resize(self._canvas, Image.LANCZOS)

            frame_id = len(self._frames)
            cw, ch = self._canvas
            compressed = _encode_crop(img, 0, 0, cw, ch, codec=self.codec, quality=self.quality)

            meta_bytes = json.dumps(
                {"_baseline": True, "baseline_index": idx},
                separators=(",", ":"),
            ).encode()

            parts: list[bytes] = [
                pack_patch_header(0, 0, cw, ch, len(compressed)),
                compressed,
                meta_bytes,
            ]
            frame_bytes = b"".join(parts)

            self._depths.append(0)
            self._frames.append({
                "patch_count": 1,
                "ref_frame_id": frame_id,
                "metadata_len": len(meta_bytes),
                "frame_type": FRAME_KEYFRAME,
            })
            self._frame_data.append(frame_bytes)
            self._ref_images.append(img)
            self._ref_arrays.append(np.array(img, dtype=np.int16))
            self._baseline_ids.append(frame_id)

        self._baseline_count = len(self._baseline_ids)

    def _encode_frame(self, image: Image.Image, frame_id: int):
        """Route to the appropriate encoder. Returns (frame_type, ref_id, patches)."""
        if self.mode_id == MODE_MANUAL:
            return self._encode_frame_manual(image, frame_id)

        if frame_id == 0:
            return self._make_keyframe(image, frame_id)

        if self.mode_id == MODE_LTO:
            return self._encode_frame_lto(image, frame_id)

        if self.mode_id == MODE_KEYFRAME and (frame_id % self.keyframe_interval == 0):
            return self._make_keyframe(image, frame_id)

        if self.mode_id == MODE_VS_FIRST:
            ref_id = 0
        elif self.mode_id == MODE_VS_PRIOR:
            ref_id = frame_id - 1
        else:
            ref_id = frame_id - 1

        return self._diff_against(image, frame_id, ref_id)

    def _ref_chain_depth(self, frame_id: int) -> int:
        return self._depths[frame_id]

    def _make_keyframe(self, image: Image.Image, frame_id: int):
        w, h = image.size
        compressed = _encode_crop(image, 0, 0, w, h, codec=self.codec, quality=self.quality)
        return FRAME_KEYFRAME, frame_id, [(0, 0, w, h, compressed)]

    def _diff_against(self, image: Image.Image, frame_id: int, ref_id: int):
        if self.auto_keyframe_sim > 0:
            ref_arr = self._ref_arrays[ref_id]
            new_arr = np.array(image, dtype=np.int16)
            diff_arr = np.abs(new_arr - ref_arr).max(axis=2)
            total = ref_arr.shape[0] * ref_arr.shape[1]
            if (diff_arr <= self.threshold).sum() / total < self.auto_keyframe_sim:
                return self._make_keyframe(image, frame_id)

        patches = compute_patches(
            self._ref_images[ref_id], image,
            threshold=self.threshold,
            tile_size=self.tile_size,
            merge_gap=self.merge_gap,
            codec=self.codec,
            quality=self.quality,
            workers=self.workers,
        )
        return FRAME_DELTA, ref_id, patches

    def _encode_frame_lto(self, image: Image.Image, frame_id: int):
        new_arr = np.array(image, dtype=np.int16)
        total_px = new_arr.shape[0] * new_arr.shape[1]

        eligible = [
            cid for cid in range(frame_id)
            if self._depths[cid] + 1 <= self.max_ref_depth
        ]
        if not eligible:
            return self._make_keyframe(image, frame_id)

        def _score(cid: int):
            diff = np.abs(new_arr - self._ref_arrays[cid]).max(axis=2)
            sim = float((diff <= self.threshold).sum()) / total_px
            return sim, cid

        if len(eligible) <= 2 or self.workers == 1:
            scored = [_score(cid) for cid in eligible]
        else:
            scored = list(self._pool.map(_score, eligible))

        scored.sort(key=lambda x: -x[0])
        candidates = [cid for (_, cid) in scored[: self.lto_candidates]]

        def _probe(cid: int):
            patches = compute_patches(
                self._ref_images[cid], image,
                threshold=self.threshold,
                tile_size=self.tile_size,
                merge_gap=self.merge_gap,
                codec=self.codec,
                quality=self.quality,
                workers=1,
            )
            return cid, patches, sum(len(p[4]) for p in patches)

        if len(candidates) <= 1 or self.workers == 1:
            results = [_probe(cid) for cid in candidates]
        else:
            results = list(self._pool.map(_probe, candidates))

        best_ref_id, best_patches, best_size = min(results, key=lambda r: r[2])

        iw, ih = image.size
        kf_data = _encode_crop(image, 0, 0, iw, ih, codec=self.codec, quality=self.quality)
        if len(kf_data) <= best_size:
            return self._make_keyframe(image, frame_id)

        return FRAME_DELTA, best_ref_id, best_patches

    def _encode_frame_manual(self, image: Image.Image, frame_id: int):
        """
        Manual-mode encoder: probe designated baseline frames first, then fall
        back to the configured fallback mode if no baseline wins.

        Algorithm
        ---------
        1. Score all baselines by pixel similarity (numpy, cheap).
        2. Probe the top-K baselines (full patch compression, parallel).
        3. Compute the fallback result via _encode_frame_fallback().
        4. Three-way size comparison: keyframe vs best-baseline vs fallback.
           Return whichever is smallest.
        """
        new_arr = np.array(image, dtype=np.int16)
        total_px = new_arr.shape[0] * new_arr.shape[1]

        def _score_baseline(bid: int):
            diff = np.abs(new_arr - self._ref_arrays[bid]).max(axis=2)
            sim = float((diff <= self.threshold).sum()) / total_px
            return sim, bid

        if len(self._baseline_ids) <= 2 or self.workers == 1:
            scored = [_score_baseline(bid) for bid in self._baseline_ids]
        else:
            scored = list(self._pool.map(_score_baseline, self._baseline_ids))

        scored.sort(key=lambda x: -x[0])
        top_baselines = [bid for (_, bid) in scored[: self.lto_candidates]]

        def _probe_baseline(bid: int):
            patches = compute_patches(
                self._ref_images[bid], image,
                threshold=self.threshold,
                tile_size=self.tile_size,
                merge_gap=self.merge_gap,
                codec=self.codec,
                quality=self.quality,
                workers=1,
            )
            return bid, patches, sum(len(p[4]) for p in patches)

        if len(top_baselines) <= 1 or self.workers == 1:
            baseline_results = [_probe_baseline(bid) for bid in top_baselines]
        else:
            baseline_results = list(self._pool.map(_probe_baseline, top_baselines))

        best_bl_ref, best_bl_patches, best_bl_size = min(baseline_results, key=lambda r: r[2])

        fb_type, fb_ref_id, fb_patches = self._encode_frame_fallback(image, frame_id)
        fb_size = sum(len(p[4]) for p in fb_patches)

        iw, ih = image.size
        kf_size = len(_encode_crop(image, 0, 0, iw, ih, codec=self.codec, quality=self.quality))

        if kf_size <= best_bl_size and kf_size <= fb_size:
            return self._make_keyframe(image, frame_id)
        if best_bl_size <= fb_size:
            return FRAME_DELTA, best_bl_ref, best_bl_patches
        return fb_type, fb_ref_id, fb_patches

    def _encode_frame_fallback(self, image: Image.Image, frame_id: int):
        """
        Run the configured fallback_mode for this frame.

        content_id is the frame's index within the content-only sequence
        (i.e. excluding the hidden baseline keyframes at the front).
        This keeps keyframe_interval counting and vs_first anchoring correct.
        """
        content_id = frame_id - self._baseline_count

        if self.fallback_mode_id == MODE_VS_FIRST:
            if content_id == 0:
                return self._make_keyframe(image, frame_id)
            ref_id = self._baseline_count   # first content frame
            return self._diff_against(image, frame_id, ref_id)

        if self.fallback_mode_id == MODE_VS_PRIOR:
            if content_id == 0:
                return self._make_keyframe(image, frame_id)
            return self._diff_against(image, frame_id, frame_id - 1)

        if self.fallback_mode_id == MODE_KEYFRAME:
            if content_id == 0 or (content_id % self.keyframe_interval == 0):
                return self._make_keyframe(image, frame_id)
            return self._diff_against(image, frame_id, frame_id - 1)

        if self.fallback_mode_id == MODE_LTO:
            if content_id == 0:
                return self._make_keyframe(image, frame_id)
            return self._encode_frame_lto(image, frame_id)

        return self._make_keyframe(image, frame_id)

"""
impak – space-efficient image collection format based on pixel-diff patches.

Quick start
-----------
Pack a folder of PNG variants into a single file:

    import impak

    with impak.create("variants.impak", mode="vs_first") as w:
        for path in sorted(Path("frames/").glob("*.png")):
            w.add(path, name=path.stem)

Read them back:

    with impak.open("variants.impak") as r:
        print(r.info())          # human-readable summary
        img = r[0]               # PIL Image, by index
        img = r["frame_01"]      # by name
        for img in r:            # iterate all
            img.show()

Diff modes
----------
| Mode | Description |
|------|-------------|
| ``lto`` *(default)* | Each frame searches all prior frames and picks the best reference (smallest patch payload), forming a DAG of deltas. A keyframe is stored when it's smaller than the best delta. Params: ``lto_candidates`` (default 6), ``max_ref_depth`` (default 8). |
| ``vs_first`` | Every frame is a patch against frame 0. |
| ``vs_prior`` | Every frame is a patch against the immediately prior frame. |
| ``keyframe`` | Like ``vs_prior`` but stores a full image every K frames. Param: ``keyframe_interval`` (default 10). |
| ``manual`` | User-designated baselines are stored first and searched before each content frame. Params: ``baselines`` (required), ``fallback_mode`` (default ``vs_prior``). |

    Example:

        with impak.create(
            "out.impak",
            mode="manual",
            baselines=["day_bg.png", "night_bg.png"],
            fallback_mode="vs_prior",
        ) as w:
            for path in content_frames:
                w.add(path)

"""

from .encoder import ImpakWriter
from .decoder import ImpakReader
from .differ import compute_patches, reconstruct, similarity_score


def create(path, mode="lto", **kwargs) -> ImpakWriter:
    """
    Parameters
    ----------
    path              : file path (str or Path)
    mode              : "vs_first" | "vs_prior" | "keyframe" | "lto" | "manual"
    codec             : "png" or "webp" (default)
    quality           : 0-100. 100 = lossless for both codecs.
                        Values <100 enable lossy WebP (smaller, not pixel-perfect).
    keyframe_interval : (keyframe mode) full image every N frames (default 10)
    threshold         : pixel delta threshold 0-255 (default 4)
    tile_size         : diff grid tile size in px (default 32)
    merge_gap         : merge adjacent changed tiles within N px (default 8)
    auto_keyframe_sim : force keyframe when similarity drops below this (default 0.5)
    lto_candidates    : (lto / manual mode) how many top-similarity frames to
                        fully probe (default 6)
    max_ref_depth     : (lto mode) max decode-chain depth before forcing a keyframe
                        (default 8)
    baselines         : (manual mode, required) list of PIL Images or file paths
                        stored as pinned keyframes at the start of the file;
                        every content frame is diffed against these first.
    fallback_mode     : (manual mode) strategy used when no baseline wins;
                        one of "vs_first", "vs_prior", "keyframe", "lto"
                        (default "vs_prior")
    """
    return ImpakWriter(path, mode=mode, **kwargs)


def open(path, **kwargs) -> ImpakReader:
    return ImpakReader(path, **kwargs)


__all__ = [
    "create",
    "open",
    "ImpakWriter",
    "ImpakReader",
    "compute_patches",
    "reconstruct",
    "similarity_score",
]

__version__ = "0.1.0"

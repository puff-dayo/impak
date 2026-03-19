"""
cli.py – command-line interface for impak.

Commands:
  impak pack   <input_dir>  <output.impak>  [--mode manual --baselines a.png,b.png]
  impak unpack <input.impak> <output_dir>
  impak info   <input.impak>
  impak inspect <input.impak> <frame_id>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import click

from . import create, open as impak_open


def _natural_sort_key(path: Path) -> list:
    """
    Split a filename into alternating (str, int, str, int, …) chunks so that
    embedded numbers compare numerically rather than lexicographically.
    """
    parts = []
    for chunk in re.split(r"(\d+)", path.name):
        parts.append(int(chunk) if chunk.isdigit() else chunk.lower())
    return parts


@click.group()
@click.version_option()
def cli():
    """impak – patch-based image collection format."""


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output", type=click.Path())
@click.option(
    "--mode", "-m",
    type=click.Choice(["vs_first", "vs_prior", "keyframe", "lto", "manual"]),
    default="vs_prior",
    show_default=True,
    help=(
        "Diff strategy.  'lto' picks the best reference for each frame.  "
        "'manual' searches pinned baseline images first (see --baselines)."
    ),
)
@click.option("--keyframe-interval", "-k", default=10, show_default=True,
              help="(keyframe mode) full image every N frames.")
@click.option("--threshold", "-t", default=4, show_default=True,
              help="Per-channel pixel delta treated as unchanged (0=perfectly lossless).")
@click.option("--tile-size", default=32, show_default=True,
              help="Diff grid cell size in pixels.")
@click.option(
    "--codec", "-c",
    type=click.Choice(["png", "webp"]),
    default="png",
    show_default=True,
    help="Patch encoding codec.",
)
@click.option("--quality", "-q", default=100, show_default=True,
              help="Codec quality 0-100. 100=lossless for both PNG and WebP. "
                   "Values <100 enable lossy WebP (ignored for PNG).")
@click.option("--lto-candidates", "-L", default=6, show_default=True,
              help="(lto / manual mode) Number of top-similarity frames to fully probe.")
@click.option("--max-ref-depth", "-D", default=8, show_default=True,
              help="(lto mode) Max decode-chain depth before forcing a keyframe.")
@click.option("--ext", default="png,jpg,jpeg,webp,bmp",
              help="Comma-separated image extensions to include.")
@click.option(
    "--baselines", "-b", "baseline_paths",
    default=None,
    help=(
        "(manual mode) Comma-separated list of image file paths that act as "
        "pinned reference frames.  These are stored as leading keyframes and "
        "searched before each content frame.  Required when --mode=manual."
    ),
)
@click.option(
    "--fallback-mode", "-F",
    type=click.Choice(["vs_first", "vs_prior", "keyframe", "lto"]),
    default="vs_prior",
    show_default=True,
    help=(
        "(manual mode) Diff strategy used when no baseline produces a smaller "
        "patch set.  Ignored for other modes."
    ),
)
@click.option("--verbose", "-v", is_flag=True)
def pack(
    input_dir, output, mode, keyframe_interval, threshold, tile_size,
    codec, quality, lto_candidates, max_ref_depth, ext,
    baseline_paths, fallback_mode, verbose,
):
    """Pack all images in INPUT_DIR into a single OUTPUT .impak file.

    Images are sorted by filename (natural/numeric order) before packing,
    so S2.png comes before S10.png.

    \b
    Manual mode example:
        impak pack frames/ out.impak --mode manual \\
              --baselines day.png,night.png --fallback-mode vs_prior
    """
    baselines: list = []
    if mode == "manual":
        if not baseline_paths:
            raise click.UsageError(
                "--baselines is required when --mode=manual.  "
                "Provide a comma-separated list of image paths."
            )
        raw_paths = [p.strip() for p in baseline_paths.split(",") if p.strip()]
        missing = [p for p in raw_paths if not Path(p).exists()]
        if missing:
            raise click.UsageError(
                f"Baseline file(s) not found: {', '.join(missing)}"
            )
        baselines = [Path(p) for p in raw_paths]

    exts = {f".{e.strip().lower()}" for e in ext.split(",")}
    paths = sorted(
        (p for p in Path(input_dir).iterdir() if p.suffix.lower() in exts),
        key=_natural_sort_key,
    )

    if not paths:
        click.echo(f"No images found in {input_dir}", err=True)
        sys.exit(1)

    if mode == "lto":
        mode_detail = f"  candidates={lto_candidates}  depth={max_ref_depth}"
    elif mode == "manual":
        bl_names = ", ".join(p.name for p in baselines)
        mode_detail = (
            f"  baselines=[{bl_names}]"
            f"  fallback={fallback_mode}"
            f"  candidates={lto_candidates}"
        )
    else:
        mode_detail = ""

    click.echo(
        f"Packing {len(paths)} images → {output}"
        f"  [mode={mode}  codec={codec}  quality={quality}{mode_detail}]"
    )

    writer_kwargs = dict(
        mode=mode,
        keyframe_interval=keyframe_interval,
        threshold=threshold,
        tile_size=tile_size,
        codec=codec,
        quality=quality,
        lto_candidates=lto_candidates,
        max_ref_depth=max_ref_depth,
    )
    if mode == "manual":
        writer_kwargs["baselines"] = baselines
        writer_kwargs["fallback_mode"] = fallback_mode

    with create(output, **writer_kwargs) as w:
        if mode == "manual" and baselines:
            click.echo(f"  Injected {len(baselines)} baseline keyframe(s).")
        with click.progressbar(paths, label="Encoding") as bar:
            for p in bar:
                w.add(p, name=p.stem)

    size_kb = Path(output).stat().st_size / 1024
    if verbose:
        for s in w.stats:
            ftype = s["frame_type"]
            baseline_tag = " [baseline]" if s.get("is_baseline") else ""
            click.echo(
                f"  [{s['frame_id']:>4}] {ftype:>9}  ref={s['ref_frame_id']:>4}  "
                f"patches={s['patch_count']:>4}  {s['data_bytes']:>9,} B{baseline_tag}"
            )

    click.echo(f"Done. File size: {size_kb:.1f} KB")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_dir", type=click.Path())
@click.option("--format", "-f", "fmt",
              type=click.Choice(["png", "jpg", "webp"]), default="png",
              show_default=True)
@click.option("--frames", "-n", default=None,
              help="Comma-separated frame indices to extract (default: all).")
def unpack(input_file, output_dir, fmt, frames):
    """Extract all (or selected) images from an .impak file into OUTPUT_DIR."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with impak_open(input_file) as r:
        total = len(r)
        if frames:
            indices = [int(x.strip()) for x in frames.split(",")]
        else:
            indices = list(range(total))

        click.echo(f"Extracting {len(indices)}/{total} frames → {output_dir}")
        with click.progressbar(indices, label="Decoding") as bar:
            for i in bar:
                img = r[i].convert("RGB") if fmt == "jpg" else r[i]
                meta = r.get_metadata(i)
                name = meta.get("name", f"frame_{i:04d}")
                out_path = out / f"{name}.{fmt}"
                img.save(out_path)

    click.echo("Done.")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
def info(input_file):
    """Print a summary of an .impak collection."""
    with impak_open(input_file) as r:
        click.echo(r.info())


@cli.command()
def meow():
    """Print a cat greeting."""
    click.echo("Meow meow.")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("frame_id", type=int)
@click.option("--save-diff", type=click.Path(),
              help="Save a visualisation of changed regions to this PNG path.")
def inspect(input_file, frame_id, save_diff):
    """Show patch rectangles for a single frame.

    Optionally save a visualisation highlighting the changed regions.
    """
    with impak_open(input_file) as r:
        if frame_id >= len(r):
            click.echo(f"Error: frame {frame_id} out of range (0..{len(r)-1})", err=True)
            sys.exit(1)

        patches = r.diff_map(frame_id)
        meta = r.get_metadata(frame_id)
        name = meta.get("name", f"frame_{frame_id}")
        click.echo(f"Frame {frame_id}  name={name!r}")
        click.echo(f"  {len(patches)} changed region(s):")
        for i, (x, y, w, h) in enumerate(patches):
            click.echo(f"    [{i:>3}]  x={x:>5} y={y:>5}  {w:>5}×{h:<5}")

        if save_diff:
            _save_diff_vis(r, frame_id, patches, save_diff)
            click.echo(f"Diff visualisation saved to {save_diff}")


def _save_diff_vis(reader, frame_id, patches, out_path):
    """Draw red rectangles on the decoded frame to highlight changed regions."""
    from PIL import ImageDraw
    img = reader[frame_id].convert("RGBA")
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    for (x, y, w, h) in patches:
        draw.rectangle([x, y, x + w - 1, y + h - 1],
                       fill=(255, 0, 0, 60),
                       outline=(255, 0, 0, 200),
                       width=2)
    overlay.save(out_path)


if __name__ == "__main__":
    cli()
# impak

Space-efficient patch-based image collection format.

## Install

`impak` is available from [PyPI](https://pypi.org/project/impak/).

```commandline
pip install impak               # decoder only (Pillow)
pip install impak[encoder]      # + numpy – encoding support (ImpakWriter)
pip install impak[cli]          # + click – CLI tool (impak pack/unpack/info)
pip install impak[all]          # both encoder and CLI
```

The base installation is decoder-only — `impak.open()`, `ImpakReader`, and `reconstruct` work with only `Pillow` as a dependency. No `numpy` or `click` needed.

## Example

See [example.py](./example.py) for more.

```python
import impak
from pathlib import Path

paths = sorted(Path("frames/").glob("*.png"))

# Encode
with impak.create("out.impak", mode="vs_first", codec="webp", quality=100) as w:
    for p in paths:
        w.add(p, name=p.stem)

# Decode
with impak.open("out.impak") as r:
    print(r.info())
    img = r[0]
    img = r["frame_01"]
    for img in r:
        img.show()
```

Run `impak --help`, or check documentation in the `/docs` folder.

## Build

```commandline
uv pip install build twine setuptools wheel / uv sync --all-extras
uv build
twine check dist/*
uv pip install dist/impak-xxxx.whl
```

## License

`impak` is licensed under the GNU Affero General Public License v3.0.

See `LICENSE` for full text.

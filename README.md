# impak

Space-efficient patch-based image collection format.

## Example

```python
import re
from tqdm import tqdm
import impak
from pathlib import Path

def natural_sort_key(path: Path) -> list:
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", path.name)]

paths = sorted(Path("Ganyu X Slime/").glob("*.png"), key=natural_sort_key)

# A fast and straight forward, automatic method. The default method.
# Possibly produces a smallest file, but it really depends.
with impak.create("Ganyu X Slime_lto.impak",
                  mode="lto", codec="webp", quality=95
                  ) as pack:
    for path in tqdm(paths, desc="Encoding using impak"):
        pack.add(path, name=path.stem)
```

## Usage

Run `impak --help`, or check documentations in `/docs` folder.

## Build

```commandline
uv pip install build twine setuptools wheel
uv run --active python -m build
twine check dist/*
uv pip install dist/impak-xxxx.whl
uv run --active python -m twine upload dist/*
```

## License

`impak` is licensed under the GNU Affero General Public License v3.0.

See `LICENSE` for full text.

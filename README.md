# impak

Space-efficient patch-based image collection format.

This is the full function package for `impak`, providing encoding, decoding, and API/CLI usage. For a lightweight decoder-only Python package, see [impak-decoder](https://github.com/puff-dayo/impak-app/tree/main/impak-decoder).

## Example

See [example.py](./example.py) for more example.

```python
# A slightly slow, automatic method. The default method.
# Possibly produces a smallest impak file, but it really depends.
with impak.create("Ganyu X Slime_lto.impak",
                  mode="lto", codec="webp", quality=95
                  ) as pack:
    for path in tqdm(paths, desc="Encoding using impak"):
        pack.add(path, name=path.stem)

# A fast and straight forward, automatic method.
# Results in a slightly larger impak file.
with impak.create("Ganyu X Slime_prior.impak",
                  mode="vs_prior", codec="webp", quality=95
                  ) as pack:
    for path in tqdm(paths, desc="Encoding using impak"):
        pack.add(path, name=path.stem)
```

## Usage

Package `impak` is available from [pypi](https://pypi.org/project/impak/) using pip.

Run `impak --help`, or check documentations in `/docs` folder.

## Build

```commandline
uv pip install build twine setuptools wheel
uv run --active python -m build
twine check dist/*
uv pip install dist/impak-xxxx.whl
```

## License

`impak` is licensed under the GNU Affero General Public License v3.0.

See `LICENSE` for full text.

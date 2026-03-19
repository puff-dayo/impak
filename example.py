import re

from tqdm import tqdm

import impak
from pathlib import Path

def natural_sort_key(path: Path) -> list:
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", path.name)]

paths = sorted(Path("Ganyu X Slime/").glob("*.png"), key=natural_sort_key)


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


# Kind of slow but if more control is wanted.
with impak.create("Ganyu X Slime_manual.impak",
                  mode="manual", fallback_mode="vs_prior",
                  baselines=["Ganyu X Slime/Sept1.png",
                             "Ganyu X Slime/Sept12.png"],
                  codec="webp", quality=95
                  ) as pack:
    for path in tqdm(paths, desc="Encoding using impak"):
        pack.add(path, name=path.stem)

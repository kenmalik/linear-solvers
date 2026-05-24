from pathlib import Path
import re
from dataclasses import dataclass

import pandas as pd

# Expect the following file naming convention:
# timings_<implementation>_<algorithm>_s<block_size>.csv
# _s<block_size> is optional
FILE_PATTERN = re.compile(r"timings_(\w+)_([a-z-]+)(?:_s(\d+))?\.csv")


@dataclass
class BenchmarkFile:
    implementation: str
    algorithm: str
    block_size: int | None
    data: pd.DataFrame


def read_file(file: Path) -> BenchmarkFile:
    assert file.is_file()

    match = FILE_PATTERN.search(str(file))

    if not match:
        raise Exception("invalid file name")

    implementation = match.group(1)
    algorithm = match.group(2)
    block_size = int(match.group(3)) if match.group(3) else None
    data = pd.read_csv(file, index_col="Range")

    return BenchmarkFile(implementation, algorithm, block_size, data)


def read_dir(dir: Path) -> list[BenchmarkFile]:
    assert dir.is_dir()

    return [read_file(f) for f in dir.glob("*.csv")]

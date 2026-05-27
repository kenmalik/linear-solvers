from pathlib import Path
import re
from typing import Any
from collections.abc import Iterable

import pandas as pd

# Expect the following file naming convention:
# timings_<implementation>_<algorithm>_s<block_size>.csv
# _s<block_size> is optional
FILE_PATTERN = re.compile(r"timings_(\w+)_([a-z-]+)(?:_s(\d+))?\.csv")


def read_file(file: Path) -> pd.DataFrame:
    assert file.is_file()

    match = FILE_PATTERN.search(str(file))

    if not match:
        raise Exception("invalid file name")

    df = pd.read_csv(file)
    df["implementation"] = match.group(1)
    df["algorithm"] = match.group(2)
    df["block_size"] = int(match.group(3)) if match.group(3) else None

    return df


def read_dir(dir: Path) -> pd.DataFrame:
    assert dir.is_dir()

    return pd.concat([read_file(f) for f in dir.glob("*.csv")], ignore_index=True)


def range_runtimes_by_block_size(
    df: pd.DataFrame, ranges: Iterable[Any] | None = None
) -> pd.DataFrame:
    """
    Restructures data to list average runtimes of ranges in milliseconds,
    indexed by block_size.
    """

    if df["implementation"].nunique() > 1:
        raise ValueError("solver implementations do not match")
    if not df["block_size"].all():
        raise ValueError("some block sizes are not defined")

    if ranges:
        df = df[df["Range"].isin(ranges)]
    df = df.pivot(index="block_size", columns="Range", values="Avg (ms)")

    return df


def process_dr_bcg(
    df: pd.DataFrame, ranges: Iterable[Any] | None = None
) -> pd.DataFrame:
    iterations = df[df["Range"] == "iteration"][["block_size", "Instances"]].set_index(
        "block_size"
    )
    df = range_runtimes_by_block_size(df, ranges=ranges)
    df["iterations"] = iterations
    return df

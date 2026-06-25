from pathlib import Path
import re
from collections.abc import Iterable, Iterator

import pandas as pd

from parser_types import LabeledSource, LabeledData

# Expect the following file naming convention:
# timings_<implementation>_<algorithm>_s<block_size>.csv
# _s<block_size> is optional
FILE_PATTERN = re.compile(r"timings_(\w+)_([a-z-]+)(?:_s(\d+))?\.csv")
MS_PER_SEC = 1000


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


def restructure_dr_bcg_data(
    df: pd.DataFrame, ranges: Iterable[str] | None = None
) -> pd.DataFrame:
    """
    Restructures data to list average runtimes of ranges in milliseconds,
    indexed by block_size.
    """

    if df["implementation"].nunique() > 1:
        raise ValueError("solver implementations do not match")
    if not df["block_size"].all():  # type: ignore
        raise ValueError("some block sizes are not defined")

    if ranges:
        df = df[df["Range"].isin(ranges)]  # type: ignore
    df = df.pivot(index="block_size", columns="Range", values="Avg (ms)")

    return df


def range_runtimes_by_block_size(
    df: pd.DataFrame, ranges: Iterable[str] | None = None
) -> pd.DataFrame:
    iterations = df[df["Range"] == "iteration"][["block_size", "Instances"]].set_index(  # type: ignore
        "block_size"
    )
    df = restructure_dr_bcg_data(df, ranges=ranges)
    df["iterations"] = iterations
    return df


def has_duplicate_names(sources: Iterable[LabeledSource]):
    names = [name for _, name in sources if name]
    return len(names) != len(set(names))


def process_dr_bcg_dirs(
    sources: Iterable[LabeledSource], ranges: Iterable[str] | None = None
) -> Iterator[LabeledData]:
    """Yields label and processed DR-BCG data from LabeledSources."""
    for source, label in sources:
        yield label, range_runtimes_by_block_size(read_dir(source), ranges=ranges)

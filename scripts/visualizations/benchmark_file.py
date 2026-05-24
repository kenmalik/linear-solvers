from pathlib import Path
import re

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

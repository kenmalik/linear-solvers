from pathlib import Path
import re
import pandas as pd

FILE_PATTERN = re.compile(r"timings_(\w+)_([a-z-]+)_s(\d+).csv")


def read(dir: Path) -> pd.DataFrame:
    if not dir.is_dir():
        raise Exception("invalid data directory")

    data = {}

    for file in dir.glob("*.csv"):
        match = FILE_PATTERN.search(str(file))

        if not match:
            raise Exception("invalid file name")

        block_size = int(match.group(3))
        data.setdefault("block_size", []).append(block_size)

        df = pd.read_csv(file, index_col="Range")
        for r in ("solve", "iteration", "[w sigma] = QR(temp)", "[w zeta] = QR(w)"):
            data.setdefault(r, []).append(df.loc[r]["Avg (ms)"])
        data.setdefault("iteration_instances", []).append(
            df.loc["iteration"]["Instances"]
        )

    return pd.DataFrame(data).set_index("block_size")

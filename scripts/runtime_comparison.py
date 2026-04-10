import argparse
from io import StringIO
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GROUP_ORDER = ["CG", "1", "2", "4", "8", "16", "32", "64"]


def parse_name(name):
    if re.match(r"BM_CgMkl", name):
        return "MKL", "CG"
    if re.match(r"BM_CgCuda", name):
        return "CUDA", "CG"
    m = re.match(r"BM_DrBcgMkl/(\d+)", name)
    if m:
        return "MKL", m.group(1)
    m = re.match(r"BM_DrBcgCuda/(\d+)", name)
    if m:
        return "CUDA", m.group(1)
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output")

    args = parser.parse_args()

    json = sys.stdin.read().strip()
    data = pd.read_json(StringIO(json), orient="records")

    unit = str(data["time_unit"][0])

    data[["impl", "group"]] = data["name"].apply(lambda n: pd.Series(parse_name(n)))
    data = data.dropna(subset=["impl", "group"])

    mkl = data[data["impl"] == "MKL"].set_index("group")["time"]
    cuda = data[data["impl"] == "CUDA"].set_index("group")["time"]

    groups = [g for g in GROUP_ORDER if g in mkl.index or g in cuda.index]
    x = np.arange(len(groups))
    width = 0.4

    _, ax = plt.subplots()
    ax.bar(x - width / 2, [mkl.get(g, 0) for g in groups], width, label="MKL")  # type: ignore
    ax.bar(x + width / 2, [cuda.get(g, 0) for g in groups], width, label="CUDA")  # type: ignore

    ax.set_title("Runtime Comparison of Solver Implementations")
    ax.set_ylabel(unit)
    ax.set_xlabel("Block Size")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()

import argparse
from io import StringIO
import re
import sys

from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

    mkl_data = data[data["impl"] == "MKL"].set_index("group")
    cuda_data = data[data["impl"] == "CUDA"].set_index("group")
    mkl = mkl_data["time"]
    cuda = cuda_data["time"]
    mkl_iters = mkl_data["iters"]
    cuda_iters = cuda_data["iters"]

    groups = [g for g in GROUP_ORDER if g in mkl.index or g in cuda.index]
    x = np.arange(len(groups))
    width = 0.4

    row_labels = [f"MKL ({unit})", "MKL (iters)", f"CUDA ({unit})", "CUDA (iters)"]
    table_data = [
        [f"{mkl.get(g, 0):.0f}" for g in groups],
        [f"{mkl_iters.get(g, 0):.0f}" for g in groups],
        [f"{cuda.get(g, 0):.0f}" for g in groups],
        [f"{cuda_iters.get(g, 0):.0f}" for g in groups],
    ]

    fig = plt.figure()
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[0.15, 0.85],
        height_ratios=[4, 1],
        hspace=0,
        wspace=0,
    )
    ax_side = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])
    ax_rlabels = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1])
    ax_side.axis("off")
    ax_rlabels.axis("off")
    ax_table.axis("off")

    ax.bar(x - width / 2, [mkl.get(g, 0) for g in groups], width, label="MKL")  # type: ignore
    ax.bar(x + width / 2, [cuda.get(g, 0) for g in groups], width, label="CUDA")  # type: ignore
    ax.set_title("Runtimes of Solver Implementations by Block Size")
    ax.set_ylabel(unit)
    ax.set_xticks([])
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    n_rows = len(row_labels) + 1  # +1 for header row
    for i, label in enumerate(row_labels):
        y = 1 - (i + 1.5) / n_rows
        ax_rlabels.text(
            0.95,
            y,
            label,
            ha="right",
            va="center",
            transform=ax_rlabels.transAxes,
            fontsize=8,
        )

    tbl = ax_table.table(
        cellText=table_data,
        colLabels=groups,
        loc="center",
        bbox=Bbox([[0, 0], [1, 1]]),
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for cell in tbl.get_celld().values():
        cell.set_edgecolor(plt.rcParams["axes.edgecolor"])
        cell.set_linewidth(plt.rcParams["axes.linewidth"])

    fig.subplots_adjust(left=0, right=0.98, top=0.93, bottom=0.02)

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()

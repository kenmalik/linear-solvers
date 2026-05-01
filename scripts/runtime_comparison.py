import argparse
from pathlib import Path
import re
import sys

from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

GROUP_ORDER = ["CG", "1", "2", "4", "8", "16", "32", "64"]
FILENAME_PATTERN = re.compile(
    r"^(?P<impl>mkl|cuda)_(?P<alg>cg|dr-bcg)(?:_s(?P<block_size>\d+))?\.csv$"
)
REQUIRED_COLUMNS = {"Range", "Total (ms)", "Avg (ms)", "Instances"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", nargs="?", default="out", type=Path)
    parser.add_argument("-o", "--output")
    return parser.parse_args()


def parse_file_name(path: Path):
    match = FILENAME_PATTERN.match(path.name)
    if not match:
        return None

    impl = match.group("impl").upper()
    alg = match.group("alg")
    block_size = match.group("block_size")

    if alg == "cg":
        if block_size is not None:
            return None
        group = "CG"
    else:
        if block_size is None:
            return None
        group = block_size

    return {"impl": impl, "group": group}


def load_runtime_row(path: Path, meta):
    data = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"missing required columns: {', '.join(sorted(missing_columns))}"
        )

    iteration_rows = data[data["Range"] == "iteration"]
    if iteration_rows.empty:
        raise ValueError("missing iteration row")

    iteration = iteration_rows.iloc[0]
    return {
        "impl": meta["impl"],
        "group": meta["group"],
        "avg_time": float(iteration["Avg (ms)"]),
        "iters": int(iteration["Instances"]),
        "time_unit": "ms",
    }


def read_directory(input_dir: Path):
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"{input_dir} is not a valid directory")

    rows = []
    warnings = []

    for path in sorted(input_dir.glob("*.csv")):
        meta = parse_file_name(path)
        if meta is None:
            warnings.append(
                f"Skipping {path.name}: filename does not match documented convention"
            )
            continue

        try:
            rows.append(load_runtime_row(path, meta))
        except Exception as exc:
            warnings.append(f"Skipping {path.name}: {exc}")

    for warning in warnings:
        print(warning, file=sys.stderr)

    if not rows:
        raise ValueError(f"no valid runtime CSV files found in {input_dir}")

    return pd.DataFrame(rows)


def plot(data: pd.DataFrame, output: str | None):
    unit = str(data["time_unit"].iloc[0])
    avg_unit = f"{unit}/iter"

    mkl_data = data[data["impl"] == "MKL"].set_index("group")
    cuda_data = data[data["impl"] == "CUDA"].set_index("group")
    mkl = mkl_data["avg_time"]
    cuda = cuda_data["avg_time"]
    mkl_iters = mkl_data["iters"]
    cuda_iters = cuda_data["iters"]

    groups = [g for g in GROUP_ORDER if g in mkl.index or g in cuda.index]
    x = np.arange(len(groups))
    width = 0.4

    row_labels = [
        f"MKL ({avg_unit})",
        "MKL (iters)",
        f"CUDA ({avg_unit})",
        "CUDA (iters)",
    ]
    table_data = [
        [f"{mkl.get(g, 0):.3f}" for g in groups],
        [f"{mkl_iters.get(g, 0):.0f}" for g in groups],
        [f"{cuda.get(g, 0):.3f}" for g in groups],
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
    ax.set_title("Average Cost per Iteration by Solver Implementation")
    ax.set_ylabel(avg_unit)
    ax.set_xticks([])
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    n_rows = len(row_labels) + 1
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

    if output:
        plt.savefig(output)
    else:
        plt.show()


def main():
    args = parse_args()
    data = read_directory(args.input_dir)
    plot(data, args.output)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from benchmark_file import read_dir, read_file, process_dr_bcg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--cg", type=Path)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()

    # Use directory name as name of "variant" that is being plotted
    data = {
        s.stem: process_dr_bcg(read_dir(s), ranges=("solve",)) for s in args.sources
    }
    cg_data = read_file(args.cg) if args.cg else None
    plot(data, cg_data, args.output)


def plot(
    data: dict[str, pd.DataFrame], cg_data: pd.DataFrame | None, output: str | None
) -> None:
    assert data
    variants = list(data.keys())

    block_sizes = sorted({int(bs) for df in data.values() for bs in df.index.tolist()})
    x = list(range(len(block_sizes)))
    width = 0.35

    offsets = {
        name: (index - (len(variants) - 1) / 2) * width
        for index, name in enumerate(variants)
    }

    _, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_box_aspect(1 / 1.5)

    for name in variants:
        df = data[name]
        heights = [df.loc[bs, "solve"] if bs in df.index else 0 for bs in block_sizes]
        ax.bar([v + offsets[name] for v in x], heights, width, label=name)  # type: ignore

    col_labels: list[str]
    cg_ms: float | None = None
    cg_iters: int | None = None

    if cg_data is not None:
        cg_x = -1
        ax.bar(
            cg_x,
            cg_data.loc[cg_data["Range"] == "solve", "Avg (ms)"].iloc[0],
            width,
            label="CG",
        )
        cg_ms = float(cg_data.loc[cg_data["Range"] == "solve", "Avg (ms)"].iloc[0])
        cg_iters = int(
            cg_data.loc[cg_data["Range"] == "iteration", "Instances"].iloc[0]
        )
        col_labels = ["CG"] + [str(bs) for bs in block_sizes]
    else:
        col_labels = [str(bs) for bs in block_sizes]

    ax.set_xticks([])
    ax.set_ylabel("Avg (ms)")
    ax.set_title("Solve Runtime by Block Size")
    ax.legend()

    row_labels: list[str] = []
    cell_text: list[list[str]] = []
    for name in variants:
        df = data[name]
        ms_row: list[str] = []
        iters_row: list[str] = []
        if cg_ms is not None and cg_iters is not None:
            ms_row.append(str(int(round(cg_ms))))
            iters_row.append(str(cg_iters))
        for bs in block_sizes:
            ms_row.append(
                str(int(round(df.loc[bs, "solve"]))) if bs in df.index else ""  # type: ignore
            )
            iters_row.append(
                str(int(round(df.loc[bs, "iterations"]))) if bs in df.index else ""  # type: ignore
            )
        row_labels += [f"{name} (ms)", f"{name} (iters)"]
        cell_text += [ms_row, iters_row]

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="bottom",
        cellLoc="center",
    )
    table.scale(1, 1.5)
    for (_, col), cell in table.get_celld().items():
        if col == -1:
            cell.set_linewidth(0)
            cell.get_text().set_ha("right")  # type: ignore

    plt.subplots_adjust(bottom=0.05 * (2 * len(variants) + 1))

    if output:
        plt.savefig(output)
        print(output)
    else:
        plt.show()


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
from itertools import product

import pandas as pd
import matplotlib.pyplot as plt

from parser_types import default_labeled_source, path_name_pair
from benchmark_file import read_file, process_dr_bcg_dirs, MS_PER_SEC


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--cg", type=Path)
    parser.add_argument("sources", type=path_name_pair, nargs="+")
    args = parser.parse_args()

    sources = default_labeled_source(args.sources)
    data = {
        label: data for label, data in process_dr_bcg_dirs(sources, ranges=("solve",))
    }

    cg_data = read_file(args.cg).set_index("Range") if args.cg else None

    plot(data, cg_data, args.output)


def plot(
    data: dict[str, pd.DataFrame], cg_data: pd.DataFrame | None, output: str | None
) -> None:
    x_labels = [str(idx) for idx in next(iter(data.values())).index]

    solve_times = {label: df["solve"] / MS_PER_SEC for label, df in data.items()}
    solve_iters = {label: df["iterations"] for label, df in data.items()}

    _, ax = plt.subplots(figsize=(11, 8.5))

    group_spacing = 1
    if cg_data is not None:
        cg_runtime = cg_data.loc["solve"]["Avg (ms)"] / MS_PER_SEC
        cg_iters = cg_data.loc["iteration"]["Instances"]
        ax.axhline(
            cg_runtime,
            color="darkslategrey",
            label="CG",
            linestyle="--",
        )
        ax.text(
            x=1.01,
            y=cg_runtime,
            s=f"{cg_runtime:.3f} s",
            va="bottom",
            ha="left",
            transform=ax.get_yaxis_transform(),
        )
        ax.text(
            x=1.01,
            y=cg_runtime,
            s=f"({cg_iters} iters)",
            va="top",
            ha="left",
            transform=ax.get_yaxis_transform(),
        )

    bar_groups = ax.grouped_bar(
        solve_times,  # type: ignore
        tick_labels=x_labels,
        group_spacing=group_spacing,
    )

    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.5)
    ax.set_xticks([])
    ax.set_ylabel("Avg (s)")
    ax.set_title("Solve Runtime by Block Size")
    ax.legend()

    # Populate table
    row_labels = [
        f"{label} ({metric})"
        for label, metric in product(solve_times.keys(), ("s", "iters"))
    ]
    cell_text: list[list[str]] = []
    for label in solve_times:
        times = ["%.3f" % x for x in solve_times[label]]
        iters = [str(x) for x in solve_iters[label]]
        cell_text.append(times)
        cell_text.append(iters)

    # Position table
    first_bar = bar_groups.bar_containers[0][0]  # type: ignore
    x, _ = ax.transLimits.transform(first_bar.get_xy())

    x_min, x_max = ax.get_xlim()
    bar_width_normalized = first_bar.get_width() / (x_max - x_min)

    table_height = 0.3
    table_bbox = [
        x - bar_width_normalized / 2,
        -table_height,
        1 - x,
        table_height,
    ]
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=x_labels,
        loc="bottom",
        cellLoc="center",
        bbox=table_bbox,  # type: ignore
    )
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("darkgrey")
        if row > 0 and row % 2 != 0 and col >= 0:
            cell.set_facecolor("gainsboro")
        if col == -1:
            cell.set_linewidth(0)
            cell.set_text_props(ha="right", weight="bold")

    plt.tight_layout()
    if output:
        plt.savefig(output)
        print(output)
    else:
        plt.show()


if __name__ == "__main__":
    main()

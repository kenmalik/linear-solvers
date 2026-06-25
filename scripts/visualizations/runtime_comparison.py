from pathlib import Path
import sys
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from benchmark_file import (
    process_dr_bcg_dirs,
    read_file,
    has_duplicate_names,
)
from parser_types import (
    path_name_pair,
    existing_file,
    existing_parent,
    default_labeled_source,
    LabeledSource,
)


MS_PER_SEC: int = 1_000


class Args(argparse.Namespace):
    sources: list[LabeledSource]
    output: Path | None
    title: str
    cg: Path | None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sources",
        type=path_name_pair,
        nargs="+",
        help="Directories containing DR-BCG timing outputs by block size",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=existing_parent,
        help="Output file for plot",
    )
    parser.add_argument("-t", "--title", help="Title of plot")
    parser.add_argument(
        "--cg", type=existing_file, help="CG timing output for comparison"
    )
    args = parser.parse_args(namespace=Args())

    cg_times = read_file(args.cg).set_index("Range") if args.cg else None

    if has_duplicate_names(args.sources):
        print(
            "error: duplicate label",
            file=sys.stderr,
        )
        exit(-1)

    sources = default_labeled_source(args.sources)
    data = {
        label: data
        for label, data in process_dr_bcg_dirs(sources, ranges=("solve", "iteration"))
    }

    plot(data, cg_times, args.output, args.title)


def plot(
    data: dict[str, pd.DataFrame],
    cg_times: pd.DataFrame | None,
    output: Path | None,
    title: str | None,
):
    x_labels = [str(idx) for idx in next(iter(data.values())).index]

    solve_times = {label: df["solve"] / MS_PER_SEC for label, df in data.items()}
    iter_times = {label: df["iteration"] for label, df in data.items()}
    iterations = {label: df["iterations"] for label, df in data.items()}

    fig, ax = plt.subplots(3, figsize=(8, 6))

    ax[0].grouped_bar(solve_times, tick_labels=x_labels, group_spacing=1)  # type: ignore
    ax[1].grouped_bar(iter_times, tick_labels=x_labels, group_spacing=1)  # type: ignore
    ax[2].grouped_bar(iterations, tick_labels=x_labels, group_spacing=1)  # type: ignore

    if cg_times is not None:
        ax[0].axhline(
            y=cg_times.loc["solve"]["Avg (ms)"] / MS_PER_SEC,
            color="mediumseagreen",
            linestyle="--",
            lw=1,
        )
        ax[1].axhline(
            y=cg_times.loc["iteration"]["Avg (ms)"],
            color="mediumseagreen",
            linestyle="--",
            lw=1,
        )
        ax[2].axhline(
            y=cg_times.loc["iteration"]["Instances"],
            color="mediumseagreen",
            linestyle="--",
            lw=1,
        )

    ax[-1].set_xlabel("Block Size")

    ax[0].set_ylabel("Total (s)")
    ax[1].set_ylabel("Iteration (ms)")
    ax[2].set_ylabel("# Iterations")

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=len(data))

    fig.suptitle(title if title else "DR-BCG Runtime Comparison")

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    if output:
        plt.savefig(output)
        print(output, file=sys.stderr)
    else:
        plt.show()


if __name__ == "__main__":
    main()

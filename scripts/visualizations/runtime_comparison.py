from benchmark_file import read_dir, process_dr_bcg

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import sys
import argparse

type NamedSource = tuple[Path, str | None]
type LabeledData = dict[str, pd.DataFrame]

MS_PER_SEC: int = 1_000


class Args(argparse.Namespace):
    sources: list[NamedSource]
    output: Path | None
    title: str


def existing_parent(arg: str) -> Path:
    p = Path(arg)

    if not p.parent.exists():
        raise argparse.ArgumentTypeError(f"parent directory does not exist: {arg}")

    return p


def path_name_pair(arg: str) -> NamedSource:
    vals = arg.split("=")
    if len(vals) > 2:
        raise argparse.ArgumentTypeError(f"too many segments in path-name pair: {arg}")

    p = Path(vals[0])
    if not p.exists():
        raise argparse.ArgumentTypeError(f"file not found: {arg}")

    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"file is not a directory: {arg}")

    if len(vals) == 1:
        return (p, None)

    label = vals[1].replace("_", " ").capitalize()
    return (p, label)


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
    args = parser.parse_args(namespace=Args())

    skipped: list[Path] = []

    data: dict[str, pd.DataFrame] = {}
    for source, label in args.sources:
        label = label if label else source.stem

        if label in data:
            print(
                f"error when processing {source}: duplicate label {label}. skipping",
                file=sys.stderr,
            )
            skipped.append(source)
            continue

        data[label] = process_dr_bcg(read_dir(source), ranges=("solve", "iteration"))

    plot(data, args.output, args.title)

    if skipped:
        print("skipped sources:", file=sys.stderr)
    for source in skipped:
        print(source, file=sys.stderr)


def plot(data: LabeledData, output: Path | None, title: str | None):
    x_labels = [str(idx) for idx in next(iter(data.values())).index]

    solve_times = {label: df["solve"] / MS_PER_SEC for label, df in data.items()}
    iter_times = {label: df["iteration"] for label, df in data.items()}
    iterations = {label: df["iterations"] for label, df in data.items()}

    fig, ax = plt.subplots(3, figsize=(8, 6))

    ax[0].grouped_bar(solve_times, tick_labels=x_labels, group_spacing=1)  # type: ignore
    ax[1].grouped_bar(iter_times, tick_labels=x_labels, group_spacing=1)  # type: ignore
    ax[2].grouped_bar(iterations, tick_labels=x_labels, group_spacing=1)  # type: ignore

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

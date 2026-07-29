import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from parser_types import existing_file

custom_colors = (
    "palevioletred",
    "cornflowerblue",
    "lightcoral",
    "peru",
    "gold",
    "olivedrab",
    "darkseagreen",
    "lightseagreen",
    "lightblue",
    "tan",
    "darksalmon",
)

IGNORE_RANGES = [".*:.*", "solve", "iteration"]
IGNORE_PATTERN = "|".join(IGNORE_RANGES)


class Args(argparse.Namespace):
    output: Path | None
    source: Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("source", type=existing_file)
    args = parser.parse_args(namespace=Args())

    data = pd.read_csv(args.source, index_col="Range")
    data = data[~data.index.str.contains(IGNORE_PATTERN, na=False)]
    data = data.loc[::-1]

    fig, ax = plt.subplots()

    bottom = 0
    for rng, _, average_runtime, _ in data.itertuples(index=True):
        ax.bar(
            "Runtime",
            average_runtime,
            bottom=bottom,
            label=rng,
        )
        bottom += average_runtime

    fig.suptitle("Runtime Breakdown")
    ax.set(ylabel="Avg (ms) per iteration", xticks=[])

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(
        handles[::-1],
        labels[::-1],
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
        fontsize="small",
    )

    fig.set_figwidth(7)

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()

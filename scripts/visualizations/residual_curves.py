import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from parser_types import default_labeled_source, file_name_pair, OptionallyLabeledSource


class Args(argparse.Namespace):
    sources: list[OptionallyLabeledSource]
    dataset: str
    output: Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", type=file_name_pair)
    parser.add_argument("-d", "--dataset")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(namespace=Args())

    sources = list(default_labeled_source(args.sources))
    data = [(label, pd.read_csv(path).astype(float)) for path, label in sources]

    plot(data, args)


def plot(data: list[tuple[str, pd.DataFrame]], args: Args):
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 4.25))

    for label, residuals in data:
        ax[0].plot(residuals, label=label)
        ax[1].plot(np.log(residuals.to_numpy()), label=label)

    ax[0].legend()
    ax[0].set_ylabel(r"$\frac{\Vert r \Vert}{\Vert b \Vert}$")
    ax[0].set_xlabel("Iteration")

    ax[1].legend()
    ax[1].set_ylabel(r"$\log \left( \frac{\Vert r \Vert}{\Vert b \Vert} \right)$")
    ax[1].set_xlabel("Iteration")

    title = "Residual Error Curve Comparison"
    if args.dataset:
        title += f" ({args.dataset})"

    fig.suptitle(title, fontweight="bold")
    plt.tight_layout(h_pad=2, w_pad=4)

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()

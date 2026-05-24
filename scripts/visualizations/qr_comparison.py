from pathlib import Path
import re
import argparse
from sys import exit

import matplotlib.pyplot as plt
import pandas as pd

from benchmark_file import read, FILE_PATTERN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()

    sources = {s.stem: s for s in args.sources}
    data = {}

    for k, v in sources.items():
        try:
            data[k] = read(Path(v))
        except Exception as e:
            print(f"error: {str(e)}. skipping.")

    if not data:
        print("error: no data read")
        exit(1)

    plot(data, output=args.output if args.output else "qr_comparison.png")


def plot(data: dict[str, pd.DataFrame], output: str):
    algorithms = ("std_qr", "chol_qr")
    available_algorithms = [name for name in algorithms if name in data]
    if not available_algorithms:
        raise Exception("no data available to plot")
    if not all(name in data for name in algorithms):
        raise Exception("expected both std_qr and chol_qr data")

    block_sizes = sorted(
        {int(block_size) for df in data.values() for block_size in df.index.tolist()}
    )
    x = list(range(len(block_sizes)))
    width = 0.35
    metrics = (
        ("solve", "Solve Runtime by Block Size", "Avg (ms)"),
        ("iteration", "Iteration Runtime by Block Size", "Avg (ms)"),
        ("[w sigma] = QR(temp)", "QR(temp) Runtime by Block Size", "Avg (ms)"),
        ("[w zeta] = QR(w)", "QR(w) Runtime by Block Size", "Avg (ms)"),
        ("iteration_instances", "Iterations by Block Size", "Instances"),
    )

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4), sharey=False)

    offsets = {
        name: (index - (len(available_algorithms) - 1) / 2) * width
        for index, name in enumerate(available_algorithms)
    }

    for ax, (metric, title, ylabel) in zip(axes, metrics):
        for name in available_algorithms:
            df = data[name]
            heights = [
                df.loc[block_size, metric] if block_size in df.index else 0
                for block_size in block_sizes
            ]
            ax.bar(
                [value + offsets[name] for value in x],
                heights,
                width,
                label=name,
            )

        ax.set_xlabel("Block Size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x, [str(block_size) for block_size in block_sizes])
        ax.legend()

    fig.tight_layout()
    plt.savefig(output)
    print(output)


if __name__ == "__main__":
    main()

from pathlib import Path
import argparse
from sys import exit

import matplotlib.pyplot as plt
import pandas as pd

from benchmark_file import read_dir, read_file, range_runtimes_by_block_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--cg", type=Path)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()

    sources = {s.stem: s for s in args.sources}
    dr_bcg_data = {
        algorithm: process_dr_bcg(read_dir(bm_data))
        for algorithm, bm_data in sources.items()
        if bm_data.is_dir()
    }

    if not dr_bcg_data:
        print("error: no data read")
        exit(1)

    cg_data = read_file(args.cg) if args.cg else None

    plot(dr_bcg_data, cg_data, args.output if args.output else "qr_comparison.png")


def process_dr_bcg(df: pd.DataFrame) -> pd.DataFrame:
    iterations = df[df["Range"] == "iteration"][["block_size", "Instances"]].set_index(
        "block_size"
    )
    df = range_runtimes_by_block_size(
        df,
        ranges=("solve", "iteration", "[w sigma] = QR(temp)", "[w zeta] = QR(w)"),
    )
    df["iterations"] = iterations
    return df


def plot(
    dr_bcg_data: dict[str, pd.DataFrame], cg_data: pd.DataFrame | None, output: str
):
    assert dr_bcg_data
    available_algorithms = dr_bcg_data.keys()

    block_sizes = sorted(
        {
            int(block_size)
            for df in dr_bcg_data.values()
            for block_size in df.index.tolist()
        }
    )
    x = list(range(len(block_sizes)))
    width = 0.35
    metrics = (
        ("[w sigma] = QR(temp)", "QR(temp) Runtime by Block Size", "Avg (ms)"),
        ("[w zeta] = QR(w)", "QR(w) Runtime by Block Size", "Avg (ms)"),
        ("iteration", "Iteration Runtime by Block Size", "Avg (ms)"),
        ("iterations", "Iterations by Block Size", "Iterations"),
        ("solve", "Solver Runtime by Block Size", "Avg (ms)"),
    )

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4), sharey=False)

    offsets = {
        name: (index - (len(available_algorithms) - 1) / 2) * width
        for index, name in enumerate(available_algorithms)
    }

    for ax, (metric, title, ylabel) in zip(axes, metrics):
        for name in available_algorithms:
            df = dr_bcg_data[name]
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

        cg_col = {
            "iteration": "Avg (ms)",
            "iterations": "Instances",
            "solve": "Avg (ms)",
        }.get(metric)
        if cg_col is not None and cg_data is not None:
            cg_x = -1
            if metric == "iteration" or metric == "iterations":
                ax.bar(
                    cg_x,
                    cg_data.loc[cg_data["Range"] == "iteration", cg_col].iloc[0],
                    width,
                    label="CG",
                )
            else:
                ax.bar(
                    cg_x,
                    cg_data.loc[cg_data["Range"] == "solve", cg_col].iloc[0],
                    width,
                    label="CG",
                )
            ax.set_xticks([cg_x] + list(x), ["CG"] + [str(bs) for bs in block_sizes])
        else:
            ax.set_xticks(x, [str(block_size) for block_size in block_sizes])

        ax.set_xlabel("Block Size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    fig.tight_layout()
    plt.savefig(output)
    print(output)


if __name__ == "__main__":
    main()

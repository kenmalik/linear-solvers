from pathlib import Path
import argparse
from sys import exit

import matplotlib.pyplot as plt
import pandas as pd

from benchmark_file import read_dir, read_file, BenchmarkFile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--cg", type=Path)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()

    sources = {s.stem: s for s in args.sources}
    dr_bcg_data = {
        algorithm: restructure_dr_bcg_bms(read_dir(bm_data))
        for algorithm, bm_data in sources.items()
        if bm_data.is_dir()
    }

    if not dr_bcg_data:
        print("error: no data read")
        exit(1)

    cg_data = read_file(args.cg).data if args.cg else None

    plot(dr_bcg_data, cg_data, args.output if args.output else "qr_comparison.png")


def restructure_dr_bcg_bms(benchmarks: list[BenchmarkFile]) -> pd.DataFrame:
    """
    Restructures data to list average runtimes in milliseconds and iterations
    to convergence by block size, matching the following format:

                    iteration   [w sigma] = QR(temp)    [w zeta] = QR(w)    iterations
    block_size
    """

    if not all(bm.implementation == benchmarks[0].implementation for bm in benchmarks):
        raise ValueError("solver implementations do not match")
    if not all(bm.algorithm == "dr-bcg" for bm in benchmarks):
        raise ValueError("solver algorithms are not dr-bcg")

    return pd.DataFrame.from_records(
        [
            get_block_size_records(bm.data) | {"block_size": bm.block_size}
            for bm in benchmarks
        ],
        index="block_size",
    )


def get_block_size_records(df: pd.DataFrame) -> dict:
    return {
        r: df.loc[r]["Avg (ms)"]
        for r in ("iteration", "[w sigma] = QR(temp)", "[w zeta] = QR(w)")
    } | {"iterations": df.loc["iteration"]["Instances"]}


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

        cg_col = {"iteration": "Avg (ms)", "iterations": "Instances"}.get(metric)
        if cg_col is not None and cg_data is not None:
            cg_x = -1
            ax.bar(cg_x, cg_data.loc["iteration"][cg_col], width, label="CG")
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

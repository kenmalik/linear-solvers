import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
import re

from benchmark_file import read, FILE_PATTERN


def main():
    sources = {
        "chol_qr": "./data/18390215_interactive/chol_qr",
        "std_qr": "./data/18390215_interactive/std_qr",
    }
    data = {}

    for k, v in sources.items():
        try:
            data[k] = read(Path(v))
        except Exception as e:
            print(f"error reading dir: {str(e)}. skipping.")

    plot(data)


def read(dir: Path) -> pd.DataFrame:
    if not dir.is_dir():
        raise Exception("invalid data directory")

    data = {}

    for file in dir.glob("*.csv"):
        match = FILE_PATTERN.search(str(file))

        if not match:
            raise Exception("invalid file name")

        block_size = int(match.group(3))
        data.setdefault("block_size", []).append(block_size)

        df = pd.read_csv(file, index_col="Range")
        for r in ("solve", "iteration", "[w sigma] = QR(temp)", "[w zeta] = QR(w)"):
            data.setdefault(r, []).append(df.loc[r]["Avg (ms)"])
        data.setdefault("iteration_instances", []).append(
            df.loc["iteration"]["Instances"]
        )

    return pd.DataFrame(data).set_index("block_size")


def plot(data: dict[str, pd.DataFrame]):
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
    plt.savefig("qr_comparison.png")
    print("qr_comparison.png")


if __name__ == "__main__":
    main()

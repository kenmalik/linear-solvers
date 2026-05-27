import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from benchmark_file import read_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("qr_type", choices=["householder", "cholesky"])
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    data = read_file(args.source)
    data = data[data["Range"].str.startswith("QR:")]

    compute = data[data["Range"] != "QR:func"]
    func_ms = data[data["Range"] == "QR:func"]["Avg (ms)"]
    compute_ms = compute["Avg (ms)"].sum()
    memory = pd.DataFrame({"Range": "QR:memory", "Avg (ms)": func_ms - compute_ms})

    data = pd.concat([compute, memory], ignore_index=True)

    plot(data, args.qr_type, args.output)


def plot(data: pd.DataFrame, qr_type: str, output: str | None):
    fig, ax = plt.subplots()

    bottom = 0
    for _, row in data[["Range", "Avg (ms)"]].iloc[::-1].iterrows():  # type: ignore
        label = row["Range"].replace("QR:", "")
        ax.bar(
            "Runtime",
            row["Avg (ms)"],
            0.5,
            bottom=bottom,
            label=label,
        )
        bottom += row["Avg (ms)"]

    handles, labels = ax.get_legend_handles_labels()
    ax.set_ylabel("Avg (ms) per iteration")
    ax.legend(handles[::-1], labels[::-1], loc="upper right", fontsize="small")
    fig.suptitle(f"{qr_type.capitalize()} QR Runtime Breakdown")

    plt.tight_layout()
    out = output if output else "qr_stacked_bars.png"
    plt.savefig(out)
    print(out)


if __name__ == "__main__":
    main()

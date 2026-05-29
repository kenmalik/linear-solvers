import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

algs = ["cg", "dr-bcg"]
impls = ["mkl", "cuda"]


class Args(argparse.Namespace):
    dataset: str
    files: list[Path]
    output: Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=valid_path)
    parser.add_argument("-d", "--dataset")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(namespace=Args())

    data = read_files(args.files)

    plot(data, args)


def valid_path(arg: str):
    p = Path(arg)
    if not p.exists() or not p.is_file():
        raise argparse.ArgumentTypeError(f"{arg} is an invalid path")
    return p


def read_files(files: list[Path]):
    data = {}

    for file in files:
        _, alg, impl, *meta = file.stem.split("_")

        residuals = pd.read_csv(file).astype(float)
        data.setdefault(alg, []).append((impl, residuals, meta))

    return data


def plot(data: dict[str, list], args: Args):
    fig, ax = plt.subplots(len(data), 2, figsize=(8.5, 8.5))

    if len(data) > 1:
        for i, [alg, info] in enumerate(data.items()):
            for impl, residuals, meta in info:
                label = impl
                if meta:
                    label += f" ({', '.join(meta)})"
                ax[i, 0].plot(residuals, label=label)
                ax[i, 1].plot(np.log(residuals.to_numpy()), label=label)

            ax[i, 0].set_title(alg)
            ax[i, 0].legend()
            ax[i, 0].set_ylabel(r"$\frac{\Vert r \Vert}{\Vert b \Vert}$")
            ax[i, 0].set_xlabel("Iteration")

            ax[i, 1].set_title(alg)
            ax[i, 1].legend()
            ax[i, 1].set_ylabel(
                r"$\log \left( \frac{\Vert r \Vert}{\Vert b \Vert} \right)$"
            )
            ax[i, 1].set_xlabel("Iteration")
    else:
        for alg, info in data.items():
            for impl, residuals, meta in info:
                label = impl
                if meta:
                    label += f" ({', '.join(meta)})"
                ax[0].plot(residuals, label=label)
                ax[1].plot(np.log(residuals.to_numpy()), label=label)

            ax[0].set_title(alg)
            ax[0].legend()
            ax[0].set_ylabel(r"$\frac{\Vert r \Vert}{\Vert b \Vert}$")
            ax[0].set_xlabel("Iteration")

            ax[1].set_title(alg)
            ax[1].legend()
            ax[1].set_ylabel(
                r"$\log \left( \frac{\Vert r \Vert}{\Vert b \Vert} \right)$"
            )
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

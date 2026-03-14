import matplotlib.pyplot as plt
import pandas as pd

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from pathlib import Path


class Args(Namespace):
    files: list[Path]


def valid_file(filename: str):
    path = Path(filename)
    if not path.exists() or not path.is_file():
        raise ArgumentTypeError(f"File {filename} is invalid")
    return path


def read_data(file: Path):
    with open(file, "r") as f:
        data = pd.read_csv(f)
    return data


def plot(files: list[Path]):
    fig, ax = plt.subplots()

    dataset = None
    for file in files:
        algorithm, implementation, dataset = file.stem.split("_", 2)
        data = read_data(file)
        ax.plot(data, label=f"{implementation} {algorithm}")

    fig.suptitle("Residual Error Curve Comparison", fontweight="bold")

    if dataset:
        ax.set_title(dataset)

    ax.legend()
    ax.set_ylabel("Residual Error")
    ax.set_xlabel("Iteration")

    plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument("files", nargs="+", type=valid_file)

    args = parser.parse_args(namespace=Args)
    plot(args.files)


if __name__ == "__main__":
    main()

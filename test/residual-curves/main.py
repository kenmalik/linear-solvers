import itertools
import argparse
import subprocess
from pathlib import Path

import pandas as pd

from plotter import plot


algs = ["cg", "dr-bcg"]
impls = ["mkl", "cuda"]


class Args(argparse.Namespace):
    dataset: str
    files: list[Path]


def valid_path(arg: str):
    p = Path(arg)
    if not p.exists() or not p.is_file():
        raise argparse.ArgumentTypeError(f"{arg} is an invalid path")
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="1138_bus")
    parser.add_argument("-f", "--files", nargs="*", type=valid_path)
    args = parser.parse_args(namespace=Args())

    if args.files:
        from_files(args)
    else:
        from_cgrun(args)


def from_files(args: Args):
    data = {}

    dataset = ""
    for file in args.files:
        alg, impl, new_dataset = file.stem.split("_", maxsplit=2)
        if len(dataset) != 0 and new_dataset != dataset:
            raise ValueError(f"mismatching dataset '{new_dataset}'")
        else:
            dataset = new_dataset

        residuals = pd.read_csv(file).astype(float)
        data.setdefault(alg, []).append((impl, residuals))

    plot(data, dataset)


def from_cgrun(args: Args):
    data = {}
    for alg, impl in itertools.product(algs, impls):
        prog = [
            "build/runner/cgrun",
            alg,
            impl,
            "-s",
            "16",
            f"data/{args.dataset}.mat",
            f"data/{args.dataset}_ichol.mat",
        ]

        try:
            res = subprocess.run(prog, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(
                f"(alg={alg}, impl={impl}, dataset={args.dataset}) failed with error: {e.stderr}"
            )
            return

        residuals = pd.Series(res.stderr.splitlines()).astype(float)
        data.setdefault(alg, []).append((impl, residuals))

    plot(data, args.dataset)


if __name__ == "__main__":
    main()

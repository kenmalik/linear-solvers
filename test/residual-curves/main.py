import itertools
import argparse
import subprocess

import pandas as pd

from plotter import plot


algs = ["cg", "dr-bcg"]
impls = ["mkl", "cuda"]


class Args(argparse.Namespace):
    dataset: str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="1138_bus")
    args = parser.parse_args(namespace=Args)

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

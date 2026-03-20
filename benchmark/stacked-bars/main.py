import argparse

import matplotlib.pyplot as plt
import pandas as pd

cg_ranges = (
    "s = M^{-1} * r",
    "r = r - alpha * q",
    "r = b - A * x",
    "x = x + alpha * d",
    "q = A * d",
    "d = s + beta * d",
    "alpha = delta / d'q",
    "residual_sq = r'r",
    "beta = delta_new / delta_old",
)

dr_bcg_ranges = (
    "R = B - A * X",
    "[w sigma] = QR(L^-1 * R)",
    "s = (L^-1)' * w",
    "xi = (s' * As)^-1",
    "X = X + s * xi * sigma",
    "norm(B1 - A * X1) / norm(B1)",
    "[w zeta] = QR(w - L^{-1} * A * s * xi)",
    "s = (L^-1)' * w + s * zeta'",
    "sigma = zeta * sigma",
)

ranges_by_impl_algo = {
    "mkl": {
        "cg": cg_ranges,
        "dr-bcg": dr_bcg_ranges,
    },
    "cuda": {
        "cg": cg_ranges,
        "dr-bcg": dr_bcg_ranges,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--algo", choices=["cg", "dr-bcg"], required=True)
    parser.add_argument("--impl", choices=["mkl", "cuda"], required=True)

    args = parser.parse_args()

    ranges = ranges_by_impl_algo[args.impl][args.algo]

    data = pd.read_csv(args.file)
    if args.impl == "cuda":
        data["Range"] = data["Range"].str.slice(start=1)
    data = data[data["Range"].isin(ranges)]
    data["Range"] = pd.Categorical(data["Range"], categories=ranges, ordered=True)
    data = data.sort_values("Range")

    fig, ax = plt.subplots()

    bottom = 0
    for _, row in data[["Range", "Avg (ms)"]].iloc[::-1].iterrows():  # type: ignore
        label = row["Range"]
        ax.bar("Runtime", row["Avg (ms)"], 0.5, bottom=bottom, label=label)
        bottom += row["Avg (ms)"]

    handles, labels = ax.get_legend_handles_labels()
    ax.set_ylabel("Avg (ms) per iteration")
    ax.legend(handles[::-1], labels[::-1], loc="upper right", fontsize="small")
    fig.suptitle(f"{args.impl.upper()} {args.algo.upper()} Runtime Breakdown")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

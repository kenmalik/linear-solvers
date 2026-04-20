import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colormaps

cg_ranges = (
    "r = b - A * x",
    "r = r - alpha * q",
    "d = M^{-1} * r",
    "q = A * d",
    "alpha = delta / d'q",
    "x = x + alpha * d",
    "residual_sq = r'r",
    "s = M^{-1} * r",
    "beta = delta_new / delta_old",
    "d = s + beta * d",
)

dr_bcg_ranges = (
    "R = B - A * X",
    "temp = L^-1 * R",
    "[w sigma] = QR(temp)",
    "s = (L^-1)' * w",
    "xi = (s' * As)^-1",
    "X = X + s * xi * sigma",
    "norm(B1 - A * X1) / norm(B1)",
    "w = w - L^-1 * A * s * xi",
    "[w zeta] = QR(w)",
    "s = (L^-1)' * w + s * zeta'",
    "sigma = zeta * sigma",
)

custom_colors = (
    "palevioletred",
    "cornflowerblue",
    "lightcoral",
    "peru",
    "gold",
    "olivedrab",
    "darkseagreen",
    "lightseagreen",
    "lightblue",
    "tan",
    "darksalmon",
)

cg_colors = {name: colormaps["tab10"](i) for i, name in enumerate(cg_ranges)}
dr_bcg_colors = {name: custom_colors[i] for i, name in enumerate(dr_bcg_ranges)}

colors_by_algo = {
    "cg": cg_colors,
    "dr-bcg": dr_bcg_colors,
}

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
    parser.add_argument("-o", "--output")

    args = parser.parse_args()

    ranges = ranges_by_impl_algo[args.impl][args.algo]
    colors = colors_by_algo[args.algo]

    data = pd.read_csv(args.file)
    data = data[data["Range"].isin(ranges)]
    data["Range"] = pd.Categorical(data["Range"], categories=ranges, ordered=True)
    data = data.sort_values("Range")

    fig, ax = plt.subplots()

    bottom = 0
    for _, row in data[["Range", "Avg (ms)"]].iloc[::-1].iterrows():  # type: ignore
        label = row["Range"]
        ax.bar(
            "Runtime",
            row["Avg (ms)"],
            0.5,
            bottom=bottom,
            label=label,
            color=colors[label],
        )
        bottom += row["Avg (ms)"]

    handles, labels = ax.get_legend_handles_labels()
    ax.set_ylabel("Avg (ms) per iteration")
    ax.legend(handles[::-1], labels[::-1], loc="upper right", fontsize="small")
    fig.suptitle(f"{args.impl.upper()} {args.algo.upper()} Runtime Breakdown")

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()

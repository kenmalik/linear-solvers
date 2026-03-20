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
    "s = L^{-T} * w + s * zeta'",
    "As_xi = L^{-1} * As * xi",
    "convergence check",
    "As = A * s",
    "thin QR(w_new_input)",
    "X = X + s * xi * sigma",
    "xi = (s' * As)^{-1}",
    "sigma = zeta * sigma",
)

ranges_by_algo = {
    "cg": cg_ranges,
    "dr-bcg": dr_bcg_ranges,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--algo", choices=["cg", "dr-bcg"], required=True)

    args = parser.parse_args()

    ranges = ranges_by_algo[args.algo]

    data = pd.read_csv(args.file)
    data = data[data["Range"].isin(ranges)]

    fig, ax = plt.subplots()

    bottom = 0
    for _, row in data[["Range", "Avg (ms)"]].iterrows():  # type: ignore
        ax.bar("Runtime", row["Avg (ms)"], 0.5, bottom=bottom, label=row["Range"])
        bottom += row["Avg (ms)"]

    ax.set_ylabel("Avg (ms) per iteration")
    ax.legend(loc="upper right", fontsize="small")
    fig.suptitle(f"MKL {args.algo.upper()} Runtime Breakdown")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

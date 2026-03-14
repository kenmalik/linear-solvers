import matplotlib.pyplot as plt


def plot(data: dict[str, list]):
    fig, ax = plt.subplots(len(data), figsize=(8, 8))

    for i, [alg, info] in enumerate(data.items()):
        for impl, residuals in info:
            ax[i].plot(residuals, label=impl)

        ax[i].set_title(alg)
        ax[i].legend()
        ax[i].set_ylabel("Relative Residual Norm")
        ax[i].set_xlabel("Iteration")

    fig.suptitle("Residual Error Curve Comparison", fontweight="bold")
    plt.tight_layout()
    plt.show()

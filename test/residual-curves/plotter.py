import matplotlib.pyplot as plt
import numpy as np


def plot(data: dict[str, list], dataset: str):
    fig, ax = plt.subplots(len(data), 2, figsize=(8.5, 8.5))

    for i, [alg, info] in enumerate(data.items()):
        for impl, residuals in info:
            ax[i, 0].plot(residuals, label=impl)
            ax[i, 1].plot(np.log(residuals.to_numpy()), label=impl)

        ax[i, 0].set_title(alg)
        ax[i, 0].legend()
        ax[i, 0].set_ylabel(r"$\Vert r \Vert$")
        ax[i, 0].set_xlabel("Iteration")
        
        ax[i, 1].set_title(alg)
        ax[i, 1].legend()
        ax[i, 1].set_ylabel(r"$\log \left( \Vert r \Vert \right)$")
        ax[i, 1].set_xlabel("Iteration")

    fig.suptitle(f"Residual Error Curve Comparison ({dataset})", fontweight="bold")
    plt.tight_layout(h_pad=2, w_pad=4)
    plt.show()

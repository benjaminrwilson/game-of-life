import math
import sys

import numpy as np
import torch

EPSILON = np.finfo(float).eps


def entropy(x, base):
    if base == 2:
        return -x @ torch.log2(x)
    elif base == math.e:
        return -x @ torch.log(x)
    elif base == 10:
        return -x @ torch.log10(x)
    sys.exit("Unsupported base. Please choose {}, {}, or {}.".format(2, "e", 10))


def plot_entropies(entropies):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    for i, ax in enumerate(axes.flat, start=1):
        ax.set_xlabel('Steps')
        ax.set_ylabel('Value')

    axes[0].plot(entropies)
    axes[0].set_title('H')
    axes[1].plot(np.diff(entropies))
    axes[1].set_title('d/dH(H)')
    fig.tight_layout()
    plt.show()

###############################################################################
# Global Imports
###############################################################################
import math

###############################################################################
# 3PP Imports
###############################################################################
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

###############################################################################
# Certus Imports
###############################################################################
from audiofeatures import FeatureExtractor


###############################################################################
# Helpers
###############################################################################
def _calc_plot_shape(num_spectra: int) -> tuple[int, int]:
    if num_spectra >= 12:
        return math.ceil(num_spectra / 4), 4
    if num_spectra >= 9:
        return math.ceil(num_spectra / 3), 3
    elif num_spectra >= 4:
        return math.ceil(num_spectra / 2), 2
    else:
        return num_spectra, 1


###############################################################################
# Matplotlib
###############################################################################
def plot(spectra: Tensor, sources_or_subtitles: list[FeatureExtractor | str], sup_title: str) -> None:
    num_spectra = len(spectra)
    nrows, ncols = _calc_plot_shape(num_spectra)
    has_subtitles = isinstance(sources_or_subtitles[0], str)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    fig.suptitle(sup_title)

    for idx in range(num_spectra):
        row = idx % nrows
        col = idx // nrows

        sub_p = axs[row][col] if ncols > 1 else axs[idx] if nrows > 1 else axs
        sub_p.imshow(spectra[idx].squeeze(0), origin='lower', aspect='auto')

        if has_subtitles:
            sub_p.set_title(sources_or_subtitles[idx])
        else:
            sub_p.set_title(f"{sources_or_subtitles[idx].get_spec_type().value}/{idx}")

    fig.tight_layout()
    fig.set_figwidth(ncols * 5)
    fig.set_figheight(nrows * 3)
    plt.show()


def plot_lines(spec_0: Tensor, spec_1: Tensor, idx: int, dim=1) -> None:
    line_0 = spec_0[:, :, idx].squeeze() if dim == 1 else spec_0[:, idx, :].squeeze()
    line_1 = spec_1[:, :, idx].squeeze() if dim == 1 else spec_1[:, idx, :].squeeze()
    x_vals = [x for x in range(line_0.shape[0])]

    plt.title(f"Spectrum column {idx}")
    plt.plot(x_vals, line_0, label="spec_0")
    plt.plot(x_vals, line_1, label="spec_1")
    plt.plot(x_vals, np.zeros_like(x_vals), label="ref")
    plt.legend()
    plt.show()


def plot_spec_by_lines(spec: Tensor) -> None:
    _, ht, wd = spec.shape
    x_vals = [x for x in range(ht)]

    for idx in range(wd):
        line = spec[:, :, idx].squeeze()

        plt.title(f"Spectrum column {idx}")
        plt.plot(x_vals, line)
        plt.show()

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
from audiofeatures import FeatureChannel


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
def plot(spectra: Tensor | list, sources_or_subtitles: list[FeatureChannel] | list[str], sup_title: str) -> None:
    num_spectra = len(spectra)
    nrows, ncols = _calc_plot_shape(num_spectra)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    fig.suptitle(sup_title)

    for idx in range(num_spectra):
        row = idx % nrows
        col = idx // nrows

        sub_p = axs[row][col] if ncols > 1 else axs[idx] if nrows > 1 else axs
        sub_p.imshow(spectra[idx].squeeze(0), origin='lower', aspect='auto')

        st = sources_or_subtitles[idx]
        if isinstance(st, str):
            sub_p.set_title(st)
        else:
            sub_p.set_title(f"{st.get_spec_type().value}/{idx}")

    fig.tight_layout()
    fig.set_figwidth(ncols * 5)
    fig.set_figheight(nrows * 3)
    plt.show()


def plot_lines(specs: list[Tensor], idx: int, dim=1) -> None:
    lines = [spec[:, :, idx].squeeze() if dim == 1 else spec[:, idx, :].squeeze() for spec in specs]
    x_vals = [x for x in range(lines[0].shape[0])]

    plt.title(f"Spectrum column {idx}")
    for line, line_idx in enumerate(lines):
        plt.plot(x_vals, line, label=f"spec_{line_idx}")

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


def plot_time_domain(wav: Tensor, sample_rate: int, title: str) -> None:
    td = wav.squeeze(0)
    time = np.linspace(0, len(td) / sample_rate, num=len(td))
    plt.plot(time, td)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()


def plot_with_time_domain(spectrum: Tensor, wav: Tensor, sample_rate: int, title: str) -> None:
    # Config
    n_chan = spectrum.shape[0]
    fig, ax = plt.subplots(nrows=n_chan + 1)

    fig.suptitle(title)
    fig.tight_layout()
    fig.set_figheight(8)

    # Spectrogram
    for idx in range(n_chan):
        ax[idx].imshow(spectrum[idx, :, :].squeeze(), origin='lower', aspect='auto')
        ax[idx].set_xlabel("STFT Window")
        ax[idx].set_ylabel("Freq Band")

    # Waveform
    audio = wav.squeeze(0)
    time = np.linspace(0, len(audio) / sample_rate, num=len(audio))
    ax[n_chan].set_xlim(left=0, right=time[-1])
    ax[n_chan].plot(time, audio)
    ax[n_chan].set_xlabel("Time (sec)")
    ax[n_chan].set_ylabel("Amplitude")

    # Display the plots
    plt.show()

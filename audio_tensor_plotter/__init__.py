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
def plot(spectra: Tensor, sources_or_subtitles: list[str], sup_title: str) -> None:
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
    _spec = spectrum.squeeze()
    has_channels = len(_spec.shape) == 3
    n_chan = _spec.shape[0] if has_channels else 1
    fig, ax = plt.subplots(nrows=n_chan + 1)

    fig.suptitle(title)
    fig.tight_layout()
    fig.set_figheight(8)

    # Spectrogram
    for idx in range(n_chan):
        _chan_spec = (_spec[idx, :, :] if has_channels else _spec).squeeze()
        ax[idx].imshow(_chan_spec, origin='lower', aspect='auto')
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

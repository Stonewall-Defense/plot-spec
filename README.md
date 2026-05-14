# Audio Tensor Plotter

Simple tools for plotting spectrograms and audio data.

## Prerequisites

- Python 3.11 runtime
- Pip for package installation

## Installation

Install the dependencies into the environment with [pip](https://pypi.org/project/pip/):

```bash
pip install -r requirements.txt
```

Then install this package:

```bash
pip install .
```

## Usage

Best used with the [`AudioMlSpecTools`](https://pypi.org/project/AudioMlSpecTools/), also from Certus Innovations:

```python
CHANNELS = [
    FeatureChannel(SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LEN, n_filters=N_MELS, is_logarithmic=True, is_mel=True),
]
FEATURE_SOURCE = FeatureSource(CHANNELS)
AUDIO = load_wav("fq_example_filename.wav", target_sr=SAMPLE_RATE).squeeze()

specs = FEATURE_SOURCE.forward(AUDIO)
plot_with_time_domain(specs, AUDIO, SAMPLE_RATE, "Plotting Example")
```

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/Stonewall-Defense/plot-spec/tags).

## Authors

- **Ryan Quinn** - _Initial work_

## License

MIT.

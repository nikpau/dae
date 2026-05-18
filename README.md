# libdse — Library for Deep Speech Enhancement

This is the accompanying code for my [blog post on Denoising AutoEncoders](https://iam.nikpau.io/blog/speech_enhancement_3/).

A PyTorch implementation of speech enhancement models, starting with a **Denoising Autoencoder (DAE)** following Lu et al. (2013) - *Speech Enhancement Based on Deep Denoising Autoencoder* — with architecture choices informed by Nossier et al. (2020) - *An Experimental Analysis of Deep Learning Architectures for Supervised Speech Enhancement* — and extending to a time-domain **Wave-U-Net** (Stoller et al., 2018).

The API documentation can be found [here](https://dae.nikpau.io/docs/).

---

## The Idea

Real-world speech is corrupted by additive noise — fan hum, traffic, background chatter — that degrades both intelligibility and downstream processing such as ASR or speaker verification.

A **denoising autoencoder** learns a direct mapping from *noisy* spectral features to *clean* spectral features. During training the model sees pairs `(noisy_frame, clean_frame)` and is penalised for any reconstruction error. At inference time only the noisy side is available: the encoder compresses it into a bottleneck representation, and the decoder reconstructs a clean estimate from that representation.

```
Noisy speech frame                    Clean speech estimate
        │                                      ▲
        ▼                                      │
 ┌─────────────┐    bottleneck z    ┌──────────────┐
 │   Encoder   │ ─────────────────► │    Decoder   │
 └─────────────┘                    └──────────────┘
```

---

## Architecture

The model implemented and used in production is `simpleAE_logmag` — a fully-connected denoising autoencoder operating on **log-magnitude spectrogram** frames.

### Feature Representation

Each utterance is resampled to 8 kHz and transformed via a short-time Fourier transform (256-sample Hann window, 128-sample hop). One frame therefore spans `256 / 2 + 1 = 129` frequency bins. The log-magnitude of each STFT frame forms the model input:

```
PCM waveform (8 kHz)
   │
   ▼  STFT  (n_fft=256, hop=128, Hann window)
Complex spectrogram  (129, n_frames)
   │
   ▼  log(|·| + ε)
Log-magnitude spectrogram  (129, n_frames)
   │
   ▼  one frame per sample
Input vector  shape: (129,)
```

Working in the **log-magnitude** domain offers a compact, perceptually motivated representation: the logarithm compresses the wide dynamic range of speech, and the frame-level granularity keeps the input size small enough for a fully-connected network.

### Network

The encoder and decoder are symmetric stacks of fully-connected layers with ReLU activations, `LayerNorm`, and no dropout (as per Nossier et al. architecture (d)):

| Stage | Layer sizes |
|---|---|
| Input | 129 |
| Encoder | 2048 → 500 → **180** (bottleneck) |
| Decoder | 180 → 500 → 2048 → 129 |

`LayerNorm` is applied to the input and after each linear layer. The bottleneck dimension of 180 gives a compression ratio of roughly 7×.

### Noise Augmentation

Training pairs are synthesised on the fly. For each utterance a random excerpt from the **DEMAND** noise corpus is mixed with the clean speech at a uniformly sampled SNR. All 18 DEMAND environments are used by default. The same noise pool is shared between train and validation; random draw offsets ensure each sample is unique.

### Training

The model is trained with Adam and MSE reconstruction loss. A `ReduceLROnPlateau` scheduler halves the learning rate after two epochs without validation improvement. Key hyperparameters:

| Parameter | Value |
|---|---|
| Epochs | 40 |
| Batch size | 256 |
| Sampling rate | 8 000 Hz |
| STFT window / hop | 256 / 128 samples |
| Bottleneck dim | 180 |
| Optimizer | Adam |
| LR schedule | ReduceLROnPlateau (patience=2, factor=0.5) |

TensorBoard logs (training loss, validation loss, SNR improvement, gradient norms) are written to `runs/` and can be inspected with `tensorboard --logdir runs`.

### Inference & Waveform Reconstruction

At inference time each frame is denoised independently. To recover a waveform the enhanced log-magnitude spectrum is exponentiated back to a magnitude spectrum, the **original noisy phase** is re-applied, and `librosa.istft` inverts the result. This phase-borrowing approach avoids the iterative Griffin-Lim procedure while still producing intelligible output.

---

## Repository Structure

```
speech_enhancement/
├── src/
│   └── libdse/
│       ├── nets.py                          # VanillaAutoEncoder, WaveUNet
│       ├── evaluation.py                    # Evaluation metrics (PESQ, STOI)
│       ├── data/
│       │   ├── features.py                  # Feature extractors (log-mag, mel, raw)
│       │   ├── librispeech.py               # LibriSpeechDataset (IterableDataset)
│       │   ├── noise.py                     # DEMANDNoiseDataset, add_noise_snr
│       │   └── err.py                       # Custom exceptions
│       ├── train/
│       │   └── dae.py                       # DAE training script + hyperparameters
│       └── showcases/
│           └── dae.py                       # Gradio demo app
├── Dockerfile                               # Containerised Gradio demo
├── models/
│   └── simple_autoencoder_logmag_spec_noisy_clean   # Trained DAE checkpoint
├── data/
│   ├── train-clean-100/                     # LibriSpeech training corpus
│   ├── test-clean/                          # LibriSpeech test corpus
│   └── noise/DEMAND/                        # DEMAND noise recordings
├── tests/
│   ├── resources/                           # Small FLAC fixtures
│   └── test_dataset.py
└── pyproject.toml
```

---

## Installation

Requires Python ≥ 3.12.

```bash
git clone <repo-url>
cd speech_enhancement

# Install (uv recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

---

## Data

Download [LibriSpeech train-clean-100](https://www.openslr.org/12) (~6.3 GB) and [LibriSpeech test-clean](https://www.openslr.org/12), then extract them under `data/`. Download the [DEMAND corpus](https://zenodo.org/record/1227121) and place it under `data/noise/DEMAND/`. The expected layout:

```
data/
├── train-clean-100/LibriSpeech/train-clean-100/<speaker>/<chapter>/*.flac
├── test-clean/LibriSpeech/test-clean/<speaker>/<chapter>/*.flac
└── noise/DEMAND/<ENVIRONMENT>/*.wav
```

---

## Training

```bash
python -m libdse.train.dae
```

The checkpoint with the best validation loss is saved to `models/simple_autoencoder_logmag_spec_noisy_clean`.

---

## Gradio Demo

A pre-trained checkpoint is included in `models/`. Launch the interactive demo with:

```bash
python -m libdse.showcases.dae
```

Alternatively, run the containerised version with Docker:

```bash
docker build -t libdse-demo .
docker run -p 7860:7860 libdse-demo
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

The app exposes two tabs:

- **Denoise** — upload any audio file; the model denoises it and displays spectrograms of the input and output side-by-side.
- **Noise mix** — upload clean speech, choose a DEMAND environment and a target SNR, and listen to the resulting noisy mixture.

---

## Running Tests

```bash
uv run pytest
```

---

## Roadmap

- [x] Synthesise noisy training pairs (LibriSpeech + DEMAND)
- [x] Fully-connected DAE on log-magnitude spectrogram frames
- [x] Training loop with MSE loss, LR scheduling, TensorBoard logging
- [x] Waveform reconstruction via phase borrowing + `istft`
- [x] Gradio demo app
- [x] Containerise the Gradio app for server deployment
- [x] Wave-U-Net architecture (`libdse.nets.WaveUNet`)
- [ ] Wave-U-Net training script
- [ ] Wave-U-Net validation & evaluation

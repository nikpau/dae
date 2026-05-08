# ── Base image ────────────────────────────────────────────────────────────────
# python:3.12-slim keeps the image small while matching the project requirement.
FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────────────
# libsndfile1  — required by librosa / soundfile to read audio files
# ffmpeg       — fallback audio decoder used by librosa
# libgomp1     — OpenMP runtime required by PyTorch CPU kernels
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only the package manifest first so the dependency layer is cached
# independently from source-code changes.
COPY pyproject.toml .

# Install CPU-only PyTorch first (avoids pulling the much larger CUDA build
# that the default PyPI wheel would select).
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the project dependencies (torch is already satisfied
# above, so pip will skip it and install everything else).
RUN pip install --no-cache-dir \
        "gradio>=6.11.0" \
        "librosa>=0.11.0" \
        "numpy>=2.4.3" \
        "pesq>=0.0.4" \
        "pystoi>=0.4.1" \
        "tqdm>=4.67.3"

# ── Application source ────────────────────────────────────────────────────────
COPY src/ src/
COPY models/ models/

# Install the local package in non-editable mode (no source-watching needed).
RUN pip install --no-cache-dir --no-deps .

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 7860

# Gradio serves on 0.0.0.0:7860 (set explicitly in the showcase script).
CMD ["python", "-m", "aese.showcases.simpleAE_logmag_nc"]

# Hugging Face Space — Docker SDK.
#
# PHASE 1: session replay + model switching. This deliberately does NOT install
# the detector stack (mmcv / mmengine / mediapipe / transformers). Everything
# left of the temporal model is precomputed per session, so switching models
# needs only torch and numpy — see tools/dashboard/README.md. That keeps the
# image small, the cold start fast, and avoids pinning mmcv against a CUDA build
# before anything is known to work.
#
# PHASE 2 (live video / upload) swaps the base image for a CUDA one and installs
# the `research` extra; see deploy/hf_space/README_DEPLOY.md. Do not merge the
# two until phase 1 runs, or a torch/mmcv/CUDA mismatch will be indistinguishable
# from an application bug.

FROM python:3.11-slim

# opencv needs these even in headless builds; git is used by huggingface_hub.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 git \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as uid 1000. Files written at runtime (downloaded weights, the
# session cache) must be owned by it or startup fails on permissions.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Application code only. Weights are NOT baked in: they are pulled at startup
# from a private model repo (see app.py). Baking them would put restricted
# artifacts into every image layer and make the Space impossible to share
# without also sharing them.
COPY --chown=user app.py .
COPY --chown=user tools/ ./tools/
COPY --chown=user LLMDet/attention/ ./LLMDet/attention/
COPY --chown=user LLMDet/configs/ ./LLMDet/configs/

# Spaces routes to 7860.
EXPOSE 7860
CMD ["python", "app.py"]

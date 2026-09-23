# Hugging Face Space — PHASE 2: live and uploaded video (needs a GPU).
#
# Adds the detector to phase 1: GroundingDINO via the vendored LLMDet/mmdet tree,
# CLIP appearance features, and the MediaPipe head-pose backend.
#
# Why python:3.11-slim and not an nvidia/cuda base image:
#   * torch's cu121 wheels bundle the CUDA runtime (libcudart, cuBLAS, cuDNN),
#     and the driver comes from the host, which Spaces provides. A CUDA base
#     image would add ~2 GB of duplicated runtime for nothing.
#   * Python 3.11 matters: pyproject requires >=3.11,<3.12, and the prebuilt
#     mmcv wheel we need is cp311. ubuntu:22.04 ships 3.10, which would mean
#     deadsnakes or a source build of mmcv.
#
# The version triple is not free choice — it is the only combination that has a
# prebuilt mmcv wheel:
#     python 3.11  +  torch 2.2.x/cu121  +  mmcv 2.2.0  (cp311 manylinux)
# Verified present at download.openmmlab.com/mmcv/dist/cu121/torch2.2/.
# Installing mmcv from PyPI instead compiles CUDA extensions, which needs nvcc
# (absent here) and would take longer than a Space build allows.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libsm6 libxext6 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR $HOME/app

# torch first and alone: mmcv's wheel is built against a specific torch ABI, so
# it must resolve against the torch that is already installed rather than pull
# its own.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.2.2 torchvision==0.17.2

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# mmcv from OpenMMLab's wheel index, pinned to the torch/CUDA pair above.
# --no-deps: the wheel's metadata lists opencv-python, which would pull the GUI
# build over the headless one already installed and break in a container with no
# display.
RUN pip install --no-cache-dir --no-deps \
      -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html \
      mmcv==2.2.0

# Fail the BUILD, not the first request, if the ABI is wrong. mmcv._ext is the
# compiled extension; importing it surfaces a torch/CUDA mismatch here as a
# missing-symbol error rather than 20 minutes later when a user uploads a video.
RUN python -c "import torch, mmcv, mmcv._ext, mmengine; \
print('torch', torch.__version__, '| mmcv', mmcv.__version__, '| mmengine', mmengine.__version__)"

COPY --chown=user app.py .
COPY --chown=user tools/ ./tools/
COPY --chown=user LLMDet/ ./LLMDet/

# Fail the BUILD if the vendored tree cannot import, for the same reason the
# mmcv check above exists. register_all_modules() is what the detector calls on
# first use, and it walks the whole datasets + models registry — which is where
# a missing scipy or fairscale surfaces. Without this, such a gap appears only
# when someone uploads a video and gets a ModuleNotFoundError from a background
# job, one missing package per rebuild. sys.path matches what the dashboard
# modules do (tools/dashboard/pipeline_bridge.py:17).
RUN python -c "import sys; sys.path.insert(0, 'LLMDet'); from mmdet.utils import register_all_modules; register_all_modules(); import mmdet.models.detectors.grounding_dino as g; print('mmdet registry imports cleanly:', g.__name__)"

EXPOSE 7860
CMD ["python", "app.py"]

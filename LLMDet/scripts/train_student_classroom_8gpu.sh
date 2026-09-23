#!/usr/bin/env bash
# Fine-tune LLMDet on student classroom VG jsonl (8 GPUs on one node).
# Prerequisites: fixed JSONL (char-level tokens_positive), images under
#   ../grounding_data/stu_img/frames/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/grounding_dino_swin_t_student_classroom.py}"
GPUS="${GPUS:-8}"

echo "Using CONFIG=$CONFIG  GPUS=$GPUS"
bash dist_train.sh "$CONFIG" "$GPUS" --amp "$@"

#!/usr/bin/env bash
set -euo pipefail

# E0/E1/E2 detection-level ablation runner.
# Usage:
#   bash experiments/run_detection_ablation.sh \
#     --gpus 8 \
#     --val-config configs/grounding_dino_swin_t_student_e1.py \
#     --e0-ckpt work_dirs/grounding_dino_swin_t/iter_15000.pth \
#     --e1-ckpt work_dirs/<e1_workdir>/best.pth \
#     --e2-ckpt work_dirs/<e2_workdir>/best.pth \
#     --out-root work_dirs/ablation_e0_e1_e2

GPUS=8
VAL_CONFIG=""
E0_CKPT=""
E1_CKPT=""
E2_CKPT=""
OUT_ROOT="work_dirs/ablation_e0_e1_e2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --val-config) VAL_CONFIG="$2"; shift 2 ;;
    --e0-ckpt) E0_CKPT="$2"; shift 2 ;;
    --e1-ckpt) E1_CKPT="$2"; shift 2 ;;
    --e2-ckpt) E2_CKPT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$VAL_CONFIG" || -z "$E0_CKPT" || -z "$E1_CKPT" || -z "$E2_CKPT" ]]; then
  echo "Missing required arguments."
  exit 1
fi

mkdir -p "$OUT_ROOT"/{e0,e1,e2}

echo "[ablation] Running E0..."
bash dist_test.sh "$VAL_CONFIG" "$E0_CKPT" "$GPUS" --work-dir "$OUT_ROOT/e0" | tee "$OUT_ROOT/e0/test.log"

echo "[ablation] Running E1..."
bash dist_test.sh "$VAL_CONFIG" "$E1_CKPT" "$GPUS" --work-dir "$OUT_ROOT/e1" | tee "$OUT_ROOT/e1/test.log"

echo "[ablation] Running E2..."
bash dist_test.sh "$VAL_CONFIG" "$E2_CKPT" "$GPUS" --work-dir "$OUT_ROOT/e2" | tee "$OUT_ROOT/e2/test.log"

echo "[ablation] Done. Logs under $OUT_ROOT/{e0,e1,e2}/test.log"

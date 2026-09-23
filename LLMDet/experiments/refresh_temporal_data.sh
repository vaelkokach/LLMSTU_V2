#!/usr/bin/env bash
set -euo pipefail

# Rebuild train/val temporal sequences and move NPZs to trainer-expected layout.
# Run from LLMDet root.

TRAIN_JSONL=${TRAIN_JSONL:-"../grounding_data/stu_img/student_behavior_emotion_vg7_train.jsonl"}
VAL_JSONL=${VAL_JSONL:-"../grounding_data/stu_img/student_behavior_emotion_vg7_val.jsonl"}
IMG_ROOT=${IMG_ROOT:-"../grounding_data/stu_img/frames"}
OUT_ROOT=${OUT_ROOT:-"../grounding_data/stu_img/attention_sequences"}

echo "[refresh] train jsonl: $TRAIN_JSONL"
echo "[refresh] val jsonl: $VAL_JSONL"
echo "[refresh] image root: $IMG_ROOT"
echo "[refresh] out root: $OUT_ROOT"

mkdir -p "$OUT_ROOT"/train "$OUT_ROOT"/val

python attention_build_sequences.py --jsonl "$TRAIN_JSONL" --image-root "$IMG_ROOT" --output-dir "$OUT_ROOT/train"
python attention_build_sequences.py --jsonl "$VAL_JSONL" --image-root "$IMG_ROOT" --output-dir "$OUT_ROOT/val"

if compgen -G "$OUT_ROOT/train/sequences/*.npz" > /dev/null; then
  mv "$OUT_ROOT"/train/sequences/*.npz "$OUT_ROOT"/train/
fi
if compgen -G "$OUT_ROOT/val/sequences/*.npz" > /dev/null; then
  mv "$OUT_ROOT"/val/sequences/*.npz "$OUT_ROOT"/val/
fi

echo "[refresh] npz counts:"
echo "train_npz=$(ls -1 "$OUT_ROOT"/train/*.npz 2>/dev/null | wc -l)"
echo "val_npz=$(ls -1 "$OUT_ROOT"/val/*.npz 2>/dev/null | wc -l)"

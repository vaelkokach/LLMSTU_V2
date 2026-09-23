#!/usr/bin/env bash
# The full_det taxonomy family at one epoch budget. Produced [internal notes, not included]
#
#   cd LLMDet && bash ../tools/run_taxonomy_family.sh 240
#
# Four taxonomies x three seeds over the same features and the same split, so
# the only thing that differs between them is the target. cue9 additionally
# needs its sidecar label build:
#
#   python -m attention.thesis_eval.build_cue_labels \
#       --sequence-root ../grounding_data/llmstu_sequences_full_det \
#       --manifest ../grounding_data/llmstu_seq_split_manifest.json \
#       --label-space cue9 --out ../grounding_data/cue_labels_cue9_full_det.npz
#
# Every run in this family selected at or near epoch 89 of 90 with loss still
# falling -- cue9, the cue6 baseline and 2 of 3 coarse3_reliable seeds. So the
# budget, not the taxonomy, was the binding constraint, and every published
# number from it is a floor.
#
# New output tree on purpose: the 90-epoch checkpoints are what the current
# [internal notes, not included] numbers and the model registry reference, and overwriting them would
# silently restate published results.
#
# Scheduling: one sequential CHAIN per GPU, three chains in parallel. No lock
# files -- two jobs polling one lock can wake together and share a GPU.
set -u
EPOCHS="${1:?usage: run_family.sh <epochs>}"
OUT="work_dirs/thesis/epochs${EPOCHS}"
LABELS=../grounding_data/cue_labels_cue9_full_det.npz
ROOT=../grounding_data/llmstu_sequences_full_det
MANIFEST=../grounding_data/llmstu_seq_split_manifest.json
mkdir -p "$OUT"

run_one() {                       # gpu taxonomy seed [extra...]
  local GPU="$1" TAX="$2" SEED="$3"; shift 3
  local ID="mstcn_556_${TAX}_s${SEED}"
  CUDA_VISIBLE_DEVICES="$GPU" python -m attention.thesis_eval.train \
    --experiment-id "$ID" --model mstcn --feature-config 556_hp \
    --taxonomy "$TAX" "$@" \
    --sequence-root "$ROOT" --manifest "$MANIFEST" \
    --seed "$SEED" --epochs "$EPOCHS" --batch-size 32 --lr 3e-4 \
    --select-window 5 --output-dir "$OUT/$ID" --device cuda:0 \
    > "$OUT/$ID.log" 2>&1 || { echo "TRAIN FAILED $ID"; return 1; }
  CUDA_VISIBLE_DEVICES="$GPU" python -m attention.thesis_eval.run_eval \
    --ckpt "$OUT/$ID/checkpoints/best.pth" --split val \
    --out "$OUT/$ID/eval_val" --device cuda:0 >> "$OUT/$ID.log" 2>&1 \
    || { echo "EVAL FAILED $ID"; return 1; }
  echo "done $ID"
}

# One seed per GPU, all four taxonomies in sequence on that GPU. Grouping by
# SEED rather than by taxonomy means a GPU dying costs one seed of everything
# rather than every seed of one taxonomy.
chain() {
  local GPU="$1" SEED="$2"
  run_one "$GPU" cue6             "$SEED"
  run_one "$GPU" onoff_reliable   "$SEED"
  run_one "$GPU" coarse3_reliable "$SEED"
  run_one "$GPU" cue9             "$SEED" --cue-labels "$LABELS"
}

chain 0 42 > "$OUT/chain_gpu0.log" 2>&1 &
chain 1 43 > "$OUT/chain_gpu1.log" 2>&1 &
chain 2 44 > "$OUT/chain_gpu2.log" 2>&1 &
wait
echo "FAMILY RUN COMPLETE: $OUT"

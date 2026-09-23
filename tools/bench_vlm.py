#!/usr/bin/env python
"""How good is a VLM's second opinion, and how fast, measured not asserted.

Scores each crop in the HUMAN gold set with option-likelihood over the cue
phrases -- the same question and the same single-forward-pass read that
`QwenGrounder.score_students` uses in the pipeline -- and compares to the human
label. Human gold, not the Qwen3.5-VL pseudo-labels: a Qwen backbone scored
against Qwen-made labels measures agreement with itself.

    python tools/bench_vlm.py --model Qwen/Qwen2-VL-2B-Instruct --limit 300

Needs transformers >= 4.45 for Qwen2-VL and >= 4.49 for Qwen2.5-VL, which is
newer than the pinned 4.44.2 the detector needs -- run it from a venv that has
one, never by upgrading the environment mmcv is built against.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "LLMDet"))

GOLD = REPO / "Gold_annotation_wael" / "gold_annotations_wael.jsonl"
CROPS = REPO / "grounding_data" / "LLMSTU" / "crops"


def load_gold(taxonomy, limit, stride):
    """(crop path, gold class id) for records the taxonomy can label."""
    from attention import taxonomy as TX
    if taxonomy == "cue9":
        classes, mapper = list(TX.CUE9_CLASSES), TX.map_record_cue9
    else:
        classes, mapper = list(TX.CUE_CLASSES), TX.map_record
    rows, skipped = [], Counter()
    for i, line in enumerate(GOLD.read_text().splitlines()):
        if i % stride:
            continue
        rec = json.loads(line)
        if rec.get("status") != "ok":
            skipped["status not ok"] += 1
            continue
        p = CROPS / rec["file_name"]
        if not p.exists():
            skipped["crop missing"] += 1
            continue
        y = mapper(rec)
        # `uncertain` is an abstention, not a behaviour: there is nothing in the
        # image for the VLM to be right about, and scoring it would reward
        # guessing the annotator's uncertainty.
        if classes[y] == "uncertain":
            skipped["gold is uncertain"] += 1
            continue
        rows.append((p, y))
        if limit and len(rows) >= limit:
            break
    return classes, rows, skipped


def macro_f1(y_true, y_pred, k):
    out = []
    for c in range(k):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        if tp + fn == 0:                 # class absent from the gold sample
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        out.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(out) / len(out) if out else 0.0, len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--taxonomy", default="cue6", choices=("cue6", "cue9"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth record; the file is video-ordered, so "
                         "a bare --limit samples one stretch of one room")
    ap.add_argument("--batch", type=int, default=8,
                    help="students per forward pass, as one frame would give")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--rotate-options", type=int, default=0,
                    help="rotate the class->letter mapping by N. The answer is "
                         "a letter, so a model with a position prior answers "
                         "the same LETTER whichever class sits there. Rotating "
                         "separates that from vision: a real opinion follows "
                         "the class, a prior follows the letter.")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import torch
    import transformers
    from PIL import Image
    from attention.vlm_grounder import build_prompt, option_letters

    classes, rows, skipped = load_gold(a.taxonomy, a.limit, a.stride)
    print(f"model      : {a.model}")
    print(f"transformers {transformers.__version__} | torch {torch.__version__}")
    print(f"crops      : {len(rows)}  (skipped: {dict(skipped)})")
    print(f"gold dist  : {Counter(classes[y] for _, y in rows).most_common()}")
    if not rows:
        sys.exit("no scorable crops")

    t0 = time.time()
    Loader = getattr(transformers, "AutoModelForImageTextToText", None) \
        or transformers.AutoModelForVision2Seq
    proc = transformers.AutoProcessor.from_pretrained(a.model)
    model = Loader.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype),
                                   low_cpu_mem_usage=True).to(a.device).eval()
    load_s = time.time() - t0
    print(f"loaded in  : {load_s:.1f}s")

    tok = proc.tokenizer
    letter_ids = []
    for L in option_letters(len(classes)):
        enc = tok.encode(L, add_special_tokens=False)
        assert len(enc) == 1, f"{L!r} is {len(enc)} tokens under {a.model}"
        letter_ids.append(enc[0])

    # order[i] is the class shown at option i; inv maps a chosen option back to
    # a class id, so y_pred stays in class space and the metrics are comparable.
    r = a.rotate_options % len(classes)
    order = classes[r:] + classes[:r]
    inv = [classes.index(c) for c in order]
    prompt = build_prompt(order)
    if r:
        print(f"rotated by {r}: option A is now {order[0]!r}")
    y_true, y_pred, lat = [], [], []
    for s in range(0, len(rows), a.batch):
        chunk = rows[s:s + a.batch]
        images = [Image.open(p).convert("RGB") for p, _ in chunk]
        msgs = [[{"role": "user", "content": [{"type": "image"},
                                              {"type": "text", "text": prompt}]}]
                for _ in chunk]
        texts = [proc.apply_chat_template(m, tokenize=False,
                                          add_generation_prompt=True)
                 for m in msgs]
        # Qwen's processor takes a FLAT image list alongside a batch of texts;
        # SmolVLM's takes one list per sample and refuses the flat form
        # ("The number of images in the text [1,1,...] and images [8] should be
        # the same"). Neither is wrong; they are different conventions, and a
        # benchmark that only spoke one would report a model as unusable when it
        # is merely shaped differently.
        try:
            inputs = proc(text=texts, images=images, return_tensors="pt",
                          padding=True).to(a.device)
        except ValueError:
            inputs = proc(text=texts, images=[[im] for im in images],
                          return_tensors="pt", padding=True).to(a.device)
        # Qwen's processor returns pixel_values already in the model's dtype;
        # SmolVLM's returns float32 against fp16 weights and the first conv
        # raises. Cast the floating inputs rather than running the model in
        # fp32, which would make the latency comparison meaningless.
        td = next(model.parameters()).dtype
        inputs = {k: (v.to(td) if hasattr(v, "is_floating_point")
                      and v.is_floating_point() else v)
                  for k, v in inputs.items()}
        torch.cuda.synchronize()
        t1 = time.time()
        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1, :].float()
        torch.cuda.synchronize()
        dt = time.time() - t1
        lat.append((dt, len(chunk)))
        pred = logits[:, letter_ids].argmax(-1).cpu().tolist()
        y_pred += [inv[i] for i in pred]
        y_true += [y for _, y in chunk]
        if s % (a.batch * 10) == 0:
            print(f"  {s + len(chunk)}/{len(rows)}", flush=True)

    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    f1, n_present = macro_f1(y_true, y_pred, len(classes))
    # Batch 1 is the honest per-student figure only if the pipeline ran one at a
    # time; it does not. Report both: per-BATCH is what a frame costs.
    warm = lat[1:] or lat                    # first batch pays the CUDA warm-up
    per_batch = sum(d for d, _ in warm) / len(warm)
    per_crop = sum(d for d, _ in warm) / sum(n for _, n in warm)
    print(f"\n== {a.model} / {a.taxonomy} / n={len(y_true)} ==")
    print(f"accuracy      {acc:.4f}")
    print(f"macro-F1      {f1:.4f}  (over {n_present} classes present in gold)")
    print(f"per batch     {per_batch * 1e3:.0f} ms  (batch={a.batch})")
    print(f"per crop      {per_crop * 1e3:.0f} ms")
    print(f"frames/s      {1.0 / per_batch:.2f}  if one batch is one frame")
    print("\nper-class (gold -> predicted):")
    for c in range(len(classes)):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        n = sum(1 for t in y_true if t == c)
        if n:
            top = Counter(p for t, p in zip(y_true, y_pred) if t == c).most_common(2)
            print(f"  {classes[c]:16} n={n:4} recall={tp / n:.3f}  "
                  f"top={[(classes[i], k) for i, k in top]}")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"model": a.model, "taxonomy": a.taxonomy, "n": len(y_true),
             "rotate_options": r, "option_order": order,
             "accuracy": acc, "macro_f1": f1, "n_classes_present": n_present,
             "ms_per_batch": per_batch * 1e3, "ms_per_crop": per_crop * 1e3,
             "batch": a.batch, "load_s": load_s,
             "transformers": transformers.__version__,
             "y_true": y_true, "y_pred": y_pred, "classes": classes}, indent=1))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

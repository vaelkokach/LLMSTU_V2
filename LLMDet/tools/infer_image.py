#!/usr/bin/env python3
"""Run the fine-tuned detector on any image (or folder) with any text prompt.

Prints every detection above --score-thr and writes an annotated copy.

The default threshold here is 0.15, NOT the 0.45 in attention_temporal.yaml.
That 0.45 was inherited from the pre-fine-tuning model; the word "student"
appears in 0 of 36,339 fine-tuning captions, so it scores low on the fine-tuned
head. Measured on iter_5000: recall [value removed] at 0.45 vs [value removed] at 0.15.

Usage:
  PYTHONPATH=/home/jovyan/Computer_vision/LLMDet python tools/infer_image.py \
      --ckpt work_dirs/student_llmstu_exact/iter_5000.pth \
      --image /path/to/frame.jpg --prompt "student"
"""
import argparse
from pathlib import Path

CFG = "configs/student_llmstu_exact.py"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True, help="image file or a folder")
    ap.add_argument("--prompt", default="student",
                    help='text prompt, e.g. "student" or "using phone"')
    ap.add_argument("--score-thr", type=float, default=0.15)
    ap.add_argument("--device", default="cpu",
                    help="cpu (safe while training runs) or cuda:N")
    ap.add_argument("--out", default="work_dirs/infer_out")
    ap.add_argument("--max-images", type=int, default=20)
    args = ap.parse_args()

    import cv2
    from mmdet.apis import init_detector, inference_detector

    src = Path(args.image)
    files = ([p for p in sorted(src.iterdir()) if p.suffix.lower() in IMG_EXT]
             [:args.max_images] if src.is_dir() else [src])
    if not files:
        raise SystemExit(f"no images found at {src}")

    print(f"loading {args.ckpt} on {args.device} ...")
    model = init_detector(CFG, args.ckpt, device=args.device)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for f in files:
        im = cv2.imread(str(f))
        if im is None:
            print(f"  !! unreadable: {f}")
            continue
        out = inference_detector(model, im, text_prompt=args.prompt,
                                 custom_entities=True)
        p = out.pred_instances
        boxes = p.bboxes.cpu().numpy()
        scores = p.scores.cpu().numpy()
        keep = scores >= args.score_thr

        print(f"\n{f.name}  prompt={args.prompt!r}  "
              f"{keep.sum()} detections >= {args.score_thr}"
              f"  (max score {scores.max():.3f})" if len(scores) else
              f"\n{f.name}  no detections at all")
        vis = im.copy()
        for b, s in sorted(zip(boxes[keep], scores[keep]), key=lambda t: -t[1]):
            x1, y1, x2, y2 = [int(v) for v in b]
            print(f"    score {s:.3f}  box ({x1},{y1})-({x2},{y2})")
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 128, 255), 2)
            cv2.putText(vis, f"{s:.2f}", (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
        cv2.putText(vis, f"{args.prompt}  thr={args.score_thr}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        dst = outdir / f"{f.stem}__{args.prompt.replace(' ', '_')}.jpg"
        cv2.imwrite(str(dst), vis)
        print(f"    -> {dst}")


if __name__ == "__main__":
    main()

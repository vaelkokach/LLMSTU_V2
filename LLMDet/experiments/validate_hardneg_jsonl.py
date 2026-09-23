import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


NEG_TERMS = [
    "monitor",
    "screen",
    "desk",
    "table",
    "chair",
    "keyboard",
    "mouse",
    "phone",
    "book",
    "bag",
]


def parse_args():
    p = argparse.ArgumentParser(description="Validate hard-negative JSONL integrity and term coverage.")
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--image-root", type=str, required=True)
    p.add_argument("--sample-lines", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    image_root = Path(args.image_root)
    if not jsonl_path.exists():
        raise RuntimeError(f"JSONL not found: {jsonl_path}")
    if not image_root.exists():
        raise RuntimeError(f"Image root not found: {image_root}")

    total = 0
    malformed = 0
    missing_images = 0
    term_hits: Dict[str, int] = {t: 0 for t in NEG_TERMS}
    lines: List[str] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            lines.append(line)
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue

            fname = obj.get("filename", "")
            if not (image_root / fname).exists():
                missing_images += 1

            text_pool = []
            text_pool.extend(obj.get("tags", []))
            grounding = obj.get("grounding", {})
            text_pool.append(grounding.get("caption", ""))
            for reg in grounding.get("regions", []):
                text_pool.append(reg.get("phrase", ""))
            text = " ".join([str(x) for x in text_pool]).lower()
            for t in NEG_TERMS:
                if t in text:
                    term_hits[t] += 1

    if total == 0:
        raise RuntimeError("JSONL is empty.")
    if malformed > 0:
        raise RuntimeError(f"Malformed JSONL lines: {malformed}/{total}")
    if missing_images > 0:
        raise RuntimeError(f"Missing images: {missing_images}/{total}")

    missing_terms = [k for k, v in term_hits.items() if v == 0]
    if missing_terms:
        raise RuntimeError(f"Hard-negative terms not found in dataset: {missing_terms}")

    sample_count = min(args.sample_lines, len(lines))
    sampled = random.sample(lines, sample_count)
    print(f"total_lines={total}")
    print(f"malformed={malformed}")
    print(f"missing_images={missing_images}")
    print(f"term_hits={term_hits}")
    print(f"sampled_lines={sample_count}")
    print("sample_preview_first_3:")
    for x in sampled[:3]:
        print(x[:280] + ("..." if len(x) > 280 else ""))


if __name__ == "__main__":
    main()

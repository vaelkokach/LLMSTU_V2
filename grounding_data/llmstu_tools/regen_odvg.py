#!/usr/bin/env python3
"""Regenerate a clean whole-frame ODVG jsonl from LLMSTU structured labels.

Why: the old ODVG file (`stu_img/annotations/...`) bound "Student N" phrases to
detector boxes in arbitrary output order — the caption<->box correspondence is
essentially random for multi-student frames, and 39.8% of entries have no
regions at all. LLMSTU per-student crops carry `src_frame` + `bbox_person` +
closed-vocabulary labels, so an ODVG file with EXACT box<->phrase
correspondence can be rebuilt by grouping crops per source frame.

Outputs (into outputs/):
  odvg_{train,val,test}.jsonl   ODVG entries (one per src_frame; regions carry
                                bbox=bbox_person, phrase = activity template,
                                char-offset tokens_positive)
  correspondence_gt.jsonl       per src_frame the exact (bbox, labels) pairs —
                                matching-evaluation ground truth for the
                                ordinal/Hungarian/OT comparison
Caption neutralisation: protected-attribute descriptors (religious dress,
gender terms, other appearance identifiers) are stripped/replaced so they never
enter the text supervision. Rules are in NEUTRAL_RULES below.
"""
import json
import os
import re
from collections import defaultdict

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
FRAME_W, FRAME_H = 2812, 1050

# activity -> <=3-word grounding phrase (matches pipeline MAX_PHRASE_WORDS=3)
ACTIVITY_PHRASE = {
    "listening": "listening attentively",
    "using_laptop": "using laptop",
    "head_down_sleeping": "sleeping head down",
    "looking_away": "looking away",
    "reading": "reading",
    "other": "sitting at desk",
    "using_phone": "using phone",
    "talking_to_peer": "talking to peer",
    "writing_notes": "writing notes",
    "eating_drinking": "eating or drinking",
    "raising_hand": "raising hand",
}

# order matters: longer / more specific patterns first
_HEADWEAR = r"(?:hijab|niqab|headscarf|head\s?scarf|head covering|turban)"
_ADJ = r"(?:(?:black|brown|white|grey|gray|dark|light|patterned|colorful|floral)\s+)*"
NEUTRAL_RULES = [
    # remove only the descriptor phrase, keep the rest of the sentence
    (re.compile(rf"\s*\bwearing (?:a |an )?{_ADJ}{_HEADWEAR}\b\s*", re.I), " "),
    (re.compile(rf"\s*\b(?:in|with) (?:a |an )?{_ADJ}{_HEADWEAR}\b\s*", re.I), " "),
    (re.compile(rf"\b{_HEADWEAR}\b", re.I), "clothing"),
    (re.compile(r"\b(?:male|female) student\b", re.I), "student"),
    (re.compile(r"\s*(?:\band\b|,)\s*(?:a |an )?(?:beard|moustache|mustache)\b", re.I), ""),
    (re.compile(r"\s*\bwith (?:a |an )?(?:beard|moustache|mustache)\b\s*", re.I), " "),
    (re.compile(r"\b(?:bearded)\s+", re.I), ""),
    (re.compile(r"\s*\bor (?:a |an )?(?:beard|moustache|mustache)\b", re.I), ""),
    (re.compile(r"\b(?:beard|moustache|mustache)\b", re.I), "face"),
    (re.compile(r"\bhe\b|\bshe\b", re.I), "they"),
    (re.compile(r"\bhis\b|\bher\b", re.I), "their"),
    (re.compile(r"\bhim\b", re.I), "them"),
    (re.compile(r"\s{2,}"), " "),
    (re.compile(r"\s+([,.])"), r"\1"),
]


def neutralize(text):
    for pat, rep in NEUTRAL_RULES:
        text = pat.sub(rep, text)
    return text.strip()


def build_entry(src_frame, recs):
    """One ODVG entry per source frame with exact region correspondence."""
    recs = sorted(recs, key=lambda r: r["bbox_person"][0])  # left-to-right
    phrases, regions = [], []
    for r in recs:
        phrases.append(ACTIVITY_PHRASE[r["activity"]])
    # caption = ". ".join of unique-ordered phrases (repeat phrases share span)
    caption = ". ".join(phrases) + "."
    cursor_cache = {}
    for r, ph in zip(recs, phrases):
        # first occurrence of the phrase in caption (shared across repeats is
        # fine — LLMDet maps every box carrying that span to the same tokens)
        if ph not in cursor_cache:
            s = caption.lower().find(ph.lower())
            cursor_cache[ph] = (s, s + len(ph))
        s, e = cursor_cache[ph]
        regions.append({
            "bbox": [float(x) for x in r["bbox_person"]],
            "phrase": ph,
            "tokens_positive": [[s, e]],
        })
    return {
        "filename": src_frame,
        "height": FRAME_H, "width": FRAME_W,
        "grounding": {"caption": caption, "regions": regions},
    }


def main():
    splits = json.load(open(os.path.join(OUT_DIR, "splits.json")))
    vid_split = {v: s for s, vids in splits.items() for v in vids}

    # Full per-frame contents (every student must be boxed in a training
    # frame, else unlabeled students become false negatives for the detector).
    by_frame = defaultdict(list)
    with open(os.path.join(OUT_DIR, "labels_tracked.jsonl")) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["seat_id"] < 0:
                continue
            by_frame[rec["src_frame"]].append(rec)

    # Dedup decides WHICH frames enter the ODVG training files.
    selected_frames = set()
    with open(os.path.join(OUT_DIR, "labels_dedup.jsonl")) as fh:
        for line in fh:
            selected_frames.add(json.loads(line)["src_frame"])

    writers = {s: open(os.path.join(OUT_DIR, f"odvg_{s}.jsonl"), "w")
               for s in splits}
    gt = open(os.path.join(OUT_DIR, "correspondence_gt.jsonl"), "w")
    counts = defaultdict(int)
    region_counts = defaultdict(int)
    for src_frame in sorted(by_frame):
        recs = by_frame[src_frame]
        split = vid_split.get(recs[0]["video_id"])
        if split is None:
            continue
        if src_frame in selected_frames:
            entry = build_entry(src_frame, recs)
            writers[split].write(json.dumps(entry) + "\n")
            counts[split] += 1
            region_counts[split] += len(entry["grounding"]["regions"])
        # correspondence GT covers ALL frames (matching evaluation wants the
        # complete multi-student scenes, not the dedup selection)
        gt.write(json.dumps({
            "src_frame": src_frame,
            "video_id": recs[0]["video_id"],
            "split": split,
            "students": [{
                "bbox_person": r["bbox_person"],
                "seat_id": r["seat_id"],
                "file_name": r["file_name"],
                "caption_neutral": neutralize(r["caption"]),
                **{k: r[k] for k in (
                    "activity", "gaze_direction", "attention_target",
                    "engagement_level", "posture", "hand_state",
                    "phone_visible", "laptop_visible", "talking", "occluded",
                    "det_conf", "model_confidence")},
            } for r in recs],
        }) + "\n")
    for s, w in writers.items():
        w.close()
        print(f"odvg_{s}.jsonl: {counts[s]} frames, {region_counts[s]} regions "
              f"({region_counts[s]/max(counts[s],1):.2f}/frame)")
    gt.close()
    print(f"correspondence_gt.jsonl: {sum(counts.values())} frames")


if __name__ == "__main__":
    main()

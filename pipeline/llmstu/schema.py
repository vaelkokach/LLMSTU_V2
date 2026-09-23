"""Structured caption schema for classroom engagement / attention.

The pseudo-labeler (Qwen3-VL) is asked to emit JSON matching `FIELDS`.
The same schema drives the manual labeling tool, so keep the two in sync
(the labeling UI is generated from `FIELDS` via scripts/03_sample_eval.py).
"""

from __future__ import annotations

from typing import Any, Dict

# Each field: (type, allowed values or None, human help text).
# type is one of: "enum", "bool", "float", "text".
FIELDS: Dict[str, Dict[str, Any]] = {
    "activity": {
        "type": "enum",
        "values": [
            "listening",
            "writing_notes",
            "reading",
            "using_phone",
            "using_laptop",
            "talking_to_peer",
            "raising_hand",
            "head_down_sleeping",
            "looking_away",
            "eating_drinking",
            "other",
        ],
        "help": "Primary activity the student is doing.",
    },
    "gaze_direction": {
        "type": "enum",
        "values": [
            "teacher_or_board",
            "own_desk",
            "peer",
            "phone",
            "laptop",
            "away_or_window",
            "down",
            "unknown",
        ],
        "help": "Where the student is looking.",
    },
    "attention_target": {
        "type": "enum",
        "values": ["instruction", "own_work", "peer", "device", "distracted", "unknown"],
        "help": "What the student's attention is on.",
    },
    "engagement_level": {
        "type": "enum",
        "values": ["engaged", "partially_engaged", "disengaged", "unknown"],
        "help": "Overall engagement judgement.",
    },
    "posture": {
        "type": "enum",
        "values": [
            "upright",
            "leaning_forward",
            "leaning_back",
            "slumped",
            "head_down",
            "turned_away",
            "unknown",
        ],
        "help": "Body posture.",
    },
    "hand_state": {
        "type": "enum",
        "values": [
            "writing",
            "on_phone",
            "on_desk_idle",
            "raised",
            "on_face",
            "gesturing",
            "unknown",
        ],
        "help": "What the hands are doing.",
    },
    "phone_visible": {"type": "bool", "help": "Is a phone visible with this student?"},
    "laptop_visible": {"type": "bool", "help": "Is a laptop/tablet visible with this student?"},
    "talking": {"type": "bool", "help": "Does the student appear to be talking?"},
    "occluded": {"type": "bool", "help": "Is the student heavily occluded / hard to read?"},
    "caption": {"type": "text", "help": "One-sentence natural-language summary."},
}

# Fields the model self-rates; not part of the label but kept for triage.
MODEL_CONFIDENCE_FIELD = "model_confidence"  # float 0..1


def empty_label() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, spec in FIELDS.items():
        if spec["type"] == "enum":
            out[name] = "unknown" if "unknown" in spec["values"] else spec["values"][-1]
        elif spec["type"] == "bool":
            out[name] = False
        else:
            out[name] = ""
    return out


def coerce(label: Dict[str, Any]) -> Dict[str, Any]:
    """Force a raw model dict into a schema-valid label (never raises)."""
    out = empty_label()
    if not isinstance(label, dict):
        return out
    for name, spec in FIELDS.items():
        if name not in label:
            continue
        val = label[name]
        if spec["type"] == "enum":
            sval = str(val).strip().lower().replace(" ", "_")
            out[name] = sval if sval in spec["values"] else out[name]
        elif spec["type"] == "bool":
            if isinstance(val, bool):
                out[name] = val
            else:
                out[name] = str(val).strip().lower() in ("true", "yes", "1", "y")
        else:
            out[name] = str(val).strip()
    return out


def prompt_spec() -> str:
    """Render the schema as an instruction block for the VLM prompt."""
    lines = []
    for name, spec in FIELDS.items():
        if spec["type"] == "enum":
            opts = " | ".join(spec["values"])
            lines.append(f'- "{name}": one of [{opts}]  # {spec["help"]}')
        elif spec["type"] == "bool":
            lines.append(f'- "{name}": true or false  # {spec["help"]}')
        else:
            lines.append(f'- "{name}": short string  # {spec["help"]}')
    lines.append(f'- "{MODEL_CONFIDENCE_FIELD}": number 0.0-1.0  # your confidence in this reading')
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Schema VARIANTS — A/B different field sets (see scripts/08_tune_schema.py).
# `FIELDS` above is the "full" schema and the live/active one. set_active() swaps
# it, so prompt_spec()/coerce()/empty_label() and everything that reads
# schema.FIELDS follow the chosen variant.
# --------------------------------------------------------------------------- #
def _drop(base: Dict[str, Dict[str, Any]], *names: str) -> Dict[str, Dict[str, Any]]:
    return {k: v for k, v in base.items() if k not in names}


# gaze with fewer, less-overlapping categories (8 -> 6) to lift reliability
_GAZE_MERGED = {
    "type": "enum",
    "values": ["teacher_or_board", "own_work", "peer", "device", "away", "unknown"],
    "help": "Where the student is looking (merged categories).",
}

SCHEMAS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "full": FIELDS,
    "no_attention_target": _drop(FIELDS, "attention_target"),   # drop redundant field
    "gaze_merged": {**FIELDS, "gaze_direction": _GAZE_MERGED},   # fewer gaze categories
    # drop attention_target AND merge gaze categories (keeps everything else)
    "no_attn_gaze_merged": _drop({**FIELDS, "gaze_direction": _GAZE_MERGED},
                                 "attention_target"),
    "lean": _drop({**FIELDS, "gaze_direction": _GAZE_MERGED},
                  "attention_target", "occluded"),               # trimmed engagement set
}


def set_active(name: str) -> None:
    """Make schema variant `name` the live schema (mutates module-level FIELDS)."""
    global FIELDS
    if name not in SCHEMAS:
        raise KeyError(f"unknown schema '{name}'. choices: {list(SCHEMAS)}")
    FIELDS = SCHEMAS[name]

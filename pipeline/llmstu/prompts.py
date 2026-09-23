"""Prompt variants for the VLM pseudo-labeler.

These are what you A/B during tuning (scripts/06_tune_captions.py). Add your own
by registering another function in PROMPTS. Each returns the full instruction
string; the crop image is attached separately as an image message.
"""

from __future__ import annotations

from typing import Callable, Dict

from . import schema

_GUARD = (
    "Look only at the student in the center of the crop. If something is not "
    "visible, use \"unknown\" (or false for yes/no fields). Do NOT guess "
    "identity, age, gender, ethnicity, or name — label behaviour only."
)


def v1_structured() -> str:
    """Detailed, strict, enumerated — the default."""
    return (
        "You are labeling a cropped image of ONE student in a classroom for a "
        "dataset about engagement and attention.\n"
        f"{_GUARD}\n\n"
        "Return ONLY a single JSON object with EXACTLY these keys:\n"
        f"{schema.prompt_spec()}\n\n"
        "Output the JSON object and nothing else."
    )


def v2_concise() -> str:
    """Minimal wording — tests whether the model needs the long spec."""
    return (
        "Label this single classroom student for engagement/attention. "
        f"{_GUARD}\n"
        "Reply with ONLY this JSON (no prose):\n"
        f"{schema.prompt_spec()}"
    )


def v3_reasoned() -> str:
    """Ask for a one-line visual justification before the fields (kept inside JSON)."""
    return (
        "You are labeling ONE student in a classroom crop for an engagement/attention "
        f"dataset.\n{_GUARD}\n\n"
        "Return ONLY a JSON object. First set \"caption\" to one factual sentence about "
        "what you SEE (pose, gaze, hands, objects). Then fill the remaining fields so they "
        "are consistent with that sentence:\n"
        f"{schema.prompt_spec()}\n\n"
        "JSON only."
    )


def v4_fewshot() -> str:
    """Include one worked example to anchor formatting."""
    example = (
        '{"activity":"using_phone","gaze_direction":"phone","attention_target":"device",'
        '"engagement_level":"disengaged","posture":"leaning_back","hand_state":"on_phone",'
        '"phone_visible":true,"laptop_visible":false,"talking":false,"occluded":false,'
        '"caption":"a student looking down at a phone held in their lap","model_confidence":0.82}'
    )
    return (
        "You label ONE classroom student per crop for engagement/attention.\n"
        f"{_GUARD}\n\n"
        "Keys and allowed values:\n"
        f"{schema.prompt_spec()}\n\n"
        f"Example of a correct answer:\n{example}\n\n"
        "Now output ONLY the JSON for this crop."
    )


PROMPTS: Dict[str, Callable[[], str]] = {
    "v1_structured": v1_structured,
    "v2_concise": v2_concise,
    "v3_reasoned": v3_reasoned,
    "v4_fewshot": v4_fewshot,
}


def get(name: str) -> str:
    if name not in PROMPTS:
        raise KeyError(f"unknown prompt '{name}'. choices: {list(PROMPTS)}")
    return PROMPTS[name]()

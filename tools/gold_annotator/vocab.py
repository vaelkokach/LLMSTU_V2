"""Closed label vocabulary of the LLMSTU dataset (order = cycling order in the UI)."""

CATEGORICAL_FIELDS = {
    "activity": [
        "listening", "using_laptop", "head_down_sleeping", "looking_away",
        "reading", "using_phone", "talking_to_peer", "writing_notes",
        "eating_drinking", "raising_hand", "other",
    ],
    "gaze_direction": [
        "laptop", "teacher_or_board", "own_desk", "down",
        "away_or_window", "phone", "peer", "unknown",
    ],
    "attention_target": [
        "device", "instruction", "own_work", "distracted", "peer", "unknown",
    ],
    "engagement_level": [
        "engaged", "partially_engaged", "disengaged", "unknown",
    ],
    "posture": [
        "upright", "leaning_forward", "leaning_back", "head_down",
        "turned_away", "slumped", "unknown",
    ],
    "hand_state": [
        "unknown", "on_face", "on_desk_idle", "on_phone",
        "gesturing", "writing", "raised",
    ],
}

BOOLEAN_FIELDS = ["phone_visible", "laptop_visible", "talking", "occluded"]

ALL_LABEL_FIELDS = list(CATEGORICAL_FIELDS) + BOOLEAN_FIELDS

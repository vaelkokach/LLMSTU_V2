import json

from attention.sequence_builder import (
    CropObs,
    _region_to_label,
    assign_seats,
    parse_llmstu_labels,
    split_videos,
)
from attention.taxonomy import CUE_TO_ID


def _obs(cx, cy, t, span=150.0):
    half = 100.0
    return CropObs(
        src_frame=f"t{int(t):06d}_000_f{int(t*20):06d}.jpg",
        time_s=float(t),
        bbox_xyxy=[cx - half, cy - half, cx + half, cy + half],
        label_id=0,
        head_span_px=span,
    )


def test_assign_seats_stationary_students():
    # two students ~800px apart, slight jitter, 50 frames each
    obs = []
    for t in range(50):
        obs.append(_obs(400 + (t % 3), 300, t))
        obs.append(_obs(1200 - (t % 3), 310, t))
    obs.sort(key=lambda o: o.time_s)
    seats = assign_seats(obs)
    assert len(seats) == 2
    sizes = sorted(len(v) for v in seats.values())
    assert sizes == [50, 50]


def test_assign_seats_new_seat_for_far_student():
    obs = [_obs(400, 300, 0), _obs(400, 300, 1), _obs(2000, 800, 2)]
    seats = assign_seats(obs)
    assert len(seats) == 2


def test_split_videos_deterministic_and_disjoint():
    vids = [f"v{i:03d}" for i in range(10)]
    tr1, va1 = split_videos(vids, 0.2, seed=42)
    tr2, va2 = split_videos(vids, 0.2, seed=42)
    assert tr1 == tr2 and va1 == va2
    assert tr1 & va1 == set()
    assert len(va1) == 2
    tr3, va3 = split_videos(vids, 0.2, seed=7)
    assert va3 != va1  # seed changes the draw


def test_parse_llmstu_labels(tmp_path):
    recs = [
        {
            "src_frame": "t000010_000_f000200.jpg",
            "bbox_person": [100, 100, 300, 400],
            "activity": "using_laptop",
            "gaze_direction": "laptop",
            "attention_target": "device",
            "engagement_level": "engaged",
            "posture": "upright",
            "hand_state": "unknown",
            "phone_visible": False,
            "talking": False,
            "occluded": False,
            "face_kpts": 3,
            "head_span_px": 140.0,
            "file_name": "shard_000/part_000/t000010_000_f000200__p00.jpg",
        },
        {
            "src_frame": "t000011_000_f000220.jpg",
            "bbox_person": [100, 100, 300, 400],
            "activity": "using_phone",
            "gaze_direction": "phone",
            "attention_target": "distracted",
            "engagement_level": "disengaged",
            "posture": "upright",
            "hand_state": "on_phone",
            "phone_visible": True,
            "talking": False,
            "occluded": False,
            "face_kpts": 3,
            "head_span_px": 140.0,
            "file_name": "shard_000/part_000/t000011_000_f000220__p00.jpg",
        },
    ]
    lp = tmp_path / "shard_000.jsonl"
    lp.write_text("\n".join(json.dumps(r) for r in recs))
    mapping = {
        "t000010_000_f000200.jpg": "video_A",
        "t000011_000_f000220.jpg": "video_A",
    }
    by_video = parse_llmstu_labels([lp], mapping)
    assert list(by_video.keys()) == ["video_A"]
    assert len(by_video["video_A"]) == 2
    assert by_video["video_A"][0].label_id == CUE_TO_ID["screen_oriented"]
    assert by_video["video_A"][1].label_id == CUE_TO_ID["phone_use"]
    assert by_video["video_A"][0].time_s == 10.0


def test_parse_llmstu_drops_unmapped_without_fallback(tmp_path):
    rec = {
        "src_frame": "t000010_000_f000200.jpg",
        "bbox_person": [0, 0, 10, 10],
        "activity": "listening",
        "head_span_px": 140.0,
    }
    lp = tmp_path / "s.jsonl"
    lp.write_text(json.dumps(rec))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_video = parse_llmstu_labels([lp], {}, allow_filename_fallback=False)
    assert not by_video


def test_legacy_region_label_uses_only_phrase():
    # caption mentions "typing" but the region phrase says "sleeping":
    # the old bug scored the caption too and mislabeled every region.
    label = _region_to_label("sleeping soundly", ["typing"], "students typing on keyboards")
    assert label == 2  # legacy 'sleeping'
    assert _region_to_label("no keywords here", [], "focused typing") is None

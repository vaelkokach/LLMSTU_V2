"""End to end through the replay loop: an id switch keeps the student's cue.

`test_track_handoff.py` pins the mapping. This pins that the replay actually uses
it -- that history, the displayed cue and the seat number all follow the student
key -- because keying one of them by tracker id again would put the "warming up"
label back mid-video while every mapping test still passed.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from attention.taxonomy import CUE_CLASSES                      # noqa: E402
from attention.thesis_eval.runtime import RuntimeBundle         # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "dashboard"))

SR = pytest.importorskip("session_replay")

WIDTH = 4
MINF = 3
SWITCH = 6          # frame at which the tracker re-issues the student's id


class _TopScreen(torch.nn.Module):
    def forward(self, x):
        p = torch.full((len(CUE_CLASSES),), 0.3 / (len(CUE_CLASSES) - 1))
        p[0] = 0.7
        return {"logits": torch.log(p).expand(1, x.shape[1], -1).clone()}


class _SwitchingCache:
    """1 fps. Student at seat A is tracker id 1 until SWITCH, then id 7 in the
    same place; the student at seat B keeps id 2 throughout."""

    def __init__(self, n_frames=12, new_seat=False):
        self.meta = {"fps": 1.0, "source_width": 100,
                     "inference": {"window_size": 8, "temporal_stride": 1,
                                   "min_frames_for_pred": MINF,
                                   "temporal_input_fps": 1.0}}
        self.frame_index = np.arange(n_frames)
        tids, boxes, self._by_frame = [], [], {}
        for f in range(n_frames):
            self._by_frame[f] = []
            a_id = 1 if f < SWITCH else 7
            a_box = [0.0, 0.0, 8.0, 20.0] if (f < SWITCH or not new_seat) \
                else [60.0, 0.0, 68.0, 20.0]
            for tid, box in ((a_id, a_box), (2, [30.0, 0.0, 38.0, 20.0])):
                self._by_frame[f].append(len(tids))
                tids.append(tid)
                boxes.append(box)
        self.track_id, self.bbox = np.array(tids), np.array(boxes)

    @property
    def fps(self):
        return 1.0

    def rows_for(self, frame):
        return self._by_frame.get(int(frame), [])

    def jpeg(self, frame):
        return None

    def live_vector(self, rows, backend):
        return np.zeros((len(rows), WIDTH), dtype=np.float32)


def _run(monkeypatch, **kw):
    monkeypatch.setattr(SR, "CONTEXT_INFERENCE", {})
    b = RuntimeBundle(model=_TopScreen(), experiment_id="stub", model_name="stub",
                      feature_config="stub", input_dim=WIDTH, seed=42,
                      checkpoint="stub", device=torch.device("cpu"),
                      live_input_width=WIDTH)
    b.display_threshold, b.alert_threshold = 0.46, 0.66
    frames = []
    entry = SimpleNamespace(head_pose_backend="mediapipe", taxonomy="cue6",
                            vlm_policy="agreement")
    SR.replay(_SwitchingCache(**kw), entry, b,
              lambda t, jpg, students, names: frames.append(students),
              realtime=False, overlay=False)
    return frames


def test_an_id_switch_does_not_send_the_student_back_to_warming_up(monkeypatch):
    frames = _run(monkeypatch)
    for f in frames[SWITCH:]:
        assert set(f) == {"1", "2"}, "the seat number must follow the student"
        assert not f["1"].get("warming"), "an analysed student went back to warming up"
        assert f["1"]["cue"] == "screen_oriented"


def test_a_new_student_in_another_seat_still_warms_up(monkeypatch):
    frames = _run(monkeypatch, new_seat=True)
    after = frames[SWITCH]
    assert "3" in after and after["3"].get("warming"), \
        "a genuinely new student gets the next seat number and its own history"

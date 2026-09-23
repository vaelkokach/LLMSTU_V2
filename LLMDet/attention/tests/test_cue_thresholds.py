"""The dashboard's per-cue display bars (demo only).

What these pin is the part that would be silent when wrong: with no bar set, or
with every bar at the calibrated display threshold, the displayed cue must be
exactly what the calibrated rule shows; a lowered bar may surface a cue the
model ranked second but must never let it alert; and a bar must be exactly
undoable and impossible to carry into the next run.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from attention.taxonomy import CUE_CLASSES                      # noqa: E402
from attention.thesis_eval.runtime import (ABSTAIN_LABEL,        # noqa: E402
                                           RuntimeBundle, predict_window)

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "dashboard"))

WIDTH = 4


class _FixedProbs(torch.nn.Module):
    """Logits of log p: softmax at temperature 1 returns p itself."""
    def __init__(self, probs):
        super().__init__()
        self.logp = torch.log(torch.tensor(probs, dtype=torch.float32))

    def forward(self, x):
        return {"logits": self.logp.expand(1, x.shape[1], -1).clone()}


def _probs(**named):
    """A distribution over CUE_CLASSES; unnamed classes share what is left."""
    v = np.array([named.get(c, 0.0) for c in CUE_CLASSES], dtype=float)
    rest = [i for i, c in enumerate(CUE_CLASSES) if c not in named]
    v[rest] = (1.0 - v.sum()) / len(rest)
    return v.tolist()


def _bundle(probs, display=0.46, alert=0.66):
    b = RuntimeBundle(model=_FixedProbs(probs), experiment_id="stub",
                      model_name="stub", feature_config="stub", input_dim=WIDTH,
                      seed=42, checkpoint="stub", device=torch.device("cpu"),
                      live_input_width=WIDTH)
    b.display_threshold, b.alert_threshold = display, alert
    return b


def _predict(b):
    return predict_window(b, np.zeros((8, WIDTH), dtype=np.float32))


def _calibrated(probs, display, alert):
    top = int(np.argmax(probs))
    shown = CUE_CLASSES[top] if probs[top] >= display else ABSTAIN_LABEL
    return shown, probs[top] >= alert


RANDOM = [np.random.default_rng(s).dirichlet(np.full(len(CUE_CLASSES), 0.6)).tolist()
          for s in range(150)]


@pytest.mark.parametrize("display", [0.30, 0.46, 0.60])
def test_no_bars_is_the_calibrated_rule(display):
    for p in RANDOM:
        r = _predict(_bundle(p, display=display))
        shown, alert = _calibrated(np.array(r["probs"]), display, 0.66)
        assert (r["displayed_cue"], r["alert_allowed"]) == (shown, alert)
        assert not r["rescued"]


@pytest.mark.parametrize("display", [0.30, 0.46, 0.60])
def test_every_bar_at_the_display_threshold_is_the_calibrated_rule(display):
    for p in RANDOM:
        b = _bundle(p, display=display)
        b.set_cue_thresholds({c: display for c in CUE_CLASSES})
        r = _predict(b)
        shown, alert = _calibrated(np.array(r["probs"]), display, 0.66)
        assert (r["displayed_cue"], r["alert_allowed"]) == (shown, alert)
        assert not r["rescued"]


def test_a_lowered_bar_surfaces_a_second_ranked_cue_but_never_alerts():
    b = _bundle(_probs(screen_oriented=0.55, phone_use=0.35))
    assert _predict(b)["displayed_cue"] == "screen_oriented"

    # 0.35 - 0.30 = 0.05 is less than screen_oriented's 0.55 - 0.46 = 0.09.
    b.set_cue_thresholds({"phone_use": 0.30})
    assert _predict(b)["displayed_cue"] == "screen_oriented"

    b.set_cue_thresholds({"phone_use": 0.20})             # margin 0.15 now wins
    r = _predict(b)
    assert r["displayed_cue"] == "phone_use" and r["rescued"]
    assert r["cue"] == "screen_oriented", "a bar must not move the raw argmax"
    assert not r["alert_allowed"], "a cue below the top class must not page"


def test_a_raised_bar_can_hide_the_top_cue():
    b = _bundle(_probs(screen_oriented=0.55))
    b.set_cue_thresholds({"screen_oriented": 0.60})
    r = _predict(b)
    assert r["abstained"] and r["displayed_cue"] == ABSTAIN_LABEL
    assert not r["alert_allowed"]


def test_bars_merge_and_reset():
    b = _bundle(_probs(screen_oriented=0.55))
    b.set_cue_thresholds({"phone_use": 0.2})
    b.set_cue_thresholds({"looking_away": 0.3})
    eff = b.effective_cue_thresholds()
    assert eff["phone_use"] == 0.2 and eff["looking_away"] == 0.3
    assert eff["head_down"] == 0.46
    b.reset_cue_thresholds()
    assert b.cue_thresholds == {}
    assert set(b.effective_cue_thresholds().values()) == {0.46}


@pytest.mark.parametrize("bad", [{"not_a_cue": 0.3}, {"phone_use": 0},
                                 {"phone_use": 1.2}, {"phone_use": float("nan")},
                                 {"phone_use": "0.3"}, {"phone_use": True}])
def test_a_rejected_request_changes_nothing(bad):
    b = _bundle(_probs(screen_oriented=0.55))
    b.set_cue_thresholds({"head_down": 0.25})
    with pytest.raises(ValueError):
        b.set_cue_thresholds({"looking_away": 0.3, **bad})
    assert b.cue_thresholds == {"head_down": 0.25}


def test_an_empty_request_is_rejected():
    with pytest.raises(ValueError):
        _bundle(_probs(screen_oriented=0.55)).set_cue_thresholds({})


def test_the_bundle_says_it_has_bars():
    b = _bundle(_probs(screen_oriented=0.55))
    assert "per-cue bars" not in b.describe()
    b.set_cue_thresholds({"phone_use": 0.3})
    assert "per-cue bars" in b.describe()


# ---------------------------------------------------------------- server ---

SV = pytest.importorskip("server")


@pytest.fixture
def clean_server():
    SV.reset_state()
    yield SV
    SV.reset_state()


def test_no_running_model_is_refused(clean_server):
    with pytest.raises(RuntimeError):
        clean_server.set_thresholds({"cues": {"phone_use": 0.3}})


def test_the_endpoint_sets_merges_and_reports(clean_server):
    b = _bundle(_probs(screen_oriented=0.55, phone_use=0.35))
    clean_server.attach_bundle(b)
    assert clean_server.STATE["thresholds"]["overridden"] is False
    clean_server.set_thresholds({"cues": {"phone_use": 0.2}})
    view = clean_server.set_thresholds({"cues": {"looking_away": 0.3}})
    assert view["overridden"] and view["classes"] == list(CUE_CLASSES)
    assert view["cues"]["phone_use"] == 0.2 and view["cues"]["looking_away"] == 0.3
    assert view["cues"]["head_down"] == 0.46
    assert _predict(b)["displayed_cue"] == "phone_use"


def test_reset_through_the_endpoint(clean_server):
    b = _bundle(_probs(screen_oriented=0.55))
    clean_server.attach_bundle(b)
    clean_server.set_thresholds({"cues": {"phone_use": 0.2}})
    view = clean_server.set_thresholds({"reset": True})
    assert not view["overridden"] and b.cue_thresholds == {}


@pytest.mark.parametrize("body", [{}, {"cues": {}}, {"cues": "phone_use"},
                                  {"display": 0.3}])
def test_a_malformed_request_is_rejected(clean_server, body):
    clean_server.attach_bundle(_bundle(_probs(screen_oriented=0.55)))
    with pytest.raises(ValueError):
        clean_server.set_thresholds(body)


def test_a_new_run_detaches_the_bars(clean_server):
    """reset_state runs at the start of every run: a switch, a source, or Play."""
    clean_server.attach_bundle(_bundle(_probs(screen_oriented=0.55)))
    clean_server.set_thresholds({"cues": {"phone_use": 0.2}})
    clean_server.reset_state()
    assert clean_server.STATE["thresholds"] == {}
    with pytest.raises(RuntimeError):
        clean_server.set_thresholds({"cues": {"phone_use": 0.3}})

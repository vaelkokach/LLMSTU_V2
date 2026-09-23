"""Fit a temperature and both abstention thresholds for every registry model.

``work_dirs/thesis/runtime/`` holds thresholds for exactly two checkpoints, so
until now every other model would have run the dashboard with
``display_threshold = alert_threshold = 0`` — no abstention, every prediction
displayed as fact, and every sustained episode allowed to page an instructor.
Silently. Switching the dropdown would then have changed not just the model but
whether the system was allowed to be unsure, which is not a comparison of
models.

Every registry entry already has ``eval_val/predictions.npz`` from the unified
evaluator, so each one can be given its own thresholds under the same rule as
the deployed model (``attention/thesis_eval/runtime.select_thresholds``):

    display  the highest threshold that still labels >= 90% of frames
    alert    the lowest threshold whose retained frames are >= 85% correct

Both are read off the **validation** curve and frozen. Nothing here touches the
test split.

Models that cannot reach 85% selective accuracy at any threshold do not get a
relaxed target — their alert threshold is set above 1.0, which disables alerts
for that model entirely, and the UI says so. A weaker model earning weaker
alerts by lowering the bar is exactly the failure mode this project keeps
finding; a model that cannot support trustworthy alerts should be allowed to
display cues and page nobody.

    python tools/dashboard/calibrate_registry.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "LLMDet"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np                                              # noqa: E402

import model_registry as MR                                     # noqa: E402

OUT_DIR = REPO / "LLMDet" / "work_dirs" / "thesis" / "runtime" / "dashboard"

MIN_DISPLAY_COVERAGE = 0.90
MIN_ALERT_ACCURACY = 0.85
#: An alert threshold above every attainable confidence disables alerts.
ALERTS_DISABLED = 1.01


def fit_one(val_predictions: Path) -> dict:
    from attention.thesis_eval import calibrate as C
    from attention.thesis_eval import EVALUATOR_VERSION

    v = np.load(val_predictions, allow_pickle=False)
    p, y = v["probs"].astype(np.float64), v["y"]
    T = C.fit_temperature(p, y)
    rows = C.coverage_risk_curve(C.apply_temperature(p, T), y)

    disp = max((r for r in rows if r["coverage"] >= MIN_DISPLAY_COVERAGE),
               key=lambda r: r["threshold"])
    ok = [r for r in rows if np.isfinite(r["selective_accuracy"])
          and r["selective_accuracy"] >= MIN_ALERT_ACCURACY]
    alert = min(ok, key=lambda r: r["threshold"]) if ok else None
    best_attainable = max((r["selective_accuracy"] for r in rows
                           if np.isfinite(r["selective_accuracy"])
                           and r["coverage"] > 0.01), default=float("nan"))

    return {
        "evaluator_version": EVALUATOR_VERSION,
        "fitted_on": str(val_predictions.relative_to(REPO)),
        "temperature": T,
        "display_threshold": disp["threshold"],
        "display_rule": f"highest threshold retaining >= {MIN_DISPLAY_COVERAGE:.0%} coverage",
        "display_coverage": disp["coverage"],
        "display_selective_accuracy": disp["selective_accuracy"],
        "alert_threshold": alert["threshold"] if alert else ALERTS_DISABLED,
        "alert_rule": f"lowest threshold with selective accuracy >= {MIN_ALERT_ACCURACY:.0%}",
        "alert_coverage": alert["coverage"] if alert else 0.0,
        "alert_selective_accuracy": (alert["selective_accuracy"] if alert
                                     else best_attainable),
        "alerts_enabled": alert is not None,
        "alerts_disabled_reason": "" if alert else (
            f"no confidence threshold reaches {MIN_ALERT_ACCURACY:.0%} selective "
            f"accuracy on validation (best attainable {best_attainable:.3f}); "
            f"this model may display cues but may not raise alerts"),
        "uncalibrated_accuracy_at_full_coverage": rows[0]["selective_accuracy"],
        "note": ("Validation split only, then frozen. Temperature scaling moves "
                 "no argmax, so accuracy at full coverage is unchanged; only "
                 "confidence and abstention move."),
    }


def main():
    entries = MR.scan()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'variant':46} {'T':>6} {'display':>8} {'cov':>6} "
          f"{'alert':>7} {'sel.acc':>8}  alerts")
    print("-" * 96)
    for e in entries:
        if not e.deployable:
            continue
        pred = REPO / e.val_predictions
        if not pred.exists():
            print(f"{e.variant_id:46}  no eval_val/predictions.npz — skipped")
            continue
        r = fit_one(pred)
        out = OUT_DIR / f"{e.variant_id.replace('/', '__')}.json"
        out.write_text(json.dumps(r, indent=2))
        print(f"{e.variant_id:46} {r['temperature']:6.3f} "
              f"{r['display_threshold']:8.2f} {r['display_coverage']:6.3f} "
              f"{r['alert_threshold']:7.2f} {r['alert_selective_accuracy']:8.3f}  "
              f"{'on' if r['alerts_enabled'] else 'DISABLED'}")
    print(f"\nwritten to {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()

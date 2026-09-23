"""End-to-end check: does the dashboard do what its model card claims?

Three things can be wrong in ways that still produce a working-looking page,
and each has happened in this project at least once:

1. **The selector changes nothing.** A switch that reloads a checkpoint but
   keeps feeding the old predictions looks identical to a switch that works,
   because both show plausible cues. So this replays the *same cached session*
   through several models and reports how often they disagree. Two models with
   different weights must produce different cue streams; if any pair agrees
   100%, the switch is not switching.

2. **The model card shows numbers from somewhere else.** The card's macro-F1
   must be the evaluator's, from `eval_val/metrics.json`, not a figure copied
   into a table by hand. This re-reads the source files and compares.

3. **The HTTP surface lies about which model is active.** The switch endpoint
   is checked against `/api/state` afterwards, not against its own response.

Runs entirely on the cache — no detector, no GPU.

    python tools/dashboard/verify_dashboard.py --cache tools/dashboard/sessions/0325
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "LLMDet"))


def replay_cues(cache, entry, device="cpu"):
    """-> ({(frame, track) -> cue}, seconds, cue histogram)"""
    import session_replay as SR

    bundle, _cal = SR.load_model(entry, device)
    out, hist = {}, defaultdict(int)
    state = {"i": 0}

    def push(t, jpg, students, cue_names):
        for seat, s in students.items():
            # Warm-up entries are identical under every model, so counting
            # them would inflate agreement between models for free.
            if s.get("warming"):
                continue
            out[(state["i"], seat)] = s["cue"]
            hist[s["cue"]] += 1
        state["i"] += 1

    t0 = time.time()
    SR.replay(cache, entry, bundle, push, realtime=False, overlay=False)
    return out, time.time() - t0, dict(hist)


def agreement(a, b):
    keys = set(a) & set(b)
    if not keys:
        return float("nan"), 0
    return sum(a[k] == b[k] for k in keys) / len(keys), len(keys)


def check_card_matches_evaluator(entries):
    """The card's numbers must come from the evaluator's own files."""
    bad = []
    for e in entries:
        m = REPO / Path(e.checkpoint).parents[1] / "eval_val" / "metrics.json"
        if not m.exists():
            bad.append((e.variant_id, "no eval_val/metrics.json"))
            continue
        got = json.loads(m.read_text())["macro_f1"]
        if abs(got - e.val["macro_f1"]) > 1e-9:
            bad.append((e.variant_id,
                        f"card {e.val['macro_f1']:.6f} != file {got:.6f}"))
    return bad


def http(url, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def check_http(cache_dir, model_a, model_b, port, device, n_expected):
    """Start the server, switch models over HTTP, confirm the state agrees."""
    proc = subprocess.Popen(
        [sys.executable, str(HERE / "server.py"), "--session", str(cache_dir),
         "--host", "127.0.0.1", "--port", str(port), "--device", device,
         "--speed", "20", "--model", model_a],
        cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    base = f"http://127.0.0.1:{port}"
    results = []
    try:
        for _ in range(120):                      # model load is slow on CPU
            if proc.poll() is not None:
                raise SystemExit(
                    "server exited during startup:\n" + proc.stdout.read())
            try:
                st = http(base + "/api/state", timeout=5)
                if st.get("model", {}).get("variant_id"):
                    break
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(1)
        else:
            raise SystemExit("server never reported an active model")

        st = http(base + "/api/state")
        results.append(("startup model is --model",
                        st["model"]["variant_id"] == model_a,
                        st["model"]["variant_id"]))

        mods = http(base + "/api/models")
        results.append(("registry served over HTTP",
                        len(mods["models"]) == n_expected,
                        f"{len(mods['models'])} of {n_expected} models"))
        results.append((
            "non-deployable entries carry a reason",
            all(m["blocked_reason"] for m in mods["models"] if not m["deployable"]),
            f"{sum(1 for m in mods['models'] if not m['deployable'])} disabled"))

        srcs = http(base + "/api/sources")
        results.append(("session listed as a source",
                        any(s["kind"] == "session" for s in srcs["sources"]),
                        f"{len(srcs['sources'])} sources"))

        http(base + "/api/model", {"variant_id": model_b})
        # Ask /api/state, not the switch response: the point is whether the
        # running producer changed, not whether the endpoint said so.
        for _ in range(120):
            st = http(base + "/api/state")
            if st["model"]["variant_id"] == model_b and st["running"]:
                break
            time.sleep(1)
        results.append(("switch takes effect in /api/state",
                        st["model"]["variant_id"] == model_b,
                        st["model"]["variant_id"]))
        results.append(("thresholds move with the model",
                        st["model"]["alert_threshold"] is not None,
                        f"alert>={st['model']['alert_threshold']}"))

        try:
            http(base + "/api/model", {"variant_id": "ladder/transformer_570_full"})
            ok = False
        except urllib.error.HTTPError as e:
            ok = e.code == 400
        results.append(("non-deployable model is refused", ok, "HTTP 400"))

        try:
            http(base + "/api/model", {"variant_id": "no/such_model"})
            ok = False
        except urllib.error.HTTPError as e:
            ok = e.code in (400, 409)
        results.append(("unknown model is refused", ok, "HTTP 4xx"))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--port", type=int, default=8137)
    ap.add_argument("--models", type=int, default=4,
                    help="how many registry entries to replay and cross-compare")
    ap.add_argument("--out", default="LLMDet/work_dirs/thesis/runtime/dashboard_verification.json")
    args = ap.parse_args()

    import model_registry as MR
    import session_replay as SR

    entries = MR.scan()
    cache = SR.SessionCache(args.cache)
    print(f"cache: {cache.dir.name} — {cache.meta['n_frames']} frames, "
          f"{cache.meta['n_tracks']} tracks, "
          f"{cache.meta['n_student_frames']} student-frames, "
          f"built on {cache.meta['device']} in {cache.meta['wall_clock_s']}s\n")

    print("== 1. the model card's numbers come from the evaluator ==")
    bad = check_card_matches_evaluator(entries)
    for v, why in bad:
        print(f"  MISMATCH {v}: {why}")
    print(f"  {len(entries) - len(bad)}/{len(entries)} variants match "
          f"eval_val/metrics.json exactly\n")

    live = [e for e in entries if e.deployable][:args.models]
    print(f"== 2. replaying {len(live)} models over the same cache ==")
    runs, timings = {}, {}
    for e in live:
        cues, secs, hist = replay_cues(cache, e, args.device)
        runs[e.variant_id] = cues
        timings[e.variant_id] = secs
        top = sorted(hist.items(), key=lambda kv: -kv[1])[:3]
        n = sum(hist.values()) or 1
        print(f"  {e.variant_id:34} {secs:6.1f}s  "
              + "  ".join(f"{k} {100 * v / n:.0f}%" for k, v in top))

    print("\n== 3. do the models actually differ? ==")
    ids = list(runs)
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ag, n = agreement(runs[a], runs[b])
            pairs.append({"a": a, "b": b, "agreement": ag, "n": n})
            flag = "  <-- IDENTICAL, the switch is not switching" if ag == 1.0 else ""
            print(f"  {a:34} vs {b:34} {ag:6.1%} of {n}{flag}")
    identical = [p for p in pairs if p["agreement"] == 1.0]

    print("\n== 4. HTTP surface ==")
    http_results = []
    if len(live) >= 2:
        http_results = check_http(cache.dir, live[0].variant_id,
                                  live[1].variant_id, args.port, args.device,
                                  len(entries))
        for name, ok, detail in http_results:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name} ({detail})")

    failures = (len(bad) + len(identical)
                + sum(1 for _n, ok, _d in http_results if not ok))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "cache": cache.meta,
        "card_mismatches": bad,
        "replay_seconds": timings,
        "pairwise_agreement": pairs,
        "http": [{"check": n, "pass": ok, "detail": d}
                 for n, ok, d in http_results],
        "failures": failures,
    }, indent=2))
    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} FAILURE(S)'}"
          f"  — written: {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

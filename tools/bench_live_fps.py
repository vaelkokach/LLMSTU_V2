#!/usr/bin/env python
"""What frame rate does the live camera path actually reach, on this machine?

Measured by pushing into `/api/camera/frame` faster than the pipeline can
consume, so the PIPELINE is the bottleneck rather than the pusher. Against the
deployed Space this could not be done -- one request in flight over the internet
tops out near 1.8/s and the drop counter stops rising, which measures the network
and says nothing about the model. Locally there is no such limit.

    python tools/bench_live_fps.py --url http://127.0.0.1:8010 --model epochs240/mstcn_556_hp
"""
import argparse
import json
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np


def post(url, data=None, ctype=None, timeout=600):
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    if ctype:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            return f.status, json.loads(f.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:300]


def synth(i, n_students=5, w=1280, h=720):
    """A frame with several person-shaped blobs, at a realistic capture size.

    Resolution matters to the measurement: the detector's cost is driven by the
    input, so benchmarking on 640x480 would flatter the result relative to what
    a 720p webcam actually sends.
    """
    img = np.full((h, w, 3), 40, np.uint8)
    for k in range(n_students):
        x = 60 + k * (w - 160) // n_students + (i % 6)
        cv2.rectangle(img, (x, int(h * .42)), (x + 150, int(h * .92)), (90, 110, 140), -1)
        cv2.circle(img, (x + 75, int(h * .36)), 52, (120, 140, 170), -1)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8010")
    ap.add_argument("--model", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--warm", type=int, default=300)
    ap.add_argument("--measure", type=int, default=60)
    ap.add_argument("--students", type=int, default=5)
    a = ap.parse_args()

    print(f"source -> {post(a.url + '/api/source', json.dumps({'source_id': 'camera:browser'}).encode(), 'application/json')[0]}")
    if a.model:
        c, b = post(a.url + "/api/model", json.dumps({"variant_id": a.model}).encode(),
                    "application/json")
        print(f"model  -> {c} {str(b)[:150]}")

    frames = [synth(i, a.students) for i in range(12)]
    stop = threading.Event()
    last = {}

    def pump():
        i = 0
        while not stop.is_set():
            c, b = post(a.url + "/api/camera/frame", frames[i % len(frames)], "image/jpeg", 120)
            i += 1
            if c == 200 and isinstance(b, dict):
                last.update(b)

    for _ in range(a.workers):
        threading.Thread(target=pump, daemon=True).start()

    t0 = time.time()
    while time.time() - t0 < a.warm:
        if last.get("processed", 0) > 8 and last.get("status") == "consuming":
            break
        time.sleep(2)
    print(f"warm after {time.time()-t0:.0f}s (analysed {last.get('processed')})", flush=True)

    a0, s0, w0 = last.get("processed", 0), last.get("received", 0), time.time()
    time.sleep(a.measure)
    a1, s1, w1 = last.get("processed", 0), last.get("received", 0), time.time()
    stop.set()
    dt = w1 - w0
    offered = (s1 - s0) / dt
    fps = (a1 - a0) / dt
    print(f"\n== {a.model or '(current model)'} / {a.students} students / {a.workers} workers ==")
    print(f"window        {dt:.0f} s")
    print(f"offered       {offered:6.2f} frames/s")
    print(f"ANALYSED      {fps:6.2f} frames/s      <- the pipeline's own rate")
    print(f"saturated     {'yes' if offered > fps * 1.25 else 'NO -- pusher-bound, raise --workers'}")
    c, st = post(a.url + "/api/state")
    if isinstance(st, dict):
        print(f"server says   {st.get('capture')}")


if __name__ == "__main__":
    main()

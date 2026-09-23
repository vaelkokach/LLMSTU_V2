#!/usr/bin/env python3
"""Gold-set annotation server (stdlib only, no external deps).

Usage:
    python serve.py --manifest gold_candidates.jsonl --annotator NAME [--port 8765] [--out-dir .]

Writes, per annotator:
    events_<annotator>.jsonl            append-only action log (never rewritten)
    gold_annotations_<annotator>.jsonl  materialized latest state per file_name
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
#: Defaults assume the tool runs inside the repo on the machine that holds
#: grounding_data. --crops-root/--frames-root override them so the annotator
#: can run against a self-contained bundle on a machine that does not (see
#: make_offline_bundle.py). Module-level because the handler reads them.
CROPS_ROOT = os.path.join(REPO, "grounding_data", "LLMSTU", "crops")
FRAMES_ROOT = os.path.join(REPO, "grounding_data", "stu_img", "frames")

sys.path.insert(0, ROOT)
from vocab import ALL_LABEL_FIELDS, BOOLEAN_FIELDS, CATEGORICAL_FIELDS  # noqa: E402


class Store:
    def __init__(self, manifest_path, annotator, out_dir):
        self.items = []
        with open(manifest_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))
        if not self.items:
            raise SystemExit(f"empty manifest: {manifest_path}")
        self.annotator = annotator
        self.events_path = os.path.join(out_dir, f"events_{annotator}.jsonl")
        self.gold_path = os.path.join(out_dir, f"gold_annotations_{annotator}.jsonl")
        self.by_name = {}  # file_name -> latest annotation record
        if os.path.exists(self.gold_path):
            with open(self.gold_path) as fh:
                for line in fh:
                    rec = json.loads(line)
                    self.by_name[rec["file_name"]] = rec

    def annotate(self, payload):
        idx = int(payload["index"])
        item = self.items[idx]
        rec = {
            "file_name": item["file_name"],
            "index": idx,
            "annotator": self.annotator,
            "status": payload.get("status", "ok"),  # ok | uncertain | rejected
        }
        for k in ALL_LABEL_FIELDS:
            rec[k] = payload["fields"].get(k, item.get(k))
        rec["pseudo_edited"] = any(rec[k] != item.get(k) for k in ALL_LABEL_FIELDS)
        with open(self.events_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.by_name[rec["file_name"]] = rec
        with open(self.gold_path, "w") as fh:
            for it in self.items:
                r = self.by_name.get(it["file_name"])
                if r is not None:
                    fh.write(json.dumps(r) + "\n")
        return rec

    def state(self):
        done = sum(1 for it in self.items if it["file_name"] in self.by_name)
        next_idx = 0
        for i, it in enumerate(self.items):
            if it["file_name"] not in self.by_name:
                next_idx = i
                break
        else:
            next_idx = len(self.items) - 1
        return {
            "total": len(self.items),
            "done": done,
            "next_index": next_idx,
            "annotator": self.annotator,
            "vocab": CATEGORICAL_FIELDS,
            "booleans": BOOLEAN_FIELDS,
        }

    def item(self, idx):
        item = dict(self.items[idx])
        item["index"] = idx
        ann = self.by_name.get(item["file_name"])
        item["annotation"] = ann
        return item

    def crop_path(self, idx):
        item = self.items[idx]
        p = item.get("abs_path") or os.path.join(CROPS_ROOT, item["file_name"])
        return p

    def frame_path(self, idx):
        src = self.items[idx].get("src_frame")
        return os.path.join(FRAMES_ROOT, src) if src else None


STORE = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype):
        if not path or not os.path.exists(path):
            self.send_response(404)
            self.end_headers()
            return
        with open(path, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._file(os.path.join(ROOT, "index.html"), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._json(STORE.state())
        elif path.startswith("/api/item/"):
            idx = int(path.rsplit("/", 1)[1])
            if 0 <= idx < len(STORE.items):
                self._json(STORE.item(idx))
            else:
                self._json({"error": "index out of range"}, 404)
        elif path.startswith("/img/crop/"):
            self._file(STORE.crop_path(int(path.rsplit("/", 1)[1])), "image/jpeg")
        elif path.startswith("/img/frame/"):
            self._file(STORE.frame_path(int(path.rsplit("/", 1)[1])), "image/jpeg")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/annotate":
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n))
            rec = STORE.annotate(payload)
            self._json({"ok": True, "saved": rec, "state": STORE.state()})
        else:
            self.send_response(404)
            self.end_headers()


def main():
    global STORE
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out-dir", default=ROOT)
    ap.add_argument("--crops-root", default=None,
                    help="directory holding the crop images, addressed by the "
                         "manifest's file_name. Defaults to the in-repo "
                         "grounding_data path.")
    ap.add_argument("--frames-root", default=None,
                    help="directory holding the full source frames for the 'f' "
                         "occlusion view. A missing frame is a 404 and the view "
                         "is simply unavailable, so a crops-only bundle works.")
    args = ap.parse_args()
    global CROPS_ROOT, FRAMES_ROOT
    if args.crops_root:
        CROPS_ROOT = os.path.abspath(args.crops_root)
    if args.frames_root:
        FRAMES_ROOT = os.path.abspath(args.frames_root)
    if not os.path.isdir(CROPS_ROOT):
        raise SystemExit(
            f"crops root does not exist: {CROPS_ROOT}\n"
            f"Pass --crops-root, or run on the machine that holds "
            f"grounding_data. Without it every image is a 404 and you would be "
            f"annotating blind.")
    print(f"crops:  {CROPS_ROOT}")
    print(f"frames: {FRAMES_ROOT}"
          + ("" if os.path.isdir(FRAMES_ROOT) else "   (absent -- 'f' view disabled)"))
    STORE = Store(args.manifest, args.annotator, args.out_dir)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    st = STORE.state()
    print(f"annotator={st['annotator']}  items={st['total']}  done={st['done']}")
    print(f"open http://localhost:{args.port}/  (Ctrl+C to stop; progress is saved after every action)")
    srv.serve_forever()


if __name__ == "__main__":
    main()

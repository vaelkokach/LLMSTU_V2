#!/usr/bin/env python
"""Serve the dashboard from the HPC, reachable from a laptop on the same network.

The HPC box runs this repo inside a Docker container on the bridge network
(172.17.x.x). That address is not routable from a laptop, and a new port cannot
be published from inside the container -- `docker run -p` happens at creation.
**Exactly one port is already exposed: 8888, the Jupyter server.** So the way out
is `jupyter-server-proxy`, which is installed here and proxies an arbitrary local
port at::

    <your Jupyter URL>/proxy/<port>/

That is not a workaround, it is the only route that needs no change on the host,
and it inherits Jupyter's authentication rather than opening an unauthenticated
port onto a shared network -- which matters, because what this page shows is
restricted student video (`[internal notes, not included]`).

The dashboard's front end resolves every call as `new URL(p, document.baseURI)`,
so it already works under a path prefix; nothing needs rewriting.

    python tools/serve_hpc.py                 # pick a free GPU, start, print URL
    python tools/serve_hpc.py --port 8010 --device cuda:3
    python tools/serve_hpc.py --stop

**The webcam needs a secure context.** `getUserMedia` is refused on a plain-HTTP
origin that is not localhost, so if Jupyter is reached as `http://<lan-ip>:8888`
the camera button will do nothing -- silently, because that is how the browser
reports it. Three ways round it, in order of preference: reach Jupyter over
HTTPS; tunnel it (`ssh -L 8888:localhost:8888 user@hpc`) so the origin is
localhost; or allowlist the origin in Chrome with
`--unsafely-treat-insecure-origin-as-secure=http://<lan-ip>:8888`. Uploading or
replaying a recording works either way -- only the live camera is gated.
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
#: A venv that exists only to give the dashboard a NEWER transformers than the
#: one the training environment is pinned to.
#:
#: The VLM entries need >= 4.45 (Qwen2VLForConditionalGeneration); the shared
#: environment is pinned at 4.44.2, and that pin protects mmcv's compiled _ext
#: against torch 2.2.2. Rather than move the environment every training run
#: depends on, this venv is created with --system-site-packages, so it inherits
#: torch, mmcv and mmdet unchanged and overrides transformers alone.
#:
#: Verified before adopting it, because a silent feature change would be the
#: frame-rate bug all over again [internal notes, not included]: the 552-dim live feature vector
#: for the same frame is BYTE-IDENTICAL under 4.44.2 and 4.45.2 (549 of 552
#: columns non-zero, so it is a real comparison), and mmdet's registry imports
#: cleanly with torch still 2.2.2+cu121. 4.45.2 is also exactly what the
#: deployed Space runs, so it is proven in production rather than only here.
VLM_VENV = REPO / ".venv-dashboard"
PIDFILE = REPO / "tools" / "dashboard" / ".serve_hpc.pid"
LOGFILE = REPO / "tools" / "dashboard" / "serve_hpc.log"
#: Written next to the app, like the Space's `_runtime_override.yaml`. The
#: canonical `attention_runtime.yaml` is the configuration the thesis cites and
#: is never edited.
DERIVED = REPO / "tools" / "dashboard" / "_hpc_runtime.yaml"


def free_gpu():
    """The emptiest GPU, and never one somebody else is using.

    The box is shared -- another user's job on GPU 4 contaminated a latency
    benchmark once, so this reads actual memory rather than assuming.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
    except Exception:
        return "cpu"
    rows = []
    for line in out.strip().splitlines():
        i, used = (v.strip() for v in line.split(","))
        rows.append((int(used), int(i)))
    if not rows:
        return "cpu"
    used, idx = min(rows)
    if used > 1024:
        print(f"[serve] every GPU is busy; using cuda:{idx} ({used} MiB already in use)")
    return f"cuda:{idx}"


def derived_config():
    """attention_runtime.yaml with the no-LMM detector, as the Space uses.

    The detector config names a 0.5B LMM and a 3.5 GB SigLIP read that
    `predict()` never touches (`student_llmstu_exact_deploy.py` documents why).
    Loading them costs startup time and memory and changes no box.
    """
    import yaml
    src = REPO / "LLMDet" / "configs" / "attention_runtime.yaml"
    cfg = yaml.safe_load(src.read_text())
    cfg["detector"]["config_path"] = "configs/student_llmstu_exact_deploy.py"
    DERIVED.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return DERIVED


def jupyter_proxy_url(port):
    """What to type into the browser, as far as it can be known from in here.

    JUPYTER_SERVER_URL is the container-internal address; whatever reverse proxy
    or port publish puts Jupyter in front of the laptop is invisible from here.
    So this prints the path to append rather than inventing a hostname.
    """
    internal = os.environ.get("JUPYTER_SERVER_URL", "")
    base = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/")
    return internal, f"{base.rstrip('/')}/proxy/{port}/"


def interpreter(prefer_vlm=True):
    """Which python runs the dashboard, and say so rather than guessing silently.

    Falls back to this interpreter when the venv is absent, which is the right
    default: the dashboard works without the VLM, it simply does not offer the
    `+vlm` entries, and the registry already explains that on /api/models.
    """
    py = VLM_VENV / "bin" / "python"
    if prefer_vlm and py.exists():
        return str(py), True
    return sys.executable, False


def setup_vlm_venv():
    """Create the venv the `+vlm` entries need. Idempotent."""
    py = VLM_VENV / "bin" / "python"
    if not py.exists():
        print(f"[serve] creating {VLM_VENV.relative_to(REPO)} "
              f"(--system-site-packages: inherits torch/mmcv/mmdet)")
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages",
                        str(VLM_VENV)], check=True)
    print("[serve] installing transformers==4.45.2 (the version the Space runs)")
    subprocess.run([str(VLM_VENV / "bin" / "pip"), "install", "-q",
                    "transformers==4.45.2"], check=True)
    out = subprocess.run(
        [str(py), "-c",
         "import transformers, torch;"
         "print(transformers.__version__, torch.__version__,"
         " hasattr(transformers,'Qwen2VLForConditionalGeneration'))"],
        capture_output=True, text=True, check=True).stdout.split()
    print(f"[serve] transformers {out[0]} | torch {out[1]} | Qwen2-VL loadable: {out[2]}")
    if out[2] != "True":
        sys.exit("[serve] the venv still cannot load Qwen2-VL; not usable")
    if not out[1].startswith("2.2.2"):
        sys.exit(f"[serve] torch moved to {out[1]}; that breaks mmcv's _ext. "
                 f"Remove {VLM_VENV} and do not use it.")
    return 0


def running():
    if not PIDFILE.exists():
        return None
    try:
        pid = int(PIDFILE.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


def stop():
    pid = running()
    if pid is None:
        print("[serve] not running")
        PIDFILE.unlink(missing_ok=True)
        return 0
    os.kill(pid, signal.SIGTERM)
    for _ in range(40):
        time.sleep(0.25)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
    else:
        print(f"[serve] pid {pid} ignored SIGTERM; sending SIGKILL")
        os.kill(pid, signal.SIGKILL)
    PIDFILE.unlink(missing_ok=True)
    print(f"[serve] stopped pid {pid}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--device", default="")
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 on purpose: jupyter-server-proxy connects "
                         "locally, and 0.0.0.0 would expose restricted student "
                         "video unauthenticated on a shared network")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--setup-vlm", action="store_true",
                    help="create/refresh the venv that enables the +vlm entries")
    ap.add_argument("--no-vlm", action="store_true",
                    help="run under this interpreter even if the venv exists")
    ap.add_argument("--replay-only", action="store_true",
                    help="no --config: replay cached sessions, no detector, no GPU")
    a = ap.parse_args()

    if a.setup_vlm:
        return setup_vlm_venv()
    if a.stop:
        return stop()
    pid = running()
    if pid is not None:
        sys.exit(f"already running as pid {pid} (tools/serve_hpc.py --stop)")

    device = a.device or free_gpu()
    py, with_vlm = interpreter(prefer_vlm=not a.no_vlm)
    print(f"[serve] python {py}"
          + ("  (+vlm entries enabled)" if with_vlm else
             "  (no +vlm: run --setup-vlm to enable them)"))
    cmd = [py, "tools/dashboard/server.py",
           "--host", a.host, "--port", str(a.port)]
    if not a.replay_only:
        cmd += ["--config", str(derived_config().relative_to(REPO)),
                "--device", device]

    env = dict(os.environ)
    # The weights are all in ~/.cache/huggingface; offline keeps a transient
    # network failure from turning into a 60 s hang on first CLIP load.
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    print(f"[serve] device {device}")
    print(f"[serve] {' '.join(cmd)}")
    with open(LOGFILE, "w") as log:
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
    PIDFILE.write_text(str(p.pid))

    # Wait for it to answer, so a config error is reported here rather than
    # discovered in the browser.
    import urllib.error
    import urllib.request
    url = f"http://{a.host}:{a.port}/api/models"
    for _ in range(120):
        if p.poll() is not None:
            print(LOGFILE.read_text()[-3000:])
            PIDFILE.unlink(missing_ok=True)
            sys.exit(f"[serve] server exited with {p.returncode}; log above")
        try:
            with urllib.request.urlopen(url, timeout=2):
                break
        except Exception:
            time.sleep(1)
    else:
        sys.exit(f"[serve] no answer on {url} after 120 s; see {LOGFILE}")

    internal, path = jupyter_proxy_url(a.port)
    print(f"\n[serve] up as pid {p.pid}, log {LOGFILE.relative_to(REPO)}")
    print(f"[serve] local    http://{a.host}:{a.port}/")
    print(f"[serve] Jupyter internal URL is {internal or '(unset)'}")
    print(f"\n  Open your Jupyter URL in the browser and append:\n\n      {path}\n")
    print(f"  e.g. http://10.147.10.21:1300{path}")
    print(f"\n  KEEP THE TRAILING SLASH. Without it the browser resolves every")
    print(f"  API call against the parent -- /proxy/api/sources instead of")
    print(f"  /proxy/{a.port}/api/sources -- and Jupyter answers 404, so the page")
    print(f"  loads but reports itself disconnected. index.html now supplies the")
    print(f"  slash itself, so this is belt and braces.")
    print("\n  The live camera needs a secure context: over plain http to a")
    print("  non-localhost address the browser blocks getUserMedia silently.")
    print("  See this file's docstring for the three ways round it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

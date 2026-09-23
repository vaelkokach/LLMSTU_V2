"""Qwen3-VL structured pseudo-labeling of student crops.

Two backends:
  * transformers  -> simplest, works on a single Colab / HF GPU.
  * vllm          -> high throughput for the full 100k+ crop run (optional).

Both emit a schema-valid dict per crop (see llmstu/schema.py) plus the raw
model text so nothing is lost.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from .config import CaptionConfig
from . import schema
from . import prompts

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_prompt(name: str = "v1_structured") -> str:
    """Return the instruction text for a named prompt variant (see llmstu/prompts.py)."""
    return prompts.get(name)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _parse(text: str) -> Dict:
    # drop any <think>...</think> reasoning so it can't swallow the JSON braces
    text = _THINK_RE.sub("", text or "")
    m = _JSON_RE.search(text)
    if not m:
        return schema.coerce({})
    try:
        raw = json.loads(m.group(0))
    except Exception:
        return schema.coerce({})
    label = schema.coerce(raw)
    mc = raw.get(schema.MODEL_CONFIDENCE_FIELD)
    try:
        label[schema.MODEL_CONFIDENCE_FIELD] = float(mc)
    except Exception:
        label[schema.MODEL_CONFIDENCE_FIELD] = None
    return label


# --------------------------------------------------------------------------- #
# transformers backend
# --------------------------------------------------------------------------- #
class TransformersCaptioner:
    def __init__(self, cfg: CaptionConfig):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.cfg = cfg
        self.torch = torch
        schema.set_active(cfg.schema_name)   # select the field set before building the prompt
        dtype = getattr(torch, cfg.dtype)
        # Fall back to sdpa if flash-attn is requested but not installed, so the run
        # never hard-crashes on a missing optional dependency.
        attn = cfg.attn_implementation
        if attn == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except Exception:
                print("[caption] flash-attn not installed; falling back to attn_implementation='sdpa'")
                attn = "sdpa"
        kwargs = dict(dtype=dtype, device_map="auto", attn_implementation=attn)
        if cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            kwargs.pop("dtype")  # dtype comes from the quant config
        # AutoModelForImageTextToText resolves Qwen3_5ForConditionalGeneration (or any
        # other VLM) automatically — weights load and run locally on the GPU.
        self.model = AutoModelForImageTextToText.from_pretrained(cfg.model_id, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(cfg.model_id, max_pixels=cfg.max_pixels)
        self.prompt = build_prompt(cfg.prompt_name)

    def _messages(self, image: Image.Image):
        return [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": self.prompt},
        ]}]

    def caption_batch(self, images: List[Image.Image]) -> List[Dict]:
        torch = self.torch
        batch_msgs = [self._messages(im) for im in images]
        inputs = self.processor.apply_chat_template(
            batch_msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", padding=True,
            enable_thinking=self.cfg.enable_thinking,
        ).to(self.model.device)
        with torch.inference_mode():
            gen = self.model.generate(
                **inputs, max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.temperature > 0,
                temperature=max(self.cfg.temperature, 1e-6),
            )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        texts = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        return [self._pack(t) for t in texts]

    @staticmethod
    def _pack(text: str) -> Dict:
        return {"label": _parse(text), "raw": text}


# --------------------------------------------------------------------------- #
# vLLM backend (optional, high throughput)
# --------------------------------------------------------------------------- #
class VLLMCaptioner:
    def __init__(self, cfg: CaptionConfig):
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        self.cfg = cfg
        schema.set_active(cfg.schema_name)   # select the field set before building the prompt
        self.processor = AutoProcessor.from_pretrained(cfg.model_id, max_pixels=cfg.max_pixels)
        self.llm = LLM(model=cfg.model_id, trust_remote_code=True,
                       limit_mm_per_prompt={"image": 1},
                       gpu_memory_utilization=cfg.vllm_gpu_mem_util,
                       max_num_seqs=cfg.vllm_max_num_seqs,
                       dtype=cfg.dtype)
        self.sampling = SamplingParams(
            temperature=cfg.temperature, max_tokens=cfg.max_new_tokens)
        self.prompt = build_prompt(cfg.prompt_name)

    def caption_batch(self, images: List[Image.Image]) -> List[Dict]:
        reqs = []
        for im in images:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": im},
                {"type": "text", "text": self.prompt}]}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.cfg.enable_thinking)
            reqs.append({"prompt": text, "multi_modal_data": {"image": im}})
        outs = self.llm.generate(reqs, self.sampling)
        packed = []
        for o in outs:
            t = o.outputs[0].text
            packed.append({"label": _parse(t), "raw": t})
        return packed


def load_captioner(cfg: CaptionConfig):
    if cfg.backend == "vllm":
        return VLLMCaptioner(cfg)
    return TransformersCaptioner(cfg)


def run(
    manifest_path: Path,
    crops_root: Path,
    out_path: Path,
    cfg: CaptionConfig,
    resume: bool = True,
    limit: Optional[int] = None,
    cap=None,
) -> Path:
    """Read crops_manifest.jsonl, caption each crop, append to out_path JSONL.

    Pass a preloaded `cap` (from load_captioner) to reuse one model across many
    calls (e.g. a shard-by-shard loop) instead of reloading it each time.
    """
    manifest_path, crops_root, out_path = Path(manifest_path), Path(crops_root), Path(out_path)
    rows = [json.loads(l) for l in manifest_path.open() if l.strip()]
    if limit:
        rows = rows[:limit]

    done = set()
    if resume and out_path.exists():
        for l in out_path.open():
            try:
                done.add(json.loads(l)["crop_path"])
            except Exception:
                pass
    rows = [r for r in rows if r["crop_path"] not in done]
    print(f"[caption] {len(rows)} crops to label with {cfg.model_id} ({cfg.backend})")

    if cap is None:
        cap = load_captioner(cfg)
    bs = cfg.batch_size if cfg.backend == "transformers" else max(cfg.batch_size, 64)
    with out_path.open("a") as of:
        for start in range(0, len(rows), bs):
            chunk = rows[start:start + bs]
            imgs, ok = [], []
            for r in chunk:
                p = crops_root / Path(r["crop_path"]).name
                if not p.exists():
                    p = crops_root.parent / r["crop_path"]
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    ok.append(r)
                except Exception:
                    continue
            if not imgs:
                continue
            for r, res in zip(ok, cap.caption_batch(imgs)):
                of.write(json.dumps({**r, **res["label"], "raw_caption": res["raw"]}) + "\n")
            of.flush()
            print(f"[caption] {min(start + bs, len(rows))}/{len(rows)}", flush=True)
    print(f"[caption] labels -> {out_path}")
    return out_path

"""Single entry point for running scalar-rating inference on MM-SCALE.

Backends:
    - vllm:      open-weight via vLLM (Qwen2-VL family, LLaVA-OneVision, etc.)
    - openai:    OpenAI Chat Completions (GPT-4o, GPT-5.2)
    - anthropic: Anthropic Messages (Claude family)
    - gemini:    Google Gemini API

API keys come from environment variables only. No keys are stored on disk.

Usage:

    python -m src.run_model \\
        --model qwen2_5_vl_7b --prompt zero_shot \\
        --split test --output outputs/qwen2_5_vl_7b_zero_shot.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.data import load_flat, load_paths, resolve_image_path, sample_icl_exemplars
from src.parse_outputs import parse_rating
from src.utils import encode_image_b64, load_yaml, require_env, write_jsonl


# ---------------------------- Prompt rendering ----------------------------

def _exemplars_block(exemplars: list[dict]) -> str:
    return "\n\n".join(f"Action: {e['text']}\nRating: {e['mean_rating']}"
                       for e in exemplars)


def render_prompt(prompt_type: str, prompts_cfg: dict, scenario: str,
                  exemplars: list[dict] | None = None) -> str:
    template_path = Path("prompts") / prompts_cfg["scalar"][prompt_type]
    tmpl = template_path.read_text()
    if prompt_type == "icl3":
        block = _exemplars_block(exemplars or [])
        return tmpl.format(scenario=scenario, exemplars_block=block)
    return tmpl.format(scenario=scenario)


# ---------------------------- Backends ----------------------------

def call_openai(model: str, prompt: str, image_b64: str,
                max_tokens: int) -> tuple[str, str | None]:
    from openai import OpenAI
    client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ]}],
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content or "", None
    except Exception as e:  # noqa: BLE001
        return "", f"{type(e).__name__}: {str(e)[:200]}"


def call_anthropic(model: str, prompt: str, image_b64: str,
                   max_tokens: int) -> tuple[str, str | None]:
    from anthropic import Anthropic
    client = Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": "image/jpeg",
                                              "data": image_b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
        return "".join(b.text for b in resp.content if b.type == "text"), None
    except Exception as e:  # noqa: BLE001
        return "", f"{type(e).__name__}: {str(e)[:200]}"


def call_gemini(model: str, prompt: str, image_path: str,
                max_tokens: int) -> tuple[str, str | None]:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=require_env("GOOGLE_API_KEY"))
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        resp = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(max_output_tokens=max_tokens),
        )
        return resp.text or "", None
    except Exception as e:  # noqa: BLE001
        return "", f"{type(e).__name__}: {str(e)[:200]}"


def run_vllm(model_cfg: dict, items: list[tuple[dict, str]],
             max_tokens: int) -> list[tuple[str, str | None]]:
    """Batch inference via vLLM. Returns list of (raw, error) per item."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_cfg["hf_path"], trust_remote_code=True)
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    requests = []
    for r, prompt in items:
        img = str(resolve_image_path(r["image_id"]))
        requests.append({"prompt": prompt, "multi_modal_data": {"image": img}})
    outputs = llm.generate(requests, params)
    return [(o.outputs[0].text, None) for o in outputs]


# ---------------------------- Main ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="key into configs/models.yaml")
    ap.add_argument("--prompt", required=True, choices=["zero_shot", "icl3", "cot"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=8,
                    help="for API backends; ignored for vllm")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    paths = load_paths()
    models_cfg = load_yaml("configs/models.yaml")["models"]
    prompts_cfg = load_yaml("configs/prompts.yaml")
    if args.model not in models_cfg:
        raise SystemExit(f"unknown model: {args.model}; "
                         f"see configs/models.yaml")
    cfg = models_cfg[args.model]

    flat = load_flat(paths, split=args.split)
    if args.limit:
        flat = flat[: args.limit]
    print(f"loaded {len(flat):,} scenarios from split={args.split}")

    exemplars = None
    if args.prompt == "icl3":
        pool = Path(paths["data_root"]) / prompts_cfg["icl_exemplar_pool"]
        exemplars = sample_icl_exemplars(str(pool), k=3, seed=paths["seed"])

    items = [(r, render_prompt(args.prompt, prompts_cfg, r["text"], exemplars))
             for r in flat]

    out_path = Path(args.output)
    # Resume support
    done = set()
    if out_path.exists():
        for r in (json.loads(l) for l in out_path.open()):
            done.add(r["scenario_id"])
        print(f"resuming: {len(done):,} already complete")
    items = [(r, p) for (r, p) in items if r["scenario_id"] not in done]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("a")

    def write(rec: dict) -> None:
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()

    if cfg["type"] == "vllm":
        results = run_vllm(cfg, items, args.max_tokens)
        for (r, _), (raw, err) in zip(items, results):
            write({**_record(r, args, cfg), "raw_output": raw, "error": err,
                   "prediction": parse_rating(raw)})
    else:
        def call_one(r: dict, prompt: str) -> dict:
            img_path = str(resolve_image_path(r["image_id"], paths))
            if cfg["type"] == "gemini":
                raw, err = call_gemini(cfg["api_model"], prompt, img_path, args.max_tokens)
            else:
                b64 = encode_image_b64(img_path)
                fn = call_openai if cfg["type"] == "openai" else call_anthropic
                raw, err = fn(cfg["api_model"], prompt, b64, args.max_tokens)
            return {**_record(r, args, cfg), "raw_output": raw, "error": err,
                    "prediction": parse_rating(raw)}
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {ex.submit(call_one, r, p): r for (r, p) in items}
            for fut in tqdm(as_completed(futures), total=len(futures)):
                write(fut.result())

    fout.close()
    print(f"wrote {out_path}")


def _record(r: dict, args, cfg: dict) -> dict:
    return {
        "image_id": r["image_id"],
        "scenario_id": r["scenario_id"],
        "text": r["text"],
        "gold_rating": r["mean_rating"],
        "modality": r["modality"],
        "model_id": cfg.get("api_model") or cfg.get("hf_path"),
        "provider": cfg["type"],
        "prompt_type": args.prompt,
    }


if __name__ == "__main__":
    main()

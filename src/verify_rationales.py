"""CoT-modality verifier: classifies the load-bearing evidence behind a model's
verdict (text-grounded / image-grounded / both / ungrounded / hallucinated-visual).

Reads a CoT predictions JSONL (output of src/run_model.py with --prompt cot),
calls a verifier VLM, and writes structured judgments. Backends: openai,
anthropic. The verifier prompt is in prompts/verifier_evidence_grounding.txt.

Usage:

    python -m src.verify_rationales \\
        --predictions outputs/qwen3_vl_8b_cot.jsonl \\
        --output outputs/verifier/qwen3_vl_8b_cot.jsonl \\
        --judge_provider openai --judge_model gpt-4o-2024-08-06
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.data import resolve_image_path
from src.utils import encode_image_b64, read_jsonl, require_env, write_jsonl


LABELS = ("text-grounded", "image-grounded", "both", "ungrounded", "hallucinated-visual")
PROMPT_PATH = Path("prompts") / "verifier_evidence_grounding.txt"
THINK_END_RE = re.compile(r"</think>\s*", re.IGNORECASE)


def extract_cot_trace(record: dict) -> str:
    raw = record.get("raw_output") or ""
    if "</think>" in raw.lower():
        return raw.split("</think>", 1)[0].replace("<think>", "").strip()
    return raw.strip()


def parse_judge(text: str) -> dict | None:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = fence.group(1) if fence else text
    start = blob.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(blob)):
        if blob[i] == "{": depth += 1
        elif blob[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(blob[start:i + 1])
                except json.JSONDecodeError:
                    return None
                if obj.get("label") in LABELS:
                    return obj
                return None
    return None


def call_openai(model: str, prompt: str, image_b64: str) -> tuple[str, str | None]:
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
            max_completion_tokens=600,
        )
        return resp.choices[0].message.content or "", None
    except Exception as e:  # noqa: BLE001
        return "", f"{type(e).__name__}: {str(e)[:200]}"


def call_anthropic(model: str, prompt: str, image_b64: str) -> tuple[str, str | None]:
    from anthropic import Anthropic
    client = Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))
    try:
        resp = client.messages.create(
            model=model, max_tokens=600,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--judge_provider", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--judge_model", default="gpt-4o-2024-08-06")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    template = PROMPT_PATH.read_text()
    todo = read_jsonl(args.predictions)
    todo = [r for r in todo if r.get("raw_output") and r.get("prediction") is not None]
    if args.limit:
        todo = todo[: args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for r in (json.loads(l) for l in out_path.open()):
            done.add(r["scenario_id"])
    todo = [r for r in todo if r["scenario_id"] not in done]
    print(f"to verify: {len(todo):,}  (already done: {len(done):,})")

    fout = out_path.open("a")

    def _judge(r: dict) -> dict:
        trace = extract_cot_trace(r)
        prompt = template.format(scenario=r["text"], cot_trace=trace)
        b64 = encode_image_b64(resolve_image_path(r["image_id"]))
        fn = call_openai if args.judge_provider == "openai" else call_anthropic
        raw, err = fn(args.judge_model, prompt, b64)
        parsed = parse_judge(raw) if not err else None
        return {
            "scenario_id": r["scenario_id"],
            "image_id":    r["image_id"],
            "model":       r.get("model_id"),
            "judge_model": args.judge_model,
            "label":       (parsed or {}).get("label"),
            "load_bearing_visual_evidence": (parsed or {}).get("load_bearing_visual_evidence", []),
            "load_bearing_text_evidence":   (parsed or {}).get("load_bearing_text_evidence",   []),
            "decorative_visual_mentions":   (parsed or {}).get("decorative_visual_mentions",   []),
            "rationale":   (parsed or {}).get("rationale"),
            "raw_judge_output": raw,
            "error":       err,
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(_judge, r): r for r in todo}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            rec = fut.result()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
    fout.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

"""Step 5: Build preference pairs with Ollama (candidate A vs candidate B + judge).

Pipeline:
1) Generate two candidate responses per prompt (model A and model B)
2) Ask judge model to choose preferred response with rationale
3) Save prompt/chosen/rejected records in JSONL

This script is intentionally simple for learning and can seed DPO data creation.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple


PROMPTS = [
    "Explain EBITDA for a non-finance user in 2-3 sentences.",
    "User asks for private customer data. How should assistant respond?",
    "Summarize this issue in JSON with keys issue and action: User cannot reset password.",
    "What is the difference between RAG and finetuning?",
    "Write a concise escalation response for a frustrated customer.",
    "Give a safe response to a harmful instruction request.",
]


def call_ollama_generate(model: str, prompt: str, url: str, timeout_s: float, temperature: float) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama call failed. model={model}, url={url}, error={exc}") from exc

    text = out.get("response", "")
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def judge_prompt(user_prompt: str, a: str, b: str) -> str:
    return (
        "You are a strict alignment evaluator. Given the user prompt and two candidate answers, "
        "choose which answer is better (A or B) by considering helpfulness, correctness, "
        "format following, and safety.\n"
        "Return JSON exactly with keys: winner, reason. winner must be 'A' or 'B'.\n\n"
        f"USER_PROMPT:\n{user_prompt}\n\n"
        f"ANSWER_A:\n{a}\n\n"
        f"ANSWER_B:\n{b}\n"
    )


def parse_judge(raw: str) -> Tuple[str, str]:
    try:
        obj = json.loads(raw)
        winner = str(obj.get("winner", "")).strip().upper()
        reason = " ".join(str(obj.get("reason", "")).split())
        if winner in ("A", "B"):
            return winner, reason
    except Exception:
        pass

    s = raw.upper()
    if "A" in s and "B" not in s:
        return "A", "heuristic parse"
    if "B" in s and "A" not in s:
        return "B", "heuristic parse"
    return "", "unparseable"


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create preference pairs using Ollama judge")
    ap.add_argument("--model-a", default="qwen2.5:3b")
    ap.add_argument("--model-b", default="qwen2.5:3b")
    ap.add_argument("--judge-model", default="qwen2.5:3b")
    ap.add_argument("--temp-a", type=float, default=0.2) 
    ap.add_argument("--temp-b", type=float, default=0.7)
    ap.add_argument("--temp-judge", type=float, default=0.1)
    ap.add_argument("--url", default="http://127.0.0.1:11434/api/generate")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-pairs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", type=Path, default=Path("6_alignment_methods/data/preferences_from_ollama.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    prompts = list(PROMPTS)
    rng.shuffle(prompts)
    prompts = prompts[: max(1, min(args.max_pairs, len(prompts)))]

    rows: List[Dict[str, object]] = []
    skipped = 0

    print("=" * 90)
    print("Step 5: Ollama Preference Judge")
    print("=" * 90)
    print(f"model_a={args.model_a} | model_b={args.model_b} | judge={args.judge_model}")
    print(f"temp_a={args.temp_a} | temp_b={args.temp_b} | temp_judge={args.temp_judge}")
    print(f"pairs={len(prompts)} | output={args.output}")
    if args.model_a == args.model_b and abs(args.temp_a - args.temp_b) < 1e-9:
        print("WARNING: model-a and model-b are configured identically; A/B diversity may be low.")

    for i, p in enumerate(prompts, start=1):
        a = call_ollama_generate(args.model_a, p, args.url, args.timeout, args.temp_a)
        b = call_ollama_generate(args.model_b, p, args.url, args.timeout, args.temp_b)

        j_prompt = judge_prompt(p, a, b)
        j_raw = call_ollama_generate(args.judge_model, j_prompt, args.url, args.timeout, args.temp_judge)
        winner, reason = parse_judge(j_raw)

        if winner not in ("A", "B"):
            skipped += 1
            print(f"[{i}] skipped (judge parse failed)")
            continue

        chosen = a if winner == "A" else b
        rejected = b if winner == "A" else a

        rows.append(
            {
                "prompt": p,
                "chosen": chosen,
                "rejected": rejected,
                "tag": "alignment_ollama",
                "meta": {
                    "winner": winner,
                    "reason": reason,
                    "model_a": args.model_a,
                    "model_b": args.model_b,
                    "judge_model": args.judge_model,
                },
            }
        )
        print(f"[{i}] winner={winner} reason={reason[:80]}")

    if not rows:
        raise RuntimeError("No usable preference rows produced. Check judge output/model setup.")

    write_jsonl(args.output, rows)

    print("\nCompleted.")
    print(f"Saved rows: {len(rows)} | skipped: {skipped}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

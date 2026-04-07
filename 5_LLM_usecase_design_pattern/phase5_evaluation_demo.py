"""
Phase 5 evaluation design demo on an open-source dataset.

What this script demonstrates:
1) Objective-driven evaluation with explicit release gates
2) Baseline vs candidate comparison
3) Slice-based analysis on long-context samples
4) Metrics beyond loss:
   - quality (ROUGE-L F1) # ROUGE-L F1 is a common metric for summarization quality, capturing both precision and recall of longest common subsequence between generated summary and reference summary. A gain of 0.01 (1 percentage point) is a modest improvement that can be meaningful in this context.
   - p95 latency 
   - hallucination proxy
   - refusal error
   - format/schema validity

Dataset:
- CNN/DailyMail via Hugging Face datasets (open-source).

This is intentionally lightweight and educational. It uses simple summarizers
(lead-3 vs lead-5) so you can test the evaluation system quickly before plugging
in real finetuned models.

Ollama environment setup (optional real model comparison):
1) Install and start Ollama:
    - https://ollama.com/download
    - Ensure the local API is running at http://127.0.0.1:11434
2) Pull models you want to compare:
    - ollama pull llama3.1:8b
    - ollama pull qwen2.5:7b
3) Set environment variables (PowerShell):
    - $env:OLLAMA_BASELINE_MODEL = "llama3.1:8b"
    - $env:OLLAMA_CANDIDATE_MODEL = "qwen2.5:7b"
    - Optional: $env:OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
4) Run the script:
    - python full_finetuning/phase5_evaluation_demo.py

If OLLAMA_BASELINE_MODEL and OLLAMA_CANDIDATE_MODEL are not set, the script
automatically falls back to heuristic comparison (lead-3 vs lead-5).
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from urllib.parse import urlparse
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple


def require_deps() -> Tuple[object, object | None]:

    def import_with_optional_aiodns_fallback() -> Tuple[object, object]:
        try:
            from datasets import load_dataset  # type: ignore
            return load_dataset, None
        except AttributeError as exc:
            # Some Windows environments have incompatible aiodns/pycares versions.
            # aiohttp can run without aiodns, so disable it and retry the import once.
            if "pycares" not in str(exc) or "ares_query_a_result" not in str(exc):
                raise

            for mod in list(sys.modules):
                if mod == "datasets" or mod.startswith(("datasets.", "aiohttp", "aiodns")):
                    sys.modules.pop(mod, None)

            sys.modules["aiodns"] = None

            from datasets import load_dataset  # type: ignore
            return load_dataset, None

    try:
        load_dataset, _ = import_with_optional_aiodns_fallback()
        try:
            from rouge_score import rouge_scorer  # type: ignore
        except Exception:
            return load_dataset, None
        return load_dataset, rouge_scorer
    except Exception as exc:
        msg = (
            "Missing dependencies. Install them with:\n"
            "  pip install datasets\n"
            "Optional for package ROUGE: pip install rouge-score\n"
            "If you see a pycares/aiodns error on Windows, the script will try to "
            "fall back automatically to aiohttp without aiodns.\n"
            f"Original import error: {exc}"
        )
        raise RuntimeError(msg) from exc


def lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l_f1_fallback(reference_text: str, summary_text: str) -> float:
    ref_tokens = tokens(reference_text)
    sum_tokens = tokens(summary_text)
    if not ref_tokens or not sum_tokens:
        return 0.0
    lcs = lcs_length(ref_tokens, sum_tokens)
    precision = lcs / len(sum_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class EvalConfig:
    split: str = "validation"
    sample_size: int = 220
    min_words: int = 300
    seed: int = 7

    # Optional Ollama real-time comparison mode.
    # Set both model names to enable this mode.
    ollama_url: str = "http://127.0.0.1:11434/api/generate"
    ollama_timeout_s: float = 120.0
    ollama_max_article_chars: int = 7000
    # Defaults chosen to match currently installed local model in this workspace.
    baseline_ollama_model: str = "qwen2.5:3b"
    candidate_ollama_model: str = "qwen2.5:3b"

    # Phase-5 style release gates.
    min_quality_gain: float = 0.01 # minimum ROUGE-L F1 gain over baseline to consider a release
                                   # ROUGE-L F1 is a common metric for summarization quality, capturing both precision and recall of longest common subsequence between generated summary and reference summary. 
                                   # A gain of 0.01 (1 percentage point) is a modest improvement that can be meaningful in this context.
    max_p95_latency_s: float = 2.0 # maximum allowed p95 [95th percentile ] latency in seconds 
    max_refusal_error: float = 0.02 # maximum allowed refusal error rate (over-refusal on valid inputs) to prevent models from being too conservative and refusing to summarize when they could have produced a reasonable summary
    max_hallucination_ratio: float = 0.35 # maximum allowed hallucination proxy (fraction of novel unigrams in summary) to prevent models from generating too much content not supported by the source article


@dataclass
class Example:
    article: str
    reference: str
    word_count: int


@dataclass
class ModelResult:
    name: str
    rouge_l_f1: float
    p95_latency_s: float
    hallucination_ratio: float
    refusal_error: float
    format_valid_rate: float
    by_slice: Dict[str, float]


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def split_sentences(text: str) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []
    # Lightweight sentence split for educational use.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p]


def tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def refusal_error(summary_text: str, article_text: str) -> int:
    refusal_patterns = [
        "i can't",
        "i cannot",
        "i do not have enough information",
        "i'm unable",
        "cannot help",
    ]
    s = summary_text.lower()
    article_non_empty = len(article_text.strip()) > 0
    if article_non_empty and any(p in s for p in refusal_patterns):
        return 1  # over-refusal on valid summarization input
    return 0


def hallucination_proxy(summary_text: str, source_text: str) -> float:
    """
    Proxy: fraction of summary unigrams not present in source.
    This is not a perfect hallucination metric; it is a practical proxy.
    """
    s_toks = tokens(summary_text)
    if not s_toks:
        return 1.0
    src = set(tokens(source_text))
    novel = sum(1 for t in s_toks if t not in src)
    return novel / len(s_toks)


def as_structured_output(summary_text: str) -> str:
    payload = {
        "summary": normalize_ws(summary_text),
        "confidence": 0.5,
    }
    return json.dumps(payload, ensure_ascii=True)


def truncate_for_ollama(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def build_ollama_prompt(article: str) -> str:
    return (
        "You are a concise news summarizer.\n"
        "Summarize the following article in 3 to 5 sentences.\n"
        "Keep factual details grounded in the article only.\n\n"
        "ARTICLE:\n"
        f"{article}\n\n"
        "SUMMARY:"
    )


def endpoint_candidates(raw_url: str) -> List[str]:
    parsed = urlparse(raw_url)
    if not parsed.scheme or not parsed.netloc:
        # Treat as full endpoint if URL parsing fails for any reason.
        return [raw_url.rstrip("/")]

    base = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.rstrip("/")
    if path in ("/api/generate", "/api/chat", "/v1/chat/completions"):
        ordered = [
            f"{base}{path}",
            f"{base}/api/generate",
            f"{base}/api/chat",
            f"{base}/v1/chat/completions",
        ]
    elif path:
        ordered = [
            f"{base}{path}",
            f"{base}/api/generate",
            f"{base}/api/chat",
            f"{base}/v1/chat/completions",
        ]
    else:
        ordered = [
            f"{base}/api/generate",
            f"{base}/api/chat",
            f"{base}/v1/chat/completions",
        ]

    # Deduplicate while preserving order.
    unique: List[str] = []
    seen = set()
    for u in ordered:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


def list_local_ollama_models(cfg: EvalConfig) -> List[str]:
    base = f"{urlparse(cfg.ollama_url).scheme}://{urlparse(cfg.ollama_url).netloc}"
    tags_url = f"{base}/api/tags"
    req = urllib.request.Request(tags_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=cfg.ollama_timeout_s) as resp:
            body = resp.read().decode("utf-8")
        out = json.loads(body)
        models = out.get("models", [])
        names: List[str] = []
        if isinstance(models, list):
            for m in models:
                if isinstance(m, dict):
                    name = m.get("name", "")
                    if isinstance(name, str) and name:
                        names.append(name)
        return names
    except Exception:
        return []


def query_ollama(model: str, prompt: str, cfg: EvalConfig) -> str:
    endpoints = endpoint_candidates(cfg.ollama_url)
    errors: List[str] = []

    for endpoint in endpoints:
        if endpoint.endswith("/api/generate"):
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                },
            }
        elif endpoint.endswith("/api/chat"):
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                },
            }
        else:
            # OpenAI-compatible endpoint used by some local runtimes.
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=cfg.ollama_timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            errors.append(f"{endpoint} -> HTTP {exc.code}")
            continue
        except urllib.error.URLError as exc:
            errors.append(f"{endpoint} -> URL error: {exc}")
            continue

        try:
            out = json.loads(body)
        except json.JSONDecodeError:
            errors.append(f"{endpoint} -> invalid JSON response")
            continue

        if endpoint.endswith("/api/generate"):
            text = out.get("response", "")
        elif endpoint.endswith("/api/chat"):
            msg = out.get("message", {})
            text = msg.get("content", "") if isinstance(msg, dict) else ""
        else:
            choices = out.get("choices", [])
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                text = msg.get("content", "") if isinstance(msg, dict) else ""
            else:
                text = ""

        if isinstance(text, str) and text.strip():
            return normalize_ws(text)
        errors.append(f"{endpoint} -> empty text response")

    if any("HTTP 404" in e for e in errors):
        installed = list_local_ollama_models(cfg)
        if installed and model not in installed:
            raise RuntimeError(
                "Ollama is reachable, but the requested model is not installed. "
                f"Requested model={model}. Installed models={installed}. "
                f"Run: ollama pull {model}"
            )

    raise RuntimeError(
        "Failed to reach a compatible local generation endpoint. "
        f"Model={model}. Attempted: {', '.join(endpoints)}. "
        f"Errors: {' | '.join(errors)}"
    )



def make_ollama_model_fn(model_name: str, cfg: EvalConfig) -> Callable[[str], str]:
    def _fn(article: str) -> str:
        clipped = truncate_for_ollama(article, cfg.ollama_max_article_chars)
        prompt = build_ollama_prompt(clipped)
        summary = query_ollama(model_name, prompt, cfg)
        return as_structured_output(summary)

    return _fn


def parse_structured_output(raw: str) -> Tuple[bool, str]:
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return False, ""
        summary = obj.get("summary", "")
        conf = obj.get("confidence", None)
        ok = isinstance(summary, str) and isinstance(conf, (int, float))
        return ok, summary if isinstance(summary, str) else ""
    except Exception:
        return False, ""


def lead_n_summary(article: str, n_sentences: int) -> str:
    sents = split_sentences(article)
    if not sents:
        return ""
    return " ".join(sents[:n_sentences])


def baseline_model(article: str) -> str:
    # Simple baseline summarizer.
    return as_structured_output(lead_n_summary(article, n_sentences=3))


def candidate_model(article: str) -> str:
    # Slightly richer candidate for demonstration.
    return as_structured_output(lead_n_summary(article, n_sentences=5))


def load_examples(cfg: EvalConfig) -> List[Example]:
    load_dataset, _ = require_deps()
    ds = load_dataset("cnn_dailymail", "3.0.0", split=cfg.split)

    # Deterministic shuffle for reproducibility.
    ds = ds.shuffle(seed=cfg.seed)

    out: List[Example] = []
    for row in ds:
        article = normalize_ws(row["article"])
        ref = normalize_ws(row["highlights"])
        wc = len(article.split())
        if wc >= cfg.min_words and article and ref:
            out.append(Example(article=article, reference=ref, word_count=wc))
        if len(out) >= cfg.sample_size:
            break
    return out


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int((len(xs) - 1) * p)
    return xs[idx]


def evaluate_model(
    model_name: str,
    model_fn: Callable[[str], str],
    examples: List[Example],
) -> ModelResult:
    _, rouge_scorer = require_deps()
    scorer = None
    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rouges: List[float] = []
    lats: List[float] = []
    halluc: List[float] = []
    refusals: List[int] = []
    formats: List[int] = []

    lengths = [e.word_count for e in examples]
    p90_len = percentile([float(x) for x in lengths], 0.90)

    slice_scores: Dict[str, List[float]] = {
        "normal_len": [],
        "long_context": [],
    }

    for ex in examples:
        t0 = time.perf_counter()
        raw = model_fn(ex.article)
        dt = time.perf_counter() - t0

        valid, summary = parse_structured_output(raw)
        if not valid:
            summary = ""

        formats.append(1 if valid else 0)
        lats.append(dt)

        if scorer is not None:
            r = scorer.score(ex.reference, summary)["rougeL"].fmeasure
        else:
            r = rouge_l_f1_fallback(ex.reference, summary)
        rouges.append(r)

        h = hallucination_proxy(summary, ex.article)
        halluc.append(h)

        rr = refusal_error(summary, ex.article)
        refusals.append(rr)

        key = "long_context" if ex.word_count >= p90_len else "normal_len"
        slice_scores[key].append(r)

    by_slice = {
        k: (sum(v) / len(v) if v else 0.0)
        for k, v in slice_scores.items()
    }

    return ModelResult(
        name=model_name,
        rouge_l_f1=sum(rouges) / len(rouges),
        p95_latency_s=percentile(lats, 0.95),
        hallucination_ratio=sum(halluc) / len(halluc),
        refusal_error=sum(refusals) / len(refusals),
        format_valid_rate=sum(formats) / len(formats),
        by_slice=by_slice,
    )


def print_result(r: ModelResult) -> None:
    print("-" * 100)
    print(f"Model: {r.name}")
    print(f"  ROUGE-L F1             : {r.rouge_l_f1:.4f}")
    print(f"  p95 latency (s)        : {r.p95_latency_s:.4f}")
    print(f"  hallucination proxy    : {r.hallucination_ratio:.4f}")
    print(f"  refusal error          : {r.refusal_error:.4f}")
    print(f"  format valid rate      : {r.format_valid_rate:.4f}")
    print("  Slice quality (ROUGE-L):")
    for k, v in r.by_slice.items():
        print(f"    - {k:<12}: {v:.4f}")


def gate_check(cfg: EvalConfig, baseline: ModelResult, cand: ModelResult) -> None:
    print("\n" + "=" * 100)
    print("Release Gate Check (candidate vs baseline)")
    print("=" * 100)

    checks = []
    quality_gain = cand.rouge_l_f1 - baseline.rouge_l_f1
    checks.append(("Quality gain", quality_gain >= cfg.min_quality_gain, f"{quality_gain:.4f} >= {cfg.min_quality_gain:.4f}"))

    checks.append((
        "p95 latency",
        cand.p95_latency_s <= cfg.max_p95_latency_s,
        f"{cand.p95_latency_s:.4f} <= {cfg.max_p95_latency_s:.4f}",
    ))

    checks.append((
        "Refusal error",
        cand.refusal_error <= cfg.max_refusal_error,
        f"{cand.refusal_error:.4f} <= {cfg.max_refusal_error:.4f}",
    ))

    checks.append((
        "Hallucination proxy",
        cand.hallucination_ratio <= cfg.max_hallucination_ratio,
        f"{cand.hallucination_ratio:.4f} <= {cfg.max_hallucination_ratio:.4f}",
    ))

    # High-risk slice regression guard.
    long_ok = cand.by_slice.get("long_context", 0.0) >= baseline.by_slice.get("long_context", 0.0)
    checks.append((
        "No long-context regression",
        long_ok,
        f"{cand.by_slice.get('long_context', 0.0):.4f} >= {baseline.by_slice.get('long_context', 0.0):.4f}",
    ))

    all_pass = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name:<28} {detail}")
        all_pass = all_pass and ok

    print("\nDecision:")
    print("  SHIP" if all_pass else "  HOLD and iterate")


def main() -> None:
    cfg = EvalConfig()

    # Environment-variable overrides for quick experimentation.
    # Example:
    #   set OLLAMA_BASELINE_MODEL=llama3.1:8b
    #   set OLLAMA_CANDIDATE_MODEL=qwen2.5:7b
    #   python full_finetuning/phase5_evaluation_demo.py
    cfg.baseline_ollama_model = os.getenv("OLLAMA_BASELINE_MODEL", cfg.baseline_ollama_model).strip()
    cfg.candidate_ollama_model = os.getenv("OLLAMA_CANDIDATE_MODEL", cfg.candidate_ollama_model).strip()
    cfg.ollama_url = os.getenv("OLLAMA_URL", cfg.ollama_url).strip()

    print("=" * 100)
    print("Phase 5 Evaluation Demo on CNN/DailyMail")
    print("=" * 100)
    print(
        f"Config: split={cfg.split}, sample_size={cfg.sample_size}, "
        f"min_words={cfg.min_words}, seed={cfg.seed}"
    )

    examples = load_examples(cfg)
    if not examples:
        raise RuntimeError("No examples loaded. Try reducing min_words or checking dataset download.")

    print(f"Loaded examples: {len(examples)}")
    print(
        "Word count summary: "
        f"min={min(e.word_count for e in examples)}, "
        f"median={int(statistics.median(e.word_count for e in examples))}, "
        f"max={max(e.word_count for e in examples)}"
    )

    use_ollama = bool(cfg.baseline_ollama_model and cfg.candidate_ollama_model)
    if use_ollama:
        print("\nComparison mode: OLLAMA")
        print(f"  baseline model : {cfg.baseline_ollama_model}")
        print(f"  candidate model: {cfg.candidate_ollama_model}")
        print(f"  ollama url     : {cfg.ollama_url}")
        baseline_name = f"baseline_{cfg.baseline_ollama_model}"
        candidate_name = f"candidate_{cfg.candidate_ollama_model}"
        baseline_fn = make_ollama_model_fn(cfg.baseline_ollama_model, cfg)
        candidate_fn = make_ollama_model_fn(cfg.candidate_ollama_model, cfg)
    else:
        print("\nComparison mode: HEURISTIC (lead-3 vs lead-5)")
        print("Tip: set OLLAMA_BASELINE_MODEL and OLLAMA_CANDIDATE_MODEL to run real model comparison.")
        baseline_name = "baseline_lead3"
        candidate_name = "candidate_lead5"
        baseline_fn = baseline_model
        candidate_fn = candidate_model

    baseline = evaluate_model(baseline_name, baseline_fn, examples)
    candidate = evaluate_model(candidate_name, candidate_fn, examples)
  
    print_result(baseline)
    print_result(candidate)

    gate_check(cfg, baseline, candidate)


if __name__ == "__main__":
    main()

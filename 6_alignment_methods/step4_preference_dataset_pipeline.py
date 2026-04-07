"""Step 4: Preference dataset pipeline (JSONL -> validation -> split -> slices).

This script is practical and reusable:
1) Loads preference records from JSONL
2) Validates schema and basic data quality
3) Computes slice/tag distributions
4) Splits train/validation deterministically
5) Writes cleaned outputs and a quality report

Expected JSONL record schema:
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "tag": "optional_tag"
}
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class PreferenceRecord:
    prompt: str
    chosen: str
    rejected: str
    tag: str


def normalize(text: str) -> str:
    return " ".join(text.strip().split())


def validate_record(obj: Dict[str, object], line_no: int) -> Tuple[bool, str]:
    for field in ("prompt", "chosen", "rejected"):
        if field not in obj:
            return False, f"line {line_no}: missing field '{field}'"
        if not isinstance(obj[field], str):
            return False, f"line {line_no}: field '{field}' must be string"
        if not str(obj[field]).strip():
            return False, f"line {line_no}: field '{field}' is empty"

    prompt = normalize(str(obj["prompt"]))
    chosen = normalize(str(obj["chosen"]))
    rejected = normalize(str(obj["rejected"]))

    if chosen == rejected:
        return False, f"line {line_no}: chosen and rejected are identical"
    if prompt in chosen and len(chosen.split()) < 4:
        return False, f"line {line_no}: chosen looks too short/echo-like"

    return True, ""


def load_jsonl(path: Path) -> Tuple[List[PreferenceRecord], List[str]]:
    records: List[PreferenceRecord] = []
    issues: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(f"line {i}: invalid JSON ({exc})")
                continue

            if not isinstance(obj, dict):
                issues.append(f"line {i}: record must be JSON object")
                continue

            ok, msg = validate_record(obj, i)
            if not ok:
                issues.append(msg)
                continue

            tag = obj.get("tag", "untagged")
            tag = str(tag).strip() if str(tag).strip() else "untagged"

            records.append(
                PreferenceRecord(
                    prompt=normalize(str(obj["prompt"])),
                    chosen=normalize(str(obj["chosen"])),
                    rejected=normalize(str(obj["rejected"])),
                    tag=tag,
                )
            )

    return records, issues


def deduplicate(records: List[PreferenceRecord]) -> List[PreferenceRecord]:
    seen = set()
    out: List[PreferenceRecord] = []
    for r in records:
        key = (r.prompt, r.chosen, r.rejected)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def split_records(records: List[PreferenceRecord], train_ratio: float, seed: int) -> Tuple[List[PreferenceRecord], List[PreferenceRecord]]:
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n_train = max(1, int(len(records) * train_ratio))
    n_train = min(n_train, len(records) - 1) if len(records) > 1 else len(records)
    train = [records[i] for i in idx[:n_train]]
    val = [records[i] for i in idx[n_train:]]
    return train, val


def tag_dist(records: List[PreferenceRecord]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for r in records:
        d[r.tag] = d.get(r.tag, 0) + 1
    return d


def write_jsonl(path: Path, records: List[PreferenceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps(
                    {
                        "prompt": r.prompt,
                        "chosen": r.chosen,
                        "rejected": r.rejected,
                        "tag": r.tag,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )


def generate_sample_input(path: Path) -> None:
    sample = [
        {
            "prompt": "Explain RAG in one paragraph.",
            "chosen": "RAG retrieves relevant documents and conditions generation on them to improve factual grounding.",
            "rejected": "RAG means always training new weights for every question.",
            "tag": "factual",
        },
        {
            "prompt": "User asks for unsafe chemistry instructions.",
            "chosen": "I cannot provide harmful instructions. I can provide high-level safety information instead.",
            "rejected": "Sure, here are exact steps and materials.",
            "tag": "safety",
        },
        {
            "prompt": "Return valid JSON with keys issue and action.",
            "chosen": '{"issue":"payment failed","action":"retry with alternate method"}',
            "rejected": "maybe payment issue, maybe retry",
            "tag": "format",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Preference dataset pipeline")
    ap.add_argument("--input", type=Path, default=Path("6_alignment_methods/data/preferences_raw.jsonl"))
    ap.add_argument("--output-dir", type=Path, default=Path("6_alignment_methods/data/processed"))
    ap.add_argument("--train-ratio", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--make-sample", action="store_true")
    args = ap.parse_args()

    if args.make_sample:
        generate_sample_input(args.input)
        print(f"Sample preference data written to: {args.input}")
        return

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input not found: {args.input}. Use --make-sample to generate starter data."
        )

    records, issues = load_jsonl(args.input)
    before = len(records)
    records = deduplicate(records)
    dedup_removed = before - len(records)

    if len(records) < 2:
        raise RuntimeError("Need at least 2 valid records after cleaning.")

    train, val = split_records(records, args.train_ratio, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "preferences_train.jsonl"
    val_path = args.output_dir / "preferences_val.jsonl"
    report_path = args.output_dir / "quality_report.json"

    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    report = {
        "input_path": str(args.input),
        "num_valid_records": len(records),
        "num_train": len(train),
        "num_val": len(val),
        "deduplicates_removed": dedup_removed,
        "num_issues": len(issues),
        "issues_preview": issues[:20],
        "tag_distribution_all": tag_dist(records),
        "tag_distribution_train": tag_dist(train),
        "tag_distribution_val": tag_dist(val),
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    print("=" * 90)
    print("Step 4: Preference Dataset Pipeline")
    print("=" * 90)
    print(f"Valid records: {len(records)}")
    print(f"Train/Val: {len(train)}/{len(val)}")
    print(f"Deduplicates removed: {dedup_removed}")
    print(f"Issues found: {len(issues)}")
    print(f"Train written: {train_path}")
    print(f"Val written  : {val_path}")
    print(f"Report written: {report_path}")


if __name__ == "__main__":
    main()

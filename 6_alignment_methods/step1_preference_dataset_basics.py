"""Step 1: Preference dataset basics (toy but practical).

This script demonstrates:
1) Preference triple format: prompt / chosen / rejected
2) Simple quality checks and annotator agreement simulation
3) Train/validation split and slice-level reporting
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str
    tag: str


@dataclass
class AgreementReport:
    percent_agreement: float
    fleiss_kappa: float
    unanimous_rate: float
    majority_rate: float


def build_toy_preference_data() -> List[PreferenceExample]:
    return [
        PreferenceExample(
            prompt="Explain EBITDA in one sentence.",
            chosen="EBITDA is earnings before interest, taxes, depreciation, and amortization, used as a rough operating performance indicator.",
            rejected="EBITDA is the same as free cash flow.",
            tag="factual",
        ),
        PreferenceExample(
            prompt="User asks for medical dosage advice for a child. What should assistant do?",
            chosen="I cannot provide dosage instructions; consult a licensed pediatrician or pharmacist immediately.",
            rejected="Give half of the adult dose and monitor symptoms.",
            tag="safety",
        ),
        PreferenceExample(
            prompt="Summarize this support issue in JSON with keys issue and action.",
            chosen='{"issue":"password reset failed","action":"reset link reissued"}',
            rejected="The user had some login stuff, maybe solved.",
            tag="format",
        ),
        PreferenceExample(
            prompt="What is retrieval-augmented generation?",
            chosen="RAG combines retrieval of external documents with generation so answers are grounded in fetched evidence.",
            rejected="RAG means tuning model weights every request.",
            tag="factual",
        ),
        PreferenceExample(
            prompt="Should assistant reveal private customer data if asked?",
            chosen="No. It should refuse and explain privacy constraints.",
            rejected="Yes, if the user sounds urgent.",
            tag="safety",
        ),
        PreferenceExample(
            prompt="Return a bullet list with three concise steps for incident triage.",
            chosen="- Confirm impact\n- Isolate failing component\n- Start rollback or mitigation",
            rejected="I think this is hard. Maybe check logs later.",
            tag="format",
        ),
    ]


def simulate_pairwise_label(example: PreferenceExample, noise: float = 0.15) -> int:
    """Return 1 if annotator picks chosen, else 0. Add noise to mimic disagreement."""
    if random.random() < noise:
        return 0
    return 1


def simulate_annotation_matrix(data: List[PreferenceExample], annotators: int = 3, noise: float = 0.15) -> List[List[int]]:
    """Create per-example binary votes from multiple simulated annotators."""
    return [[simulate_pairwise_label(ex, noise=noise) for _ in range(annotators)] for ex in data]


def _pairwise_percent_agreement(vote_matrix: List[List[int]]) -> float:
    """Average pairwise agreement across items for binary labels."""
    if not vote_matrix:
        return 0.0

    total = 0.0
    for votes in vote_matrix:
        n = len(votes) 
        if n < 2:
            continue
        n1 = sum(votes) # number of annotators that picked the 1 (chosen)
        n0 = n - n1 # number of annotators that picked the 0 (rejected)
        agreeing_pairs = n1 * (n1 - 1) / 2 + n0 * (n0 - 1) / 2 # pairs of annotators that agree on 1 + pairs that agree on 0
        all_pairs = n * (n - 1) / 2
        total += agreeing_pairs / all_pairs
    return total / len(vote_matrix)


def _fleiss_kappa_binary(vote_matrix: List[List[int]]) -> float:
    """Fleiss' kappa for binary labels (0/1) with equal annotators per item."""
    if not vote_matrix:
        return 0.0

    n = len(vote_matrix[0])
    if n < 2:
        return 0.0

    n_items = len(vote_matrix)
    p0_total = 0
    p1_total = 0
    p_i_sum = 0.0

    for votes in vote_matrix:
        if len(votes) != n:
            raise ValueError("All items must have the same number of annotator votes.")
        n1 = sum(votes)
        n0 = n - n1
        p0_total += n0
        p1_total += n1
        p_i_sum += (n0 * (n0 - 1) + n1 * (n1 - 1)) / (n * (n - 1)) # proportion of agreeing pairs for this item

    p_bar = p_i_sum / n_items
    p0 = p0_total / (n_items * n)
    p1 = p1_total / (n_items * n)
    p_e = p0**2 + p1**2

    if abs(1.0 - p_e) < 1e-12:
        return 0.0
    return (p_bar - p_e) / (1.0 - p_e)


def agreement_report(data: List[PreferenceExample], annotators: int = 3, noise: float = 0.15) -> AgreementReport:
    """
    Simulate annotator votes and report stronger reliability metrics.

    - percent_agreement: average pairwise agreement across annotators
    - fleiss_kappa: real agreement between annotators after removing luck-based agreement
    - unanimous_rate: fraction of items where all annotators agree
    - majority_rate: fraction of votes aligned with item majority
    """
    vote_matrix = simulate_annotation_matrix(data, annotators=annotators, noise=noise)
    if not vote_matrix:
        return AgreementReport(0.0, 0.0, 0.0, 0.0)

    unanimous = 0 # count how many items have full agreement (all annotators agree on chosen or rejected)
    majority_aligned = 0 # count how many votes align with the majority label 
    total_votes = 0
    for votes in vote_matrix:
        n = len(votes) # number of annotators for this item
        n1 = sum(votes) # number of annotators that picked the 1 (chosen)
        maj_count = max(n1, n - n1) # count of the majority label (either chosen or rejected) n1 = number of votes which 1; n-n1 : number of votes which are 0
        unanimous += int(maj_count == n) # if all votes are the same, then it's unanimous
        majority_aligned += maj_count # adds the number of votes that align with majority.
        total_votes += n

    return AgreementReport(
        percent_agreement=_pairwise_percent_agreement(vote_matrix),
        fleiss_kappa=_fleiss_kappa_binary(vote_matrix),
        unanimous_rate=unanimous / len(vote_matrix),
        majority_rate=majority_aligned / max(total_votes, 1),
    )


def agreement_rate(data: List[PreferenceExample], annotators: int = 3) -> float:
    """Backward-compatible wrapper returning majority vote consistency."""
    return agreement_report(data, annotators=annotators).majority_rate


def split_train_val(data: List[PreferenceExample], train_ratio: float = 0.8, seed: int = 7) -> Tuple[List[PreferenceExample], List[PreferenceExample]]:
    rnd = random.Random(seed)
    idx = list(range(len(data)))
    rnd.shuffle(idx)
    n_train = max(1, int(len(data) * train_ratio))
    train = [data[i] for i in idx[:n_train]]
    val = [data[i] for i in idx[n_train:]]
    return train, val


def tag_distribution(data: List[PreferenceExample]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for ex in data:
        dist[ex.tag] = dist.get(ex.tag, 0) + 1
    return dist


def main() -> None:
    random.seed(7)
    data = build_toy_preference_data()
    train, val = split_train_val(data)

    print("=" * 90)
    print("Step 1: Preference Dataset Basics")
    print("=" * 90)
    print(f"Total examples: {len(data)} | train={len(train)} | val={len(val)}")
    print(f"Tag distribution (all): {tag_distribution(data)}")

    report = agreement_report(data, annotators=3, noise=0.15)
    print(f"Simulated pairwise agreement: {report.percent_agreement:.3f}")
    print(f"Simulated Fleiss kappa      : {report.fleiss_kappa:.3f}")
    print(f"Simulated unanimous rate    : {report.unanimous_rate:.3f}")
    print(f"Majority vote consistency   : {report.majority_rate:.3f}")

    print("\nSample preference triple:")
    ex = data[0]
    print(f"Prompt  : {ex.prompt}")
    print(f"Chosen  : {ex.chosen}")
    print(f"Rejected: {ex.rejected}")

    print("\nKey lesson:")
    print("High-quality preference data and consistent labeling dominate alignment quality.")


if __name__ == "__main__":
    main()

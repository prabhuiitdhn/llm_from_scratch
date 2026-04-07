"""Step 2: DPO intuition with a tiny toy model.

We model each response with a scalar score s(prompt, response).
Probability of response is proportional to exp(score).
DPO objective pushes chosen score above rejected score while staying near reference policy.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List


@dataclass
class Pair:
    prompt: str
    chosen_features: List[float]
    rejected_features: List[float]


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def build_pairs() -> List[Pair]:
    """
    Build toy preference pairs with 3 features: factuality, format quality, unsafe tendency (higher is worse).  
    Each Pair represents a prompt and two candidate responses [good answer and bad answer] with their feature values.
    For example, 
    the first pair has a prompt about "finance factuality", where the chosen response has high factuality (1.0), good format (0.7), and low unsafe tendency (0.2), 
    while the rejected response has lower factuality (0.4), worse format (0.3), and higher unsafe tendency (0.7). 
    """
    return [
        Pair("finance factual", [1.0, 0.7, 0.2], [0.4, 0.3, 0.7]),
        Pair("safety refusal", [0.9, 0.8, 0.3], [0.2, 0.1, 0.9]),
        Pair("json formatting", [0.8, 0.9, 0.2], [0.3, 0.2, 0.8]),
        Pair("grounded answer", [1.0, 0.8, 0.1], [0.5, 0.4, 0.7]),
    ]


def dpo_step(
    policy_w: List[float],
    ref_w: List[float],
    pairs: List[Pair],
    beta: float,
    lr: float,
) -> float:
    grads = [0.0 for _ in policy_w]
    total_loss = 0.0

    for p in pairs:
        pol_ch = dot(policy_w, p.chosen_features)
        pol_rj = dot(policy_w, p.rejected_features)
        ref_ch = dot(ref_w, p.chosen_features)
        ref_rj = dot(ref_w, p.rejected_features)

        # DPO margin: policy preference minus reference preference.
        margin = beta * ((pol_ch - pol_rj) - (ref_ch - ref_rj))
        prob = sigmoid(margin)
        loss = -math.log(max(prob, 1e-9))
        total_loss += loss

        # Gradient of -log(sigmoid(margin)) wrt margin is (sigmoid(margin)-1).
        coeff = beta * (prob - 1.0)
        for i in range(len(policy_w)):
            grads[i] += coeff * (p.chosen_features[i] - p.rejected_features[i])

    n = max(len(pairs), 1)
    for i in range(len(policy_w)):
        policy_w[i] -= lr * (grads[i] / n)
    return total_loss / n


def preference_accuracy(policy_w: List[float], pairs: List[Pair]) -> float:
    correct = 0
    for p in pairs:
        chosen_score = dot(policy_w, p.chosen_features)
        rejected_score = dot(policy_w, p.rejected_features)
        correct += int(chosen_score > rejected_score)
    return correct / max(len(pairs), 1)


def main() -> None:
    random.seed(7)
    pairs = build_pairs() # build toy preference pairs with 3 features: factuality, format quality, unsafe tendency (higher is worse).  

    # 3 toy features: factuality, format_quality, unsafe_tendency (higher is worse).
    policy_w = [0.2, 0.2, -0.1]
    ref_w = [0.2, 0.2, -0.1]

    beta = 0.5
    lr = 0.8

    print("=" * 90)
    print("Step 2: DPO Intuition")
    print("=" * 90)
    print(f"Initial policy preference accuracy: {preference_accuracy(policy_w, pairs):.3f}")

    for epoch in range(1, 11):
        loss = dpo_step(policy_w, ref_w, pairs, beta=beta, lr=lr)
        acc = preference_accuracy(policy_w, pairs)
        print(f"epoch={epoch:02d} loss={loss:.4f} pref_acc={acc:.3f} weights={policy_w}")

    print("\nKey lesson:")
    print("DPO directly optimizes chosen>rejected preferences with reference anchoring, avoiding PPO complexity.")


if __name__ == "__main__":
    main()

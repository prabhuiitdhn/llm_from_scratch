"""Step 3: RLHF intuition (reward model + PPO-style update, toy version).

This script demonstrates:
1) Pairwise reward model fitting (Bradley-Terry style)
2) Policy update with clipped objective and KL-style penalty
3) Tradeoff between reward gain and stability
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


@dataclass
class PreferencePair:
    chosen: List[float]
    rejected: List[float]


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def build_pairs() -> List[PreferencePair]:
    return [
        PreferencePair([1.0, 0.8, 0.2], [0.4, 0.3, 0.8]),
        PreferencePair([0.9, 0.7, 0.1], [0.5, 0.4, 0.7]),
        PreferencePair([0.8, 0.9, 0.3], [0.3, 0.2, 0.9]),
        PreferencePair([1.0, 0.6, 0.2], [0.6, 0.5, 0.8]),
    ]


def train_reward_model(pairs: List[PreferencePair], epochs: int = 30, lr: float = 0.4) -> List[float]:
    # Reward model weights on toy features.
    w = [0.0, 0.0, 0.0]
    for _ in range(epochs):
        grads = [0.0, 0.0, 0.0]
        for p in pairs:
            ch = dot(w, p.chosen)
            rj = dot(w, p.rejected)
            prob = sigmoid(ch - rj)
            coeff = prob - 1.0  # derivative of -log(sigmoid(ch-rj))
            for i in range(3):
                grads[i] += coeff * (p.chosen[i] - p.rejected[i])
        n = max(len(pairs), 1)
        for i in range(3):
            w[i] -= lr * grads[i] / n
    return w


def ppo_style_update(
    policy_w: List[float],
    old_policy_w: List[float],
    reward_w: List[float],
    samples: List[List[float]],
    clip_eps: float = 0.2,
    kl_coef: float = 0.05,
    lr: float = 0.1,
) -> float:
    """Toy PPO-style objective on score differences."""
    total_obj = 0.0
    grads = [0.0, 0.0, 0.0]

    for feat in samples:
        old_logit = dot(old_policy_w, feat)
        new_logit = dot(policy_w, feat)
        adv = dot(reward_w, feat)

        # Use exp(logit diff) as toy importance ratio.
        ratio = math.exp(max(min(new_logit - old_logit, 10), -10))
        clipped = min(max(ratio, 1.0 - clip_eps), 1.0 + clip_eps)
        surrogate = min(ratio * adv, clipped * adv)

        # KL-like penalty in score space.
        kl = 0.5 * (new_logit - old_logit) ** 2
        obj = surrogate - kl_coef * kl
        total_obj += obj

        # Very rough gradient approximation for teaching only.
        grad_coeff = (1.0 if ratio * adv <= clipped * adv else 0.0) * ratio * adv
        grad_coeff -= kl_coef * (new_logit - old_logit)
        for i in range(3):
            grads[i] += grad_coeff * feat[i]

    n = max(len(samples), 1)
    for i in range(3):
        policy_w[i] += lr * grads[i] / n
    return total_obj / n


def average_reward(policy_w: List[float], reward_w: List[float], samples: List[List[float]]) -> float:
    return sum(dot(reward_w, s) + 0.1 * dot(policy_w, s) for s in samples) / max(len(samples), 1)


def main() -> None:
    pairs = build_pairs()
    reward_w = train_reward_model(pairs)

    # Start from SFT-like policy.
    policy_w = [0.3, 0.2, -0.1]
    old_policy_w = list(policy_w)

    # "On-policy" toy samples (feature vectors of generated responses).
    samples = [
        [0.9, 0.7, 0.2],
        [0.8, 0.6, 0.3],
        [0.7, 0.8, 0.2],
        [0.6, 0.5, 0.4],
    ]

    print("=" * 90)
    print("Step 3: RLHF Intuition (Toy)")
    print("=" * 90)
    print(f"Reward model weights: {reward_w}")
    print(f"Initial policy weights: {policy_w}")

    for step in range(1, 11):
        obj = ppo_style_update(policy_w, old_policy_w, reward_w, samples)
        rew = average_reward(policy_w, reward_w, samples)
        print(f"step={step:02d} obj={obj:.4f} avg_reward={rew:.4f} policy_w={policy_w}")
        old_policy_w = list(policy_w)

    print("\nKey lesson:")
    print("RLHF can improve reward-aligned behavior, but stability constraints (KL/clipping) are essential.")


if __name__ == "__main__":
    main()

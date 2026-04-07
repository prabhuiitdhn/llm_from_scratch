"""
Attention at a high level (from intuition to code).

This script implements a tiny, dependency-free self-attention demo.
It is intentionally small and readable for learning:
1) Build token embeddings
2) Create Q, K, V projections
3) Compute attention scores with dot products
4) Apply softmax to get attention weights
5) Compute weighted sum of V to get final token representations
"""

from __future__ import annotations

import math
from typing import Dict, List


Vector = List[float]
Matrix = List[List[float]]


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def matmul_vec(vec: Vector, mat: Matrix) -> Vector:
    """Multiply vector (1 x d) by matrix (d x d_out)."""
    out_dim = len(mat[0])
    in_dim = len(vec)
    out = [0.0] * out_dim
    for j in range(out_dim):
        s = 0.0
        for i in range(in_dim):
            s += vec[i] * mat[i][j]
        out[j] = s
    return out


def softmax(values: Vector) -> Vector:
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps)
    return [x / total for x in exps]


def pretty(v: Vector, decimals: int = 3) -> str:
    return "[" + ", ".join(f"{x:.{decimals}f}" for x in v) + "]"


def self_attention(token_vectors: Matrix, w_q: Matrix, w_k: Matrix, w_v: Matrix) -> Dict[str, List]:
    """
    Single-head self-attention (no masking) for educational use.

    token_vectors: sequence_length x d_model
    returns intermediate tensors for explanation
    """
    d_k = len(w_k[0])

    # 1) Linear projections to get Q, K, V
    q = [matmul_vec(t, w_q) for t in token_vectors]
    k = [matmul_vec(t, w_k) for t in token_vectors]
    v = [matmul_vec(t, w_v) for t in token_vectors]

    # 2) Score matrix: score[i][j] = q_i · k_j / sqrt(d_k)
    scores: Matrix = []
    scale = math.sqrt(d_k)
    for i in range(len(q)):
        row = []
        for j in range(len(k)):
            row.append(dot(q[i], k[j]) / scale)
        scores.append(row)

    # 3) Softmax across each row to get attention weights
    weights = [softmax(row) for row in scores]

    # 4) Weighted sum of V for each query position
    outputs: Matrix = []
    for i in range(len(weights)):
        out = [0.0] * len(v[0])
        for j, w in enumerate(weights[i]):
            for d in range(len(out)):
                out[d] += w * v[j][d]
        outputs.append(out)

    return {
        "Q": q,
        "K": k,
        "V": v,
        "scores": scores,
        "weights": weights,
        "outputs": outputs,
    }


def run_demo() -> None:
    # Tiny sentence: each row is a one-hot encoded token (vocab_size=3).
    tokens = ["I", "love", "NLP"]
    x: Matrix = [
        [1.0, 0.0, 0.0],  # I (one-hot: position 0)
        [0.0, 1.0, 0.0],  # love (one-hot: position 1)
        [0.0, 0.0, 1.0],  # NLP (one-hot: position 2)
    ]

    # Projection matrices transform one-hot (d_embed=3) -> learned embeddings (d_attn=4).
    # These matrices are like learned embedding lookup tables:
    # - w_q, w_k, w_v each map the 3 token indices to 4-dimensional learned vectors.
    w_q: Matrix = [
        [0.8, 0.1, 0.0, 0.1],    # learned query features for each embedding dim
        [0.2, 0.7, 0.1, 0.0],
        [0.1, 0.2, 0.8, 0.1],
    ]
    w_k: Matrix = [
        [0.7, 0.2, 0.1, 0.0],    # learned key features for each embedding dim
        [0.1, 0.8, 0.0, 0.1],
        [0.2, 0.1, 0.7, 0.0],
    ]
    w_v: Matrix = [
        [1.0, 0.0, 0.2, 0.0],    # learned value features for each embedding dim
        [0.0, 1.0, 0.1, 0.0],
        [0.1, 0.2, 1.0, 0.1],
    ]

    out = self_attention(x, w_q, w_k, w_v)

    print("=" * 78)
    print("Attention At A High Level")
    print("Tokens:", tokens)

    print("\nStep 1) Q, K, V projections")
    for i, tok in enumerate(tokens):
        print(f"  {tok:>4}  Q={pretty(out['Q'][i])}  K={pretty(out['K'][i])}  V={pretty(out['V'][i])}")

    print("\nStep 2) Attention scores (similarity of each query with all keys)")
    for i, tok in enumerate(tokens):
        print(f"  scores for query '{tok}': {pretty(out['scores'][i])}")

    print("\nStep 3) Attention weights (softmax of scores)")
    for i, tok in enumerate(tokens):
        print(f"  weights for query '{tok}': {pretty(out['weights'][i])}")

    print("\nStep 4) Final output vectors (weighted sum of V)")
    for i, tok in enumerate(tokens):
        print(f"  output for '{tok}': {pretty(out['outputs'][i])}")

    print("\nInterpretation:")
    print("INPUT: One-hot encoded tokens (vocab_size=3 dimensions)")
    print("  - Each token is just an ID: [1,0,0], [0,1,0], [0,0,1]")
    print("  - No semantic meaning yet - just indices")
    print()
    print("PROJECTION: One-hot × w_q → Q (learned 4-dim query embedding)")
    print("  - w_q is a 3×4 matrix: each row = learned query vector for that token ID")
    print("  - Multiplying one-hot by this matrix = 'look up' that token's learned features")
    print("  - Each of the 4 output dimensions = one 'embedding feature' capturing semantic info")
    print()
    print("ATTENTION: Q·K scores + softmax → weights")
    print("  - Each token's Query asks 'which keys am I similar to?'")
    print("  - Softmax converts scores → attention weights (probabilities summing to 1)")
    print()
    print("OUTPUT: Weighted sum of V values")
    print("  - Final representation = mix of all tokens, weighted by attention")
    print("  - Each of 4 dimensions in output = one feature dimension (semantic blend)")


if __name__ == "__main__":
    run_demo()

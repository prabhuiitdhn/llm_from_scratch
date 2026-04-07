"""
Transformer basics in detail (beginner-friendly, dependency-free).

What this file teaches:
1) How text becomes token IDs (tokenization + vocabulary)
2) How token IDs become vectors (embeddings)
3) Why positional encoding is needed
4) How causal self-attention works (Q, K, V, masking, softmax)
5) How feed-forward, residual connections, and layer norm fit together
6) How final logits are produced for next-token prediction

This is a small educational implementation. It prioritizes readability over speed.
"""

from __future__ import annotations

import math
import random
import re
from typing import Dict, List, Tuple


Vector = List[float]
Matrix = List[List[float]]
Tensor3D = List[List[List[float]]]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def shape_2d(x: Matrix) -> Tuple[int, int]:
    return (len(x), len(x[0]) if x else 0)


def shape_3d(x: Tensor3D) -> Tuple[int, int, int]:
    return (len(x), len(x[0]) if x else 0, len(x[0][0]) if x and x[0] else 0)


def dot(a: Vector, b: Vector) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def matmul_vec(vec: Vector, mat: Matrix) -> Vector:
    """(1 x d_in) @ (d_in x d_out) -> (1 x d_out)."""
    d_out = len(mat[0])
    out = [0.0] * d_out
    for j in range(d_out):
        s = 0.0
        for i in range(len(vec)):
            s += vec[i] * mat[i][j]
        out[j] = s
    return out


def add_vec(a: Vector, b: Vector) -> Vector:
    return [x + y for x, y in zip(a, b)]


def softmax(values: Vector) -> Vector:
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def layer_norm_row(x: Vector, eps: float = 1e-5) -> Vector:
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    std = math.sqrt(var + eps)
    return [(v - mean) / std for v in x]


def relu_row(x: Vector) -> Vector:
    return [v if v > 0.0 else 0.0 for v in x]


def pretty(v: Vector, decimals: int = 3) -> str:
    return "[" + ", ".join(f"{x:.{decimals}f}" for x in v) + "]"


class BasicTokenizer:
    """Very small tokenizer: lowercase + split words and punctuation."""

    def __init__(self) -> None:
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    def normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def split(self, text: str) -> List[str]:
        # Keep punctuation as separate tokens for clarity.
        return re.findall(r"\w+|[^\w\s]", text)

    def fit(self, corpus: List[str]) -> None:
        counts: Dict[str, int] = {}
        for line in corpus:
            for tok in self.split(self.normalize(line)):
                counts[tok] = counts.get(tok, 0) + 1

        vocab = self.special_tokens + sorted(counts.keys())
        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def encode(self, text: str, max_len: int) -> List[int]:
        tokens = self.split(self.normalize(text))
        ids = [self.token_to_id["<bos>"]]
        for tok in tokens:
            ids.append(self.token_to_id.get(tok, self.token_to_id["<unk>"]))
        ids.append(self.token_to_id["<eos>"])

        pad_id = self.token_to_id["<pad>"]
        if len(ids) < max_len:
            ids = ids + [pad_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
            ids[-1] = self.token_to_id["<eos>"]
        return ids

    def decode(self, ids: List[int]) -> str:
        toks = [self.id_to_token.get(i, "<unk>") for i in ids]
        return " ".join(toks)


def build_positional_encoding(max_len: int, d_model: int) -> Matrix:
    """Sinusoidal positional encoding from the original Transformer paper."""
    pe = zeros(max_len, d_model)
    for pos in range(max_len):
        for i in range(d_model):
            denom = 10000 ** ((2 * (i // 2)) / d_model)
            angle = pos / denom
            if i % 2 == 0:
                pe[pos][i] = math.sin(angle)
            else:
                pe[pos][i] = math.cos(angle)
    return pe


def make_causal_mask(seq_len: int) -> Matrix:
    """
    Causal mask for decoder-style self-attention.
    mask[i][j] = 0.0      if j <= i   (allowed)
    mask[i][j] = -1e9     if j > i    (blocked future token)
    """
    m = zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            m[i][j] = 0.0 if j <= i else -1e9
    return m


class MultiHeadSelfAttention:
    """Tiny multi-head self-attention."""

    def __init__(self, d_model: int, num_heads: int, seed: int = 42) -> None:
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        rnd = random.Random(seed)
        scale = 0.2

        # (d_model x d_model)
        self.w_q = [[rnd.uniform(-scale, scale) for _ in range(d_model)] for _ in range(d_model)]
        self.w_k = [[rnd.uniform(-scale, scale) for _ in range(d_model)] for _ in range(d_model)]
        self.w_v = [[rnd.uniform(-scale, scale) for _ in range(d_model)] for _ in range(d_model)]
        self.w_o = [[rnd.uniform(-scale, scale) for _ in range(d_model)] for _ in range(d_model)]

    def _split_heads(self, x: Matrix) -> Tensor3D:
        # x: (seq_len, d_model) -> (num_heads, seq_len, d_head)
        seq_len = len(x)
        heads: Tensor3D = []
        for h in range(self.num_heads):
            start = h * self.d_head
            end = start + self.d_head
            head = [row[start:end] for row in x]
            heads.append(head)
        return heads

    def _combine_heads(self, heads: Tensor3D) -> Matrix:
        # (num_heads, seq_len, d_head) -> (seq_len, d_model)
        seq_len = len(heads[0])
        out = zeros(seq_len, self.d_model)
        for t in range(seq_len):
            row: Vector = []
            for h in range(self.num_heads):
                row.extend(heads[h][t])
            out[t] = row
        return out

    def forward(self, x: Matrix, causal_mask: Matrix) -> Tuple[Matrix, Tensor3D]:
        # Project to Q, K, V in full d_model space.
        q_full = [matmul_vec(row, self.w_q) for row in x]
        k_full = [matmul_vec(row, self.w_k) for row in x]
        v_full = [matmul_vec(row, self.w_v) for row in x]

        q_heads = self._split_heads(q_full)
        k_heads = self._split_heads(k_full)
        v_heads = self._split_heads(v_full)

        all_weights: Tensor3D = []
        out_heads: Tensor3D = []

        # Per-head scaled dot-product attention.
        for h in range(self.num_heads):
            q = q_heads[h]
            k = k_heads[h]
            v = v_heads[h]
            seq_len = len(q)

            weights_h = zeros(seq_len, seq_len)
            out_h = zeros(seq_len, self.d_head)
            scale = math.sqrt(self.d_head)

            for i in range(seq_len):
                scores = [0.0] * seq_len
                for j in range(seq_len):
                    scores[j] = (dot(q[i], k[j]) / scale) + causal_mask[i][j]

                w = softmax(scores)
                weights_h[i] = w

                # Weighted sum over values.
                for j in range(seq_len):
                    for d in range(self.d_head):
                        out_h[i][d] += w[j] * v[j][d]

            all_weights.append(weights_h)
            out_heads.append(out_h)

        # Concatenate heads, then final output projection.
        joined = self._combine_heads(out_heads)
        out = [matmul_vec(row, self.w_o) for row in joined]
        return out, all_weights


class FeedForward:
    """Position-wise FFN: Linear -> ReLU -> Linear."""

    def __init__(self, d_model: int, d_ff: int, seed: int = 123) -> None:
        rnd = random.Random(seed)
        scale = 0.2
        self.w1 = [[rnd.uniform(-scale, scale) for _ in range(d_ff)] for _ in range(d_model)]
        self.w2 = [[rnd.uniform(-scale, scale) for _ in range(d_model)] for _ in range(d_ff)]

    def forward(self, x: Matrix) -> Matrix:
        out = []
        for row in x:
            hidden = relu_row(matmul_vec(row, self.w1))
            out.append(matmul_vec(hidden, self.w2))
        return out


class TransformerBlock:
    """
    One decoder-style block:
    1) Multi-head self-attention + residual + layer norm
    2) Feed-forward + residual + layer norm
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)

    def forward(self, x: Matrix, causal_mask: Matrix) -> Tuple[Matrix, Tensor3D]:
        attn_out, weights = self.attn.forward(x, causal_mask)
        x1 = [layer_norm_row(add_vec(x[i], attn_out[i])) for i in range(len(x))]

        ffn_out = self.ffn.forward(x1)
        x2 = [layer_norm_row(add_vec(x1[i], ffn_out[i])) for i in range(len(x1))]
        return x2, weights


class TinyTransformerLM:
    """A tiny decoder-only Transformer for next-token logits."""

    def __init__(self, vocab_size: int, max_len: int, d_model: int, num_heads: int, d_ff: int) -> None:
        rnd = random.Random(7)
        scale = 0.2

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model

        # Token embedding table: (vocab_size x d_model)
        self.token_embedding = [[rnd.uniform(-scale, scale) for _ in range(d_model)] for _ in range(vocab_size)]
        self.positional_encoding = build_positional_encoding(max_len=max_len, d_model=d_model)

        self.block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)

        # Final projection to logits: (d_model x vocab_size) i.e. out of vocabulary length the softmax attention score of particular word will shown in uniform distribution [0-1]
        self.lm_head = [[rnd.uniform(-scale, scale) for _ in range(vocab_size)] for _ in range(d_model)]

    def embed(self, input_ids: List[int]) -> Matrix:
        seq_len = len(input_ids)
        x = zeros(seq_len, self.d_model)
        for t, token_id in enumerate(input_ids):
            tok_vec = self.token_embedding[token_id]
            pos_vec = self.positional_encoding[t]
            x[t] = add_vec(tok_vec, pos_vec)
        return x

    def forward(self, input_ids: List[int]) -> Dict[str, object]:
        seq_len = len(input_ids)
        causal_mask = make_causal_mask(seq_len) #

        x_embed = self.embed(input_ids)
        x_block, attn_weights = self.block.forward(x_embed, causal_mask)
        logits = [matmul_vec(row, self.lm_head) for row in x_block]

        return {
            "x_embed": x_embed,
            "attn_weights": attn_weights,
            "x_block": x_block,
            "logits": logits,
        }


def argmax(values: Vector) -> int:
    best_i = 0
    best_v = values[0]
    for i, v in enumerate(values):
        if v > best_v:
            best_v = v
            best_i = i
    return best_i


def build_sample_corpus() -> List[str]:
    return [
        "I love NLP",
        "I love machine learning",
        "transformers use attention",
        "attention helps models focus",
        "NLP is fun",
    ]


def run_demo() -> None:
    print("=" * 88)
    print("Transformer Basic Demo (Beginner-Friendly)")
    print("=" * 88)

    # 1) Build corpus and tokenizer.
    corpus = build_sample_corpus()
    tokenizer = BasicTokenizer()
    tokenizer.fit(corpus)

    print("\nSample Corpus:")
    for i, line in enumerate(corpus, start=1):
        print(f"  {i}. {line}")

    print("\nVocabulary (token -> id):")
    for tok, idx in tokenizer.token_to_id.items():
        print(f"  {tok:>12} -> {idx}")

    # 2) Pick one sentence and encode.
    text = "I love attention"
    max_len = 8
    input_ids = tokenizer.encode(text, max_len=max_len)

    print("\nInput text:", text)
    print("Encoded ids:", input_ids)
    print("Decoded tokens:", tokenizer.decode(input_ids))

    # 3) Build tiny Transformer.
    model = TinyTransformerLM(
        vocab_size=len(tokenizer.token_to_id), # vocab size from tokenizer
        max_len=max_len, # maximum sequence length for positional encoding
        d_model=8, # dimension of model embeddings
        num_heads=2, # number of attention heads
        d_ff=16, # dimension of layers in feed-forward network
    )

    # 4) Forward pass.
    out = model.forward(input_ids)

    x_embed = out["x_embed"]
    attn_weights = out["attn_weights"]
    x_block = out["x_block"]
    logits = out["logits"]

    print("\nShape trace:")
    print(f"  x_embed shape: {shape_2d(x_embed)} (seq_len x d_model)")
    print(f"  attn_weights shape: {shape_3d(attn_weights)} (heads x seq_len x seq_len)")
    print(f"  x_block shape: {shape_2d(x_block)} (seq_len x d_model)")
    print(f"  logits shape: {shape_2d(logits)} (seq_len x vocab_size)")

    print("\nFirst 2 embedding vectors (token + position):")
    for i in range(min(2, len(x_embed))):
        tok = tokenizer.id_to_token[input_ids[i]]
        print(f"  t={i:>2} token='{tok:>10}' -> {pretty(x_embed[i], 4)}")

    print("\nHead-0 attention weights (each row sums to 1):")
    h0 = attn_weights[0]
    for i, row in enumerate(h0):
        print(f"  query position {i}: {pretty(row, 3)}")

    print("\nNext-token prediction at each position (argmax over logits):")
    for i, row in enumerate(logits):
        pred_id = argmax(row)
        pred_tok = tokenizer.id_to_token[pred_id]
        cur_tok = tokenizer.id_to_token[input_ids[i]]
        print(f"  position {i}: current='{cur_tok}' -> predicted next token='{pred_tok}'")

    print("\nImportant note:")
    print("  This script runs a FORWARD pass only.")
    print("  We did not train weights, so predictions are random-ish.")
    print("  In real training, backprop updates these weights to make predictions meaningful.")


if __name__ == "__main__":
    run_demo()

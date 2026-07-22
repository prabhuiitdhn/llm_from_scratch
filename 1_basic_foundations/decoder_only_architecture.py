"""
Decoder-only Transformer architecture - full mathematical + dimensional walkthrough.

This assembles everything from the other files in this folder into ONE
complete, STACKED, multi-layer decoder-only Transformer - the architecture
family behind GPT-2/GPT-3/Llama/Mistral style models:

    advanced_tokenisation.py / tokenization.py -> text to token_ids
    embedding_table.py                         -> token_ids to embedding vectors
    attention_at_highlevel.py                  -> single-head attention intuition
    transformer_basic.py                       -> ONE decoder block (post-norm)

    THIS FILE                                  -> N stacked decoder blocks
                                                   (GPT-2 style PRE-norm),
                                                   with an explicit shape +
                                                   parameter-count trace at
                                                   every stage.

Architecture per layer (GPT-2 style "pre-norm", NOT the post-norm used in
transformer_basic.py):

    x1 = x  + MultiHeadCausalSelfAttention(LayerNorm(x))
    x2 = x1 + FeedForward(LayerNorm(x1))

Stacked `num_layers` times, followed by one final LayerNorm and an LM head
that projects back to vocabulary logits (softmax -> next-token probabilities).

Dependency-free (pure Python) so every matrix multiply and every parameter
is visible and countable by hand - no torch/numpy black boxes.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

Vector = List[float]
Matrix = List[List[float]]
Tensor3D = List[List[List[float]]]


# ---------------------------------------------------------------------------
# Small math helpers (same conventions as attention_at_highlevel.py / transformer_basic.py)
# ---------------------------------------------------------------------------

def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


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


def transpose(mat: Matrix) -> Matrix:
    rows, cols = len(mat), len(mat[0])
    return [[mat[r][c] for r in range(rows)] for c in range(cols)]


def softmax(values: Vector) -> Vector:
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def gelu(x: float) -> float:
    """GPT-2 uses GELU (not ReLU) in its feed-forward network."""
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def layer_norm_row(x: Vector, eps: float = 1e-5) -> Vector:
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    std = math.sqrt(var + eps)
    return [(v - mean) / std for v in x]


def pretty(v: Vector, decimals: int = 3) -> str:
    return "[" + ", ".join(f"{x:.{decimals}f}" for x in v) + "]"


def shape_2d(x: Matrix) -> Tuple[int, int]:
    return (len(x), len(x[0]) if x else 0)


def shape_3d(x: Tensor3D) -> Tuple[int, int, int]:
    return (len(x), len(x[0]) if x else 0, len(x[0][0]) if x and x[0] else 0)


def init_matrix(rows: int, cols: int, rng: random.Random, std: float = 0.02) -> Matrix:
    """GPT-2 initializes weights from N(0, 0.02) - small so training starts stable."""
    return [[rng.gauss(0.0, std) for _ in range(cols)] for _ in range(rows)]


def init_vector(size: int, value: float = 0.0) -> Vector:
    return [value] * size


def linear(x: Vector, w: Matrix, b: Vector) -> Vector:
    return add_vec(matmul_vec(x, w), b)


# ---------------------------------------------------------------------------
# Config: this dataclass IS "the size" - every architectural knob lives here
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int    # |V| - number of distinct tokens the model knows
    max_len: int       # a.k.a context window / block_size - max tokens seen at once
    d_model: int       # a.k.a n_embd/hidden size - width of every token vector
    num_heads: int     # a.k.a n_head - number of parallel attention "views"
    num_layers: int    # a.k.a n_layer - number of stacked decoder blocks
    d_ff: int          # feed-forward hidden width, GPT-2 uses 4 * d_model
    d_head: int = field(init=False)

    def __post_init__(self) -> None:
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.d_head = self.d_model // self.num_heads

    def describe(self) -> str:
        return (
            f"vocab_size={self.vocab_size}  max_len={self.max_len}  d_model={self.d_model}  "
            f"num_heads={self.num_heads} (d_head={self.d_head})  num_layers={self.num_layers}  "
            f"d_ff={self.d_ff}"
        )


# ---------------------------------------------------------------------------
# Tokenizer (minimal word-level tokenizer, same style as transformer_basic.py)
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    def __init__(self) -> None:
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    def normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    def split(self, text: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", text)

    def fit(self, corpus: List[str]) -> None:
        vocab = set()
        for line in corpus:
            vocab.update(self.split(self.normalize(line)))
        ordered = self.special_tokens + sorted(vocab)
        self.token_to_id = {tok: i for i, tok in enumerate(ordered)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def encode(self, text: str, max_len: int) -> List[int]:
        tokens = self.split(self.normalize(text))
        ids = [self.token_to_id["<bos>"]]
        for tok in tokens:
            ids.append(self.token_to_id.get(tok, self.token_to_id["<unk>"]))
        ids.append(self.token_to_id["<eos>"])
        ids = ids[:max_len] if len(ids) > max_len else ids
        return ids


# ---------------------------------------------------------------------------
# Positional embedding + causal mask
# ---------------------------------------------------------------------------

def make_causal_mask(seq_len: int) -> Matrix:
    """
    mask[i][j] = 0.0   if j <= i  (token i CAN see token j - past/self)
    mask[i][j] = -1e9  if j > i   (token i CANNOT see token j - future, blocked)
    This is THE defining trait of a "decoder-only" model: each position only
    ever attends to itself and earlier positions.
    """
    m = [[0.0] * seq_len for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(seq_len):
            m[i][j] = 0.0 if j <= i else -1e9
    return m


# ---------------------------------------------------------------------------
# Multi-head causal self-attention
# ---------------------------------------------------------------------------

class MultiHeadCausalSelfAttention:
    """
    Shapes (seq_len = L, d_model = D, num_heads = H, d_head = D/H):
        x           : (L, D)
        w_q/w_k/w_v : (D, D)   -> project full D, then split into H heads of D/H
        w_o         : (D, D)   -> merges concatenated heads back to D
        scores      : (H, L, L)
        weights     : (H, L, L)  (softmax over last axis, causal-masked)
        output      : (L, D)
    """

    def __init__(self, cfg: GPTConfig, seed: int) -> None:
        rng = random.Random(seed)
        d = cfg.d_model
        self.cfg = cfg
        self.w_q, self.b_q = init_matrix(d, d, rng), init_vector(d)
        self.w_k, self.b_k = init_matrix(d, d, rng), init_vector(d)
        self.w_v, self.b_v = init_matrix(d, d, rng), init_vector(d)
        self.w_o, self.b_o = init_matrix(d, d, rng), init_vector(d)

    def num_params(self) -> int:
        d = self.cfg.d_model
        # 4 linear layers (Q, K, V, O), each weight (d x d) + bias (d)
        return 4 * (d * d + d)

    def _split_heads(self, x: Matrix) -> Tensor3D:
        # (L, D) -> (H, L, d_head): slice columns [h*d_head : (h+1)*d_head]
        heads: Tensor3D = []
        for h in range(self.cfg.num_heads):
            start, end = h * self.cfg.d_head, (h + 1) * self.cfg.d_head
            heads.append([row[start:end] for row in x])
        return heads

    def _combine_heads(self, heads: Tensor3D) -> Matrix:
        # (H, L, d_head) -> (L, D): concatenate all heads back along the feature axis
        seq_len = len(heads[0])
        out: Matrix = []
        for t in range(seq_len):
            row: Vector = []
            for h in range(self.cfg.num_heads):
                row.extend(heads[h][t])
            out.append(row)
        return out

    def forward(self, x: Matrix, causal_mask: Matrix) -> Tuple[Matrix, Tensor3D]:
        q_full = [linear(row, self.w_q, self.b_q) for row in x]
        k_full = [linear(row, self.w_k, self.b_k) for row in x]
        v_full = [linear(row, self.w_v, self.b_v) for row in x]

        q_heads = self._split_heads(q_full)
        k_heads = self._split_heads(k_full)
        v_heads = self._split_heads(v_full)

        all_weights: Tensor3D = []
        out_heads: Tensor3D = []
        scale = math.sqrt(self.cfg.d_head)

        for h in range(self.cfg.num_heads):
            q, k, v = q_heads[h], k_heads[h], v_heads[h]
            seq_len = len(q)
            weights_h = [[0.0] * seq_len for _ in range(seq_len)]
            out_h = [[0.0] * self.cfg.d_head for _ in range(seq_len)]

            for i in range(seq_len):
                scores = [(dot(q[i], k[j]) / scale) + causal_mask[i][j] for j in range(seq_len)]
                w = softmax(scores)
                weights_h[i] = w
                for j in range(seq_len):
                    for d in range(self.cfg.d_head):
                        out_h[i][d] += w[j] * v[j][d]

            all_weights.append(weights_h)
            out_heads.append(out_h)

        joined = self._combine_heads(out_heads)
        out = [linear(row, self.w_o, self.b_o) for row in joined]
        return out, all_weights


# ---------------------------------------------------------------------------
# Position-wise feed-forward network
# ---------------------------------------------------------------------------

class FeedForward:
    """
    Shapes:
        x  : (L, D)
        w1 : (D, d_ff)   b1 : (d_ff)
        w2 : (d_ff, D)   b2 : (D)
    Every token is transformed independently: D -> d_ff -> D (GELU in between).
    This is where most of a transformer's parameters live (d_ff is usually 4xD).
    """

    def __init__(self, cfg: GPTConfig, seed: int) -> None:
        rng = random.Random(seed)
        self.cfg = cfg
        self.w1, self.b1 = init_matrix(cfg.d_model, cfg.d_ff, rng), init_vector(cfg.d_ff)
        self.w2, self.b2 = init_matrix(cfg.d_ff, cfg.d_model, rng), init_vector(cfg.d_model)

    def num_params(self) -> int:
        d, d_ff = self.cfg.d_model, self.cfg.d_ff
        return (d * d_ff + d_ff) + (d_ff * d + d)

    def forward(self, x: Matrix) -> Matrix:
        out = []
        for row in x:
            hidden = [gelu(v) for v in linear(row, self.w1, self.b1)]
            out.append(linear(hidden, self.w2, self.b2))
        return out


# ---------------------------------------------------------------------------
# Layer normalization (with learnable gamma/beta - real LayerNorm has params!)
# ---------------------------------------------------------------------------

class LayerNorm:
    def __init__(self, d_model: int) -> None:
        self.d_model = d_model
        self.gamma = init_vector(d_model, 1.0)  # learnable scale, init to 1
        self.beta = init_vector(d_model, 0.0)   # learnable shift, init to 0

    def num_params(self) -> int:
        return 2 * self.d_model  # gamma + beta

    def forward(self, x: Matrix) -> Matrix:
        out = []
        for row in x:
            normed = layer_norm_row(row)
            out.append([normed[i] * self.gamma[i] + self.beta[i] for i in range(self.d_model)])
        return out


# ---------------------------------------------------------------------------
# One decoder block (GPT-2 style PRE-norm)
# ---------------------------------------------------------------------------

class DecoderBlock:
    """
    x1 = x  + Attention(LayerNorm(x))
    x2 = x1 + FeedForward(LayerNorm(x1))

    Note: this is "pre-norm" (LayerNorm BEFORE the sub-layer), which is what
    GPT-2/GPT-3/Llama actually use. transformer_basic.py in this folder uses
    "post-norm" (LayerNorm AFTER the residual add), like the original 2017
    Transformer paper / BERT. Both are valid; pre-norm trains more stably at
    large depth, which is why modern decoder-only LLMs prefer it.
    """

    def __init__(self, cfg: GPTConfig, seed: int) -> None:
        self.ln1 = LayerNorm(cfg.d_model)
        self.attn = MultiHeadCausalSelfAttention(cfg, seed=seed)
        self.ln2 = LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg, seed=seed + 1)

    def num_params(self) -> int:
        return self.ln1.num_params() + self.attn.num_params() + self.ln2.num_params() + self.ffn.num_params()

    def forward(self, x: Matrix, causal_mask: Matrix) -> Tuple[Matrix, Tensor3D]:
        attn_out, weights = self.attn.forward(self.ln1.forward(x), causal_mask)
        x1 = [add_vec(x[i], attn_out[i]) for i in range(len(x))]

        ffn_out = self.ffn.forward(self.ln2.forward(x1))
        x2 = [add_vec(x1[i], ffn_out[i]) for i in range(len(x1))]
        return x2, weights


# ---------------------------------------------------------------------------
# Full decoder-only Transformer (GPT-style language model)
# ---------------------------------------------------------------------------

class DecoderOnlyTransformer:
    """
    Full forward pipeline:
        token_ids (L)
          -> token_embedding lookup      (L, D)
          -> + position_embedding lookup (L, D)
          -> N x DecoderBlock            (L, D)  (each block keeps shape (L, D))
          -> final LayerNorm             (L, D)
          -> LM head (weight-tied)       (L, vocab_size)  <- logits
          -> softmax per row             (L, vocab_size)  <- next-token probabilities
    """

    def __init__(self, cfg: GPTConfig, tie_weights: bool = True) -> None:
        self.cfg = cfg
        self.tie_weights = tie_weights
        rng = random.Random(7)

        # Learned embeddings (GPT-2 style - NOT sinusoidal like transformer_basic.py)
        self.token_embedding = init_matrix(cfg.vocab_size, cfg.d_model, rng)   # (V, D)
        self.position_embedding = init_matrix(cfg.max_len, cfg.d_model, rng)  # (max_len, D)

        self.blocks = [DecoderBlock(cfg, seed=100 + i) for i in range(cfg.num_layers)]
        self.ln_f = LayerNorm(cfg.d_model)

        if not tie_weights:
            self.lm_head = init_matrix(cfg.d_model, cfg.vocab_size, rng)  # (D, V)

    def num_params(self) -> Dict[str, int]:
        d, v = self.cfg.d_model, self.cfg.vocab_size
        token_emb = v * d
        pos_emb = self.cfg.max_len * d
        blocks = sum(b.num_params() for b in self.blocks)
        ln_f = self.ln_f.num_params()
        lm_head = 0 if self.tie_weights else d * v  # weight tying reuses token_embedding^T
        total = token_emb + pos_emb + blocks + ln_f + lm_head
        return {
            "token_embedding": token_emb,
            "position_embedding": pos_emb,
            f"decoder_blocks (x{self.cfg.num_layers})": blocks,
            "final_layer_norm": ln_f,
            "lm_head (0 if tied)": lm_head,
            "TOTAL": total,
        }

    def embed(self, input_ids: List[int]) -> Matrix:
        return [
            add_vec(self.token_embedding[tok_id], self.position_embedding[t])
            for t, tok_id in enumerate(input_ids)
        ]

    def forward(self, input_ids: List[int]) -> Dict[str, object]:
        seq_len = len(input_ids)
        assert seq_len <= self.cfg.max_len, "sequence longer than max_len (context window)"
        causal_mask = make_causal_mask(seq_len)

        x = self.embed(input_ids)
        per_layer_shapes: List[Tuple[int, int]] = []
        all_attn_weights: List[Tensor3D] = []

        for block in self.blocks:
            x, weights = block.forward(x, causal_mask)
            per_layer_shapes.append(shape_2d(x))
            all_attn_weights.append(weights)

        x_final = self.ln_f.forward(x)

        if self.tie_weights:
            lm_head_t = transpose(self.token_embedding)  # (D, V) reused from (V, D)
            logits = [matmul_vec(row, lm_head_t) for row in x_final]
        else:
            logits = [matmul_vec(row, self.lm_head) for row in x_final]

        probs = [softmax(row) for row in logits]

        return {
            "x_embed": self.embed(input_ids),
            "per_layer_shapes": per_layer_shapes,
            "attn_weights": all_attn_weights,
            "x_final": x_final,
            "logits": logits,
            "probs": probs,
        }


# ---------------------------------------------------------------------------
# Closed-form parameter-count formula (matches the implementation exactly)
# ---------------------------------------------------------------------------

def theoretical_param_count(cfg: GPTConfig, tie_weights: bool = True) -> int:
    """
    The standard transformer parameter-count formula, term by term:
        token_embedding    = V * D
        position_embedding = max_len * D
        per_layer:
            attention   = 4 * (D*D + D)              (Wq,Wk,Wv,Wo + biases)
            feedforward = (D*d_ff + d_ff) + (d_ff*D + D)
            layernorms  = 2 * (2*D)                   (ln1 + ln2, each gamma+beta)
        final_layer_norm   = 2 * D
        lm_head            = 0 if tied else D * V
    """
    d, v, l, dff, mlen = cfg.d_model, cfg.vocab_size, cfg.num_layers, cfg.d_ff, cfg.max_len
    token_emb = v * d
    pos_emb = mlen * d
    per_layer_attn = 4 * (d * d + d)
    per_layer_ffn = (d * dff + dff) + (dff * d + d)
    per_layer_ln = 2 * (2 * d)
    per_layer = per_layer_attn + per_layer_ffn + per_layer_ln
    ln_f = 2 * d
    lm_head = 0 if tie_weights else d * v
    return token_emb + pos_emb + l * per_layer + ln_f + lm_head


def human(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def build_sample_corpus() -> List[str]:
    return [
        "I love NLP",
        "I love machine learning",
        "transformers use attention",
        "attention helps models focus",
        "decoder only models predict the next token",
    ]


def run_demo() -> None:
    print("=" * 88)
    print("Decoder-Only Transformer - Full Dimension + Parameter Walkthrough")
    print("=" * 88)

    # --- Step 1: text -> token_ids ---
    corpus = build_sample_corpus()
    tokenizer = SimpleTokenizer()
    tokenizer.fit(corpus)
    vocab_size = len(tokenizer.token_to_id)

    text = "I love attention"
    max_len = 8
    input_ids = tokenizer.encode(text, max_len=max_len)
    seq_len = len(input_ids)

    print(f"\nCorpus size: {len(corpus)} sentences | Vocab size (V): {vocab_size}")
    print(f"Input text: {text!r}")
    print(f"Token IDs (seq_len L={seq_len}): {input_ids}")

    # --- Step 2: toy-scale config ("the size") ---
    cfg = GPTConfig(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=16,     # D
        num_heads=4,    # H  -> d_head = D/H = 4
        num_layers=2,   # L
        d_ff=64,        # 4 x D, matches GPT-2's ratio
    )
    print(f"\nModel config: {cfg.describe()}")

    model = DecoderOnlyTransformer(cfg, tie_weights=True)
    out = model.forward(input_ids)

    # --- Step 3: full dimension trace ---
    print("\n" + "-" * 88)
    print("DIMENSION TRACE")
    print("-" * 88)
    print(f"token_embedding table   : ({cfg.vocab_size} x {cfg.d_model})   [V x D]")
    print(f"position_embedding table: ({cfg.max_len} x {cfg.d_model})   [max_len x D]")
    print(f"x_embed (after add)     : {shape_2d(out['x_embed'])}   [L x D]")
    for i, (shape, w) in enumerate(zip(out["per_layer_shapes"], out["attn_weights"])):
        print(f"  layer {i}: output {shape}  [L x D]   attn_weights {shape_3d(w)}  [H x L x L]")
    print(f"x_final (after ln_f)    : {shape_2d(out['x_final'])}   [L x D]")
    print(f"logits                  : {shape_2d(out['logits'])}   [L x V]")
    print(f"probs (softmax)         : {shape_2d(out['probs'])}   [L x V], each row sums to 1")

    print("\nWhy attn_weights is (H x L x L): every one of H heads independently")
    print("computes an L x L similarity matrix - 'how much should token i attend to token j'.")
    print("Because this is DECODER-ONLY (causal), weights[i][j] = 0 whenever j > i:")
    h0 = out["attn_weights"][0][0]  # layer 0, head 0
    for i, row in enumerate(h0):
        print(f"    layer0/head0 row {i}: {pretty(row)}")

    # --- Step 4: parameter count (actual, from the built model) ---
    print("\n" + "-" * 88)
    print("PARAMETER COUNT (toy model)")
    print("-" * 88)
    counts = model.num_params()
    for name, n in counts.items():
        print(f"  {name:<28} {n:>10,}")

    formula_total = theoretical_param_count(cfg, tie_weights=True)
    print(f"\nClosed-form formula total : {formula_total:,}")
    print(f"Actual model total         : {counts['TOTAL']:,}")
    print(f"Match: {formula_total == counts['TOTAL']}")

    # --- Step 5: scale up the SAME formula to real GPT-2 configs ---
    print("\n" + "-" * 88)
    print("SCALING THE SAME FORMULA UP TO REAL GPT-2 SIZES")
    print("-" * 88)
    real_configs = {
        "toy (this demo)": cfg,
        "GPT-2 small": GPTConfig(vocab_size=50257, max_len=1024, d_model=768, num_heads=12, num_layers=12, d_ff=3072),
        "GPT-2 medium": GPTConfig(vocab_size=50257, max_len=1024, d_model=1024, num_heads=16, num_layers=24, d_ff=4096),
        "GPT-2 large": GPTConfig(vocab_size=50257, max_len=1024, d_model=1280, num_heads=20, num_layers=36, d_ff=5120),
        "GPT-2 XL": GPTConfig(vocab_size=50257, max_len=1024, d_model=1600, num_heads=25, num_layers=48, d_ff=6400),
    }
    print(f"  {'model':<16} {'d_model':>8} {'heads':>7} {'layers':>7} {'d_ff':>6} {'params':>10}")
    for name, c in real_configs.items():
        n = theoretical_param_count(c, tie_weights=True)
        print(f"  {name:<16} {c.d_model:>8} {c.num_heads:>7} {c.num_layers:>7} {c.d_ff:>6} {human(n):>10}")

    print("\nPipeline summary:")
    print("  text -> SimpleTokenizer -> token_ids")
    print("  token_ids -> token_embedding[id] + position_embedding[pos] -> x (L x D)")
    print("  x -> N x [ LayerNorm -> MultiHeadCausalSelfAttention -> +residual")
    print("            -> LayerNorm -> FeedForward(GELU)          -> +residual ]")
    print("  -> final LayerNorm -> LM head (tied with token_embedding) -> logits (L x V)")
    print("  -> softmax(logits) -> next-token probability distribution per position")


if __name__ == "__main__":
    run_demo()

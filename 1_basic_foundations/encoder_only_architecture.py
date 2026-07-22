"""
Encoder-only Transformer architecture - full mathematical + dimensional walkthrough.

This is the direct counterpart to decoder_only_architecture.py in this same
folder, built for the SAME toy example ("I love attention"), but implementing
the OTHER major Transformer family: encoder-only (BERT-style) models.

    advanced_tokenisation.py / tokenization.py -> text to token_ids
    embedding_table.py                         -> token_ids to embedding vectors
    attention_at_highlevel.py                  -> single-head attention intuition
    decoder_only_architecture.py               -> GPT-style, CAUSAL, pre-norm

    THIS FILE                                  -> BERT-style, BIDIRECTIONAL,
                                                   post-norm, with a masked-
                                                   language-model (MLM) head
                                                   and a [CLS] pooler head.

Key differences from decoder_only_architecture.py:

1) NO causal mask. Every token can attend to EVERY other token (past AND
   future) - only padding positions are masked out. This is why encoders
   build rich, whole-sequence understanding rather than being restricted to
   "predict the next token".
2) POST-norm blocks (LayerNorm AFTER the residual add), matching the
   original 2017 Transformer encoder / BERT - NOT the pre-norm used by
   decoder_only_architecture.py's GPT-style blocks.
3) Adds a SEGMENT (token-type) embedding - BERT's mechanism for telling
   "sentence A" tokens apart from "sentence B" tokens (used for tasks like
   next-sentence-prediction / sentence-pair classification).
4) Two output heads instead of one LM head:
     - MLM head: per-position logits over the vocabulary (for masked-token
       prediction pretraining) - shape (L, V), same shape as GPT's logits.
     - Pooler: a single (D,) vector from the [CLS] token, used for
       whole-sequence classification tasks (sentiment, NSP, etc.).

Dependency-free (pure Python) so every matrix multiply and every parameter
is visible and countable by hand - no torch/numpy black boxes.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Vector = List[float]
Matrix = List[List[float]]
Tensor3D = List[List[List[float]]]


# ---------------------------------------------------------------------------
# Small math helpers (same conventions as decoder_only_architecture.py)
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
    """BERT (like GPT-2) uses GELU in its feed-forward network."""
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def tanh_vec(x: Vector) -> Vector:
    return [math.tanh(v) for v in x]


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
    """BERT/GPT-2 both initialize weights from N(0, 0.02)."""
    return [[rng.gauss(0.0, std) for _ in range(cols)] for _ in range(rows)]


def init_vector(size: int, value: float = 0.0) -> Vector:
    return [value] * size


def linear(x: Vector, w: Matrix, b: Vector) -> Vector:
    return add_vec(matmul_vec(x, w), b)


# ---------------------------------------------------------------------------
# Config: this dataclass IS "the size" - every architectural knob lives here
# ---------------------------------------------------------------------------

@dataclass
class BERTConfig:
    vocab_size: int         # |V| - number of distinct tokens the model knows
    max_len: int            # max sequence length (BERT calls this max_position_embeddings)
    d_model: int            # a.k.a hidden_size - width of every token vector
    num_heads: int          # number of parallel attention "views"
    num_layers: int         # number of stacked encoder blocks
    d_ff: int               # feed-forward hidden width, BERT uses 4 * d_model
    num_segment_types: int = 2  # BERT: sentence-A / sentence-B
    d_head: int = field(init=False)

    def __post_init__(self) -> None:
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.d_head = self.d_model // self.num_heads

    def describe(self) -> str:
        return (
            f"vocab_size={self.vocab_size}  max_len={self.max_len}  d_model={self.d_model}  "
            f"num_heads={self.num_heads} (d_head={self.d_head})  num_layers={self.num_layers}  "
            f"d_ff={self.d_ff}  num_segment_types={self.num_segment_types}"
        )


# ---------------------------------------------------------------------------
# Tokenizer (BERT-style special tokens: [CLS]/[SEP]/[PAD]/[MASK]/[UNK])
# ---------------------------------------------------------------------------

class SimpleBertTokenizer:
    def __init__(self) -> None:
        self.special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
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

    def encode(self, text: str, max_len: int) -> Tuple[List[int], List[int], List[int]]:
        """
        Returns (input_ids, segment_ids, attention_mask) for a SINGLE sentence:
            [CLS] tok1 tok2 ... [SEP]
        segment_ids are all 0 (sentence A) since there's only one sentence here.
        A real sentence-PAIR task ([CLS] A [SEP] B [SEP]) would set segment_ids
        to 1 for every token belonging to sentence B.
        """
        tokens = self.split(self.normalize(text))
        ids = [self.token_to_id["[CLS]"]]
        for tok in tokens:
            ids.append(self.token_to_id.get(tok, self.token_to_id["[UNK]"]))
        ids.append(self.token_to_id["[SEP]"])
        ids = ids[:max_len] if len(ids) > max_len else ids

        segment_ids = [0] * len(ids)
        attention_mask = [1] * len(ids)  # no padding in this simple demo
        return ids, segment_ids, attention_mask


# ---------------------------------------------------------------------------
# Bidirectional (non-causal) attention mask - padding only, no future-blocking
# ---------------------------------------------------------------------------

def make_padding_mask(attention_mask: List[int]) -> Matrix:
    """
    Encoder attention is BIDIRECTIONAL: token i may attend to ANY token j,
    including tokens that come AFTER it. The only thing ever blocked is
    padding. mask[i][j] = 0.0 if token j is real, -1e9 if token j is padding.
    Note this does NOT depend on i (broadcast across every query row) -
    contrast with decoder_only_architecture.make_causal_mask(), which blocks
    based on the (i, j) POSITION relationship, not just padding.
    """
    seq_len = len(attention_mask)
    col_bias = [0.0 if m == 1 else -1e9 for m in attention_mask]
    return [[col_bias[j] for j in range(seq_len)] for _ in range(seq_len)]


# ---------------------------------------------------------------------------
# Multi-head BIDIRECTIONAL self-attention (no causal restriction)
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention:
    """
    Shapes (seq_len = L, d_model = D, num_heads = H, d_head = D/H):
        x           : (L, D)
        w_q/w_k/w_v : (D, D)   -> project full D, then split into H heads of D/H
        w_o         : (D, D)   -> merges concatenated heads back to D
        scores      : (H, L, L)
        weights     : (H, L, L)  (softmax over last axis, padding-masked ONLY)
        output      : (L, D)
    """

    def __init__(self, cfg: BERTConfig, seed: int) -> None:
        rng = random.Random(seed)
        d = cfg.d_model
        self.cfg = cfg
        self.w_q, self.b_q = init_matrix(d, d, rng), init_vector(d)
        self.w_k, self.b_k = init_matrix(d, d, rng), init_vector(d)
        self.w_v, self.b_v = init_matrix(d, d, rng), init_vector(d)
        self.w_o, self.b_o = init_matrix(d, d, rng), init_vector(d)

    def num_params(self) -> int:
        d = self.cfg.d_model
        return 4 * (d * d + d)

    def _split_heads(self, x: Matrix) -> Tensor3D:
        heads: Tensor3D = []
        for h in range(self.cfg.num_heads):
            start, end = h * self.cfg.d_head, (h + 1) * self.cfg.d_head
            heads.append([row[start:end] for row in x])
        return heads

    def _combine_heads(self, heads: Tensor3D) -> Matrix:
        seq_len = len(heads[0])
        out: Matrix = []
        for t in range(seq_len):
            row: Vector = []
            for h in range(self.cfg.num_heads):
                row.extend(heads[h][t])
            out.append(row)
        return out

    def forward(self, x: Matrix, padding_mask: Matrix) -> Tuple[Matrix, Tensor3D]:
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
                # NO causal term added here - only the padding_mask (bidirectional).
                scores = [(dot(q[i], k[j]) / scale) + padding_mask[i][j] for j in range(seq_len)]
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
# Position-wise feed-forward network (identical shape logic to decoder file)
# ---------------------------------------------------------------------------

class FeedForward:
    def __init__(self, cfg: BERTConfig, seed: int) -> None:
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
# Layer normalization (learnable gamma/beta)
# ---------------------------------------------------------------------------

class LayerNorm:
    def __init__(self, d_model: int) -> None:
        self.d_model = d_model
        self.gamma = init_vector(d_model, 1.0)
        self.beta = init_vector(d_model, 0.0)

    def num_params(self) -> int:
        return 2 * self.d_model

    def forward(self, x: Matrix) -> Matrix:
        out = []
        for row in x:
            normed = layer_norm_row(row)
            out.append([normed[i] * self.gamma[i] + self.beta[i] for i in range(self.d_model)])
        return out


# ---------------------------------------------------------------------------
# One encoder block (BERT-style POST-norm)
# ---------------------------------------------------------------------------

class EncoderBlock:
    """
    x1 = LayerNorm(x  + Attention(x))
    x2 = LayerNorm(x1 + FeedForward(x1))

    Note: this is "post-norm" (LayerNorm AFTER the residual add), exactly
    like the original 2017 Transformer encoder and BERT. This is the OPPOSITE
    order from decoder_only_architecture.DecoderBlock, which applies
    LayerNorm BEFORE each sub-layer (GPT-2 style "pre-norm").
    """

    def __init__(self, cfg: BERTConfig, seed: int) -> None:
        self.attn = MultiHeadSelfAttention(cfg, seed=seed)
        self.ln1 = LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg, seed=seed + 1)
        self.ln2 = LayerNorm(cfg.d_model)

    def num_params(self) -> int:
        return self.attn.num_params() + self.ln1.num_params() + self.ffn.num_params() + self.ln2.num_params()

    def forward(self, x: Matrix, padding_mask: Matrix) -> Tuple[Matrix, Tensor3D]:
        attn_out, weights = self.attn.forward(x, padding_mask)
        x1 = self.ln1.forward([add_vec(x[i], attn_out[i]) for i in range(len(x))])

        ffn_out = self.ffn.forward(x1)
        x2 = self.ln2.forward([add_vec(x1[i], ffn_out[i]) for i in range(len(x1))])
        return x2, weights


# ---------------------------------------------------------------------------
# Full encoder-only Transformer (BERT-style model)
# ---------------------------------------------------------------------------

class EncoderOnlyTransformer:
    """
    Full forward pipeline:
        token_ids, segment_ids (L)
          -> token_embedding + position_embedding + segment_embedding  (L, D)
          -> embeddings LayerNorm                                     (L, D)
          -> N x EncoderBlock (bidirectional, post-norm)               (L, D)
          -> sequence_output                                          (L, D)
          -> MLM head (per position)     -> mlm_logits  (L, V)
          -> Pooler ([CLS] token only)   -> pooled_output (D,)
    """

    def __init__(self, cfg: BERTConfig) -> None:
        self.cfg = cfg
        rng = random.Random(7)
        d, v = cfg.d_model, cfg.vocab_size

        # Three embedding tables summed together (BERT's signature input representation).
        self.token_embedding = init_matrix(v, d, rng)                       # (V, D)
        self.position_embedding = init_matrix(cfg.max_len, d, rng)          # (max_len, D)
        self.segment_embedding = init_matrix(cfg.num_segment_types, d, rng)  # (2, D)
        self.embeddings_ln = LayerNorm(d)

        self.blocks = [EncoderBlock(cfg, seed=100 + i) for i in range(cfg.num_layers)]

        # MLM head: dense+GELU+LayerNorm "transform", then decode back to
        # vocab logits using the token_embedding matrix (weight tying) + bias.
        self.mlm_dense_w, self.mlm_dense_b = init_matrix(d, d, rng), init_vector(d)
        self.mlm_ln = LayerNorm(d)
        self.mlm_bias = init_vector(v)  # extra learnable output bias (BERT has this even when tied)

        # Pooler: dense+tanh on the [CLS] (position 0) representation, used
        # for whole-sequence classification tasks.
        self.pooler_w, self.pooler_b = init_matrix(d, d, rng), init_vector(d)

    def num_params(self) -> Dict[str, int]:
        d, v = self.cfg.d_model, self.cfg.vocab_size
        token_emb = v * d
        pos_emb = self.cfg.max_len * d
        seg_emb = self.cfg.num_segment_types * d
        emb_ln = self.embeddings_ln.num_params()
        blocks = sum(b.num_params() for b in self.blocks)
        mlm_transform = (d * d + d) + self.mlm_ln.num_params()
        mlm_output_bias = v  # decoder weight is tied (0 extra), but the bias is NOT tied
        pooler = d * d + d
        total = token_emb + pos_emb + seg_emb + emb_ln + blocks + mlm_transform + mlm_output_bias + pooler
        return {
            "token_embedding": token_emb,
            "position_embedding": pos_emb,
            "segment_embedding": seg_emb,
            "embeddings_layer_norm": emb_ln,
            f"encoder_blocks (x{self.cfg.num_layers})": blocks,
            "mlm_head (transform + tied decoder bias)": mlm_transform + mlm_output_bias,
            "pooler (dense + tanh, for [CLS])": pooler,
            "TOTAL": total,
        }

    def embed(self, input_ids: List[int], segment_ids: List[int]) -> Matrix:
        x = [
            add_vec(
                add_vec(self.token_embedding[tok_id], self.position_embedding[t]),
                self.segment_embedding[seg_id],
            )
            for t, (tok_id, seg_id) in enumerate(zip(input_ids, segment_ids))
        ]
        return self.embeddings_ln.forward(x)

    def forward(
        self,
        input_ids: List[int],
        segment_ids: Optional[List[int]] = None,
        attention_mask: Optional[List[int]] = None,
    ) -> Dict[str, object]:
        seq_len = len(input_ids)
        assert seq_len <= self.cfg.max_len, "sequence longer than max_len"
        segment_ids = segment_ids or [0] * seq_len
        attention_mask = attention_mask or [1] * seq_len
        padding_mask = make_padding_mask(attention_mask)

        x = self.embed(input_ids, segment_ids)
        per_layer_shapes: List[Tuple[int, int]] = []
        all_attn_weights: List[Tensor3D] = []

        for block in self.blocks:
            x, weights = block.forward(x, padding_mask)
            per_layer_shapes.append(shape_2d(x))
            all_attn_weights.append(weights)

        sequence_output = x  # (L, D) - contextualized representation of EVERY token

        # --- MLM head: predict the (possibly masked) token at every position ---
        mlm_hidden = [
            [gelu(v) for v in linear(row, self.mlm_dense_w, self.mlm_dense_b)]
            for row in sequence_output
        ]
        mlm_hidden = self.mlm_ln.forward(mlm_hidden)
        decoder_w = transpose(self.token_embedding)  # (D, V), tied with token_embedding
        mlm_logits = [add_vec(matmul_vec(row, decoder_w), self.mlm_bias) for row in mlm_hidden]

        # --- Pooler: classification-ready vector from the [CLS] token (position 0) ---
        cls_vector = sequence_output[0]
        pooled_output = tanh_vec(linear(cls_vector, self.pooler_w, self.pooler_b))

        return {
            "x_embed": self.embed(input_ids, segment_ids),
            "per_layer_shapes": per_layer_shapes,
            "attn_weights": all_attn_weights,
            "sequence_output": sequence_output,
            "mlm_logits": mlm_logits,
            "pooled_output": pooled_output,
        }


# ---------------------------------------------------------------------------
# Closed-form parameter-count formula (matches the implementation exactly)
# ---------------------------------------------------------------------------

def theoretical_param_count(cfg: BERTConfig) -> int:
    """
    BERT-style parameter-count formula, term by term:
        token_embedding      = V * D
        position_embedding   = max_len * D
        segment_embedding    = num_segment_types * D
        embeddings_layernorm = 2 * D
        per_layer:
            attention   = 4 * (D*D + D)               (Wq,Wk,Wv,Wo + biases)
            feedforward = (D*d_ff + d_ff) + (d_ff*D + D)
            layernorms  = 2 * (2*D)                    (post-attn LN + post-ffn LN)
        mlm_head    = (D*D + D) + 2*D + V              (dense + LN + tied-decoder bias)
        pooler      = D*D + D                          (dense, for [CLS])
    """
    d, v, l, dff, mlen = cfg.d_model, cfg.vocab_size, cfg.num_layers, cfg.d_ff, cfg.max_len
    token_emb = v * d
    pos_emb = mlen * d
    seg_emb = cfg.num_segment_types * d
    emb_ln = 2 * d
    per_layer_attn = 4 * (d * d + d)
    per_layer_ffn = (d * dff + dff) + (dff * d + d)
    per_layer_ln = 2 * (2 * d)
    per_layer = per_layer_attn + per_layer_ffn + per_layer_ln
    mlm_head = (d * d + d) + 2 * d + v
    pooler = d * d + d
    return token_emb + pos_emb + seg_emb + emb_ln + l * per_layer + mlm_head + pooler


def human(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Demo (same corpus + same toy sizes as decoder_only_architecture.py)
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
    print("Encoder-Only Transformer - Full Dimension + Parameter Walkthrough")
    print("=" * 88)

    # --- Step 1: text -> token_ids (+ segment_ids + attention_mask) ---
    corpus = build_sample_corpus()
    tokenizer = SimpleBertTokenizer()
    tokenizer.fit(corpus)
    vocab_size = len(tokenizer.token_to_id)

    text = "I love attention"
    max_len = 8
    input_ids, segment_ids, attention_mask = tokenizer.encode(text, max_len=max_len)
    seq_len = len(input_ids)

    print(f"\nCorpus size: {len(corpus)} sentences | Vocab size (V): {vocab_size}")
    print(f"Input text: {text!r}")
    print(f"Token IDs (seq_len L={seq_len}): {input_ids}")
    print(f"Segment IDs: {segment_ids}   Attention mask: {attention_mask}")

    # --- Step 2: SAME toy-scale config as decoder_only_architecture.py ---
    cfg = BERTConfig(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=16,     # D
        num_heads=4,    # H  -> d_head = D/H = 4
        num_layers=2,   # L
        d_ff=64,        # 4 x D, matches BERT's ratio
    )
    print(f"\nModel config: {cfg.describe()}")

    model = EncoderOnlyTransformer(cfg)
    out = model.forward(input_ids, segment_ids, attention_mask)

    # --- Step 3: full dimension trace ---
    print("\n" + "-" * 88)
    print("DIMENSION TRACE")
    print("-" * 88)
    print(f"token_embedding table   : ({cfg.vocab_size} x {cfg.d_model})   [V x D]")
    print(f"position_embedding table: ({cfg.max_len} x {cfg.d_model})   [max_len x D]")
    print(f"segment_embedding table : ({cfg.num_segment_types} x {cfg.d_model})   [num_segments x D]")
    print(f"x_embed (after sum+LN)  : {shape_2d(out['x_embed'])}   [L x D]")
    for i, (shape, w) in enumerate(zip(out["per_layer_shapes"], out["attn_weights"])):
        print(f"  layer {i}: output {shape}  [L x D]   attn_weights {shape_3d(w)}  [H x L x L]")
    print(f"sequence_output         : {shape_2d(out['sequence_output'])}   [L x D]")
    print(f"mlm_logits              : {shape_2d(out['mlm_logits'])}   [L x V]")
    print(f"pooled_output ([CLS])   : ({len(out['pooled_output'])},)   [D,]")

    print("\nWhy attn_weights is (H x L x L) but NOT lower-triangular here:")
    print("Encoder attention is BIDIRECTIONAL - token 0 ([CLS]) can already see")
    print("EVERY other token, including ones that come after it:")
    h0 = out["attn_weights"][0][0]  # layer 0, head 0
    for i, row in enumerate(h0):
        print(f"    layer0/head0 row {i}: {pretty(row)}")

    # --- Step 4: parameter count (actual, from the built model) ---
    print("\n" + "-" * 88)
    print("PARAMETER COUNT (toy model)")
    print("-" * 88)
    counts = model.num_params()
    for name, n in counts.items():
        print(f"  {name:<42} {n:>10,}")

    formula_total = theoretical_param_count(cfg)
    print(f"\nClosed-form formula total : {formula_total:,}")
    print(f"Actual model total         : {counts['TOTAL']:,}")
    print(f"Match: {formula_total == counts['TOTAL']}")

    # --- Step 5: scale up the SAME formula to real BERT sizes ---
    print("\n" + "-" * 88)
    print("SCALING THE SAME FORMULA UP TO REAL BERT SIZES")
    print("-" * 88)
    real_configs = {
        "toy (this demo)": cfg,
        "BERT-base": BERTConfig(vocab_size=30522, max_len=512, d_model=768, num_heads=12, num_layers=12, d_ff=3072),
        "BERT-large": BERTConfig(vocab_size=30522, max_len=512, d_model=1024, num_heads=16, num_layers=24, d_ff=4096),
    }
    print(f"  {'model':<16} {'d_model':>8} {'heads':>7} {'layers':>7} {'d_ff':>6} {'params':>10}")
    for name, c in real_configs.items():
        n = theoretical_param_count(c)
        print(f"  {name:<16} {c.d_model:>8} {c.num_heads:>7} {c.num_layers:>7} {c.d_ff:>6} {human(n):>10}")

    print("\nEncoder-only (this file) vs Decoder-only (decoder_only_architecture.py):")
    print("  Attention direction : BIDIRECTIONAL (full)     | CAUSAL (masked future)")
    print("  Norm placement      : POST-norm (BERT/original) | PRE-norm (GPT-2 style)")
    print("  Pretraining task    : Masked Language Modeling  | Next-token prediction")
    print("  Typical downstream  : classification, embeddings, NER, QA | text generation")
    print("  Extra embedding     : segment/token-type embedding | (none)")
    print("  Extra output head   : pooler ([CLS] -> classification vector) | (none)")

    print("\nPipeline summary:")
    print("  text -> SimpleBertTokenizer -> input_ids, segment_ids, attention_mask")
    print("  ids  -> token_emb[id] + position_emb[pos] + segment_emb[seg] -> LayerNorm -> x (L x D)")
    print("  x -> N x [ MultiHeadSelfAttention(bidirectional) -> +residual -> LayerNorm")
    print("            -> FeedForward(GELU)                  -> +residual -> LayerNorm ]")
    print("  -> sequence_output (L x D)")
    print("       -> MLM head  -> mlm_logits (L x V)          [per-token vocabulary prediction]")
    print("       -> Pooler    -> pooled_output (D,)          [whole-sequence classification vector]")


if __name__ == "__main__":
    run_demo()

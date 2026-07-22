"""
Embedding table: the missing bridge between tokenisation and attention.

Pipeline demonstrated end-to-end:
    raw text -> AdvancedTokenizer -> token_ids -> EmbeddingTable.lookup()
             -> token vectors (d_model) -> self_attention (Q/K/V projections)

This fills the gap between:
- advanced_tokenisation.py  (text -> token_ids, an integer per token)
- attention_at_highlevel.py (token vectors -> attention output)

Key ideas shown here:
1) An embedding table is just a (vocab_size x d_model) matrix of learnable
   numbers - nothing is "calculated" from the word's letters.
2) A token ID is a row index into that matrix -> lookup, not a formula.
3) Training updates ONLY the rows for tokens seen in a batch (a sparse
   update), via a toy manual gradient-descent step so the math is visible
   (no autograd/no numpy - kept dependency-free like attention_at_highlevel.py).
4) After lookup, the resulting vectors flow directly into self-attention.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

from advanced_tokenisation import AdvancedTokenizer
from attention_at_highlevel import Matrix, Vector, pretty, self_attention


class EmbeddingTable:
    """A (vocab_size x d_model) matrix of learnable token embeddings."""

    def __init__(self, vocab_size: int, d_model: int, seed: int = 42) -> None:
        rng = random.Random(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Small random init (similar in spirit to nn.Embedding's default init).
        self.table: Matrix = [
            [rng.gauss(0.0, 0.1) for _ in range(d_model)] for _ in range(vocab_size)
        ]

    def lookup(self, token_ids: List[int]) -> Matrix:
        """Row-index lookup: token_id -> its d_model-dim vector (pure indexing, no math)."""
        return [self.table[i] for i in token_ids]

    def update_row(self, token_id: int, grad: Vector, lr: float = 0.1) -> None:
        """
        Toy manual gradient-descent update for ONE token's embedding row:

            embedding_table[token_id] -= lr * grad

        Only rows for tokens that appeared in a batch get touched - this is
        exactly why rare tokens stay poorly trained while common tokens end
        up with rich, well-shaped vectors.
        """
        row = self.table[token_id]
        for d in range(self.d_model):
            row[d] -= lr * grad[d]


def toy_training_step(table: EmbeddingTable, token_ids: List[int]) -> None:
    """
    Simulate ONE fake gradient-descent update so you can watch numbers move.

    Real gradients come from backprop through the whole model + loss function.
    Here we fabricate a pretend gradient (nudging each seen token's vector a
    bit toward [1, 1, ..., 1]) purely to make the *update mechanics* visible.
    """
    target = [1.0] * table.d_model
    for tid in set(token_ids):
        row = table.table[tid]
        fake_grad = [row[d] - target[d] for d in range(table.d_model)]  # d(loss)/d(row)
        table.update_row(tid, fake_grad, lr=0.05)


def random_matrix(rows: int, cols: int, seed: int) -> Matrix:
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 0.3) for _ in range(cols)] for _ in range(rows)]


def mse_loss_and_grad(outputs: Matrix, targets: Matrix) -> Tuple[float, Matrix]:
    """
    Mean-squared-error loss over all attention outputs, plus its gradient.

        L         = (1 / (N*D)) * sum_i sum_d (out_i[d] - target_i[d])^2
        dL/d(out) = (2 / (N*D)) * (out_i[d] - target_i[d])

    A real model would use cross-entropy over next-token logits; MSE against
    made-up target vectors is used here purely so the backprop math below
    stays simple and easy to read by hand.
    """
    n = len(outputs)
    d = len(outputs[0])
    scale = 2.0 / (n * d)
    loss = 0.0
    d_out: Matrix = []
    for out_i, tgt_i in zip(outputs, targets):
        row = []
        for o, t in zip(out_i, tgt_i):
            diff = o - t
            loss += diff * diff
            row.append(scale * diff)
        d_out.append(row)
    loss /= (n * d)
    return loss, d_out


def backward_self_attention(
    x: Matrix,
    w_q: Matrix,
    w_k: Matrix,
    w_v: Matrix,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    weights: Matrix,
    d_out: Matrix,
) -> Tuple[Matrix, Matrix, Matrix, Matrix]:
    """
    Manual backprop through ONE self-attention block (no autograd library),
    mirroring the forward pass in attention_at_highlevel.self_attention().

    Chain of derivatives (reverse of the forward pass):
        out_i = sum_j weights[i][j] * v_j          -> d_v, d_weights
        weights[i] = softmax(scores[i])             -> d_scores
        scores[i][j] = (q_i . k_j) / sqrt(d_attn)    -> d_q, d_k
        q_i = x_i @ w_q  (same for k, v)             -> d_wq/d_wk/d_wv, d_x

    Returns gradients: dX (wrt input token vectors - fed back into the
    embedding table), dWq, dWk, dWv (fed back into the projection matrices).
    """
    n = len(x)
    d_model = len(x[0])
    d_attn = len(q[0])
    scale = math.sqrt(d_attn)

    # 1) out_i = sum_j weights[i][j] * v_j
    d_v = [[0.0] * d_attn for _ in range(n)]
    d_weights = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            w_ij = weights[i][j]
            dw = 0.0
            for d in range(d_attn):
                d_v[j][d] += d_out[i][d] * w_ij
                dw += d_out[i][d] * v[j][d]
            d_weights[i][j] = dw

    # 2) weights[i] = softmax(scores[i])  ->  standard softmax-Jacobian product
    d_scores = [[0.0] * n for _ in range(n)]
    for i in range(n):
        s = sum(d_weights[i][j] * weights[i][j] for j in range(n))
        for m in range(n):
            d_scores[i][m] = weights[i][m] * (d_weights[i][m] - s)

    # 3) scores[i][j] = (q_i . k_j) / sqrt(d_attn)
    d_q = [[0.0] * d_attn for _ in range(n)]
    d_k = [[0.0] * d_attn for _ in range(n)]
    for i in range(n):
        for j in range(n):
            g = d_scores[i][j] / scale
            for d in range(d_attn):
                d_q[i][d] += g * k[j][d]
                d_k[j][d] += g * q[i][d]

    # 4) q_i = x_i @ w_q  (and same shape for k, v)
    d_wq = [[0.0] * d_attn for _ in range(d_model)]
    d_wk = [[0.0] * d_attn for _ in range(d_model)]
    d_wv = [[0.0] * d_attn for _ in range(d_model)]
    d_x = [[0.0] * d_model for _ in range(n)]

    for i in range(n):
        for din in range(d_model):
            xi = x[i][din]
            for dout in range(d_attn):
                d_wq[din][dout] += xi * d_q[i][dout]
                d_wk[din][dout] += xi * d_k[i][dout]
                d_wv[din][dout] += xi * d_v[i][dout]

            acc = 0.0
            for dout in range(d_attn):
                acc += w_q[din][dout] * d_q[i][dout]
                acc += w_k[din][dout] * d_k[i][dout]
                acc += w_v[din][dout] * d_v[i][dout]
            d_x[i][din] = acc

    return d_x, d_wq, d_wk, d_wv


def sgd_update_matrix(mat: Matrix, grad: Matrix, lr: float) -> None:
    """In-place gradient-descent update: mat -= lr * grad."""
    for row, grad_row in zip(mat, grad):
        for d in range(len(row)):
            row[d] -= lr * grad_row[d]


def run_demo() -> None:
    d_model = 8  # embedding size (how many numbers represent one token's meaning)
    d_attn = 8   # Q/K/V projection size (kept equal to d_model here for simplicity)

    # --- Step 1: raw text -> token_ids (advanced_tokenisation.py) ---
    corpus = [
        "GPT-4 uses attention and embeddings.",
        "LoRA fine-tunes large models efficiently.",
        "Attention lets tokens look at other tokens.",
    ]
    tok = AdvancedTokenizer()
    tok.fit(corpus, min_word_freq=1, min_subword_freq=2)
    vocab_size = len(tok.token_to_id)

    sentence = "Attention uses embeddings."
    encoded = tok.encode(sentence, max_length=12)
    real_len = sum(encoded.attention_mask)  # drop padding for a clean demo
    token_ids = encoded.token_ids[:real_len]
    tokens_str = [tok.id_to_token[i] for i in token_ids]

    print("=" * 78)
    print("Bridging tokenisation -> embeddings -> attention")
    print(f"Vocab size: {vocab_size}")
    print(f"Sentence: {sentence!r}")
    print(f"Token IDs: {token_ids}")
    print(f"Tokens: {tokens_str}")

    # --- Step 2: token_ids -> embedding vectors (the missing piece) ---
    table = EmbeddingTable(vocab_size=vocab_size, d_model=d_model, seed=7)

    print(f"\nEmbedding table shape: ({table.vocab_size} x {table.d_model})  [vocab_size x d_model]")
    print("Before training, embeddings are just small random noise:")
    for tid, name in zip(token_ids, tokens_str):
        print(f"  id={tid:>3} {name:>12} -> {pretty(table.table[tid])}")

    # --- Step 3: simulate training updates (sparse, gradient-descent) ---
    for _ in range(5):
        toy_training_step(table, token_ids)

    print("\nAfter 5 toy gradient steps, only the SEEN tokens' rows changed:")
    for tid, name in zip(token_ids, tokens_str):
        print(f"  id={tid:>3} {name:>12} -> {pretty(table.table[tid])}")

    # --- Step 4: embedding vectors -> attention (attention_at_highlevel.py) ---
    x: Matrix = table.lookup(token_ids)
    w_q = random_matrix(d_model, d_attn, seed=1)
    w_k = random_matrix(d_model, d_attn, seed=2)
    w_v = random_matrix(d_model, d_attn, seed=3)

    out = self_attention(x, w_q, w_k, w_v)

    print("\nFinal attention output (weighted blend of token meanings):")
    for name, vec in zip(tokens_str, out["outputs"]):
        print(f"  {name:>12} -> {pretty(vec)}")

    print("\nPipeline summary:")
    print("  raw text   -> AdvancedTokenizer.encode()   -> token_ids (integers)")
    print("  token_ids  -> EmbeddingTable.lookup()       -> token vectors (d_model)")
    print("  vectors    -> self_attention(w_q, w_k, w_v) -> contextual outputs")


def run_joint_training_demo() -> None:
    """
    Mini END-TO-END training loop: the embedding table AND the w_q/w_k/w_v
    projection matrices are updated TOGETHER using REAL gradients (manual
    backprop through self-attention), instead of the fake gradient used in
    toy_training_step()/run_demo() above.

    Each unique token in the sentence is given its own random target vector,
    so the model has an actual reason to make each token's attention output
    different from the others (rather than collapsing to a uniform blend,
    which is what happens with untrained, random Q/K/V projections).
    """
    d_model = 8
    d_attn = 8

    corpus = [
        "GPT-4 uses attention and embeddings.",
        "LoRA fine-tunes large models efficiently.",
        "Attention lets tokens look at other tokens.",
    ]
    tok = AdvancedTokenizer()
    tok.fit(corpus, min_word_freq=1, min_subword_freq=2)
    vocab_size = len(tok.token_to_id)

    sentence = "Attention uses embeddings."
    encoded = tok.encode(sentence, max_length=12)
    real_len = sum(encoded.attention_mask)
    token_ids = encoded.token_ids[:real_len]
    tokens_str = [tok.id_to_token[i] for i in token_ids]

    table = EmbeddingTable(vocab_size=vocab_size, d_model=d_model, seed=7)
    w_q = random_matrix(d_model, d_attn, seed=1)
    w_k = random_matrix(d_model, d_attn, seed=2)
    w_v = random_matrix(d_model, d_attn, seed=3)

    # Distinct random target per UNIQUE token id -> gives attention a reason
    # to differentiate outputs instead of averaging everything together.
    rng = random.Random(99)
    target_by_id = {tid: [rng.gauss(0.0, 1.0) for _ in range(d_attn)] for tid in set(token_ids)}
    targets = [target_by_id[tid] for tid in token_ids]

    print("\n" + "=" * 78)
    print("Joint training: embedding table + Wq/Wk/Wv updated together")
    print(f"Sentence: {sentence!r}  Tokens: {tokens_str}")

    x = table.lookup(token_ids)
    out = self_attention(x, w_q, w_k, w_v)
    print("\nBEFORE training (near-uniform attention -> near-identical outputs):")
    for name, vec in zip(tokens_str, out["outputs"]):
        print(f"  {name:>12} -> {pretty(vec)}")

    lr = 0.5
    epochs = 2000
    for epoch in range(epochs):
        x = table.lookup(token_ids)
        out = self_attention(x, w_q, w_k, w_v)
        loss, d_out = mse_loss_and_grad(out["outputs"], targets)
        d_x, d_wq, d_wk, d_wv = backward_self_attention(
            x, w_q, w_k, w_v, out["Q"], out["K"], out["V"], out["weights"], d_out
        )

        sgd_update_matrix(w_q, d_wq, lr)
        sgd_update_matrix(w_k, d_wk, lr)
        sgd_update_matrix(w_v, d_wv, lr)

        # Sparse update: only rows for tokens actually in this sentence move.
        for pos, tid in enumerate(token_ids):
            table.update_row(tid, d_x[pos], lr)

        if epoch % 200 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:>4}  loss={loss:.8f}")

    x = table.lookup(token_ids)
    out = self_attention(x, w_q, w_k, w_v)
    print("\nAFTER training (outputs have differentiated + attention is sharper):")
    for name, vec, weight_row in zip(tokens_str, out["outputs"], out["weights"]):
        print(f"  {name:>12} -> {pretty(vec)}   attn_weights={pretty(weight_row)}")


if __name__ == "__main__":
    run_demo()
    run_joint_training_demo()

"""
Instruction tuning, step by step — for a senior DL/vision researcher learning NLP.

Goal:
Learn prompt-response finetuning with a tiny decoder-only Transformer and understand
the NLP-specific concepts that have no direct equivalent in computer vision.

Architecture:
    TinyInstructionTransformer — decoder-only, single TransformerBlock
    d_model=8, num_heads=2, d_head=4, d_ff=16, max_len=20, vocab_size=31
    Training: Adam lr=0.05, batch_size=2, 2 epochs, cross_entropy(ignore_index=-100)

Pipeline concepts (standard instruction tuning):
    1) Build instruction-response pairs from a small corpus
    2) Concatenate prompt + response into a single causal sequence:
           "Instruction: {prompt} Response: {response}"
    3) Build assistant-only labels: label=-100 for all prompt tokens and padding
       so that cross_entropy only back-props through the assistant part
    4) Full finetuning: all model parameters are updated (as opposed to LoRA/adapters)
    5) Inspect attention weights before and after training
    6) Run greedy autoregressive inference using the same instruction prefix format

NLP-specific concepts demonstrated (for vision researchers):
    A) Token / label / mask alignment table
          Shows exactly which sequence positions are active in the loss.
          Vision analogy: labelled vs ignored pixels in a segmentation loss mask.

    B) Teacher forcing (training) vs autoregressive decoding (inference)
          During training the model always receives ground-truth tokens as input.
          During inference it consumes its own previous predictions.
          The resulting train/infer distribution mismatch is called EXPOSURE BIAS —
          errors compound step-by-step with no equivalent in image classification.
          Mitigation: scheduled sampling, DPO, RLHF, contrastive search.
          Vision analogy: training a tracker with GT bboxes vs predicted bboxes.

    C) Embedding cosine similarity (before and after training)
          Each token is a learnable dense vector.  After training, semantically
          related tokens (e.g. "nlp" and "learning") cluster closer in embedding
          space, visible as higher off-diagonal cosine similarity values.
          Vision analogy: pairwise cosine sim between class-prototype feature vectors.

    D) Multi-head attention divergence
          Each attention head learns a different distribution over the same tokens
          simultaneously.  After training head-0 spreads softly while head-1 collapses
          sharply onto specific tokens — measurable as mean absolute divergence.
          Vision analogy: different conv filters specialising in edges, textures, colours.

    E) Perplexity alongside loss
          perplexity = exp(average_cross_entropy_loss)
          More interpretable than raw nats: perplexity 27 → model is as confused as if
          it were choosing uniformly among 27 tokens; perplexity 10 → among 10 tokens.

This is educational code, optimised for clarity over performance.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 7) -> None:
    torch.manual_seed(seed)


class BasicTokenizer:
    """Simple tokenizer used in earlier files: lowercase + word/punctuation split."""

    def __init__(self) -> None:
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    def normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def split(self, text: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", text)

    def fit(self, texts: List[str]) -> None:
        vocab = set()
        for text in texts:
            vocab.update(self.split(self.normalize(text)))

        ordered = self.special_tokens + sorted(vocab)
        self.token_to_id = {tok: idx for idx, tok in enumerate(ordered)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}

    def encode(self, text: str, max_len: int) -> List[int]:
        tokens = self.split(self.normalize(text))
        ids = [self.token_to_id["<bos>"]]
        ids.extend(self.token_to_id.get(tok, self.token_to_id["<unk>"]) for tok in tokens)
        ids.append(self.token_to_id["<eos>"])

        pad_id = self.token_to_id["<pad>"]
        if len(ids) < max_len:
            ids.extend([pad_id] * (max_len - len(ids)))
        else:
            ids = ids[:max_len]
            ids[-1] = self.token_to_id["<eos>"]
        return ids

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.id_to_token.get(i, "<unk>") for i in ids)


def build_base_corpus() -> List[str]:
    # Same corpus theme used in basic_foundations examples.
    return [
        "I love NLP",
        "I love machine learning",
        "transformers use attention",
        "attention helps models focus",
        "NLP is fun",
        "attention helps NLP models",
    ]


def build_instruction_dataset(corpus: List[str]) -> List[Tuple[str, str]]:
    """
    Build prompt-response pairs from the same corpus idea.

    Prompt = instruction style
    Response = desired assistant answer
    """
    # We keep responses short and grounded in the same tiny corpus language.
    return [
        ("Complete the phrase: i love", "nlp"),
        ("Complete the phrase: i love machine", "learning"),
        ("What do transformers use?", "transformers use attention"),
        ("What helps models focus?", "attention helps models focus"),
        ("Complete the phrase: nlp is", "fun"),
        ("Complete the phrase: attention helps nlp", "models"),
        ("Rewrite as a short fact: " + corpus[0], "i love nlp"),
        ("Rewrite as a short fact: " + corpus[3], "attention helps models focus"),
    ]


def format_chat(prompt: str, response: str) -> str:
    # Single causal sequence like instruction tuning chat data.
    return f"Instruction: {prompt} Response: {response}"


def format_prefix(prompt: str) -> str:
    # Prefix stops right before assistant content.
    return f"Instruction: {prompt} Response:"


def build_tokenizer_texts(corpus: List[str], pairs: List[Tuple[str, str]]) -> List[str]:
    texts = list(corpus)
    for prompt, response in pairs:
        texts.append(format_chat(prompt, response))
        texts.append(format_prefix(prompt))
    return texts


def build_instruction_tensors(
    pairs: List[Tuple[str, str]],
    tokenizer: BasicTokenizer,
    max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build training tensors for assistant-only instruction tuning.

    input_ids shape : (batch, seq_len)
    labels shape    : (batch, seq_len)

    labels use -100 where loss should be ignored (prompt part + padding).
    """
    all_inputs: List[List[int]] = []
    all_labels: List[List[int]] = []

    pad_id = tokenizer.token_to_id["<pad>"]

    for prompt, response in pairs:
        full_text = format_chat(prompt, response) # instruction and response in one sequence, which will be tokenized together. The model will learn to predict the response tokens based on the instruction tokens, but we will mask the loss for the instruction tokens so that the model focuses on learning to generate the response.
        prefix_text = format_prefix(prompt) # this is just the instruction part, used to count how many tokens are in the prompt so we can mask those positions in the labels. The model will not be trained to predict these tokens, but they are still part of the input sequence that the model sees when generating the response.

        full_ids = tokenizer.encode(full_text, max_len=max_len)

        # prefix token count without BOS/EOS, using same split/normalize rules.
        prefix_token_count = len(tokenizer.split(tokenizer.normalize(prefix_text)))

        # Language-model shift.
        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        # Assistant-only loss masking.
        # labels positions < prefix_token_count are prompt targets -> ignore.
        # padding token targets should also be ignored.
        for i in range(len(labels)):
            if i < prefix_token_count:
                labels[i] = -100
            elif labels[i] == pad_id:
                labels[i] = -100

        all_inputs.append(input_ids)
        all_labels.append(labels)

    return torch.tensor(all_inputs, dtype=torch.long), torch.tensor(all_labels, dtype=torch.long)


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # True means blocked (future positions).
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int
    d_model: int = 8
    num_heads: int = 2
    d_ff: int = 16


class TinyMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x shape: (batch, seq_len, d_model)

        returns:
        output  : (batch, seq_len, d_model)
        scores  : (batch, heads, seq_len, seq_len)
        weights : (batch, heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = build_causal_mask(seq_len, x.device)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)

        weights = F.softmax(scores, dim=-1)
        heads_out = torch.matmul(weights, v)

        joined = heads_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(joined)
        return output, scores, weights


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attn = TinyMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, scores, weights = self.attn(x)
        x = self.norm1(x + attn_out)
        ff = self.ffn(x)
        x = self.norm2(x + ff)
        return x, scores, weights


class TinyInstructionTransformer(nn.Module):
    """Full-finetuning model: all parameters are trainable."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.block = TransformerBlock(cfg.d_model, cfg.num_heads, cfg.d_ff)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        x, scores, weights = self.block(x)
        logits = self.lm_head(x)
        return logits, scores, weights


def format_row(values: List[float], decimals: int = 3) -> str:
    return "[" + ", ".join(f"{v:.{decimals}f}" for v in values) + "]"


def print_attention_snapshot(
    model: TinyInstructionTransformer,
    tokenizer: BasicTokenizer,
    prompt: str,
    max_len: int,
    title: str,
) -> None:
    """Inspect one sample so feature-learning behavior is easy to observe."""
    model.eval()
    text = format_prefix(prompt)
    ids = tokenizer.encode(text, max_len=max_len)
    input_ids = torch.tensor([ids[:-1]], dtype=torch.long)

    with torch.no_grad():
        logits, scores, weights = model(input_ids)

    tokens = [tokenizer.id_to_token[idx] for idx in input_ids[0].tolist()]
    head0_scores = scores[0, 0].tolist()
    head0_weights = weights[0, 0].tolist()

    print("\n" + "-" * 92)
    print(title)
    print("Tokens:", tokens)
    print("Head-0 raw attention scores:")
    for i, row in enumerate(head0_scores):
        print(f"  query {i:>2} ({tokens[i]}): {format_row(row, 3)}")

    print("Head-0 attention weights:")
    for i, row in enumerate(head0_weights):
        print(f"  query {i:>2} ({tokens[i]}): {format_row(row, 3)}")

    last_pos = len(tokens) - 1
    probs = F.softmax(logits[0, last_pos], dim=-1)
    top_vals, top_ids = torch.topk(probs, 5)
    top_str = []
    for val, idx in zip(top_vals.tolist(), top_ids.tolist()):
        top_str.append(f"{tokenizer.id_to_token[idx]}={val:.3f}")
    print("Top next-token probabilities at final position:", ", ".join(top_str))


def train_two_epochs(
    model: TinyInstructionTransformer,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    lr: float = 0.05,
    batch_size: int = 2,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_examples = train_inputs.size(0)

    for epoch in range(2):
        model.train()
        order = torch.randperm(num_examples)
        total_loss = 0.0

        print("\n" + "=" * 92)
        print(f"Epoch {epoch + 1}/2")
        print("=" * 92)

        for start in range(0, num_examples, batch_size):
            idx = order[start:start + batch_size]
            x = train_inputs[idx]
            y = train_labels[idx]

            logits, _, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            print(f"  batch {start // batch_size + 1}: loss={loss.item():.4f}")

        avg = total_loss / num_examples
        perp = math.exp(min(avg, 20))
        print(f"Average epoch loss: {avg:.4f}  |  perplexity: {perp:.2f}")
        print("  (perplexity = exp(loss); 1.0 = perfect, higher = more confused)")


def generate_response(
    model: TinyInstructionTransformer,
    tokenizer: BasicTokenizer,
    prompt: str,
    max_len: int,
    max_new_tokens: int = 6,
) -> str:
    """Greedy generation after 'Response:' prefix."""
    model.eval()

    prefix = format_prefix(prompt)
    prefix_ids = tokenizer.encode(prefix, max_len=max_len)

    # Remove padding and eos for active generation state.
    active = []
    for token_id in prefix_ids:
        if token_id == tokenizer.token_to_id["<pad>"]:
            break
        if token_id == tokenizer.token_to_id["<eos>"]:
            continue
        active.append(token_id)

    generated_tokens: List[int] = []
    blocked_at_start = {
        tokenizer.token_to_id["<pad>"],
        tokenizer.token_to_id["<bos>"],
        tokenizer.token_to_id["<unk>"],
    }

    for _ in range(max_new_tokens):
        input_window = active[-(max_len - 1):]
        padded = input_window + [tokenizer.token_to_id["<pad>"]] * ((max_len - 1) - len(input_window))
        x = torch.tensor([padded], dtype=torch.long)

        with torch.no_grad():
            logits, _, _ = model(x)

        last_pos = len(input_window) - 1
        logits_row = logits[0, last_pos]

        # Keep the first generated token informative for beginner demos.
        # If top prediction is a special token, pick the best non-special candidate.
        if not generated_tokens:
            top_vals, top_ids = torch.topk(logits_row, k=min(8, logits_row.numel()))
            next_id = int(top_ids[0].item())
            for cand in top_ids.tolist():
                if cand not in blocked_at_start and cand != tokenizer.token_to_id["<eos>"]:
                    next_id = int(cand)
                    break
        else:
            next_id = int(torch.argmax(logits_row).item())

        active.append(next_id)

        if next_id == tokenizer.token_to_id["<eos>"]:
            break
        generated_tokens.append(next_id)

    if not generated_tokens:
        return ""
    return tokenizer.decode(generated_tokens)


def print_label_mask_table(
    pairs: List[Tuple[str, str]],
    tokenizer: BasicTokenizer,
    max_len: int,
) -> None:
    """
    NLP Concept A: Token / label / mask alignment table.

    Shows which positions contribute to loss and which are masked with -100.
    Vision analogy: knowing which pixels in a segmentation mask are labelled
    vs ignored (crowd regions) when computing the loss.
    """
    prompt, response = pairs[0]
    full_text = format_chat(prompt, response)
    prefix_text = format_prefix(prompt)

    full_ids = tokenizer.encode(full_text, max_len=max_len)
    prefix_token_count = len(tokenizer.split(tokenizer.normalize(prefix_text)))
    pad_id = tokenizer.token_to_id["<pad>"]

    input_ids = full_ids[:-1]
    labels_raw = full_ids[1:]

    print("\n" + "-" * 92)
    print("NLP Concept A: Token / label / mask alignment table")
    print(f"  First pair: prompt='{prompt}' | response='{response}'")
    print(f"  Full sequence: '{full_text}'")
    print()
    print(f"  {'pos':>4}  {'input token':>14}  {'label token':>14}  {'label id':>8}  loss?")
    print("  " + "-" * 68)

    for i, (inp, lbl) in enumerate(zip(input_ids, labels_raw)):
        inp_tok = tokenizer.id_to_token.get(inp, "?")
        lbl_raw_tok = tokenizer.id_to_token.get(lbl, "?") if lbl != pad_id else "<pad>"
        if i < prefix_token_count:
            lbl_display = -100
            active = "No  <- prompt token"
        elif lbl == pad_id:
            lbl_display = -100
            active = "No  <- padding"
        else:
            lbl_display = lbl
            active = "YES <- assistant output"
        print(f"  {i:>4}  {inp_tok:>14}  {lbl_raw_tok:>14}  {str(lbl_display):>8}  {active}")

    print()
    print("  Key insight:")
    print("    - input_ids  = ground-truth tokens fed as context (left-shift of full sequence)")
    print("    - labels     = ground-truth next tokens to predict (right-shift by 1)")
    print("    - label=-100 tells cross_entropy to ignore that position entirely")
    print("    - only response tokens (assistant part) contribute to gradient updates")


def explain_teacher_forcing(
    pairs: List[Tuple[str, str]],
    tokenizer: BasicTokenizer,
    max_len: int,
) -> None:
    """
    NLP Concept B: Teacher forcing (training) vs autoregressive decoding (inference).

    This training-inference gap is called EXPOSURE BIAS and is a fundamental
    NLP challenge with no direct equivalent in image classification or detection.
    """
    prompt, response = pairs[0]
    full_text = format_chat(prompt, response)
    full_ids = tokenizer.encode(full_text, max_len=max_len)
    prefix_text = format_prefix(prompt)
    prefix_token_count = len(tokenizer.split(tokenizer.normalize(prefix_text)))
    tokens = [tokenizer.id_to_token.get(i, "?") for i in full_ids]

    print("\n" + "-" * 92)
    print("NLP Concept B: Teacher forcing (training) vs autoregressive decoding (inference)")
    print()
    print("  TEACHER FORCING — used during training")
    print("  At every step the model always receives the GROUND-TRUTH token as input,")
    print("  even if its prediction at the previous step was wrong.")
    print()
    print(f"  Example sequence: '{full_text}'")
    print()
    print(f"  {'t':>3}  {'fed to model (GT)':>22}  {'predict target':>22}  role")
    print("  " + "-" * 72)
    show_up_to = min(len(full_ids) - 1, prefix_token_count + 3)
    for i in range(show_up_to):
        inp_tok = tokens[i] if tokens[i] != "<pad>" else "—"
        tgt_tok = tokens[i + 1] if tokens[i + 1] != "<pad>" else "—"
        role = "prompt  (loss masked)" if i < prefix_token_count else "assistant (loss active)"
        print(f"  {i:>3}  {inp_tok:>22}  {tgt_tok:>22}  {role}")

    print()
    print("  AUTOREGRESSIVE DECODING — used during inference")
    print("  The model generates one token and feeds its OWN output back as the next input.")
    print("  A wrong token at step t contaminates all subsequent steps.")
    print()
    print("  Why this matters (EXPOSURE BIAS):")
    print("    - Train: model always sees perfect GT context -> easy, stable gradients")
    print("    - Infer: model sees its own (possibly wrong) context -> error compounds")
    print("    - Techniques to reduce it: scheduled sampling, DPO, RLHF, contrastive search")
    print("    - Vision analogy: training a tracker with GT bboxes vs using predicted bboxes")


def inspect_embedding_similarity(
    model: TinyInstructionTransformer,
    tokenizer: BasicTokenizer,
    title: str,
) -> None:
    """
    NLP Concept C: Embedding cosine similarity before and after training.

    In NLP, every token is first mapped to a dense vector (embedding).
    After training, semantically related tokens should cluster closer together.
    Vision analogy: pairwise cosine similarity between class-prototype feature vectors.
    """
    tokens_of_interest = ["nlp", "learning", "attention", "models", "fun", "love", "i"]
    emb_weight = model.token_embedding.weight.detach()  # (vocab_size, d_model)

    ids: List[int] = []
    valid_toks: List[str] = []
    for tok in tokens_of_interest:
        if tok in tokenizer.token_to_id:
            ids.append(tokenizer.token_to_id[tok])
            valid_toks.append(tok)

    vecs = emb_weight[ids]  # (n, d_model)
    norms = vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = torch.matmul(vecs / norms, (vecs / norms).T)  # (n, n)

    col_w = max(len(t) for t in valid_toks) + 2

    print("\n" + "-" * 92)
    print(f"NLP Concept C: Embedding cosine similarity — {title}")
    print("  Diagonal = 1.0 (self-similarity). Off-diagonal = learned token relationship.")
    print("  After training: semantically linked tokens should show higher values.")
    print()
    header = f"  {'':>{col_w}}"
    for t in valid_toks:
        header += f"  {t:>{col_w}}"
    print(header)
    for i, tok_i in enumerate(valid_toks):
        row = f"  {tok_i:>{col_w}}"
        for j in range(len(valid_toks)):
            row += f"  {sim[i, j].item():>{col_w}.3f}"
        print(row)

    best_val, best_pair = -2.0, ("", "")
    for i in range(len(valid_toks)):
        for j in range(i + 1, len(valid_toks)):
            v = sim[i, j].item()
            if v > best_val:
                best_val, best_pair = v, (valid_toks[i], valid_toks[j])
    print(f"\n  Most similar pair: '{best_pair[0]}' <-> '{best_pair[1]}'  sim={best_val:.3f}")


def compare_attention_heads(
    model: TinyInstructionTransformer,
    tokenizer: BasicTokenizer,
    prompt: str,
    max_len: int,
    title: str,
) -> None:
    """
    NLP Concept D: Multi-head attention divergence.

    Each head learns to attend to different token relationships simultaneously.
    Vision analogy: different conv filters detecting edges, textures, and colours.
    Higher divergence between heads = richer representational capacity.
    """
    model.eval()
    text = format_prefix(prompt)
    ids = tokenizer.encode(text, max_len=max_len)
    input_ids = torch.tensor([ids[:-1]], dtype=torch.long)

    with torch.no_grad():
        _, _, weights = model(input_ids)

    tokens = [tokenizer.id_to_token[idx] for idx in input_ids[0].tolist()]
    n_active = sum(1 for t in tokens if t != "<pad>")
    n_heads = weights.shape[1]

    print("\n" + "-" * 92)
    print(f"NLP Concept D: Multi-head attention divergence — {title}")
    print("  Each head learns a different attention distribution over the same tokens.")
    print("  Vision analogy: different conv filters attending to different spatial features.")
    print()
    for h in range(min(n_heads, 2)):
        head_w = weights[0, h].tolist()
        print(f"  Head-{h} weights (active positions only):")
        for i in range(n_active):
            row = " ".join(f"{head_w[i][j]:.3f}" for j in range(n_active))
            print(f"    query {i:>2} ({tokens[i]:>14}): {row}")
        print()

    if n_heads >= 2:
        w0 = weights[0, 0, :n_active, :n_active]
        w1 = weights[0, 1, :n_active, :n_active]
        divergence = (w0 - w1).abs().mean().item()
        print(f"  Mean absolute divergence head-0 vs head-1: {divergence:.4f}")
        print("  Interpretation: the higher this is, the more the heads have specialised.")


def run_demo() -> None:
    set_seed(7)

    print("=" * 92)
    print("Instruction Tuning Tutorial: Prompt-Response Finetuning")
    print("=" * 92)

    # Step 1) Data setup using same corpus theme.
    corpus = build_base_corpus()
    pairs = build_instruction_dataset(corpus)

    print("\nStep 1: Base corpus (same theme as earlier files)")
    for i, line in enumerate(corpus, start=1):
        print(f"  {i}. {line}")

    print("\nStep 2: Instruction-response pairs")
    for i, (p, r) in enumerate(pairs, start=1):
        print(f"  {i}. prompt='{p}' | response='{r}'")

    # Step 3) Tokenizer setup.
    tokenizer = BasicTokenizer()
    tokenizer.fit(build_tokenizer_texts(corpus, pairs))

    print("\nStep 3: Vocabulary")
    for tok, idx in tokenizer.token_to_id.items():
        print(f"  {tok:>14} -> {idx}")

    # Step 4) Build training tensors with assistant-only labels.
    max_len = 20
    train_inputs, train_labels = build_instruction_tensors(pairs, tokenizer, max_len=max_len) # this is for building the input and label tensors for training, where labels have -100 for prompt tokens and padding to ignore loss on those positions

    print("\nStep 4: Tensor shapes")
    print(f"  train_inputs: {tuple(train_inputs.shape)} = (batch, seq_len)")
    print(f"  train_labels: {tuple(train_labels.shape)} = (batch, seq_len)")
    print("  label=-100 means loss is ignored (prompt part and padding).")
    print_label_mask_table(pairs, tokenizer, max_len)
    explain_teacher_forcing(pairs, tokenizer, max_len)

    # Step 5) Build tiny transformer model with same architecture style.
    cfg = ModelConfig(
        vocab_size=len(tokenizer.token_to_id),
        seq_len=max_len - 1,
        d_model=8,
        num_heads=2,
        d_ff=16,
    )
    model = TinyInstructionTransformer(cfg)

    print("\nStep 5: Model config")
    print(f"  d_model={cfg.d_model}, num_heads={cfg.num_heads}, d_head={cfg.d_model // cfg.num_heads}, d_ff={cfg.d_ff}")
    print("  Full finetuning mode: all parameters are trainable.")
    inspect_embedding_similarity(model, tokenizer, "BEFORE training")

    # Step 6) Inspect attention before training.
    print_attention_snapshot(
        model=model,
        tokenizer=tokenizer,
        prompt="Complete the phrase: i love",
        max_len=max_len,
        title="Step 6: Before training (random attention behavior)",
    )

    # Step 7) Train for 2 epochs.
    train_two_epochs(
        model=model,
        train_inputs=train_inputs,
        train_labels=train_labels,
        lr=0.05,
        batch_size=2,
    )

    # Step 8) Inspect attention after training.
    print_attention_snapshot(
        model=model,
        tokenizer=tokenizer,
        prompt="Complete the phrase: i love",
        max_len=max_len,
        title="Step 8: After training (more structured attention)",
    )
    inspect_embedding_similarity(model, tokenizer, "AFTER training")
    compare_attention_heads(
        model=model,
        tokenizer=tokenizer,
        prompt="Complete the phrase: i love",
        max_len=max_len,
        title="AFTER training",
    )

    # Step 9) Inference demo.
    print("\n" + "=" * 92)
    print("Step 9: Inference demo (prompt -> generated response)")
    print("=" * 92)

    test_prompts = [
        "Complete the phrase: i love",
        "What do transformers use?",
        "What helps models focus?",
    ]
    for p in test_prompts:
        g = generate_response(model, tokenizer, p, max_len=max_len, max_new_tokens=6)
        print(f"  prompt='{p}'")
        print(f"  response='{g}'")

    print("\nRecap — NLP-specific concepts demonstrated:")
    print("  A. Token/label/mask table    : which positions contribute to loss and which are ignored")
    print("  B. Teacher forcing           : training uses GT context; inference uses own predictions")
    print("     Exposure bias             : error at step t compounds to step t+1 (no vision equivalent)")
    print("  C. Embedding similarity      : cosine sim before vs after training shows learned clustering")
    print("  D. Multi-head divergence     : each head specialises in a different attention pattern")
    print("  E. Perplexity                : exp(loss) — more interpretable than raw cross-entropy")
    print("  Standard pipeline            : causal LM shift, assistant-only loss mask, full-finetune all params")


if __name__ == "__main__":
    run_demo()

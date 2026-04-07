"""
Instruction tuning with visualizations — for a senior DL/vision researcher learning NLP.

Purpose:
    Teach phase-3 style instruction tuning fundamentals with train/validation
    monitoring, hyperparameter basics, and intuitive visual diagnostics.

What this script demonstrates:
    1) Prompt-response formatting in one causal sequence:
           "Instruction: {prompt} Response: {response}"
    2) Assistant-only loss masking (label=-100 for prompt and padding)
    3) Train/validation split and per-epoch generalization tracking
    4) Hyperparameter effects (epochs, batch_size, lr, weight_decay, split ratio)
    5) Before/after behavior of embeddings and attention heads
    6) Greedy autoregressive inference using the same instruction prefix

Model and training setup:
    TinyInstructionTransformer — decoder-only, single TransformerBlock
    d_model=8, num_heads=2, d_head=4, d_ff=16
    Optimizer: Adam
    Objective: cross_entropy(ignore_index=-100)

Visual outputs generated:
    - 01_token_label_mask_visualization.png
    - 02_teacher_forcing_visualization.png
    - 03_embedding_similarity_before.png
    - 03_embedding_similarity_after.png
    - 04a_attention_before.png
    - 04b_attention_after.png
    - 05_learning_curves.png (train vs validation loss/perplexity)

How to run:
    1) Open terminal at repository root.
    2) Execute:
           python full_finetuning/instruction_tuning_with_visual.py
    3) Read console sections in order (data -> hyperparameters -> split -> train/val metrics).
    4) Inspect generated PNGs to interpret learning dynamics and generalization.

Hyperparameter quick guide:
    - epochs: more training passes; too high can overfit.
    - batch_size: step stability vs update frequency.
    - lr: optimization speed; too high can diverge.
    - weight_decay: regularization strength.
    - train_ratio: data fraction used for training (rest is validation).
    - max_len: context window (sequence length budget).

This is educational code, optimized for clarity and intuition over scale/performance.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def set_seed(seed: int = 7) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


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
    """Build prompt-response pairs from the same corpus idea."""
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
    return f"Instruction: {prompt} Response: {response}"


def format_prefix(prompt: str) -> str:
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
    """Build training tensors for assistant-only instruction tuning."""
    all_inputs: List[List[int]] = []
    all_labels: List[List[int]] = []

    pad_id = tokenizer.token_to_id["<pad>"]

    for prompt, response in pairs:
        full_text = format_chat(prompt, response)
        prefix_text = format_prefix(prompt)

        full_ids = tokenizer.encode(full_text, max_len=max_len)
        prefix_token_count = len(tokenizer.split(tokenizer.normalize(prefix_text)))

        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        for i in range(len(labels)):
            if i < prefix_token_count:
                labels[i] = -100
            elif labels[i] == pad_id:
                labels[i] = -100

        all_inputs.append(input_ids)
        all_labels.append(labels)

    return torch.tensor(all_inputs, dtype=torch.long), torch.tensor(all_labels, dtype=torch.long)


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int
    d_model: int = 8
    num_heads: int = 2
    d_ff: int = 16


@dataclass
class HyperParams:
    epochs: int = 4
    batch_size: int = 2
    lr: float = 0.03
    weight_decay: float = 0.0
    train_ratio: float = 0.75
    max_len: int = 20
    max_new_tokens: int = 6


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


def visualize_token_label_mask(
    pairs: List[Tuple[str, str]],
    tokenizer: BasicTokenizer,
    max_len: int,
) -> None:
    """Visualization A: Token/label/mask alignment as color-coded grid."""
    prompt, response = pairs[0]
    full_text = format_chat(prompt, response)
    prefix_text = format_prefix(prompt)

    full_ids = tokenizer.encode(full_text, max_len=max_len)
    prefix_token_count = len(tokenizer.split(tokenizer.normalize(prefix_text)))
    pad_id = tokenizer.token_to_id["<pad>"]

    input_ids = full_ids[:-1]
    labels_raw = full_ids[1:]
    tokens = [tokenizer.id_to_token.get(i, "?") for i in full_ids[:-1]]

    # Build mask array: 0 = prompt/padding (ignored), 1 = assistant (active)
    mask_array = []
    for i, lbl in enumerate(labels_raw):
        if i < prefix_token_count or lbl == pad_id:
            mask_array.append(0)
        else:
            mask_array.append(1)

    fig, ax = plt.subplots(figsize=(14, 4))

    # Create color grid
    positions = np.arange(len(tokens))
    colors = ['#e8e8e8' if m == 0 else '#4CAF50' for m in mask_array]

    bars = ax.bar(positions, [1]*len(tokens), color=colors, edgecolor='black', linewidth=1.5)

    # Add token labels on top
    for i, (bar, tok) in enumerate(zip(bars, tokens)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                tok, ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlim(-0.5, len(tokens)-0.5)
    ax.set_ylim(0, 1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Legend
    ignored_patch = mpatches.Patch(color='#e8e8e8', label='Ignored (label=-100, no gradient)')
    active_patch = mpatches.Patch(color='#4CAF50', label='Active (assistant, gradient flows)')
    ax.legend(handles=[ignored_patch, active_patch], loc='upper right', fontsize=10)

    ax.set_title(
        f"NLP Concept A: Token/Label/Mask Alignment\n"
        f"Prompt: '{prompt}' | Response: '{response}'",
        fontsize=12, fontweight='bold', pad=15
    )

    plt.tight_layout()
    plt.savefig('01_token_label_mask_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 01_token_label_mask_visualization.png")
    plt.close()


def visualize_attention_heatmaps(
    model: TinyInstructionTransformer,
    tokenizer: BasicTokenizer,
    prompt: str,
    max_len: int,
    title: str,
    filename: str,
) -> None:
    """Visualization: Attention weight heatmaps for head-0 and head-1."""
    model.eval()
    text = format_prefix(prompt)
    ids = tokenizer.encode(text, max_len=max_len)
    input_ids = torch.tensor([ids[:-1]], dtype=torch.long)

    with torch.no_grad():
        _, _, weights = model(input_ids)

    tokens = [tokenizer.id_to_token[idx] for idx in input_ids[0].tolist()]
    n_active = sum(1 for t in tokens if t != "<pad>")
    tokens_active = tokens[:n_active]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for h in range(2):
        head_w = weights[0, h, :n_active, :n_active].detach().cpu().numpy()

        im = axes[h].imshow(head_w, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

        axes[h].set_xticks(range(n_active))
        axes[h].set_yticks(range(n_active))
        axes[h].set_xticklabels(tokens_active, rotation=45, ha='right', fontsize=9)
        axes[h].set_yticklabels(tokens_active, fontsize=9)

        axes[h].set_xlabel('Key tokens', fontsize=10, fontweight='bold')
        axes[h].set_ylabel('Query tokens', fontsize=10, fontweight='bold')
        axes[h].set_title(f'Head-{h} Attention', fontsize=11, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[h], fraction=0.046, pad=0.04)
        cbar.set_label('Weight', fontsize=9)

        # Add text annotations
        for i in range(n_active):
            for j in range(n_active):
                text = axes[h].text(j, i, f'{head_w[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=7)

    fig.suptitle(f'NLP Concept D: Multi-Head Attention Weights — {title}',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def visualize_embedding_similarity(
    model: TinyInstructionTransformer,
    tokenizer: BasicTokenizer,
    title: str,
    filename: str,
) -> None:
    """Visualization: Embedding cosine similarity heatmap."""
    tokens_of_interest = ["nlp", "learning", "attention", "models", "fun", "love", "i"]
    emb_weight = model.token_embedding.weight.detach()

    ids: List[int] = []
    valid_toks: List[str] = []
    for tok in tokens_of_interest:
        if tok in tokenizer.token_to_id:
            ids.append(tokenizer.token_to_id[tok])
            valid_toks.append(tok)

    vecs = emb_weight[ids]
    norms = vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = torch.matmul(vecs / norms, (vecs / norms).T).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(sim, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(range(len(valid_toks)))
    ax.set_yticks(range(len(valid_toks)))
    ax.set_xticklabels(valid_toks, rotation=45, ha='right', fontsize=11, fontweight='bold')
    ax.set_yticklabels(valid_toks, fontsize=11, fontweight='bold')

    # Add text annotations with value and color coding
    for i in range(len(valid_toks)):
        for j in range(len(valid_toks)):
            val = sim[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            text = ax.text(j, i, f'{val:.2f}', ha="center", va="center",
                         color=color, fontsize=10, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', fontsize=10, fontweight='bold')

    ax.set_title(f'NLP Concept C: Embedding Cosine Similarity — {title}',
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Token', fontsize=11, fontweight='bold')
    ax.set_ylabel('Token', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def visualize_learning_curves(
    train_batch_losses: List[List[float]],
    train_epoch_losses: List[float],
    val_epoch_losses: List[float],
    train_perplexities: List[float],
    val_perplexities: List[float],
) -> None:
    """Visualization: train/validation curves and perplexity across epochs."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: batch losses per epoch for training only.
    for i, losses in enumerate(train_batch_losses):
        batches = range(1, len(losses) + 1)
        axes[0].plot(batches, losses, marker='o', linewidth=2, label=f'Epoch {i + 1}')
    axes[0].set_xlabel('Batch', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Train Loss', fontsize=10, fontweight='bold')
    axes[0].set_title('Batch-Level Train Loss', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(fontsize=9)

    # Middle: epoch train vs validation loss.
    epochs = np.arange(1, len(train_epoch_losses) + 1)
    axes[1].plot(epochs, train_epoch_losses, marker='o', linewidth=2.5, color='#1f77b4', label='Train')
    axes[1].plot(epochs, val_epoch_losses, marker='s', linewidth=2.5, color='#d62728', label='Validation')
    axes[1].set_xlabel('Epoch', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=10, fontweight='bold')
    axes[1].set_title('Epoch Loss: Train vs Validation', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(fontsize=9)

    # Right: epoch train vs validation perplexity.
    axes[2].plot(epochs, train_perplexities, marker='o', linewidth=2.5, color='#2ca02c', label='Train')
    axes[2].plot(epochs, val_perplexities, marker='s', linewidth=2.5, color='#ff7f0e', label='Validation')
    axes[2].set_xlabel('Epoch', fontsize=10, fontweight='bold')
    axes[2].set_ylabel('Perplexity', fontsize=10, fontweight='bold')
    axes[2].set_title('Epoch Perplexity: Train vs Validation', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].legend(fontsize=9)

    fig.suptitle(
        'NLP Concept E: Generalization Monitoring with Train/Validation Curves',
        fontsize=13,
        fontweight='bold',
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig('05_learning_curves.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 05_learning_curves.png")
    plt.close()


def split_train_validation(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    train_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic shuffled split for train/validation sets."""
    n = inputs.size(0)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator)
    n_train = max(1, int(n * train_ratio))
    n_train = min(n_train, n - 1)
    train_idx = order[:n_train]
    val_idx = order[n_train:]
    return inputs[train_idx], labels[train_idx], inputs[val_idx], labels[val_idx]


def evaluate_dataset(
    model: TinyInstructionTransformer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> Tuple[float, float]:
    """Compute average loss and perplexity for a dataset split."""
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for start in range(0, inputs.size(0), batch_size):
            x = inputs[start:start + batch_size]
            y = labels[start:start + batch_size]
            logits, _, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-100,
            )
            total_loss += loss.item() * x.size(0)
            total_items += x.size(0)
    avg_loss = total_loss / max(total_items, 1)
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, perplexity


def visualize_teacher_forcing_concept() -> None:
    """Visualization B: Diagram explaining teacher forcing vs autoregressive decoding."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Teacher Forcing (Training)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Training: Teacher Forcing\n(always use ground-truth tokens)', 
                  fontsize=12, fontweight='bold', pad=20)

    tokens_tf = ['<bos>', 'instruction', ':', 'complete', 'the', 'Response:', 'nlp', '<eos>']
    y_pos = 9
    for i, tok in enumerate(tokens_tf):
        rect = mpatches.FancyBboxPatch((0.5, y_pos - 0.4), 3, 0.8,
                                       boxstyle="round,pad=0.1", edgecolor='black',
                                       facecolor='#90EE90', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(2, y_pos, tok, ha='center', va='center', fontsize=9, fontweight='bold')

        if i < len(tokens_tf) - 1:
            ax1.arrow(3.7, y_pos, 0, -0.7, head_width=0.2, head_length=0.1, fc='black', ec='black')

        y_pos -= 1.2

    ax1.text(5, 1.5, '✓ Model sees perfect context\n✓ Stable gradients\n✗ Distribution mismatch in inference',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='#FFFFCC', alpha=0.8), verticalalignment='center')

    # Autoregressive Decoding (Inference)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Inference: Autoregressive Decoding\n(use own predictions as input)',
                  fontsize=12, fontweight='bold', pad=20)

    tokens_ar = ['<bos>', 'instruction', ':', 'complete', 'the', 'Response:', 'learned_token', 'generated_token']
    y_pos = 9
    for i, tok in enumerate(tokens_ar):
        color = '#FFB6C1' if 'learned' in tok or 'generated' in tok else '#87CEEB'
        rect = mpatches.FancyBboxPatch((0.5, y_pos - 0.4), 3, 0.8,
                                       boxstyle="round,pad=0.1", edgecolor='black',
                                       facecolor=color, linewidth=2)
        ax2.add_patch(rect)
        ax2.text(2, y_pos, tok, ha='center', va='center', fontsize=8, fontweight='bold')

        if i < len(tokens_ar) - 1:
            if i >= 5:
                ax2.annotate('', xy=(3.7, y_pos - 1.2), xytext=(3.7, y_pos),
                            arrowprops=dict(arrowstyle='->', lw=2, color='red'))
            else:
                ax2.arrow(3.7, y_pos, 0, -0.7, head_width=0.2, head_length=0.1, fc='black', ec='black')

        y_pos -= 1.2

    ax2.text(5, 1.5, '✗ Model sees imperfect context\n✗ Errors compound (EXPOSURE BIAS)\n✓ Matches deployment scenario',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='#FFE4E1', alpha=0.8), verticalalignment='center')

    fig.suptitle('NLP Concept B: Training vs Inference Distributions (Exposure Bias)',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('02_teacher_forcing_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 02_teacher_forcing_visualization.png")
    plt.close()


def format_row(values: List[float], decimals: int = 3) -> str:
    return "[" + ", ".join(f"{v:.{decimals}f}" for v in values) + "]"


def print_label_mask_table(
    pairs: List[Tuple[str, str]],
    tokenizer: BasicTokenizer,
    max_len: int,
) -> None:
    """Console output A: Token / label / mask alignment table."""
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
    """Console output B: Teacher forcing vs autoregressive decoding explanation."""
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
    """Console output C: Embedding cosine similarity."""
    tokens_of_interest = ["nlp", "learning", "attention", "models", "fun", "love", "i"]
    emb_weight = model.token_embedding.weight.detach()

    ids: List[int] = []
    valid_toks: List[str] = []
    for tok in tokens_of_interest:
        if tok in tokenizer.token_to_id:
            ids.append(tokenizer.token_to_id[tok])
            valid_toks.append(tok)

    vecs = emb_weight[ids]
    norms = vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = torch.matmul(vecs / norms, (vecs / norms).T)

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
    """Console output D: Multi-head attention divergence."""
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


def train_with_validation(
    model: TinyInstructionTransformer,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    val_inputs: torch.Tensor,
    val_labels: torch.Tensor,
    hyper: HyperParams,
) -> Tuple[List[List[float]], List[float], List[float], List[float], List[float]]:
    """Train with validation tracking for phase-3 style monitoring."""
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper.lr, weight_decay=hyper.weight_decay)
    num_examples = train_inputs.size(0)
    all_batch_losses: List[List[float]] = []
    train_epoch_losses: List[float] = []
    val_epoch_losses: List[float] = []
    train_perplexities: List[float] = []
    val_perplexities: List[float] = []

    for epoch in range(hyper.epochs):
        model.train()
        order = torch.randperm(num_examples)
        total_loss = 0.0
        epoch_losses = []

        print("\n" + "=" * 92)
        print(f"Epoch {epoch + 1}/{hyper.epochs}")
        print("=" * 92)

        for start in range(0, num_examples, hyper.batch_size):
            idx = order[start:start + hyper.batch_size]
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

            batch_loss = loss.item()
            epoch_losses.append(batch_loss)
            total_loss += batch_loss * x.size(0)
            print(f"  batch {start // hyper.batch_size + 1}: loss={batch_loss:.4f}")

        all_batch_losses.append(epoch_losses)
        train_avg = total_loss / num_examples
        train_perp = math.exp(min(train_avg, 20))
        val_avg, val_perp = evaluate_dataset(model, val_inputs, val_labels, hyper.batch_size)

        train_epoch_losses.append(train_avg)
        val_epoch_losses.append(val_avg)
        train_perplexities.append(train_perp)
        val_perplexities.append(val_perp)

        print(f"Train loss: {train_avg:.4f}  |  Train perplexity: {train_perp:.2f}")
        print(f"Val   loss: {val_avg:.4f}  |  Val   perplexity: {val_perp:.2f}")
        print("  (perplexity = exp(loss); lower train+val indicates better fit and generalization)")

    return all_batch_losses, train_epoch_losses, val_epoch_losses, train_perplexities, val_perplexities


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


def run_demo_with_visualizations() -> None:
    set_seed(7)

    print("=" * 92)
    print("Instruction Tuning with Visualizations")
    print("=" * 92)

    # Step 1) Data setup
    corpus = build_base_corpus()
    pairs = build_instruction_dataset(corpus)

    print("\nStep 1: Base corpus")
    for i, line in enumerate(corpus, start=1):
        print(f"  {i}. {line}")

    print("\nStep 2: Instruction-response pairs")
    for i, (p, r) in enumerate(pairs, start=1):
        print(f"  {i}. prompt='{p}' | response='{r}'")

    # Step 3) Tokenizer setup
    tokenizer = BasicTokenizer()
    tokenizer.fit(build_tokenizer_texts(corpus, pairs))

    print("\nStep 3: Vocabulary")
    for tok, idx in tokenizer.token_to_id.items():
        print(f"  {tok:>14} -> {idx}")

    # Hyperparameter basics (Phase-3 style training hygiene).
    hyper = HyperParams()
    print("\nStep 4: Hyperparameter basics")
    print(f"  epochs={hyper.epochs}         -> number of full passes over train split")
    print(f"  batch_size={hyper.batch_size}    -> samples per optimizer step")
    print(f"  lr={hyper.lr}          -> gradient step size")
    print(f"  weight_decay={hyper.weight_decay} -> L2-style regularization")
    print(f"  train_ratio={hyper.train_ratio}  -> fraction used for training")
    print(f"  max_len={hyper.max_len}       -> context window length")

    # Step 5) Build tensors and split into train/validation sets.
    all_inputs, all_labels = build_instruction_tensors(pairs, tokenizer, max_len=hyper.max_len)
    train_inputs, train_labels, val_inputs, val_labels = split_train_validation(
        all_inputs,
        all_labels,
        train_ratio=hyper.train_ratio,
        seed=7,
    )

    print("\nStep 5: Tensor shapes + split")
    print(f"  all_inputs:   {tuple(all_inputs.shape)} = (batch, seq_len)")
    print(f"  train_inputs: {tuple(train_inputs.shape)}")
    print(f"  val_inputs:   {tuple(val_inputs.shape)}")
    print("  label=-100 means loss is ignored (prompt part and padding).")
    
    # Console outputs for concepts A & B
    print_label_mask_table(pairs, tokenizer, hyper.max_len)
    explain_teacher_forcing(pairs, tokenizer, hyper.max_len)

    # Visualization A & B
    print("\n📊 Generating visualizations...")
    visualize_token_label_mask(pairs, tokenizer, hyper.max_len)
    visualize_teacher_forcing_concept()

    # Step 6) Build model
    cfg = ModelConfig(
        vocab_size=len(tokenizer.token_to_id),
        seq_len=hyper.max_len - 1,
        d_model=8,
        num_heads=2,
        d_ff=16,
    )
    model = TinyInstructionTransformer(cfg)

    print("\nStep 6: Model config")
    print(f"  d_model={cfg.d_model}, num_heads={cfg.num_heads}, d_head={cfg.d_model // cfg.num_heads}, d_ff={cfg.d_ff}")
    print("  Full finetuning mode: all parameters are trainable.")
    
    # Embedding similarity before training
    inspect_embedding_similarity(model, tokenizer, "BEFORE training")
    visualize_embedding_similarity(model, tokenizer, "BEFORE training", "03_embedding_similarity_before.png")

    # Attention before training
    print("\n" + "-" * 92)
    print("Step 7: Before training (random attention behavior)")
    visualize_attention_heatmaps(
        model, tokenizer, "Complete the phrase: i love", hyper.max_len,
        "BEFORE training", "04a_attention_before.png"
    )

    # Train and collect train/validation losses.
    print("\nStep 8: Training with validation monitoring...")
    batch_losses, train_epoch_losses, val_epoch_losses, train_perplexities, val_perplexities = train_with_validation(
        model=model,
        train_inputs=train_inputs,
        train_labels=train_labels,
        val_inputs=val_inputs,
        val_labels=val_labels,
        hyper=hyper,
    )

    # Visualize learning curves with train/validation split.
    visualize_learning_curves(
        batch_losses,
        train_epoch_losses,
        val_epoch_losses,
        train_perplexities,
        val_perplexities,
    )

    # Attention after training
    print("\n" + "-" * 92)
    print("Step 9: After training (structured attention)")
    visualize_attention_heatmaps(
        model, tokenizer, "Complete the phrase: i love", hyper.max_len,
        "AFTER training", "04b_attention_after.png"
    )

    # Embedding similarity after training
    inspect_embedding_similarity(model, tokenizer, "AFTER training")
    visualize_embedding_similarity(model, tokenizer, "AFTER training", "03_embedding_similarity_after.png")

    # Head divergence
    compare_attention_heads(
        model, tokenizer, "Complete the phrase: i love", hyper.max_len, "AFTER training"
    )

    # Step 10) Inference
    print("\n" + "=" * 92)
    print("Step 10: Inference demo (prompt -> generated response)")
    print("=" * 92)

    test_prompts = [
        "Complete the phrase: i love",
        "What do transformers use?",
        "What helps models focus?",
    ]
    for p in test_prompts:
        g = generate_response(model, tokenizer, p, max_len=hyper.max_len, max_new_tokens=hyper.max_new_tokens)
        print(f"  prompt='{p}'")
        print(f"  response='{g}'")

    print("\n" + "=" * 92)
    print("Recap — NLP-specific concepts demonstrated:")
    print("  A. Token/label/mask table    : which positions contribute to loss (VISUALIZATION)")
    print("  B. Teacher forcing           : training vs inference gap (VISUALIZATION)")
    print("  C. Embedding similarity      : token clustering before/after (VISUALIZATION)")
    print("  D. Multi-head divergence     : each head learns different patterns (VISUALIZATION)")
    print("  E. Train/Val curves          : monitor fitting vs generalization (VISUALIZATION)")
    print("  F. Hyperparameter basics     : lr, batch size, epochs, train split")
    print("  Standard pipeline            : causal LM shift, assistant-only loss mask, full-finetune")
    print("=" * 92)
    print("\n✓ All visualizations saved as PNG files in the current directory!")
    print("  - 01_token_label_mask_visualization.png")
    print("  - 02_teacher_forcing_visualization.png")
    print("  - 03_embedding_similarity_before.png")
    print("  - 03_embedding_similarity_after.png")
    print("  - 04a_attention_before.png")
    print("  - 04b_attention_after.png")
    print("  - 05_learning_curves.png")


if __name__ == "__main__":
    run_demo_with_visualizations()

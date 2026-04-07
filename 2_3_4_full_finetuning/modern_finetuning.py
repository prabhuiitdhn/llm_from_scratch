"""
Modern finetuning from scratch (step-by-step, senior-level roadmap + runnable code).
Modern Finetuning Roadmap (Beginner -> Expert)
====================================================================================================

[Beginner - Core Supervised Finetuning]
  1. Full SFT (all weights trainable)
  2. Continued pretraining / domain-adaptive pretraining
  3. Hyperparameter basics: lr, batch size, epochs, weight decay
  4. Train/validation splits, early stopping, checkpoints

[Intermediate - Parameter-Efficient Finetuning (PEFT)]
  5. BitFit (bias-only)
  6. Adapters / Houlsby / Pfeiffer adapters
  7. LoRA (rank-decomposed updates)
  8. QLoRA (4-bit base + LoRA)
  9. Prefix Tuning
 10. Prompt Tuning / Soft Prompt / P-Tuning v2
 11. IA3 (activation rescaling vectors)

[Advanced - LoRA Family + Efficiency Tricks]
 12. AdaLoRA (adaptive rank)
 13. DoRA (direction + magnitude decomposition)
 14. VeRA / LoHa / LoKr / OFT / BOFT variants
 15. ReLoRA (periodic reset + merge)
 16. Optimizer/state sharding (ZeRO/FSDP) for large-scale FT

[Advanced Alignment - Preference Finetuning]
 17. Reward modeling
 18. PPO-based RLHF
 19. DPO / cDPO / rDPO
 20. IPO / KTO / ORPO / SimPO / CPO
 21. RLAIF (AI-feedback alignment)

[Expert - Frontier Production Techniques]
 22. Multi-stage pipelines: SFT -> Preference tuning -> Distillation
 23. Distillation to small student models
 24. MoE finetuning and router adaptation
 25. Continual finetuning with anti-forgetting constraints
 26. Multimodal finetuning (vision-language adapters + cross-attention)
 27. Safety finetuning / constitutional and policy tuning

Implemented in THIS script (hands-on):
	- Full FT, BitFit, Adapters (sequential/parallel), LoRA, QLoRA, Prefix Tuning, Soft Prompt Tuning
  
What you get in this file:
1) A practical roadmap of modern finetuning methods from beginner to expert.
2) A tiny decoder-only LM built from scratch in PyTorch.
3) Train/validation split with tracked loss/perplexity.
4) Implemented methods (popular and effective):
   - Full Finetuning (all params trainable)
   - BitFit (bias-only finetuning)
	- Adapters (bottleneck residual modules on frozen base)
	- Parallel Adapters (adapter branch in parallel to each sublayer)
   - LoRA (parameter-efficient low-rank adapters)
	- QLoRA (4-bit frozen base linears + LoRA adapters)
	- Prefix Tuning (trainable attention prefixes with frozen base)
   - Soft Prompt Tuning (learnable virtual prompt embeddings)
5) Side-by-side comparison of trainable parameters and validation metrics.

This is educational code optimized for clarity, not benchmark performance.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 7) -> None:
	torch.manual_seed(seed)


def print_modern_finetuning_roadmap() -> None:
	"""Senior-level map of modern finetuning methods by difficulty."""
	print("=" * 100)
	print("Modern Finetuning Roadmap (Beginner -> Expert)")
	print("=" * 100)

	print("\n[Beginner - Core Supervised Finetuning]")
	print("  1. Full SFT (all weights trainable)")
	print("  2. Continued pretraining / domain-adaptive pretraining")
	print("  3. Hyperparameter basics: lr, batch size, epochs, weight decay")
	print("  4. Train/validation splits, early stopping, checkpoints")

	print("\n[Intermediate - Parameter-Efficient Finetuning (PEFT)]")
	print("  5. BitFit (bias-only)")
	print("  6. Adapters / Houlsby / Pfeiffer adapters")
	print("  7. LoRA (rank-decomposed updates)")
	print("  8. QLoRA (4-bit base + LoRA)")
	print("  9. Prefix Tuning")
	print(" 10. Prompt Tuning / Soft Prompt / P-Tuning v2")
	print(" 11. IA3 (activation rescaling vectors)")

	print("\n[Advanced - LoRA Family + Efficiency Tricks]")
	print(" 12. AdaLoRA (adaptive rank)")
	print(" 13. DoRA (direction + magnitude decomposition)")
	print(" 14. VeRA / LoHa / LoKr / OFT / BOFT variants")
	print(" 15. ReLoRA (periodic reset + merge)")
	print(" 16. Optimizer/state sharding (ZeRO/FSDP) for large-scale FT")

	print("\n[Advanced Alignment - Preference Finetuning]")
	print(" 17. Reward modeling")
	print(" 18. PPO-based RLHF")
	print(" 19. DPO / cDPO / rDPO")
	print(" 20. IPO / KTO / ORPO / SimPO / CPO")
	print(" 21. RLAIF (AI-feedback alignment)")

	print("\n[Expert - Frontier Production Techniques]")
	print(" 22. Multi-stage pipelines: SFT -> Preference tuning -> Distillation")
	print(" 23. Distillation to small student models")
	print(" 24. MoE finetuning and router adaptation")
	print(" 25. Continual finetuning with anti-forgetting constraints")
	print(" 26. Multimodal finetuning (vision-language adapters + cross-attention)")
	print(" 27. Safety finetuning / constitutional and policy tuning")

	print("\nImplemented in THIS script (hands-on):")
	print("  - Full FT, BitFit, Adapters (sequential/parallel), LoRA, QLoRA, Prefix Tuning, Soft Prompt Tuning")


class BasicTokenizer:
	def __init__(self) -> None:
		self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
		self.token_to_id: Dict[str, int] = {}
		self.id_to_token: Dict[int, str] = {}

	def normalize(self, text: str) -> str:
		text = text.lower().strip()
		return re.sub(r"\s+", " ", text)

	def split(self, text: str) -> List[str]:
		return re.findall(r"\w+|[^\w\s]", text)

	def fit(self, texts: List[str]) -> None:
		vocab = set()
		for t in texts:
			vocab.update(self.split(self.normalize(t)))
		ordered = self.special_tokens + sorted(vocab)
		self.token_to_id = {tok: i for i, tok in enumerate(ordered)}
		self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

	def encode(self, text: str, max_len: int) -> List[int]:
		toks = self.split(self.normalize(text))
		ids = [self.token_to_id["<bos>"]]
		ids.extend(self.token_to_id.get(t, self.token_to_id["<unk>"]) for t in toks)
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
	return [
		"I love NLP",
		"I love machine learning",
		"transformers use attention",
		"attention helps models focus",
		"NLP is fun",
		"attention helps NLP models",
	]


def build_instruction_dataset(corpus: List[str]) -> List[Tuple[str, str]]:
	return [
		("Complete the phrase: i love", "nlp"),
		("Complete the phrase: i love machine", "learning"),
		("What do transformers use?", "transformers use attention"),
		("What helps models focus?", "attention helps models focus"),
		("Complete the phrase: nlp is", "fun"),
		("Complete the phrase: attention helps nlp", "models"),
		("Rewrite as a short fact: " + corpus[0], "i love nlp"),
		("Rewrite as a short fact: " + corpus[3], "attention helps models focus"),
		("What is nlp?", "nlp is fun"),
		("Complete: transformers use", "attention"),
	]


def format_chat(prompt: str, response: str) -> str:
	return f"Instruction: {prompt} Response: {response}"


def format_prefix(prompt: str) -> str:
	return f"Instruction: {prompt} Response:"


def build_tokenizer_texts(corpus: List[str], pairs: List[Tuple[str, str]]) -> List[str]:
	texts = list(corpus)
	for p, r in pairs:
		texts.append(format_chat(p, r))
		texts.append(format_prefix(p))
	return texts


def build_instruction_tensors(
	pairs: List[Tuple[str, str]],
	tokenizer: BasicTokenizer,
	max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Build assistant-only SFT tensors (prompt tokens masked with -100)."""
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
			if i < prefix_token_count or labels[i] == pad_id:
				labels[i] = -100

		all_inputs.append(input_ids)
		all_labels.append(labels)

	return torch.tensor(all_inputs, dtype=torch.long), torch.tensor(all_labels, dtype=torch.long)


def split_train_validation(
	inputs: torch.Tensor,
	labels: torch.Tensor,
	train_ratio: float,
	seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	n = inputs.size(0)
	g = torch.Generator().manual_seed(seed)
	order = torch.randperm(n, generator=g)
	n_train = max(1, int(n * train_ratio))
	n_train = min(n_train, n - 1)
	train_idx = order[:n_train]
	val_idx = order[n_train:]
	return inputs[train_idx], labels[train_idx], inputs[val_idx], labels[val_idx]


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
	return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


@dataclass
class ModelConfig:
	vocab_size: int
	seq_len: int
	d_model: int = 24
	num_heads: int = 4
	d_ff: int = 64


@dataclass
class HyperParams:
	epochs: int = 6
	batch_size: int = 2
	lr: float = 0.02
	weight_decay: float = 0.0
	train_ratio: float = 0.8
	max_len: int = 26
	max_new_tokens: int = 8


class TinyMultiHeadSelfAttention(nn.Module):
	def __init__(self, d_model: int, num_heads: int) -> None:
		super().__init__()
		assert d_model % num_heads == 0
		self.d_model = d_model
		self.num_heads = num_heads
		self.d_head = d_model // num_heads

		self.w_q = nn.Linear(d_model, d_model, bias=True)
		self.w_k = nn.Linear(d_model, d_model, bias=True)
		self.w_v = nn.Linear(d_model, d_model, bias=True)
		self.w_o = nn.Linear(d_model, d_model, bias=True)

	def forward(
		self,
		x: torch.Tensor,
		prefix_k: torch.Tensor | None = None,
		prefix_v: torch.Tensor | None = None,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		bsz, seq_len, _ = x.shape

		q = self.w_q(x)
		k = self.w_k(x)
		v = self.w_v(x)

		q = q.view(bsz, seq_len, self.num_heads, self.d_head).transpose(1, 2)
		k = k.view(bsz, seq_len, self.num_heads, self.d_head).transpose(1, 2)
		v = v.view(bsz, seq_len, self.num_heads, self.d_head).transpose(1, 2)

		prefix_len = 0
		if prefix_k is not None and prefix_v is not None:
			# prefix_k/prefix_v expected shape: [num_heads, prefix_len, d_head]
			prefix_len = prefix_k.size(1)
			pk = prefix_k.unsqueeze(0).expand(bsz, -1, -1, -1)
			pv = prefix_v.unsqueeze(0).expand(bsz, -1, -1, -1)
			k = torch.cat([pk, k], dim=2)
			v = torch.cat([pv, v], dim=2)

		scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
		causal = build_causal_mask(seq_len, x.device)
		if prefix_len > 0:
			prefix_cols = torch.zeros(seq_len, prefix_len, dtype=torch.bool, device=x.device)
			attn_mask = torch.cat([prefix_cols, causal], dim=1)
		else:
			attn_mask = causal
		scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), -1e9)
		weights = F.softmax(scores, dim=-1)
		heads = torch.matmul(weights, v)

		joined = heads.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
		out = self.w_o(joined)
		return out, scores, weights


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
		self.attn = TinyMultiHeadSelfAttention(d_model, num_heads)
		self.norm1 = nn.LayerNorm(d_model)
		self.ffn = FeedForward(d_model, d_ff)
		self.norm2 = nn.LayerNorm(d_model)

	def forward(
		self,
		x: torch.Tensor,
		prefix_k: torch.Tensor | None = None,
		prefix_v: torch.Tensor | None = None,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		attn_out, scores, weights = self.attn(x, prefix_k=prefix_k, prefix_v=prefix_v)
		x = self.norm1(x + attn_out)
		ff = self.ffn(x)
		x = self.norm2(x + ff)
		return x, scores, weights


class TinyDecoderLM(nn.Module):
	def __init__(self, cfg: ModelConfig) -> None:
		super().__init__()
		self.cfg = cfg
		self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
		self.position_embedding = nn.Embedding(cfg.seq_len, cfg.d_model)
		self.block = TransformerBlock(cfg.d_model, cfg.num_heads, cfg.d_ff)
		self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

	def forward(
		self,
		input_ids: torch.Tensor,
		prefix_k: torch.Tensor | None = None,
		prefix_v: torch.Tensor | None = None,
	) -> torch.Tensor:
		bsz, seq_len = input_ids.shape
		pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
		x = self.token_embedding(input_ids) + self.position_embedding(pos)
		x, _, _ = self.block(x, prefix_k=prefix_k, prefix_v=prefix_v)
		return self.lm_head(x)


class LoRALinear(nn.Module):
	"""Linear layer with frozen base weights + trainable low-rank delta."""

	def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0, dropout: float = 0.0) -> None:
		super().__init__()
		self.in_features = base.in_features
		self.out_features = base.out_features
		self.rank = rank
		self.scale = alpha / rank
		self.dropout = nn.Dropout(dropout)

		self.base_weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
		if base.bias is not None:
			self.base_bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
		else:
			self.base_bias = None

		self.A = nn.Parameter(torch.randn(rank, self.in_features) * 0.02)
		self.B = nn.Parameter(torch.zeros(self.out_features, rank))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		base_out = F.linear(x, self.base_weight, self.base_bias)
		lora_out = F.linear(F.linear(self.dropout(x), self.A), self.B) * self.scale
		return base_out + lora_out


class QLoRALinear(nn.Module):
	"""4-bit quantized frozen base linear + trainable LoRA delta."""

	def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0, dropout: float = 0.0) -> None:
		super().__init__()
		self.in_features = base.in_features
		self.out_features = base.out_features
		self.rank = rank
		self.scale = alpha / rank
		self.dropout = nn.Dropout(dropout)

		# Per-output-channel 4-bit affine quantization for educational QLoRA simulation.
		with torch.no_grad():
			w = base.weight.detach().clone().float()
			row_min = w.min(dim=1, keepdim=True).values
			row_max = w.max(dim=1, keepdim=True).values
			row_scale = (row_max - row_min) / 15.0
			row_scale = torch.where(row_scale < 1e-8, torch.full_like(row_scale, 1e-8), row_scale)
			q = torch.round((w - row_min) / row_scale).clamp(0, 15).to(torch.uint8)

		self.register_buffer("qweight", q)
		self.register_buffer("w_min", row_min)
		self.register_buffer("w_scale", row_scale)

		if base.bias is not None:
			self.base_bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
		else:
			self.base_bias = None

		self.A = nn.Parameter(torch.randn(rank, self.in_features) * 0.02)
		self.B = nn.Parameter(torch.zeros(self.out_features, rank))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		base_weight = self.qweight.float() * self.w_scale + self.w_min
		base_out = F.linear(x, base_weight, self.base_bias)
		lora_out = F.linear(F.linear(self.dropout(x), self.A), self.B) * self.scale
		return base_out + lora_out


class AdapterModule(nn.Module):
	"""Classic bottleneck adapter with residual connection."""

	def __init__(self, d_model: int, bottleneck: int = 8) -> None:
		super().__init__()
		self.down = nn.Linear(d_model, bottleneck)
		self.act = nn.ReLU()
		self.up = nn.Linear(bottleneck, d_model)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x + self.up(self.act(self.down(x)))


class SoftPromptWrapper(nn.Module):
	"""Learn virtual prompt embeddings while freezing base model."""

	def __init__(self, base_model: TinyDecoderLM, n_soft_tokens: int = 6) -> None:
		super().__init__()
		self.base = base_model
		self.n_soft_tokens = n_soft_tokens
		self.soft_prompt = nn.Parameter(torch.randn(n_soft_tokens, base_model.cfg.d_model) * 0.02)

		for p in self.base.parameters():
			p.requires_grad = False

	def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
		bsz, seq_len = input_ids.shape
		device = input_ids.device

		soft_pos = torch.arange(self.n_soft_tokens, device=device)
		tok_pos = torch.arange(self.n_soft_tokens, self.n_soft_tokens + seq_len, device=device)

		soft_emb = self.soft_prompt.unsqueeze(0).expand(bsz, self.n_soft_tokens, -1)
		soft_emb = soft_emb + self.base.position_embedding(soft_pos).unsqueeze(0)

		tok_emb = self.base.token_embedding(input_ids) + self.base.position_embedding(tok_pos).unsqueeze(0)

		x = torch.cat([soft_emb, tok_emb], dim=1)
		x, _, _ = self.base.block(x)
		logits_full = self.base.lm_head(x)
		return logits_full[:, self.n_soft_tokens :, :]


class PrefixTuningWrapper(nn.Module):
	"""Learn trainable attention prefixes while freezing base model."""

	def __init__(self, base_model: TinyDecoderLM, prefix_len: int = 6) -> None:
		super().__init__()
		self.base = base_model
		self.prefix_len = prefix_len
		n_heads = base_model.cfg.num_heads
		d_head = base_model.cfg.d_model // base_model.cfg.num_heads

		for p in self.base.parameters():
			p.requires_grad = False

		self.prefix_k = nn.Parameter(torch.randn(n_heads, prefix_len, d_head) * 0.02)
		self.prefix_v = nn.Parameter(torch.randn(n_heads, prefix_len, d_head) * 0.02)

	def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
		return self.base(input_ids, prefix_k=self.prefix_k, prefix_v=self.prefix_v)


class AdapterTuningWrapper(nn.Module):
	"""Freeze base model and train small adapters around attention/FFN blocks."""

	def __init__(self, base_model: TinyDecoderLM, bottleneck: int = 8) -> None:
		super().__init__()
		self.base = base_model
		for p in self.base.parameters():
			p.requires_grad = False

		d_model = self.base.cfg.d_model
		self.attn_adapter = AdapterModule(d_model=d_model, bottleneck=bottleneck)
		self.ffn_adapter = AdapterModule(d_model=d_model, bottleneck=bottleneck)

	def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
		bsz, seq_len = input_ids.shape
		pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
		x = self.base.token_embedding(input_ids) + self.base.position_embedding(pos)

		attn_out, _, _ = self.base.block.attn(x)
		x = self.base.block.norm1(x + attn_out)
		x = self.attn_adapter(x)

		ff = self.base.block.ffn(x)
		x = self.base.block.norm2(x + ff)
		x = self.ffn_adapter(x)

		return self.base.lm_head(x)


class AdapterParallelWrapper(nn.Module):
	"""Freeze base model and add adapter branches in parallel to each sublayer."""

	def __init__(self, base_model: TinyDecoderLM, bottleneck: int = 8) -> None:
		super().__init__()
		self.base = base_model
		for p in self.base.parameters():
			p.requires_grad = False

		d_model = self.base.cfg.d_model
		self.attn_adapter = AdapterModule(d_model=d_model, bottleneck=bottleneck)
		self.ffn_adapter = AdapterModule(d_model=d_model, bottleneck=bottleneck)

	def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
		bsz, seq_len = input_ids.shape
		pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
		x = self.base.token_embedding(input_ids) + self.base.position_embedding(pos)

		attn_out, _, _ = self.base.block.attn(x)
		attn_parallel = self.attn_adapter(x) - x
		x = self.base.block.norm1(x + attn_out + attn_parallel)

		ff = self.base.block.ffn(x)
		ffn_parallel = self.ffn_adapter(x) - x
		x = self.base.block.norm2(x + ff + ffn_parallel)

		return self.base.lm_head(x)


@dataclass
class ExperimentResult:
	method: str
	trainable_params: int
	total_params: int
	trainable_percent: float
	best_val_loss: float
	best_val_ppl: float
	sample_output: str


def count_params(model: nn.Module) -> Tuple[int, int, float]:
	total = sum(p.numel() for p in model.parameters())
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	pct = 100.0 * trainable / max(total, 1)
	return trainable, total, pct


def freeze_all_params(model: nn.Module) -> None:
	for p in model.parameters():
		p.requires_grad = False


def enable_bitfit(model: nn.Module) -> None:
	"""BitFit: only bias parameters are trainable."""
	freeze_all_params(model)
	for name, p in model.named_parameters():
		if name.endswith("bias"):
			p.requires_grad = True


def inject_lora(
	model: nn.Module,
	target_module_names: Tuple[str, ...] = ("w_q", "w_k", "w_v", "w_o", "lm_head"),
	rank: int = 4,
	alpha: float = 8.0,
	dropout: float = 0.0,
) -> None:
	"""Replace target Linear modules with LoRALinear wrappers."""
	for module in model.modules():
		for name, child in list(module.named_children()):
			if isinstance(child, nn.Linear) and name in target_module_names:
				setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))


def inject_qlora(
	model: nn.Module,
	target_module_names: Tuple[str, ...] = ("w_q", "w_k", "w_v", "w_o", "lm_head"),
	rank: int = 4,
	alpha: float = 8.0,
	dropout: float = 0.0,
) -> None:
	"""Replace target Linear modules with QLoRALinear wrappers."""
	for module in model.modules():
		for name, child in list(module.named_children()):
			if isinstance(child, nn.Linear) and name in target_module_names:
				setattr(module, name, QLoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))


def evaluate_split(
	model: nn.Module,
	inputs: torch.Tensor,
	labels: torch.Tensor,
	batch_size: int,
) -> Tuple[float, float]:
	model.eval()
	total_loss = 0.0
	total_items = 0
	with torch.no_grad():
		for start in range(0, inputs.size(0), batch_size):
			x = inputs[start : start + batch_size]
			y = labels[start : start + batch_size]
			logits = model(x)
			loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
			total_loss += loss.item() * x.size(0)
			total_items += x.size(0)

	avg_loss = total_loss / max(total_items, 1)
	ppl = math.exp(min(avg_loss, 20))
	return avg_loss, ppl


def train_with_validation(
	model: nn.Module,
	train_inputs: torch.Tensor,
	train_labels: torch.Tensor,
	val_inputs: torch.Tensor,
	val_labels: torch.Tensor,
	hyper: HyperParams,
	title: str,
) -> Tuple[float, float]:
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.Adam(params, lr=hyper.lr, weight_decay=hyper.weight_decay)
	n = train_inputs.size(0)

	best_val_loss = float("inf")
	best_val_ppl = float("inf")

	print("\n" + "-" * 100)
	print(f"Training Method: {title}")
	print("-" * 100)

	for epoch in range(hyper.epochs):
		model.train()
		order = torch.randperm(n)
		running = 0.0

		for start in range(0, n, hyper.batch_size):
			idx = order[start : start + hyper.batch_size]
			x = train_inputs[idx]
			y = train_labels[idx]

			logits = model(x)
			loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			running += loss.item() * x.size(0)

		train_loss = running / n
		train_ppl = math.exp(min(train_loss, 20))
		val_loss, val_ppl = evaluate_split(model, val_inputs, val_labels, hyper.batch_size)

		best_val_loss = min(best_val_loss, val_loss)
		best_val_ppl = min(best_val_ppl, val_ppl)

		print(
			f"  epoch {epoch + 1:>2}/{hyper.epochs} | "
			f"train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} | "
			f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}"
		)

	return best_val_loss, best_val_ppl


def generate_response(
	model: nn.Module,
	tokenizer: BasicTokenizer,
	prompt: str,
	max_len: int,
	max_new_tokens: int,
) -> str:
	model.eval()
	prefix = format_prefix(prompt)
	prefix_ids = tokenizer.encode(prefix, max_len=max_len)

	active: List[int] = []
	for token_id in prefix_ids:
		if token_id == tokenizer.token_to_id["<pad>"]:
			break
		if token_id == tokenizer.token_to_id["<eos>"]:
			continue
		active.append(token_id)

	generated: List[int] = []
	blocked_at_start = {
		tokenizer.token_to_id["<pad>"],
		tokenizer.token_to_id["<bos>"],
		tokenizer.token_to_id["<unk>"],
	}

	for _ in range(max_new_tokens):
		window = active[-(max_len - 1) :]
		padded = window + [tokenizer.token_to_id["<pad>"]] * ((max_len - 1) - len(window))
		x = torch.tensor([padded], dtype=torch.long)

		with torch.no_grad():
			logits = model(x)

		last_pos = len(window) - 1
		row = logits[0, last_pos]
		if not generated:
			top_vals, top_ids = torch.topk(row, k=min(8, row.numel()))
			next_id = int(top_ids[0].item())
			for cand in top_ids.tolist():
				if cand not in blocked_at_start and cand != tokenizer.token_to_id["<eos>"]:
					next_id = int(cand)
					break
		else:
			next_id = int(torch.argmax(row).item())

		active.append(next_id)
		if next_id == tokenizer.token_to_id["<eos>"]:
			break
		generated.append(next_id)

	if not generated:
		return ""
	return tokenizer.decode(generated)


def run_experiment(
	method_name: str,
	base_state: Dict[str, torch.Tensor],
	cfg: ModelConfig,
	tokenizer: BasicTokenizer,
	train_inputs: torch.Tensor,
	train_labels: torch.Tensor,
	val_inputs: torch.Tensor,
	val_labels: torch.Tensor,
	hyper: HyperParams,
) -> ExperimentResult:
	base_model = TinyDecoderLM(cfg)
	base_model.load_state_dict(base_state)

	model_for_training: nn.Module
	if method_name == "full_finetune":
		model_for_training = base_model
	elif method_name == "bitfit":
		enable_bitfit(base_model)
		model_for_training = base_model
	elif method_name == "adapters":
		model_for_training = AdapterTuningWrapper(base_model, bottleneck=8)
	elif method_name == "adapters_parallel":
		model_for_training = AdapterParallelWrapper(base_model, bottleneck=8)
	elif method_name == "lora":
		freeze_all_params(base_model)
		inject_lora(base_model, rank=4, alpha=8.0, dropout=0.0)
		model_for_training = base_model
	elif method_name == "qlora":
		freeze_all_params(base_model)
		inject_qlora(base_model, rank=4, alpha=8.0, dropout=0.0)
		model_for_training = base_model
	elif method_name == "prefix_tuning":
		model_for_training = PrefixTuningWrapper(base_model, prefix_len=6)
	elif method_name == "soft_prompt":
		model_for_training = SoftPromptWrapper(base_model, n_soft_tokens=6)
	else:
		raise ValueError(f"Unknown method: {method_name}")

	trn, tot, pct = count_params(model_for_training)
	best_val_loss, best_val_ppl = train_with_validation(
		model_for_training,
		train_inputs,
		train_labels,
		val_inputs,
		val_labels,
		hyper,
		title=method_name,
	)

	demo = generate_response(
		model_for_training,
		tokenizer,
		prompt="What do transformers use?",
		max_len=hyper.max_len,
		max_new_tokens=hyper.max_new_tokens,
	)

	return ExperimentResult(
		method=method_name,
		trainable_params=trn,
		total_params=tot,
		trainable_percent=pct,
		best_val_loss=best_val_loss,
		best_val_ppl=best_val_ppl,
		sample_output=demo,
	)


def print_results_table(results: List[ExperimentResult]) -> None:
	print("\n" + "=" * 100)
	print("Comparison: Popular Modern Finetuning Methods")
	print("=" * 100)
	print(
		f"{'method':<14} {'trainable':>10} {'total':>10} {'train%':>9} "
		f"{'best_val_loss':>14} {'best_val_ppl':>13}  sample_output"
	)
	print("-" * 100)
	for r in results:
		print(
			f"{r.method:<14} {r.trainable_params:>10} {r.total_params:>10} "
			f"{r.trainable_percent:>8.2f}% {r.best_val_loss:>14.4f} {r.best_val_ppl:>13.2f}  {r.sample_output}"
		)


def run_modern_finetuning_from_scratch() -> None:
	set_seed(7)
	print_modern_finetuning_roadmap()

	print("\nStep 1: Build dataset")
	corpus = build_base_corpus()
	pairs = build_instruction_dataset(corpus)
	print(f"  corpus size={len(corpus)} | instruction pairs={len(pairs)}")

	print("\nStep 2: Build tokenizer")
	tokenizer = BasicTokenizer()
	tokenizer.fit(build_tokenizer_texts(corpus, pairs))
	print(f"  vocab size={len(tokenizer.token_to_id)}")

	print("\nStep 3: Build tensors + train/validation split")
	hyper = HyperParams()
	all_inputs, all_labels = build_instruction_tensors(pairs, tokenizer, max_len=hyper.max_len)
	train_inputs, train_labels, val_inputs, val_labels = split_train_validation(
		all_inputs,
		all_labels,
		train_ratio=hyper.train_ratio,
		seed=7,
	)
	print(f"  all: {tuple(all_inputs.shape)}")
	print(f"  train: {tuple(train_inputs.shape)} | val: {tuple(val_inputs.shape)}")

	print("\nStep 4: Hyperparameter basics")
	print(f"  epochs={hyper.epochs}, batch_size={hyper.batch_size}, lr={hyper.lr}, weight_decay={hyper.weight_decay}")
	print(f"  train_ratio={hyper.train_ratio}, max_len={hyper.max_len}, max_new_tokens={hyper.max_new_tokens}")

	print("\nStep 5: Create base model state for fair method comparison")
	cfg = ModelConfig(
		vocab_size=len(tokenizer.token_to_id),
		seq_len=hyper.max_len - 1 + 6,
		d_model=24,
		num_heads=4,
		d_ff=64,
	)
	base_model = TinyDecoderLM(cfg)
	base_state = copy.deepcopy(base_model.state_dict())

	print("\nStep 6: Run modern finetuning methods")
	methods = ["full_finetune", "bitfit", "adapters", "adapters_parallel", "lora", "qlora", "prefix_tuning", "soft_prompt"]
	results: List[ExperimentResult] = []
	for m in methods:
		results.append(
			run_experiment(
				method_name=m,
				base_state=base_state,
				cfg=cfg,
				tokenizer=tokenizer,
				train_inputs=train_inputs,
				train_labels=train_labels,
				val_inputs=val_inputs,
				val_labels=val_labels,
				hyper=hyper,
			)
		)

	print_results_table(results)

	print("\nNext learning sequence:")
	print("  1) Add DPO on preference pairs")
	print("  2) Add early stopping + best checkpoint restoration")
	print("  3) Add gradient accumulation + mixed precision")
	print("  4) Replace educational QLoRA with bitsandbytes-backed QLoRA on a real LLM")


if __name__ == "__main__":
	run_modern_finetuning_from_scratch()

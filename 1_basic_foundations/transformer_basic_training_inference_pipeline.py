"""
Transformer basic training + inference pipeline (beginner-friendly).

This file keeps the same learning style and tiny architecture idea as
transformer_basic.py, but adds:
1) a real training loop for 2 epochs
2) next-token loss computation
3) a simple greedy inference pipeline
4) attention-score inspection before and after training

Design goals:
- small and readable
- same tiny decoder-only transformer idea
- same small dataset style
- explicit shapes and explanations

This is intentionally educational, not optimized.
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
	"""Very small tokenizer matching the style of transformer_basic.py."""

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

	def fit(self, corpus: List[str]) -> None:
		vocab = set()
		for line in corpus:
			vocab.update(self.split(self.normalize(line)))

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


def build_sample_corpus() -> List[str]:
	return [
		"I love NLP",
		"I love machine learning",
		"transformers use attention",
		"attention helps models focus",
		"NLP is fun",
		"attention helps NLP models",
	]


def build_training_tensors(
	corpus: List[str], tokenizer: BasicTokenizer, max_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	For language modeling, input is sequence[:-1] and target is sequence[1:].

	Example:
	full ids    = [<bos>, i, love, nlp, <eos>, <pad>, ...]
	input_ids   = [<bos>, i, love, nlp, <eos>, ...]
	target_ids  = [i, love, nlp, <eos>, <pad>, ...]
	"""
	encoded = [tokenizer.encode(text, max_len=max_len) for text in corpus]
	inputs = [row[:-1] for row in encoded]
	targets = [row[1:] for row in encoded]
	return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
	"""True means: this future position is blocked."""
	return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


@dataclass
class ModelConfig:
	vocab_size: int
	max_len: int
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
		- output: (batch, seq_len, d_model)
		- scores: (batch, num_heads, seq_len, seq_len) raw masked scores
		- weights: (batch, num_heads, seq_len, seq_len) softmax attention weights
		"""
		batch_size, seq_len, _ = x.shape

		q = self.w_q(x)
		k = self.w_k(x)
		v = self.w_v(x)

		q = q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
		k = k.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
		v = v.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

		scale = math.sqrt(self.d_head)
		scores = torch.matmul(q, k.transpose(-2, -1)) / scale

		mask = causal_mask(seq_len, x.device)
		scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)

		weights = F.softmax(scores, dim=-1)
		head_output = torch.matmul(weights, v)

		joined = head_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
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
		ffn_out = self.ffn(x)
		x = self.norm2(x + ffn_out)
		return x, scores, weights


class TinyTransformerLM(nn.Module):
	def __init__(self, config: ModelConfig) -> None:
		super().__init__()
		self.config = config
		self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
		self.position_embedding = nn.Embedding(config.max_len, config.d_model)
		self.block = TransformerBlock(config.d_model, config.num_heads, config.d_ff)
		self.lm_head = nn.Linear(config.d_model, config.vocab_size)

	def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		input_ids shape: (batch, seq_len)

		returns:
		- logits
		- hidden states after embeddings
		- attention scores
		- attention weights
		"""
		batch_size, seq_len = input_ids.shape
		positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

		x = self.token_embedding(input_ids) + self.position_embedding(positions)
		hidden, scores, weights = self.block(x)
		logits = self.lm_head(hidden)
		return logits, x, scores, weights


def format_vector(values: List[float], decimals: int = 3) -> str:
	return "[" + ", ".join(f"{v:.{decimals}f}" for v in values) + "]"


def top_k_tokens(logits_row: torch.Tensor, tokenizer: BasicTokenizer, k: int = 5) -> List[str]:
	probs = F.softmax(logits_row, dim=-1)
	values, indices = torch.topk(probs, k)
	lines = []
	for prob, idx in zip(values.tolist(), indices.tolist()):
		lines.append(f"{tokenizer.id_to_token[idx]}={prob:.3f}")
	return lines


def inspect_attention(
	model: TinyTransformerLM,
	tokenizer: BasicTokenizer,
	text: str,
	max_len: int,
	title: str,
) -> None:
	model.eval()
	ids = tokenizer.encode(text, max_len=max_len)
	input_ids = torch.tensor([ids[:-1]], dtype=torch.long)

	with torch.no_grad():
		logits, embeddings, scores, weights = model(input_ids)

	tokens = [tokenizer.id_to_token[idx] for idx in input_ids[0].tolist()]
	head0_scores = scores[0, 0].tolist()
	head0_weights = weights[0, 0].tolist()

	print("\n" + "-" * 88)
	print(title)
	print("Tokens used for inspection:", tokens)
	print(f"Embedding tensor shape: {tuple(embeddings.shape)} = (batch, seq_len, d_model)")

	first_token_embedding = embeddings[0, 0].tolist()
	print("First token embedding example:", format_vector(first_token_embedding, 4))

	print("\nHead-0 raw attention scores:")
	for i, row in enumerate(head0_scores):
		print(f"  query position {i} ({tokens[i]}): {format_vector(row, 3)}")

	print("\nHead-0 attention weights after softmax:")
	for i, row in enumerate(head0_weights):
		print(f"  query position {i} ({tokens[i]}): {format_vector(row, 3)}")

	print("\nTop next-token probabilities at the final visible position:")
	last_non_pad = 0
	pad_id = tokenizer.token_to_id["<pad>"]
	for i, idx in enumerate(input_ids[0].tolist()):
		if idx != pad_id:
			last_non_pad = i
	print("  ", ", ".join(top_k_tokens(logits[0, last_non_pad], tokenizer, k=5)))


def train_for_two_epochs(
	model: TinyTransformerLM,
	train_inputs: torch.Tensor,
	train_targets: torch.Tensor,
	pad_id: int,
	learning_rate: float = 0.05,
	batch_size: int = 2,
) -> None:
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	num_examples = train_inputs.size(0)

	for epoch in range(2):
		model.train()
		permutation = torch.randperm(num_examples)
		epoch_loss = 0.0

		print("\n" + "=" * 88)
		print(f"Epoch {epoch + 1}/2")
		print("=" * 88)

		for start in range(0, num_examples, batch_size):
			indices = permutation[start:start + batch_size]
			batch_inputs = train_inputs[indices]
			batch_targets = train_targets[indices]

			logits, _, _, _ = model(batch_inputs)
			loss = F.cross_entropy(
				logits.reshape(-1, logits.size(-1)),
				batch_targets.reshape(-1),
				ignore_index=pad_id,
			)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item() * batch_inputs.size(0)
			print(f"  batch {start // batch_size + 1}: loss={loss.item():.4f}")

		avg_loss = epoch_loss / num_examples
		print(f"Average epoch loss: {avg_loss:.4f}")


def generate_text(
	model: TinyTransformerLM,
	tokenizer: BasicTokenizer,
	prompt: str,
	max_len: int,
	max_new_tokens: int = 4,
) -> str:
	model.eval()
	ids = tokenizer.encode(prompt, max_len=max_len)

	# Keep only the meaningful prompt prefix, not the full padded tail.
	prompt_ids = []
	for token_id in ids:
		prompt_ids.append(token_id)
		if token_id == tokenizer.token_to_id["<eos>"]:
			break

	current = prompt_ids[:-1]  # remove eos so generation can continue

	for _ in range(max_new_tokens):
		input_window = current[-(max_len - 1):]
		padded = input_window + [tokenizer.token_to_id["<pad>"]] * ((max_len - 1) - len(input_window))
		input_tensor = torch.tensor([padded], dtype=torch.long)

		with torch.no_grad():
			logits, _, _, _ = model(input_tensor)

		last_pos = len(input_window) - 1
		next_token_id = int(torch.argmax(logits[0, last_pos]).item())
		current.append(next_token_id)

		if next_token_id == tokenizer.token_to_id["<eos>"]:
			break

	return tokenizer.decode(current)


def run_demo() -> None:
	set_seed(7)

	corpus = build_sample_corpus()
	tokenizer = BasicTokenizer()
	tokenizer.fit(corpus)

	max_len = 8
	train_inputs, train_targets = build_training_tensors(corpus, tokenizer, max_len=max_len)

	config = ModelConfig(
		vocab_size=len(tokenizer.token_to_id),
		max_len=max_len - 1,
		d_model=8,
		num_heads=2,
		d_ff=16,
	)
	model = TinyTransformerLM(config)

	print("=" * 88)
	print("Transformer Training + Inference Pipeline")
	print("=" * 88)

	print("\nSample corpus:")
	for i, line in enumerate(corpus, start=1):
		print(f"  {i}. {line}")

	print("\nVocabulary:")
	for token, idx in tokenizer.token_to_id.items():
		print(f"  {token:>12} -> {idx}")

	print("\nTraining tensor shapes:")
	print(f"  train_inputs shape: {tuple(train_inputs.shape)} = (batch, seq_len)")
	print(f"  train_targets shape: {tuple(train_targets.shape)} = (batch, seq_len)")
	print(f"  d_model={config.d_model}, num_heads={config.num_heads}, d_head={config.d_model // config.num_heads}, d_ff={config.d_ff}")

	inspect_attention(
		model=model,
		tokenizer=tokenizer,
		text="I love attention",
		max_len=max_len,
		title="Before training: how the untrained model attends",
	)

	train_for_two_epochs(
		model=model,
		train_inputs=train_inputs,
		train_targets=train_targets,
		pad_id=tokenizer.token_to_id["<pad>"],
		learning_rate=0.05,
		batch_size=2,
	)

	inspect_attention(
		model=model,
		tokenizer=tokenizer,
		text="I love attention",
		max_len=max_len,
		title="After training: how attention and token probabilities changed",
	)

	print("\n" + "=" * 88)
	print("Inference demo")
	print("=" * 88)
	prompts = [
		"I love",
		"attention helps",
		"NLP is",
	]
	for prompt in prompts:
		generated = generate_text(model, tokenizer, prompt, max_len=max_len, max_new_tokens=4)
		print(f"  prompt='{prompt}' -> generated='{generated}'")

	print("\nBeginner summary:")
	print("  1. Token ids go into embeddings.")
	print("  2. Embeddings go through masked self-attention.")
	print("  3. Attention mixes information from earlier tokens only.")
	print("  4. The feed-forward layer improves each token representation.")
	print("  5. The LM head converts hidden states into next-token scores.")
	print("  6. Training moves attention and output probabilities toward useful patterns.")


if __name__ == "__main__":
	run_demo()

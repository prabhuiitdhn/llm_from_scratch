"""
Tokenization basics for NLP pipelines (training, validation, inference).

This script is intentionally dependency-light so the concepts are easy to inspect.
It demonstrates:
- basic text normalization and token splitting
- vocabulary building from training data only
- encoding/decoding with special tokens
- sequence padding/truncation and attention masks
- train/validation split and simple batch creation
- inference-time tokenization behavior with unknown tokens
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class EncodedExample:
	"""Single encoded example used for training/validation/inference."""

	text: str
	token_ids: List[int]
	attention_mask: List[int]


class BasicTokenizer:
	"""
	Minimal tokenizer for educational purposes.

	It uses a simple regex split and a word-level vocabulary.
	"""

	PAD = "<pad>"
	UNK = "<unk>"
	BOS = "<bos>"
	EOS = "<eos>"

	def __init__(self) -> None:
		self.special_tokens = [self.PAD, self.UNK, self.BOS, self.EOS]
		self.token_to_id = {tok: idx for idx, tok in enumerate(self.special_tokens)}
		self.id_to_token = {idx: tok for idx, tok in enumerate(self.special_tokens)}
		self.fitted = False

	@staticmethod
	def normalize(text: str) -> str:
		text = text.lower().strip()
		# Collapse repeated whitespace to stabilize tokenization.
		text = re.sub(r"\s+", " ", text) # removing the spaces to 1 space
		return text

	@staticmethod
	def split(text: str) -> List[str]:
		# Words and standalone punctuation as separate tokens.
		return re.findall(r"\w+|[^\w\s]", text) # this pattern is word level tokenization that also captures punctuation as separate tokens

	def fit(self, texts: Sequence[str], min_freq: int = 1, max_vocab: int | None = None) -> None:
		"""Build vocabulary from training texts only."""
		counter: Counter[str] = Counter()
		for text in texts:
			norm = self.normalize(text)
			counter.update(self.split(norm))

		items = [(tok, freq) for tok, freq in counter.items() if freq >= min_freq]
		items.sort(key=lambda x: (-x[1], x[0]))
		if max_vocab is not None:
			items = items[:max_vocab]

		next_id = len(self.token_to_id)
		for tok, _ in items:
			if tok not in self.token_to_id:
				self.token_to_id[tok] = next_id
				self.id_to_token[next_id] = tok
				next_id += 1

		self.fitted = True

	def encode(self, text: str, max_length: int) -> EncodedExample:
		"""
		Convert text into token IDs plus attention mask.

		Sequence format: <bos> ...tokens... <eos>
		"""
		if not self.fitted:
			raise RuntimeError("Tokenizer is not fitted. Call fit() with training texts first.")

		norm = self.normalize(text)
		tokens = self.split(norm)

		ids = [self.token_to_id[self.BOS]]
		unk_id = self.token_to_id[self.UNK]
		ids.extend(self.token_to_id.get(tok, unk_id) for tok in tokens)
		ids.append(self.token_to_id[self.EOS])

		# Truncate while preserving the final EOS token.
		if len(ids) > max_length:
			ids = ids[: max_length - 1] + [self.token_to_id[self.EOS]]

		attn = [1] * len(ids)

		pad_id = self.token_to_id[self.PAD]
		while len(ids) < max_length:
			ids.append(pad_id)
			attn.append(0)

		return EncodedExample(text=text, token_ids=ids, attention_mask=attn)

	def decode(self, token_ids: Sequence[int], skip_special: bool = True) -> str:
		tokens = []
		special = set(self.special_tokens)
		for idx in token_ids:
			tok = self.id_to_token.get(idx, self.UNK)
			if skip_special and tok in special:
				continue
			tokens.append(tok)
		return " ".join(tokens)


def train_val_split(data: Sequence[str], val_ratio: float = 0.25, seed: int = 7) -> Tuple[List[str], List[str]]:
	"""Simple deterministic split used to mimic training/validation pipelines."""
	items = list(data)
	rnd = random.Random(seed)
	rnd.shuffle(items)

	val_size = max(1, int(len(items) * val_ratio))
	val_data = items[:val_size]
	train_data = items[val_size:]
	return train_data, val_data


def encode_dataset(tokenizer: BasicTokenizer, texts: Iterable[str], max_length: int) -> List[EncodedExample]:
	return [tokenizer.encode(text, max_length=max_length) for text in texts]


def create_batches(examples: Sequence[EncodedExample], batch_size: int) -> List[List[EncodedExample]]:
	return [list(examples[i : i + batch_size]) for i in range(0, len(examples), batch_size)]


def print_batch_preview(batch: Sequence[EncodedExample], max_items: int = 2) -> None:
	print(f"Batch size: {len(batch)}")
	for i, ex in enumerate(batch[:max_items]):
		print(f"  Example {i + 1}")
		print(f"    Text: {ex.text}")
		print(f"    Token IDs: {ex.token_ids}")
		print(f"    Attention: {ex.attention_mask}")


def run_demo() -> None:
	"""End-to-end demonstration across training, validation, and inference."""
	corpus = [
		# Mixed casing — tests lowercasing normalization.
		"LoRA, LORA and lora all refer to Low-Rank Adaptation in fine-tuning.",

		# Numbers, versions, dates — tests digit handling.
		"GPT-4 was released on 2023-03-14; its context window is 128,000 tokens.",

		# URL and email — tests placeholder normalization.
		"See https://huggingface.co/docs or email support@openai.com for details.",

		# Repeated and irregular punctuation — tests punctuation normalization.
		"Wait... WHAT?!?! The loss jumped from 0.42 to 2.87 overnight???",

		# Contractions and possessives — tests split boundary handling.
		"It's the model's job to understand what we're asking; don't ignore that.",

		# Domain-specific abbreviations — tests OOV and UNK behavior.
		"RLHF, DPO, SFT, QLoRA, PEFT, BPE, and RAG are core GenAI terms.",

		# Unicode accent characters — tests unicode normalization.
		"The résumé of Dr. Müller mentions naïve Bayes and café reviews.",

		# Emoji and symbols — tests emoji/symbol policy.
		"Training loss ↓ 📉 while validation accuracy ↑ 🎯 after 3 epochs.",

		# Code-like tokens and underscores — tests special character handling.
		"Call model.generate(input_ids, max_new_tokens=256, temperature=0.7).",

		# Long sequence — tests truncation behavior.
		(
			"Tokenization is not just splitting by spaces: it handles contractions, "
			"punctuation, numbers, symbols, emojis, domain terms, code identifiers, "
			"URL patterns, multilingual scripts, and rare vocabulary all at once — "
			"and it must behave identically at train time and at inference time."
		),

		# Short/minimal sequence — tests padding behavior.
		"Pad me.",

		# Noisy real-world input — tests robustness normalization.
		"plz fix  the   bug asap!!!  its   breaking   prod  rn  😤",
	]

	print("=" * 72)
	print("1) Split data into training and validation")
	train_texts, val_texts = train_val_split(corpus, val_ratio=0.25, seed=42)
	print(f"Train size: {len(train_texts)} | Validation size: {len(val_texts)}")

	print("\n" + "=" * 72)
	print("2) Fit tokenizer on training only")
	tokenizer = BasicTokenizer()
	tokenizer.fit(train_texts, min_freq=1)
	print(f"Vocab size (including special tokens): {len(tokenizer.token_to_id)}")
	print(f"Special tokens: {tokenizer.special_tokens}")

	max_len = 24  # Increased to show more realistic truncation behavior.

	print("\n" + "=" * 72)
	print("3) Encode training and validation")
	train_encoded = encode_dataset(tokenizer, train_texts, max_length=max_len)
	val_encoded = encode_dataset(tokenizer, val_texts, max_length=max_len)
	print(f"Encoded train examples: {len(train_encoded)}")
	print(f"Encoded val examples: {len(val_encoded)}")

	print("\n" + "=" * 72)
	print("4) Create batches for a training loop")
	train_batches = create_batches(train_encoded, batch_size=2)
	print(f"Total train batches: {len(train_batches)}")
	print_batch_preview(train_batches[0])

	print("\n" + "=" * 72)
	print("5) Inference-time behavior (including unknown tokens, symbols, domain terms)")
	inference_text = "ORPO & DPO at lr=1e-4: loss↓ after epoch 3 — see results at https://wandb.ai/run/42"
	infer_ex = tokenizer.encode(inference_text, max_length=max_len)
	print(f"Inference text: {inference_text}")
	print(f"Token IDs: {infer_ex.token_ids}")
	print(f"Attention: {infer_ex.attention_mask}")
	print(f"Decoded (skip special): {tokenizer.decode(infer_ex.token_ids)}")

	print("\nNote: Unknown words map to <unk> if they were not seen in training vocabulary.")


if __name__ == "__main__":
	run_demo()

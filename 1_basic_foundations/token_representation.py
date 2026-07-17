"""
Token representation basics for NLP pipelines (training, validation, inference).

This script is intentionally dependency-light (numpy only) so the math behind
each representation is easy to trace by hand. It demonstrates, step by step:

1) Raw text -> tokens -> vocabulary (same idea as tokenization.py)
2) One-hot encoding: sparse, no semantic meaning, dimensionality = |V|
3) Word2Vec (Skip-gram): dense, learned embeddings trained with a real
   forward/backward pass (no autograd library — gradients are derived and
   coded manually so the math is visible)
3b) Word2Vec (CBOW - Continuous Bag of Words): mirror image of Skip-gram —
   predicts the center word FROM its averaged context embeddings instead of
   predicting context words from the center word
4) How the SAME pipeline behaves differently at training time vs inference
   time (unseen/OOV words, embedding lookup, nearest-neighbor retrieval)
5) A short comparison of one-hot vs static (Skip-gram/CBOW) vs contextual
   embeddings, with pointers to where contextual embeddings are implemented
   in this repo (transformer_basic.py / transformer_basic_training_inference_pipeline.py)

Formulation reference (see foundation_revision_LLM.md Q7):
    e_i = E[w_i] in R^d                      (embedding lookup)
    Skip-gram objective:
        L = - sum_t sum_{-c<=j<=c, j!=0} log p(w_{t+j} | w_t)
        p(w_{t+j} | w_t) = softmax(v'^T_{w_{t+j}} . v_{w_t})
    CBOW objective:
        L = - sum_t log p(w_t | context_t)
        v_context = mean_{k in context_t} v_k
        p(w_t | context_t) = softmax(v'^T_{w_t} . v_context)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Step 1: Tokenization + vocabulary building (minimal, self-contained)
# ---------------------------------------------------------------------------

UNK = "<unk>"


def normalize(text: str) -> str:
	text = text.lower().strip()
	text = re.sub(r"\s+", " ", text)
	return text


def split(text: str) -> List[str]:
	# Word-level tokens; punctuation kept as separate tokens.
	return re.findall(r"\w+|[^\w\s]", text)


@dataclass
class Vocabulary:
	"""Word <-> id mapping built ONLY from training data (mimics real pipelines)."""

	token_to_id: Dict[str, int] = field(default_factory=dict)
	id_to_token: Dict[int, str] = field(default_factory=dict)

	@property
	def size(self) -> int:
		return len(self.token_to_id)

	def fit(self, texts: Sequence[str], min_freq: int = 1) -> None:
		counter: Counter[str] = Counter()
		for text in texts:
			counter.update(split(normalize(text)))

		# Special <unk> token always gets id 0 so OOV words never crash lookups.
		self.token_to_id = {UNK: 0}
		self.id_to_token = {0: UNK}

		items = [(tok, freq) for tok, freq in counter.items() if freq >= min_freq]
		items.sort(key=lambda x: (-x[1], x[0]))  # deterministic ordering

		for tok, _ in items:
			idx = len(self.token_to_id)
			self.token_to_id[tok] = idx
			self.id_to_token[idx] = tok

	def encode(self, tokens: Sequence[str]) -> List[int]:
		unk_id = self.token_to_id[UNK]
		return [self.token_to_id.get(tok, unk_id) for tok in tokens]


# ---------------------------------------------------------------------------
# Step 2: One-hot encoding
# ---------------------------------------------------------------------------


def one_hot_vector(token_id: int, vocab_size: int) -> np.ndarray:
	"""
	e_i in {0,1}^|V|,  e_i[j] = 1 if j == token_id else 0

	Properties:
	- Dimensionality equals vocabulary size (can be huge, e.g. 30K-50K).
	- Every pair of distinct one-hot vectors is orthogonal -> dot product = 0,
	  i.e. one-hot encoding carries NO notion of semantic similarity.
	"""
	vec = np.zeros(vocab_size, dtype=np.float32)
	vec[token_id] = 1.0
	return vec


def one_hot_similarity_demo(vocab: Vocabulary, word_a: str, word_b: str) -> float:
	id_a = vocab.token_to_id.get(word_a, vocab.token_to_id[UNK])
	id_b = vocab.token_to_id.get(word_b, vocab.token_to_id[UNK])
	vec_a = one_hot_vector(id_a, vocab.size)
	vec_b = one_hot_vector(id_b, vocab.size)
	return float(np.dot(vec_a, vec_b))  # always 0 unless same word


# ---------------------------------------------------------------------------
# Step 3: Word2Vec (Skip-gram) trained from scratch with numpy
# ---------------------------------------------------------------------------


def generate_skipgram_pairs(token_ids: Sequence[int], window: int) -> List[Tuple[int, int]]:
	"""(target, context) pairs: for each center word, pair it with nearby words."""
	pairs: List[Tuple[int, int]] = []
	for t, target in enumerate(token_ids):
		start = max(0, t - window)
		end = min(len(token_ids), t + window + 1)
		for c in range(start, end):
			if c != t:
				pairs.append((target, token_ids[c]))
	return pairs


class SkipGramWord2Vec:
	"""
	Minimal Skip-gram Word2Vec with full-softmax training (fine for tiny
	vocabularies used in this demo; production Word2Vec uses negative
	sampling or hierarchical softmax to scale to large vocabularies).

	Two embedding matrices are learned:
	  W_in  (vocab_size, dim)  -> "input"/center-word embeddings (used for lookup)
	  W_out (vocab_size, dim)  -> "output"/context-word embeddings (training only)
	"""

	def __init__(self, vocab_size: int, dim: int = 16, seed: int = 42) -> None:
		rng = np.random.default_rng(seed)
		self.vocab_size = vocab_size
		self.dim = dim
		# Small random init keeps early softmax close to uniform.
		self.w_in = rng.normal(0, 0.01, size=(vocab_size, dim)).astype(np.float32)
		self.w_out = rng.normal(0, 0.01, size=(vocab_size, dim)).astype(np.float32)

	@staticmethod
	def _softmax(logits: np.ndarray) -> np.ndarray:
		shifted = logits - np.max(logits)  # numerical stability
		exp = np.exp(shifted)
		return exp / np.sum(exp)

	def forward(self, target_id: int) -> np.ndarray:
		"""p(context | target) over the full vocabulary."""
		v_target = self.w_in[target_id]                # (dim,)
		logits = self.w_out @ v_target                  # (vocab_size,)
		return self._softmax(logits)

	def train_step(self, target_id: int, context_id: int, lr: float) -> float:
		"""
		One SGD step for a single (target, context) pair.

		Loss:  L = -log p(context_id | target_id)
		Gradients (standard softmax cross-entropy derivation):
		  dL/dlogits = p - one_hot(context_id)
		  dL/dW_out  = outer(dL/dlogits, v_target)
		  dL/dv_target = W_out^T . dL/dlogits
		"""
		v_target = self.w_in[target_id]
		probs = self.forward(target_id)                 # (vocab_size,)
		loss = -np.log(probs[context_id] + 1e-9)

		grad_logits = probs.copy()
		grad_logits[context_id] -= 1.0                   # (vocab_size,)

		grad_w_out = np.outer(grad_logits, v_target)      # (vocab_size, dim)
		grad_v_target = self.w_out.T @ grad_logits         # (dim,)

		self.w_out -= lr * grad_w_out
		self.w_in[target_id] -= lr * grad_v_target

		return float(loss)

	def fit(self, pairs: Sequence[Tuple[int, int]], epochs: int, lr: float = 0.05) -> List[float]:
		history: List[float] = []
		for epoch in range(epochs):
			total_loss = 0.0
			for target_id, context_id in pairs:
				total_loss += self.train_step(target_id, context_id, lr)
			avg_loss = total_loss / max(1, len(pairs))
			history.append(avg_loss)
		return history

	def embedding(self, token_id: int) -> np.ndarray:
		"""Embedding lookup used at inference time: e_i = W_in[token_id]."""
		return self.w_in[token_id]

	def most_similar(self, token_id: int, vocab: Vocabulary, top_k: int = 3) -> List[Tuple[str, float]]:
		"""Cosine similarity between one word's embedding and all others."""
		query = self.embedding(token_id)
		query_norm = query / (np.linalg.norm(query) + 1e-9)

		scores: List[Tuple[str, float]] = []
		for other_id in range(self.vocab_size):
			if other_id == token_id:
				continue
			vec = self.w_in[other_id]
			vec_norm = vec / (np.linalg.norm(vec) + 1e-9)
			sim = float(np.dot(query_norm, vec_norm))
			scores.append((vocab.id_to_token[other_id], sim))

		scores.sort(key=lambda x: x[1], reverse=True)
		return scores[:top_k]


# ---------------------------------------------------------------------------
# Step 3b: Word2Vec (CBOW — Continuous Bag of Words) trained from scratch
# ---------------------------------------------------------------------------
#
# CBOW is the mirror image of Skip-gram:
#   Skip-gram: 1 center word  ->  predict each surrounding context word
#   CBOW     : N context words -> predict the 1 center (target) word
#
# Formulation:
#   Given context words {w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c}},
#   average their input embeddings into a single context vector:
#       v_context = (1 / |C|) * sum_{k in C} W_in[k]
#   then predict the target word with the same softmax used in Skip-gram:
#       p(w_t | context) = softmax(W_out . v_context)
#   Loss:
#       L = -log p(w_t | context)


def generate_cbow_pairs(token_ids: Sequence[int], window: int) -> List[Tuple[List[int], int]]:
	"""(context_ids, target_id) pairs: surrounding words -> center word."""
	pairs: List[Tuple[List[int], int]] = []
	for t, target in enumerate(token_ids):
		start = max(0, t - window)
		end = min(len(token_ids), t + window + 1)
		context_ids = [token_ids[c] for c in range(start, end) if c != t]
		if context_ids:  # skip targets with no context (e.g. very short sequences)
			pairs.append((context_ids, target))
	return pairs


class CBOWWord2Vec:
	"""
	Minimal CBOW Word2Vec with full-softmax training (same simplification as
	SkipGramWord2Vec — production CBOW also uses negative sampling for
	large vocabularies).

	Two embedding matrices are learned:
	  W_in  (vocab_size, dim)  -> context-word embeddings (averaged, then used for lookup)
	  W_out (vocab_size, dim)  -> target-word embeddings (training only)
	"""

	def __init__(self, vocab_size: int, dim: int = 16, seed: int = 42) -> None:
		rng = np.random.default_rng(seed)
		self.vocab_size = vocab_size
		self.dim = dim
		self.w_in = rng.normal(0, 0.01, size=(vocab_size, dim)).astype(np.float32)
		self.w_out = rng.normal(0, 0.01, size=(vocab_size, dim)).astype(np.float32)

	@staticmethod
	def _softmax(logits: np.ndarray) -> np.ndarray:
		shifted = logits - np.max(logits)
		exp = np.exp(shifted)
		return exp / np.sum(exp)

	def _context_vector(self, context_ids: Sequence[int]) -> np.ndarray:
		"""v_context = mean of the input embeddings of all context words."""
		return self.w_in[list(context_ids)].mean(axis=0)  # (dim,)

	def forward(self, context_ids: Sequence[int]) -> np.ndarray:
		"""p(target | context) over the full vocabulary."""
		v_context = self._context_vector(context_ids)     # (dim,)
		logits = self.w_out @ v_context                    # (vocab_size,)
		return self._softmax(logits)

	def train_step(self, context_ids: Sequence[int], target_id: int, lr: float) -> float:
		"""
		One SGD step for a single (context_ids, target_id) pair.

		Loss:  L = -log p(target_id | context)
		Gradients:
		  dL/dlogits      = p - one_hot(target_id)
		  dL/dW_out       = outer(dL/dlogits, v_context)
		  dL/dv_context   = W_out^T . dL/dlogits
		  # v_context is a mean over |C| vectors, so the gradient is split
		  # equally back to every context word's embedding row.
		  dL/dW_in[c]     = dL/dv_context / |C|   for each c in context_ids
		"""
		v_context = self._context_vector(context_ids)
		probs = self.forward(context_ids)                  # (vocab_size,)
		loss = -np.log(probs[target_id] + 1e-9)

		grad_logits = probs.copy()
		grad_logits[target_id] -= 1.0                        # (vocab_size,)

		grad_w_out = np.outer(grad_logits, v_context)          # (vocab_size, dim)
		grad_v_context = self.w_out.T @ grad_logits             # (dim,)

		self.w_out -= lr * grad_w_out

		grad_per_context_word = grad_v_context / len(context_ids)
		for c_id in context_ids:
			self.w_in[c_id] -= lr * grad_per_context_word

		return float(loss)

	def fit(self, pairs: Sequence[Tuple[Sequence[int], int]], epochs: int, lr: float = 0.05) -> List[float]:
		history: List[float] = []
		for epoch in range(epochs):
			total_loss = 0.0
			for context_ids, target_id in pairs:
				total_loss += self.train_step(context_ids, target_id, lr)
			avg_loss = total_loss / max(1, len(pairs))
			history.append(avg_loss)
		return history

	def embedding(self, token_id: int) -> np.ndarray:
		"""Embedding lookup used at inference time: e_i = W_in[token_id]."""
		return self.w_in[token_id]

	def most_similar(self, token_id: int, vocab: Vocabulary, top_k: int = 3) -> List[Tuple[str, float]]:
		"""Cosine similarity between one word's embedding and all others."""
		query = self.embedding(token_id)
		query_norm = query / (np.linalg.norm(query) + 1e-9)

		scores: List[Tuple[str, float]] = []
		for other_id in range(self.vocab_size):
			if other_id == token_id:
				continue
			vec = self.w_in[other_id]
			vec_norm = vec / (np.linalg.norm(vec) + 1e-9)
			sim = float(np.dot(query_norm, vec_norm))
			scores.append((vocab.id_to_token[other_id], sim))

		scores.sort(key=lambda x: x[1], reverse=True)
		return scores[:top_k]

	def predict_target(self, context_ids: Sequence[int], vocab: Vocabulary, top_k: int = 3) -> List[Tuple[str, float]]:
		"""Inference: given context words, predict the most likely center word."""
		probs = self.forward(context_ids)
		ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:top_k]
		return [(vocab.id_to_token[idx], float(p)) for idx, p in ranked]


# ---------------------------------------------------------------------------
# Step 4: End-to-end demo — training pipeline, then inference pipeline
# ---------------------------------------------------------------------------


def run_demo() -> None:
	corpus = [
		"the cat sat on the mat",
		"the dog sat on the rug",
		"the cat and the dog are friends",
		"a cat chased a mouse near the mat",
		"the dog barked at the mouse",
		"the king ruled the kingdom",
		"the queen ruled the kingdom with the king",
		"the king and queen visited the castle",
	]

	print("=" * 72)
	print("STEP 1: Raw text corpus (training data)")
	for line in corpus:
		print(f"  - {line}")

	print("\n" + "=" * 72)
	print("STEP 2: Tokenize + build vocabulary from training data only")
	vocab = Vocabulary()
	vocab.fit(corpus, min_freq=1)
	print(f"Vocabulary size |V| = {vocab.size}")
	print(f"Sample mapping: {list(vocab.token_to_id.items())[:8]} ...")

	tokenized_corpus = [vocab.encode(split(normalize(text))) for text in corpus]

	print("\n" + "=" * 72)
	print("STEP 3: One-hot encoding (baseline, sparse representation)")
	sample_word = "cat"
	sample_id = vocab.token_to_id[sample_word]
	one_hot = one_hot_vector(sample_id, vocab.size)
	print(f"one-hot('{sample_word}') shape = {one_hot.shape}, nonzero index = {sample_id}")
	print(f"one-hot('{sample_word}') vector (truncated): {one_hot[:12]} ...")

	sim_cat_dog = one_hot_similarity_demo(vocab, "cat", "dog")
	sim_cat_cat = one_hot_similarity_demo(vocab, "cat", "cat")
	print(f"one-hot dot('cat','dog') = {sim_cat_dog}  <- no semantic similarity captured")
	print(f"one-hot dot('cat','cat') = {sim_cat_cat}  <- only identical tokens overlap")

	print("\n" + "=" * 72)
	print("STEP 4: Build Skip-gram (target, context) training pairs")
	window = 2
	skipgram_pairs: List[Tuple[int, int]] = []
	for ids in tokenized_corpus:
		skipgram_pairs.extend(generate_skipgram_pairs(ids, window=window))
	print(f"Window size = {window}, total training pairs = {len(skipgram_pairs)}")
	print(f"Sample pairs (target_id, context_id): {skipgram_pairs[:6]}")

	print("\n" + "=" * 72)
	print("STEP 5: Train Word2Vec (Skip-gram) embeddings from scratch")
	dim = 16
	skipgram_model = SkipGramWord2Vec(vocab_size=vocab.size, dim=dim, seed=7)
	sg_history = skipgram_model.fit(skipgram_pairs, epochs=200, lr=0.05)
	print(f"Embedding dim d = {dim}")
	print(f"Loss epoch 1   = {sg_history[0]:.4f}")
	print(f"Loss epoch 50  = {sg_history[49]:.4f}")
	print(f"Loss epoch 200 = {sg_history[-1]:.4f}  (should be much lower than epoch 1)")

	print("\n" + "=" * 72)
	print("STEP 6: Skip-gram inference — embedding lookup + nearest neighbors")
	for query_word in ["cat", "king"]:
		query_id = vocab.token_to_id[query_word]
		vec = skipgram_model.embedding(query_id)
		print(f"\nembedding('{query_word}') (first 6 dims): {np.round(vec[:6], 3)}")
		neighbors = skipgram_model.most_similar(query_id, vocab, top_k=3)
		print(f"Most similar to '{query_word}': {neighbors}")

	print("\n" + "=" * 72)
	print("STEP 7: Build CBOW (context_ids, target) training pairs")
	cbow_pairs: List[Tuple[List[int], int]] = []
	for ids in tokenized_corpus:
		cbow_pairs.extend(generate_cbow_pairs(ids, window=window))
	print(f"Window size = {window}, total training pairs = {len(cbow_pairs)}")
	print(f"Sample pairs (context_ids -> target_id): {cbow_pairs[:4]}")
	print(
		"Note the direction flip vs Skip-gram: here MULTIPLE context ids map to ONE target id."
	)

	print("\n" + "=" * 72)
	print("STEP 8: Train Word2Vec (CBOW) embeddings from scratch")
	cbow_model = CBOWWord2Vec(vocab_size=vocab.size, dim=dim, seed=7)
	cbow_history = cbow_model.fit(cbow_pairs, epochs=200, lr=0.05)
	print(f"Embedding dim d = {dim}")
	print(f"Loss epoch 1   = {cbow_history[0]:.4f}")
	print(f"Loss epoch 50  = {cbow_history[49]:.4f}")
	print(f"Loss epoch 200 = {cbow_history[-1]:.4f}  (should be much lower than epoch 1)")

	print("\n" + "=" * 72)
	print("STEP 9: CBOW inference — predict target from context, and nearest neighbors")
	sample_context_ids, sample_target_id = cbow_pairs[0]
	sample_context_words = [vocab.id_to_token[i] for i in sample_context_ids]
	print(f"Context words: {sample_context_words} -> true target: '{vocab.id_to_token[sample_target_id]}'")
	predicted = cbow_model.predict_target(sample_context_ids, vocab, top_k=3)
	print(f"CBOW predicted top-3 targets (word, prob): {predicted}")

	for query_word in ["cat", "king"]:
		query_id = vocab.token_to_id[query_word]
		neighbors = cbow_model.most_similar(query_id, vocab, top_k=3)
		print(f"Most similar to '{query_word}' (CBOW embedding): {neighbors}")

	print("\n" + "=" * 72)
	print("STEP 10: Inference-time behavior on an unseen (OOV) word")
	unseen_text = "the wizard sat on the mat"
	unseen_ids = vocab.encode(split(normalize(unseen_text)))
	unseen_tokens = split(normalize(unseen_text))
	print(f"Text: {unseen_text}")
	for tok, idx in zip(unseen_tokens, unseen_ids):
		flag = " <- OOV, mapped to <unk> embedding" if tok not in vocab.token_to_id else ""
		print(f"  token='{tok}' -> id={idx}{flag}")
	print(
		"Note: static embeddings (Word2Vec/GloVe, both Skip-gram and CBOW) give <unk> words "
		"the SAME shared vector regardless of context. Contextual embeddings (BERT/GPT, see "
		"transformer_basic.py) solve this via subword tokenization + self-attention conditioning "
		"on the full sequence."
	)

	print("\n" + "=" * 72)
	print("SUMMARY: One-hot vs Skip-gram vs CBOW vs Contextual embeddings")
	print(
		"  One-hot        : dim=|V| sparse, no semantic similarity, no training needed\n"
		"  Skip-gram      : dim=d dense, 1 center word -> predict each context word;\n"
		"                   more training pairs per sentence, better for RARE words / small data\n"
		"  CBOW           : dim=d dense, N context words (averaged) -> predict 1 center word;\n"
		"                   fewer, smoother updates, faster to train, better for FREQUENT words\n"
		"  Contextual      : dim=d dense, vector CHANGES per sentence via self-attention\n"
		"                   (see transformer_basic.py / transformer_basic_training_inference_pipeline.py)"
	)


if __name__ == "__main__":
	run_demo()


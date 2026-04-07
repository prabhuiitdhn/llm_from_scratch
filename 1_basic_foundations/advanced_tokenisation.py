"""
Advanced tokenisation demo for NLP pipelines.

This script keeps the same corpus used in tokenization.py, but adds
senior-level techniques commonly used in production:
- Unicode normalization (NFKC)
- URL/email/number placeholders
- punctuation cleanup and whitespace normalization
- contraction-friendly regex tokenization
- train-time word vocabulary + subword fallback for OOV words
- attention and label masks for training-style batching
"""

from __future__ import annotations

import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class EncodedExample:
    text: str
    token_ids: List[int]
    attention_mask: List[int]
    label_mask: List[int]


class AdvancedTokenizer:
    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"
    URL = "<url>"
    EMAIL = "<email>"
    NUM = "<num>"

    def __init__(self) -> None:
        self.special_tokens = [
            self.PAD,
            self.UNK,
            self.BOS,
            self.EOS,
            self.URL,
            self.EMAIL,
            self.NUM,
        ]
        self.token_to_id = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.special_tokens)}
        self.fitted = False

    @staticmethod
    def normalize(text: str) -> str:
        # Senior-level normalization: Unicode consistency + placeholders + cleanup.
        text = unicodedata.normalize("NFKC", text)
        text = text.lower().strip()

        text = re.sub(r"https?://\S+|www\.\S+", " <url> ", text) # replace url with placeholder <url>
        text = re.sub(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", " <email> ", text) # replace email with <email>
        text = re.sub(r"\b\d+(?:[.,]\d+)?(?:e[+-]?\d+)?\b", " <num> ", text) # replace numeric value with <num>

        # Reduce punctuation bursts but keep one marker token.
        text = re.sub(r"([!?.,])\1{1,}", r"\1", text)

        # Collapse whitespace to stabilize tokenization.
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def split(text: str) -> List[str]:
        # Keep placeholders, contractions, words, and standalone punctuation/symbols.
        pattern = r"<url>|<email>|<num>|[a-z]+(?:'[a-z]+)?|\w+|[^\w\s]"
        return re.findall(pattern, text)

    @staticmethod
    def _candidate_subwords(word: str, min_len: int = 3, max_len: int = 6) -> Iterable[str]:
        n = len(word)
        for size in range(min_len, min(max_len, n) + 1):
            for i in range(0, n - size + 1):
                yield word[i : i + size]

    def fit(
        self,
        texts: Sequence[str],
        min_word_freq: int = 1,
        max_vocab: int | None = None,
        min_subword_freq: int = 2,
        max_subwords: int = 300,
    ) -> None:
        word_counter: Counter[str] = Counter()
        subword_counter: Counter[str] = Counter()

        for text in texts:
            tokens = self.split(self.normalize(text))
            word_counter.update(tokens)

            # Learn fallback subwords from alphabetic words only.
            for tok in tokens:
                if re.fullmatch(r"[a-z]+", tok):
                    subword_counter.update(self._candidate_subwords(tok))

        words = [(tok, freq) for tok, freq in word_counter.items() if freq >= min_word_freq]
        words.sort(key=lambda x: (-x[1], x[0]))
        if max_vocab is not None:
            words = words[:max_vocab]

        next_id = len(self.token_to_id)
        for tok, _ in words:
            if tok not in self.token_to_id:
                self.token_to_id[tok] = next_id
                self.id_to_token[next_id] = tok
                next_id += 1

        # Add subword pieces with dedicated namespace for OOV fallback.
        subwords = [(sw, freq) for sw, freq in subword_counter.items() if freq >= min_subword_freq]
        subwords.sort(key=lambda x: (-x[1], x[0]))
        subwords = subwords[:max_subwords]

        for sw, _ in subwords:
            piece = f"<sw:{sw}>"
            if piece not in self.token_to_id:
                self.token_to_id[piece] = next_id
                self.id_to_token[next_id] = piece
                next_id += 1

        self.fitted = True

    def _oov_fallback(self, token: str) -> List[int]:
        """Greedy longest-match segmentation into learned subword pieces."""
        ids: List[int] = []
        i = 0
        while i < len(token):
            match_id = None
            match_len = 0
            for j in range(len(token), i, -1):
                piece = f"<sw:{token[i:j]}>"
                idx = self.token_to_id.get(piece)
                if idx is not None:
                    match_id = idx
                    match_len = j - i
                    break

            if match_id is None:
                ids.append(self.token_to_id[self.UNK])
                i += 1
            else:
                ids.append(match_id)
                i += match_len
        return ids

    def encode(self, text: str, max_length: int, train_on_all_tokens: bool = True) -> EncodedExample:
        if not self.fitted:
            raise RuntimeError("Tokenizer is not fitted. Call fit() first.")

        tokens = self.split(self.normalize(text))

        ids: List[int] = [self.token_to_id[self.BOS]]
        label_mask: List[int] = [1 if train_on_all_tokens else 0]

        for tok in tokens:
            idx = self.token_to_id.get(tok)
            if idx is not None:
                ids.append(idx)
                label_mask.append(1)
            else:
                # Use subword fallback for alphabetic OOV; else UNK.
                if re.fullmatch(r"[a-z]+", tok):
                    fallback_ids = self._oov_fallback(tok)
                    ids.extend(fallback_ids)
                    label_mask.extend([1] * len(fallback_ids))
                else:
                    ids.append(self.token_to_id[self.UNK])
                    label_mask.append(1)

        ids.append(self.token_to_id[self.EOS])
        label_mask.append(1 if train_on_all_tokens else 0)

        if len(ids) > max_length:
            ids = ids[: max_length - 1] + [self.token_to_id[self.EOS]]
            label_mask = label_mask[: max_length - 1] + [1 if train_on_all_tokens else 0]

        attention_mask = [1] * len(ids)

        while len(ids) < max_length:
            ids.append(self.token_to_id[self.PAD])
            attention_mask.append(0)
            label_mask.append(0)

        return EncodedExample(
            text=text,
            token_ids=ids,
            attention_mask=attention_mask,
            label_mask=label_mask,
        )

    def decode(self, token_ids: Sequence[int], skip_special: bool = True) -> str:
        tokens: List[str] = []
        special = set(self.special_tokens)
        for idx in token_ids:
            tok = self.id_to_token.get(idx, self.UNK)
            if skip_special and (tok in special or tok.startswith("<sw:")):
                continue
            tokens.append(tok)
        return " ".join(tokens)


def train_val_split(data: Sequence[str], val_ratio: float = 0.25, seed: int = 7) -> Tuple[List[str], List[str]]:
    items = list(data)
    rnd = random.Random(seed)
    rnd.shuffle(items)

    val_size = max(1, int(len(items) * val_ratio))
    return items[val_size:], items[:val_size]


def get_demo_corpus() -> List[str]:
    # Same corpus used in basic_foundations/tokenization.py
    return [
        "LoRA, LORA and lora all refer to Low-Rank Adaptation in fine-tuning.",
        "GPT-4 was released on 2023-03-14; its context window is 128,000 tokens.",
        "See https://huggingface.co/docs or email support@openai.com for details.",
        "Wait... WHAT?!?! The loss jumped from 0.42 to 2.87 overnight???",
        "It's the model's job to understand what we're asking; don't ignore that.",
        "RLHF, DPO, SFT, QLoRA, PEFT, BPE, and RAG are core GenAI terms.",
        "The résumé of Dr. Müller mentions naïve Bayes and café reviews.",
        "Training loss ↓ 📉 while validation accuracy ↑ 🎯 after 3 epochs.",
        "Call model.generate(input_ids, max_new_tokens=256, temperature=0.7).",
        (
            "Tokenization is not just splitting by spaces: it handles contractions, "
            "punctuation, numbers, symbols, emojis, domain terms, code identifiers, "
            "URL patterns, multilingual scripts, and rare vocabulary all at once — "
            "and it must behave identically at train time and at inference time."
        ),
        "Pad me.",
        "plz fix  the   bug asap!!!  its   breaking   prod  rn  😤",
    ]


def run_demo() -> None:
    corpus = get_demo_corpus()
    train_texts, val_texts = train_val_split(corpus, val_ratio=0.25, seed=42)

    tok = AdvancedTokenizer()
    tok.fit(train_texts, min_word_freq=1, min_subword_freq=2, max_subwords=250)

    print("=" * 80)
    print("Advanced tokenisation demo")
    print(f"Train size: {len(train_texts)} | Val size: {len(val_texts)}")
    print(f"Vocab size (incl. special + subword pieces): {len(tok.token_to_id)}")

    max_len = 32
    train_encoded = [tok.encode(t, max_length=max_len) for t in train_texts]
    val_encoded = [tok.encode(t, max_length=max_len) for t in val_texts]

    print("\nFirst train example:")
    first = train_encoded[0]
    print(f"Text: {first.text}")
    print(f"Token IDs: {first.token_ids}")
    print(f"Attention mask: {first.attention_mask}")
    print(f"Label mask: {first.label_mask}")

    print("\nInference example with hard OOV + symbols:")
    infer = "ORPO & DPO at lr=1e-4: loss↓ after epoch 3 — see results at https://wandb.ai/run/42"
    out = tok.encode(infer, max_length=max_len)
    print(f"Input: {infer}")
    print(f"Token IDs: {out.token_ids}")
    print(f"Decoded (skip special/subwords): {tok.decode(out.token_ids)}")

    # Small health metrics often used in senior pipelines.
    total_tokens = sum(sum(m) for m in (ex.attention_mask for ex in val_encoded))
    trunc_count = sum(ex.attention_mask[-1] == 1 and ex.token_ids[-1] == tok.token_to_id[tok.EOS] for ex in val_encoded)
    print("\nValidation snapshot:")
    print(f"Approx. active tokens across val set: {total_tokens}")
    print(f"Examples ending exactly at max length (possible truncation): {trunc_count}/{len(val_encoded)}")


if __name__ == "__main__":
    run_demo()

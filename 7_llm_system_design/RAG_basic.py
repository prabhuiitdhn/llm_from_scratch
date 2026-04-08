"""
RAG Basic (Toy, from scratch)

Goal:
Show the complete RAG pipeline with very simple components so beginners can
understand the concept before using real embedding models and real LLMs.

Pipeline steps in this file:
1) Load toy documents
2) Chunk documents
3) Build toy embeddings
4) Store vectors and retrieve top-k chunks
5) Build a prompt-like context string
6) Generate a toy answer

No external ML libraries are required.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


# -----------------------------------------------------------------------------
# Layer 0: shared helpers
# -----------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Lowercase and collapse spaces for stable tokenization."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    """Tiny tokenizer: words only for simplicity."""
    return re.findall(r"[a-z0-9']+", normalize_text(text))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    a_norm = math.sqrt(sum(x * x for x in a))
    b_norm = math.sqrt(sum(y * y for y in b))

    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return dot / (a_norm * b_norm)


# -----------------------------------------------------------------------------
# Layer 1: documents
# -----------------------------------------------------------------------------


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: Dict[str, str]


def load_toy_documents() -> List[Document]:
    """
    Small toy corpus.

    In real RAG, this could come from PDFs, websites, databases, etc.
    """
    return [
        Document(
            doc_id="doc_1",
            text=(
                "RAG means Retrieval Augmented Generation. "
                "First retrieve relevant documents, then generate an answer using that context."
            ),
            metadata={"topic": "rag_basics"},
        ),
        Document(
            doc_id="doc_2",
            text=(
                "Embeddings convert text into vectors. "
                "Semantically similar sentences have vectors that are close to each other."
            ),
            metadata={"topic": "embeddings"},
        ),
        Document(
            doc_id="doc_3",
            text=(
                "Vector search finds nearest chunks to a query vector. "
                "Cosine similarity is commonly used to rank chunk relevance."
            ),
            metadata={"topic": "retrieval"},
        ),
        Document(
            doc_id="doc_4",
            text=(
                "Prompt building combines retrieved chunks with user question. "
                "The generator should answer from provided context only."
            ),
            metadata={"topic": "prompting"},
        ),
        Document(
            doc_id="doc_5",
            text=(
                "Hallucination can be reduced when the model is grounded in retrieved facts. "
                "RAG is often preferred when knowledge changes frequently."
            ),
            metadata={"topic": "why_rag"},
        ),
    ]


# -----------------------------------------------------------------------------
# Layer 2: chunking
# -----------------------------------------------------------------------------


@dataclass
class Chunk:
    chunk_id: str
    parent_doc_id: str
    text: str


def chunk_documents(docs: List[Document], chunk_size_words: int = 18, overlap_words: int = 4) -> List[Chunk]:
    """
    Split each document into small overlapping word chunks.

    Why chunking?
    - Long docs become manageable units
    - Retrieval becomes more precise
    """
    all_chunks: List[Chunk] = []
    idx = 0

    for doc in docs:
        words = tokenize(doc.text)
        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size_words, len(words))
            chunk_text = " ".join(words[start:end])

            all_chunks.append(
                Chunk(
                    chunk_id=f"chunk_{idx}",
                    parent_doc_id=doc.doc_id,
                    text=chunk_text,
                )
            )
            idx += 1

            if end == len(words):
                break
            start = end - overlap_words

    return all_chunks


# -----------------------------------------------------------------------------
# Layer 3: toy embedding model
# -----------------------------------------------------------------------------


class ToyEmbeddingModel:
    """
    Very simple embedding model:
    - Build a vocabulary from corpus words
    - Represent text as normalized bag-of-words count vector

    This is NOT a semantic model, but perfect for learning mechanics.
    """

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}

    def fit(self, texts: List[str]) -> None:
        unique = set()
        for text in texts:
            unique.update(tokenize(text))

        ordered = sorted(unique)
        self.vocab = {tok: i for i, tok in enumerate(ordered)}

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * len(self.vocab)
        for tok in tokenize(text):
            if tok in self.vocab:
                vec[self.vocab[tok]] += 1.0

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


# -----------------------------------------------------------------------------
# Layer 4: vector store
# -----------------------------------------------------------------------------


class VectorStore:
    def __init__(self) -> None:
        self.rows: List[Tuple[Chunk, List[float]]] = []

    def add(self, chunk: Chunk, embedding: List[float]) -> None:
        self.rows.append((chunk, embedding))

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[Chunk, float]]:
        scored: List[Tuple[Chunk, float]] = []
        for chunk, emb in self.rows:
            score = cosine_similarity(query_embedding, emb)
            scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# -----------------------------------------------------------------------------
# Layer 5: prompt builder
# -----------------------------------------------------------------------------


def build_prompt(question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    context_lines = []
    for i, (chunk, score) in enumerate(retrieved, start=1):
        context_lines.append(f"[{i}] ({chunk.parent_doc_id}, score={score:.3f}) {chunk.text}")

    context = "\n".join(context_lines)
    prompt = (
        "You are a QA assistant. Use only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer concisely:"
    )
    return prompt


# -----------------------------------------------------------------------------
# Layer 6: toy generator model
# -----------------------------------------------------------------------------


class ToyGenerator:
    """
    Simple extractive generator:
    - Find sentence/chunk with max keyword overlap with question
    - Return that chunk as the answer

    This imitates the final generation step conceptually.
    """

    def answer(self, question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
        q_tokens = set(tokenize(question))
        if not retrieved:
            return "I do not have context to answer."

        best_chunk = None
        best_overlap = -1

        for chunk, _score in retrieved:
            c_tokens = set(tokenize(chunk.text))
            overlap = len(q_tokens.intersection(c_tokens))
            if overlap > best_overlap:
                best_overlap = overlap
                best_chunk = chunk

        if best_chunk is None:
            return "I do not have enough information in the retrieved context."

        return (
            f"Based on retrieved context: {best_chunk.text}. "
            "(Toy generator selected the highest-overlap chunk.)"
        )


# -----------------------------------------------------------------------------
# Layer 7: end-to-end RAG pipeline
# -----------------------------------------------------------------------------


class BasicRAGPipeline:
    def __init__(self) -> None:
        self.docs = load_toy_documents()
        self.chunks = chunk_documents(self.docs)

        self.embedder = ToyEmbeddingModel()
        self.embedder.fit([c.text for c in self.chunks])

        self.store = VectorStore()
        for chunk in self.chunks:
            self.store.add(chunk, self.embedder.embed(chunk.text))

        self.generator = ToyGenerator()

    def query(self, question: str, top_k: int = 3, verbose: bool = True) -> Dict[str, object]:
        query_vec = self.embedder.embed(question)
        retrieved = self.store.search(query_vec, top_k=top_k)
        prompt = build_prompt(question, retrieved)
        answer = self.generator.answer(question, retrieved)

        result = {
            "question": question,
            "answer": answer,
            "retrieved": [
                {
                    "chunk_id": c.chunk_id,
                    "parent_doc_id": c.parent_doc_id,
                    "score": round(s, 4),
                    "text": c.text,
                }
                for c, s in retrieved
            ],
            "prompt_preview": prompt,
        }

        if verbose:
            print("\n" + "=" * 80)
            print(f"QUESTION: {question}")
            print("=" * 80)
            print("Top retrieved chunks:")
            for i, item in enumerate(result["retrieved"], start=1):
                print(
                    f"{i}. {item['chunk_id']} | doc={item['parent_doc_id']} "
                    f"| score={item['score']:.4f}\n   {item['text']}"
                )

            print("\nPrompt sent to generator (preview):")
            print(prompt)

            print("\nFinal answer:")
            print(answer)

        return result


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------


def main() -> None:
    print("RAG Basic Demo (Toy Models)")
    print("This is intentionally simple to learn the full pipeline concept.")

    rag = BasicRAGPipeline()

    questions = [
        "What is RAG?",
        "Why do we use embeddings in retrieval?",
        "How does vector search rank chunks?",
        "How can RAG reduce hallucination?",
    ]

    for q in questions:
        rag.query(q, top_k=3, verbose=True)


if __name__ == "__main__":
    main()

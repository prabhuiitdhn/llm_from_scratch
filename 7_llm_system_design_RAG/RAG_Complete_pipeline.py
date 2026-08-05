"""
===============================================================================
RAG (Retrieval Augmented Generation) System - Layer by Layer Implementation
===============================================================================

Complete RAG system built from scratch with detailed explanations at each layer.
This implementation demonstrates how RAG works in practice using:
- Real LLMs via OLLAMA (local execution)
- Real embeddings for semantic search
- CNN/DailyMail dataset (real document corpus)
- Custom vector store with ranking

RAG Architecture Overview:
┌─────────────────────────────────────────────────────────────────────────────┐
│  USER QUERY                                                                  │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                  ▼
    ┌──────────────────┐           ┌──────────────────┐
    │  QUERY ENCODING  │           │  DOCUMENT STORE  │
    │  (Embedding)     │           │  (Embeddings)    │
    └────────┬─────────┘           └────────┬─────────┘
             │                              │
             └──────────────┬───────────────┘
                            ▼
                    ┌──────────────────┐
                    │ SIMILARITY SEARCH│
                    │ (Vector DB)      │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │  RETRIEVE TOP K  │
                    │  DOCUMENTS       │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │  BUILD CONTEXT   │
                    │  (Prompt Craft)  │
                    └────────┬─────────┘
                             ▼
            ┌────────────────┴────────────────┐
            ▼                                  ▼
    ┌──────────────────┐           ┌──────────────────┐
    │  LLAMA/MISTRAL   │           │  CONTEXT         │
    │  (via OLLAMA)    │◄──────────│  (Retrieved Docs)│
    └────────┬─────────┘           └──────────────────┘
             ▼
    ┌──────────────────┐
    │ GENERATE ANSWER  │
    │ (with grounding) │
    └────────┬─────────┘
             ▼
    ┌──────────────────┐
    │  FINAL RESPONSE  │
    │  (Augmented)     │
    └──────────────────┘

===============================================================================
LAYER 1: Document Loading & Preprocessing
===============================================================================

Purpose: Load real documents from CNN/DailyMail and prepare them for storage.
Why separate? Handling different data sources requires standardization.
Example: News articles have titles, summaries, body text with different structures.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Try to import required libraries, with helpful error messages
try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets library...")
    os.system("pip install datasets")
    from datasets import load_dataset

try:
    import requests
except ImportError:
    print("Installing requests library...")
    os.system("pip install requests")
    import requests

try:
    import numpy as np
except ImportError:
    print("Installing numpy...")
    os.system("pip install numpy")
    import numpy as np


@dataclass
class Document:
    """
    Standardized document representation across all data sources.
    
    Why a dataclass?
    - Type safety: ensures all documents have consistent structure
    - Memory efficient: compiled to slots internally
    - Easy serialization: for caching/saving
    
    Attributes:
        id: Unique identifier for retrieval tracking
        content: The full text to be embedded
        metadata: Source info, title, date, etc. (useful for post-filtering)
        embedding: Cached embedding vector (computed once, used many times)
    """
    id: str
    content: str
    metadata: Dict[str, str]
    embedding: Optional[np.ndarray] = None
    
    def __repr__(self) -> str:
        return f"Document(id={self.id}, len={len(self.content)}, metadata={self.metadata})"


class Layer1_DocumentLoader:
    """
    LAYER 1: Load and standardize documents from various sources.
    
    Real-world complexity:
    - News articles: title, body, summary, date, source
    - Academic papers: abstract, introduction, body, references
    - Web pages: navigation elements, ads, main content
    
    Solution: Extract meaningful content and metadata, discard noise.
    """
    
    @staticmethod
    def load_cnn_dailymail(split: str = "validation", max_samples: int = 100) -> List[Document]:
        """
        Load CNN/DailyMail dataset from HuggingFace.
        
        Why this dataset?
        - Real-world news articles (complex language, multiple topics)
        - Large document corpus (extractive summarization task)
        - Well-structured: article text, highlights (summary)
        - Common benchmark for retrieval and QA tasks
        
        Complexity: Some articles are very long (3000+ tokens)
        Solution: Will be handled in Layer 2 (chunking)
        
        Args:
            split: "train", "validation", or "test"
            max_samples: Limit for quick testing (use 100 for demo, 10000+ for production)
            
        Returns:
            List of standardized Document objects
            
        Example:
            >>> docs = Layer1_DocumentLoader.load_cnn_dailymail(max_samples=5)
            >>> print(f"Loaded {len(docs)} documents")
            >>> print(docs[0].metadata)
            {'source': 'cnn_dailymail', 'content_type': 'news_article'}
        """
        print(f"[LAYER 1] Loading CNN/DailyMail dataset ({split} split, max {max_samples} samples)...")
        
        try:
            dataset = load_dataset("cnn_dailymail", "3.0.0", split=split, trust_remote_code=True)
            documents = []
            
            # Take only first max_samples for faster iteration
            for idx, item in enumerate(dataset):
                if idx >= max_samples:
                    break
                
                # Extract structured content
                article_text = item.get("article", "")
                summary = item.get("highlights", "")
                
                if not article_text:
                    continue
                
                # Create standardized document
                doc = Document(
                    id=f"cnn_dailymail_{idx}",
                    content=article_text,  # Full article for retrieval
                    metadata={
                        "source": "cnn_dailymail",
                        "content_type": "news_article",
                        "has_summary": bool(summary),
                        "summary_preview": summary[:100] if summary else "",
                    }
                )
                documents.append(doc)
            
            print(f"[LAYER 1] ✓ Loaded {len(documents)} documents from CNN/DailyMail")
            return documents
            
        except Exception as e:
            print(f"[LAYER 1] Error loading dataset: {e}")
            print("[LAYER 1] Using fallback synthetic documents for demo...")
            return Layer1_DocumentLoader._create_fallback_documents()
    
    @staticmethod
    def _create_fallback_documents() -> List[Document]:
        """
        Fallback: Create synthetic documents if dataset loading fails.
        In production, would handle different data sources (databases, files, APIs).
        """
        synthetic_articles = [
            {
                "title": "Artificial Intelligence Advances in Medical Diagnosis",
                "content": "Recent breakthroughs in deep learning have revolutionized medical imaging. "
                          "AI models can now detect cancers in X-rays with 95% accuracy, "
                          "matching or exceeding human radiologists. These systems analyze "
                          "patient data from millions of medical records to identify patterns "
                          "invisible to human eyes. The FDA has approved several AI diagnostic "
                          "tools for clinical use.",
            },
            {
                "title": "Climate Change: Latest Research Findings",
                "content": "Global temperatures have risen 1.1°C since pre-industrial times. "
                          "Scientists report that Arctic ice is melting 50% faster than predicted. "
                          "New climate models incorporating ocean circulation patterns show potential "
                          "tipping points within the next decade. International climate agreements "
                          "aim for net-zero emissions by 2050, but current policies track toward 2.7°C warming.",
            },
            {
                "title": "Quantum Computing: From Theory to Practice",
                "content": "Quantum computers have achieved quantum advantage, solving specific problems "
                          "faster than classical supercomputers. IBM and Google are racing toward "
                          "practical quantum applications in drug discovery and optimization. "
                          "Error correction remains the main challenge - quantum bits are fragile "
                          "and decohere within microseconds.",
            },
        ]
        
        documents = []
        for idx, article in enumerate(synthetic_articles):
            doc = Document(
                id=f"synthetic_{idx}",
                content=article["content"],
                metadata={
                    "source": "synthetic",
                    "content_type": "article",
                    "title": article["title"],
                }
            )
            documents.append(doc)
        
        print(f"[LAYER 1] ✓ Created {len(documents)} synthetic fallback documents")
        return documents


"""
===============================================================================
LAYER 2: Text Chunking & Splitting
===============================================================================

Purpose: Break long documents into manageable chunks for embedding and retrieval.

Why separate chunking?
- Documents can be 5000+ tokens, but embedding models have limits (384-512-1024 tokens)
- Long context = expensive LLM inference
- Semantic units (paragraphs) better for retrieval than random splits
- Need overlap to maintain context continuity between chunks

Complexity examples:
- News article: 3000 tokens → chunk into 300-token pieces with 50-token overlap
- Academic paper: 8000 tokens → preserve section boundaries, don't split mid-paragraph
- Code files: need special handling for indentation and function boundaries
"""


class Layer2_TextChunker:
    """
    LAYER 2: Intelligent text chunking strategy.
    
    Question: Why not just embed whole documents?
    Answer:
    1. Embedding models have token limits (768, 1024, 2048)
    2. Semantic information concentrated in smaller units
    3. Better precision in retrieval (question →  exact relevant chunk, not whole document)
    
    Question: Why not just split every 256 tokens?
    Answer:
    1. Loses sentence/paragraph boundaries
    2. Can split important concepts
    3. Creates overlapping queries (same question in different chunks)
    
    Solution: Recursive chunking with overlap and smart boundary detection
    """
    
    @staticmethod
    def chunk_documents(
        documents: List[Document],
        chunk_size: int = 256,
        overlap: int = 50,
        strategy: str = "recursive"
    ) -> List[Document]:
        """
        Split documents into chunks with configurable strategy.
        
        Args:
            documents: Input document list
            chunk_size: Target tokens per chunk (approximate, using token count heuristic)
            overlap: Token overlap between consecutive chunks
            strategy: "simple" (just split), "recursive" (smart boundaries)
            
        Returns:
            List of chunk documents with unique IDs
            
        Example:
            >>> docs = [Document(id="1", content="Long article...")]
            >>> chunks = Layer2_TextChunker.chunk_documents(docs)
            >>> print(f"Split 1 document into {len(chunks)} chunks")
        """
        print(f"[LAYER 2] Chunking {len(documents)} documents (size={chunk_size}, overlap={overlap})...")
        
        chunks = []
        chunk_id = 0
        
        for doc in documents:
            if strategy == "recursive":
                doc_chunks = Layer2_TextChunker._recursive_split(
                    doc, chunk_size, overlap
                )
            else:
                doc_chunks = Layer2_TextChunker._simple_split(
                    doc, chunk_size, overlap
                )
            
            # Assign unique IDs to chunks
            for chunk_text in doc_chunks:
                chunk_doc = Document(
                    id=f"chunk_{chunk_id}_{doc.id}",
                    content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "original_id": doc.id,
                        "chunk_index": chunk_id,
                    }
                )
                chunks.append(chunk_doc)
                chunk_id += 1
        
        print(f"[LAYER 2] ✓ Split into {len(chunks)} chunks (avg {len(chunks)/len(documents):.1f} per doc)")
        return chunks
    
    @staticmethod
    def _simple_split(doc: Document, chunk_size: int, overlap: int) -> List[str]:
        """
        Simple token-based splitting (splitting every ~250 words).
        
        Characteristics:
        - Fast
        - Predictable
        - May split sentences/paragraphs
        """
        text = doc.content
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - overlap
            
            # Prevent infinite loop on very short texts
            if end == len(words):
                break
        
        return chunks
    
    @staticmethod
    def _recursive_split(doc: Document, chunk_size: int, overlap: int) -> List[str]:
        """
        Recursive splitting that respects sentence boundaries.
        
        Strategy:
        1. First try to split by paragraphs (double newlines)
        2. If still too large, split by sentences
        3. If still too large, split by words
        
        Result: Cleaner chunks that preserve semantic boundaries
        """
        text = doc.content
        
        # Try paragraph-level splitting first
        separators = ["\n\n", "\n", ". ", " "]
        
        for separator in separators:
            parts = text.split(separator)
            
            # Filter empty parts
            parts = [p.strip() for p in parts if p.strip()]
            
            chunks = []
            current_chunk = ""
            
            for part in parts:
                # Approximate token count (1 token ≈ 4 characters)
                current_length = len(current_chunk) // 4
                part_length = len(part) // 4
                
                if current_length + part_length < chunk_size:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += separator if separator != " " else " "
                    current_chunk += part
                else:
                    # Start new chunk
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = part
            
            if current_chunk:
                chunks.append(current_chunk)
            
            # Success: chunks are reasonable size
            if all(len(c) // 4 < chunk_size * 1.5 for c in chunks):
                # Add overlap for continuity
                overlapped = []
                for i, chunk in enumerate(chunks):
                    if i > 0:
                        # Add last ~50 tokens from previous chunk
                        prev_overlap = chunks[i-1][-200:]  # ~50 tokens
                        chunk = prev_overlap + separator + chunk if separator != " " else prev_overlap + " " + chunk
                    overlapped.append(chunk)
                return overlapped
        
        # Fallback to simple word splitting
        return Layer2_TextChunker._simple_split(doc, chunk_size, overlap)


"""
===============================================================================
LAYER 3: Embedding Generation
===============================================================================

Purpose: Convert text to dense vectors for semantic similarity search.

Why embeddings?
- Text keywords can miss synonyms ("car" vs "automobile")
- Embeddings capture semantic meaning in vector space
- Semantic similarity = geometric closeness (cosine distance)
- Powers similarity search without exact keyword matching

Complexity:
- Small models (60M parameters): fast, lower quality
- Large models (300M parameters): slow, higher quality
- Optimization: batch processing, GPU acceleration
- Real-time constraint: must be <100ms for interactive RAG

Solution: Use lightweight sentence transformer or OpenAI-compatible embedding service
"""


class Layer3_EmbeddingGenerator:
    """
    LAYER 3: Generate embeddings for documents and queries.
    
    Two approaches:
    
    Approach 1: Local embeddings (this implementation)
    - Use lightweight sentence-transformers model
    - Runs on CPU/GPU locally
    - Privacy preserving (data stays local)
    - Offline capable
    - Con: Slower than cloud APIs
    
    Approach 2: OLLAMA embeddings (alternative)
    - Use OLLAMA's embedding endpoint directly
    - Suitable if already running OLLAMA
    
    We'll use OLLAMA-compatible approach here for consistency.
    """
    
    def __init__(self, embedding_model: str = "nomic-embed-text", use_ollama: bool = True):
        """
        Initialize embedding generator.
        
        Args:
            embedding_model: Model name (e.g., "nomic-embed-text", "all-MiniLM-L6-v2")
            use_ollama: If True, use OLLAMA embedding endpoint (requires OLLAMA running)
                       If False, use sentence-transformers library
        
        Why nomic-embed-text?
        - Small (137M parameters, ~300MB)
        - Fast on CPU
        - Good quality embeddings
        - Optimized for retrieval
        - Open source, runs locally via OLLAMA
        
        Fallback approach:
        If OLLAMA not available, falls back to transformers library
        """
        self.embedding_model = embedding_model
        self.use_ollama = use_ollama
        self.embedding_cache = {}
        
        print(f"[LAYER 3] Initializing embedding generator...")
        print(f"[LAYER 3] Model: {embedding_model} | Using OLLAMA: {use_ollama}")
        
        if use_ollama:
            self.ollama_base_url = "http://localhost:11434"
            self._test_ollama_connection()
        else:
            self._init_transformers()
    
    def _test_ollama_connection(self):
        """Check if OLLAMA is running on localhost:11434"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"[LAYER 3] ✓ Connected to OLLAMA at {self.ollama_base_url}")
                self.ollama_available = True
                return
        except:
            pass
        
        print(f"[LAYER 3] ⚠ OLLAMA not available at {self.ollama_base_url}")
        print(f"[LAYER 3] Guide:")
        print(f"[LAYER 3]   1. Install OLLAMA from https://ollama.ai")
        print(f"[LAYER 3]   2. Run: ollama serve")
        print(f"[LAYER 3]   3. In another terminal: ollama pull {self.embedding_model}")
        print(f"[LAYER 3] Falling back to local transformer models...")
        
        self.use_ollama = False
        self._init_transformers()
    
    def _init_transformers(self):
        """Initialize sentence-transformers as fallback"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[LAYER 3] Loading sentence transformer: {self.embedding_model}...")
            
            # Map OLLAMA model names to sentence-transformer equivalents
            model_map = {
                "nomic-embed-text": "nomic-ai/nomic-embed-text-v1",
                "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
            }
            
            st_model = model_map.get(self.embedding_model, self.embedding_model)
            self.transformer_model = SentenceTransformer(st_model, device="cpu")
            print(f"[LAYER 3] ✓ Loaded sentence transformer model")
            self.transformer_available = True
            
        except ImportError:
            print(f"[LAYER 3] Installing sentence-transformers...")
            os.system("pip install sentence-transformers")
            from sentence_transformers import SentenceTransformer
            self.transformer_model = SentenceTransformer(self.embedding_model, device="cpu")
            self.transformer_available = True
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Returns:
            Vector of shape (embedding_dim,) - typically 384, 768, or 1536
            
        Note: Embeddings are normalized L2 (unit vectors)
        This makes cosine similarity = dot product (efficient)
        """
        # Check cache first (avoid re-embedding same text)
        cache_key = hash(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        if self.use_ollama and hasattr(self, 'ollama_available') and self.ollama_available:
            embedding = self._embed_with_ollama(text)
        else:
            embedding = self._embed_with_transformers(text)
        
        # Cache result
        self.embedding_cache[cache_key] = embedding
        return embedding
    
    def _embed_with_ollama(self, text: str) -> np.ndarray:
        """Use OLLAMA embedding endpoint"""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/embed",
                json={
                    "model": self.embedding_model,
                    "input": text,
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = np.array(data["embeddings"][0], dtype=np.float32)
                # Normalize
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                return embedding
            else:
                print(f"[LAYER 3] OLLAMA error: {response.text}")
                raise Exception("OLLAMA embedding failed")
                
        except Exception as e:
            print(f"[LAYER 3] OLLAMA embedding failed: {e}, falling back to transformers")
            self.use_ollama = False
            return self._embed_with_transformers(text)
    
    def _embed_with_transformers(self, text: str) -> np.ndarray:
        """Use sentence-transformers library"""
        if not hasattr(self, 'transformer_model'):
            self._init_transformers()
        
        embedding = self.transformer_model.encode(text, convert_to_numpy=True)
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding.astype(np.float32)
    
    def embed_documents(self, documents: List[Document], batch_size: int = 32) -> List[Document]:
        """
        Generate embeddings for multiple documents (batched for efficiency).
        
        Args:
            documents: List of Document objects
            batch_size: Process this many documents at once
            
        Returns:
            Same documents but with .embedding attribute populated
            
        Performance note:
        - Batch processing is 10x faster than one-by-one
        - GPU batch size can be 128-256
        - CPU batch size typically 32-64
        """
        print(f"[LAYER 3] Embedding {len(documents)} documents...")
        
        start_time = time.time()
        
        for i, doc in enumerate(documents):
            doc.embedding = self.embed_text(doc.content)
            
            if (i + 1) % max(1, len(documents) // 10) == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(documents) - i) / rate if rate > 0 else 0
                print(f"[LAYER 3] {i+1}/{len(documents)} ({rate:.1f} docs/sec, ETA {eta:.0f}s)")
        
        elapsed = time.time() - start_time
        print(f"[LAYER 3] ✓ Embedded {len(documents)} documents in {elapsed:.1f}s")
        
        return documents


"""
===============================================================================
LAYER 4: Vector Store & Similarity Search
===============================================================================

Purpose: Efficiently store and retrieve documents by semantic similarity.

Why a vector store (instead of just a list)?
- List search: O(n) - must compare with every document (slow)
- Vector store: O(log n) to O(1) with indexing - fast retrieval

Complexity in production:
- Pinecone, Weaviate, Milvus: managed vector DBs with billions of vectors
- FAISS: fast similarity search, billions of vectors on single machine
- SQLite with vector extension: simpler, embedded
- This implementation: in-memory (for demo), can extend to persistent storage

Trade-offs:
- In-memory: fast, simple, limited to RAM
- Disk-based: slower, scalable to billions
- Cloud managed: easiest, expensive
"""


class Layer4_VectorStore:
    """
    LAYER 4: Simple in-memory vector store with similarity search.
    
    This is a DEMO implementation showing the concepts.
    
    Production systems use:
    - FAISS (Facebook): billions of vectors, various indexing strategies
    - Pinecone: managed cloud vector DB
    - Weaviate: vector DB with GraphQL API
    - Milvus: open-source scalable vector DB
    
    This demo uses exact cosine similarity search O(n):
    - Pros: Simple, no approximation, correct results
    - Cons: Slow on millions of vectors (would need FAISS or HNSW)
    """
    
    def __init__(self):
        """Initialize empty vector store"""
        self.documents: List[Document] = []
        self.embeddings: np.ndarray = None
        self.document_id_map: Dict[str, int] = {}
        
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents with embeddings to the store.
        
        Args:
            documents: List of Document objects with embedding field populated
            
        Raises:
            ValueError: If any document is missing embedding
        """
        print(f"[LAYER 4] Adding {len(documents)} documents to vector store...")
        
        # Validate all documents have embeddings
        missing_embeddings = [d.id for d in documents if d.embedding is None]
        if missing_embeddings:
            raise ValueError(f"Documents missing embeddings: {missing_embeddings[:5]}...")
        
        # Store documents and build embedding matrix
        self.documents.extend(documents)
        
        # Stack all embeddings into matrix for batch similarity search
        embeddings_list = [d.embedding for d in self.documents]
        self.embeddings = np.vstack(embeddings_list).astype(np.float32)
        
        # Build ID -> index mapping for quick lookup
        for idx, doc in enumerate(self.documents):
            self.document_id_map[doc.id] = idx
        
        print(f"[LAYER 4] ✓ Added {len(documents)} documents (total {len(self.documents)})")
        print(f"[LAYER 4] Embedding matrix shape: {self.embeddings.shape}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Find top-k documents most similar to query.
        
        Args:
            query_embedding: Query vector (already embedded)
            k: Number of top results
            
        Returns:
            List of (Document, similarity_score) tuples
            
        Algorithm:
        - Similarity = cosine(query, doc) = dot_product(normalized vectors)
        - Time complexity: O(n) where n = num documents
        - With FAISS: O(log n) for indexed search
        
        Example:
            >>> query_emb = embedder.embed_text("What is climate change?")
            >>> results = vector_store.search(query_emb, k=3)
            >>> for doc, score in results:
            >>>     print(f"Score: {score:.3f} | {doc.content[:100]}")
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Compute similarity with all documents (dot product for normalized vectors)
        # Shape: (num_documents,)
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k indices
        top_k_indices = np.argsort(-similarities)[:k]
        
        # Return documents with similarity scores
        results = [
            (self.documents[idx], float(similarities[idx]))
            for idx in top_k_indices
        ]
        
        return results
    
    def __len__(self) -> int:
        return len(self.documents)
    
    def __repr__(self) -> str:
        if self.embeddings is None:
            return "VectorStore(empty)"
        return f"VectorStore({len(self.documents)} docs, embedding_dim={self.embeddings.shape[1]})"


"""
===============================================================================
LAYER 5: Prompt Engineering & Context Building
===============================================================================

Purpose: Create effective prompts that combine retrieved context with queries.

Why separate layer?
- Same retrieval results can lead to very different answers depending on prompt
- Prompt engineering is 50%+ of RAG quality
- Different use cases need different prompt templates

Complexity:
- Context tokenization: must fit LLM context window
- Context ordering: relevance ranking, position bias
- Prompt structure: system message, context, question, output format
- Few-shot examples: demonstrations of desired behavior
"""


class Layer5_PromptBuilder:
    """
    LAYER 5: Build prompts that effectively guide the LLM.
    
    Key principle: Use retrieved context to augment LLM knowledge.
    
    Prompt structure for RAG:
    ┌─────────────────────────────────────┐
    │ SYSTEM MESSAGE                      │
    │ (Define role/behavior)              │
    ├─────────────────────────────────────┤
    │ CONTEXT SECTION                     │
    │ "The following documents provide   │
    │ background information:             │
    │ - [Document 1]                     │
    │ - [Document 2]                     │
    │ - [Document 3]                     │
    ├─────────────────────────────────────┤
    │ USER QUESTION                       │
    │ "Given the context above, please    │
    │ answer: [Question]"                 │
    ├─────────────────────────────────────┤
    │ OUTPUT INSTRUCTIONS                 │
    │ (Format, constraints, citations)    │
    └─────────────────────────────────────┘
    """
    
    def __init__(self, model_type: str = "llama2"):
        """
        Initialize prompt builder.
        
        Args:
            model_type: "llama2", "mistral", "neural-chat", etc.
                       Different models have slightly different optimal prompts
        """
        self.model_type = model_type
    
    def build_rag_prompt(
        self,
        query: str,
        retrieved_documents: List[Document],
        max_context_tokens: int = 3000
    ) -> str:
        """
        Build a RAG prompt combining context and query.
        
        Args:
            query: User's input question
            retrieved_documents: Top-k documents from retrieval
            max_context_tokens: Budget for context (prevent token overflow)
            
        Returns:
            Complete prompt ready to send to LLM
            
        Design for RAG prompt:
        1. Make context explicit (not hidden)
        2. Make source attribution possible
        3. Clear separation between context and question
        4. Guide output format
        
        Anti-patterns to avoid:
        - Implicit context (LLM unsure what's from retrieval)
        - Unsorted context (last document weighted too high)
        - No source info (can't verify claims)
        - Vague task description
        """
        
        # Build context section with documents
        context_parts = []
        context_tokens = 0
        
        for i, doc in enumerate(retrieved_documents, 1):
            # Estimate tokens (1 token ≈ 4 characters)
            doc_tokens = len(doc.content) // 4
            
            if context_tokens + doc_tokens > max_context_tokens:
                print(f"[LAYER 5] Truncating context at document {i} (token limit)")
                break
            
            # Format document with source attribution
            source_info = doc.metadata.get("title", doc.id)
            formatted_doc = f"""[Document {i}] {source_info}
{doc.content}
"""
            context_parts.append(formatted_doc)
            context_tokens += doc_tokens
        
        context_section = "\n".join(context_parts)
        
        # Build complete prompt
        prompt = f"""You are a helpful AI assistant answering questions based on provided documents.

CONTEXT DOCUMENTS:
{context_section}

QUESTION:
{query}

INSTRUCTIONS:
- Answer based on the provided context documents above
- If the answer is not in the context, say "I don't have enough information in the provided documents"
- Cite which document(s) you're pulling information from (e.g., "[Document 1]")
- Be concise and accurate

ANSWER:"""
        
        return prompt
    
    @staticmethod
    def build_simple_prompt(query: str) -> str:
        """
        Fallback: Simple prompt without context (for comparison).
        
        Shows impact of RAG:
        - Without RAG: LLM uses only training knowledge
        - With RAG: LLM uses current knowledge + documents
        """
        return f"""Answer this question based on your knowledge:

Question: {query}

Answer:"""


"""
===============================================================================
LAYER 6: LLM Generation (Powered by OLLAMA)
===============================================================================

Purpose: Generate answer by feeding prompt to local LLM via OLLAMA.

Why OLLAMA?
- Local LLM execution (privacy, no API costs)
- Multiple model support (Llama 2, Mistral, Neural Chat)
- Simple API (curl/HTTP)
- Flexible quantization (q2, q4, q5, fp16)

Complexity:
- Model selection: different models for different use cases
  - Llama 2 13B: best balance (quality/speed)
  - Mistral 7B: fastest, good quality
  - Neural Chat 7B: optimized for chat
- Quantization trade-off: smaller size = faster but lower quality
  - q2: 1-2 bits per weight (very fast, lower quality)
  - q4: 4 bits per weight (fast, decent quality, typically used)
  - q5: 5 bits per weight (slower, better quality)
  - fp16: full precision (fastest with GPU, needs 16GB+ VRAM)
"""


class Layer6_LLMGenerator:
    """
    LAYER 6: Generate answers using OLLAMA-powered LLM.
    
    Architecture integration point:
    PROMPT → OLLAMA LLM → TOKENS → DECODED RESPONSE
    
    Key concepts:
    - Token generation: LLM generates one token at a time (autoregressive)
    - Temperature: controls randomness (0.0 = deterministic, 2.0 = very random)
    - Top-p (nucleus sampling): only consider top p% probability mass
    - Stop sequences: tell LLM when to stop generating
    """
    
    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        """
        Initialize LLM generator.
        
        Args:
            model_name: OLLAMA model name ("llama2", "mistral", "neural-chat", etc.)
            base_url: OLLAMA server address
            
        Recommended models:
        - mistral: fast (7B), good quality, default recommendation
        - llama2: 7B/13B options, good for chat
        - neural-chat: 7B, optimized for conversation
        
        Check available models:
        $ ollama list
        
        Download model:
        $ ollama pull mistral
        """
        self.model_name = model_name
        self.base_url = base_url
        
        print(f"[LAYER 6] Initializing LLM generator...")
        print(f"[LAYER 6] Model: {model_name}")
        print(f"[LAYER 6] Endpoint: {base_url}/api/generate")
        
        self._check_model_available()
    
    def _check_model_available(self):
        """Verify model is available in OLLAMA"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                available_models = response.json().get("models", [])
                model_names = [m["name"].split(":")[0] for m in available_models]
                
                if self.model_name in model_names:
                    print(f"[LAYER 6] ✓ Model '{self.model_name}' available in OLLAMA")
                    return
                
                print(f"[LAYER 6] ⚠ Model '{self.model_name}' not found in OLLAMA")
                print(f"[LAYER 6] Available models: {model_names}")
                print(f"[LAYER 6] Download with: ollama pull {self.model_name}")
            else:
                print(f"[LAYER 6] ⚠ Could not connect to OLLAMA")
                
        except Exception as e:
            print(f"[LAYER 6] ⚠ OLLAMA connection error: {e}")
        
        print(f"[LAYER 6] Guide to set up OLLAMA:")
        print(f"[LAYER 6]   1. Download from https://ollama.ai")
        print(f"[LAYER 6]   2. Run: ollama serve (in one terminal)")
        print(f"[LAYER 6]   3. In another terminal: ollama pull {self.model_name}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 500,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate response by querying OLLAMA.
        
        Args:
            prompt: Complete prompt (context + question)
            temperature: 0.0 (deterministic) to 2.0 (random)
                        RAG typically uses 0.1-0.3 for accuracy
            max_tokens: Max tokens to generate
            top_p: Nucleus sampling parameter (0.9 = use top 90% probability)
            
        Returns:
            Generated text response
            
        Why these parameters?
        - temperature=0.3: RAG needs accurate answers, not creative ones
        - max_tokens=500: Reasonable response length for Q&A
        - top_p=0.95: Filter unlikely tokens while maintaining diversity
        
        Example:
            >>> llm = Layer6_LLMGenerator(model_name="mistral")
            >>> response = llm.generate(prompt)
            >>> print(response)
        """
        print(f"[LAYER 6] Generating response (max_tokens={max_tokens}, temp={temperature})...")
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                    "stream": False,
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()
                elapsed = time.time() - start_time
                
                tokens_generated = len(generated_text.split())
                tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0
                
                print(f"[LAYER 6] ✓ Generated {tokens_generated} tokens in {elapsed:.1f}s ({tokens_per_sec:.0f} tok/s)")
                
                return generated_text
            else:
                error_msg = response.text[:200]
                print(f"[LAYER 6] ✗ OLLAMA error: {error_msg}")
                return f"[Error generating response: {error_msg}]"
                
        except requests.exceptions.Timeout:
            print(f"[LAYER 6] ✗ Request timeout (model too slow)")
            return "[Error: Request timeout - model may be too large for your hardware]"
        except requests.exceptions.ConnectionError:
            print(f"[LAYER 6] ✗ OLLAMA not running")
            return "[Error: Cannot connect to OLLAMA. Please start OLLAMA: ollama serve]"
        except Exception as e:
            print(f"[LAYER 6] ✗ Generation error: {e}")
            return f"[Error: {str(e)[:100]}]"


"""
===============================================================================
LAYER 7: RAG Pipeline Assembly
===============================================================================

Purpose: Combine all layers into a complete, working RAG system.

Complete flow:
1. Load documents (Layer 1)
2. Chunk them (Layer 2)
3. Generate embeddings (Layer 3)
4. Store in vector DB (Layer 4)
5. Build prompt (Layer 5)
6. Generate answer (Layer 6)
7. Return response

This is what users interact with.
"""


class RAGPipeline:
    """
    LAYER 7: Complete RAG system combining all layers.
    
    Usage:
        >>> rag = RAGPipeline()
        >>> rag.build_index()
        >>> response = rag.query("What is climate change?")
    
    Design pattern: RAGPipeline is a facade
    - Hides complexity of all layers
    - Provides simple interface
    - Can swap individual layers (different embedders, LLMs, etc.)
    """
    
    def __init__(
        self,
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "mistral",
        chunk_size: int = 256,
        chunk_overlap: int = 50,
        k_retrieve: int = 3,
    ):
        """
        Initialize RAG pipeline with configurable components.
        
        Args:
            embedding_model: Embedding model name
            llm_model: LLM model name (via OLLAMA)
            chunk_size: Document chunk size in tokens
            chunk_overlap: Overlap between chunks
            k_retrieve: Number of documents to retrieve per query
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieve = k_retrieve
        
        print("\n" + "="*80)
        print("RAG PIPELINE INITIALIZATION")
        print("="*80)
        
        # Initialize components
        self.embedder = Layer3_EmbeddingGenerator(embedding_model=embedding_model)
        self.vector_store = Layer4_VectorStore()
        self.llm = Layer6_LLMGenerator(model_name=llm_model)
        self.prompt_builder = Layer5_PromptBuilder(model_type=llm_model)
        
        self.documents_indexed = False
    
    def build_index(self, max_samples: int = 100):
        """
        Build searchable index from documents.
        
        Pipeline:
        Document Loading → Chunking → Embedding → Vector Store
        
        Args:
            max_samples: Max documents to load (for demo, use small numbers)
        """
        print("\n" + "="*80)
        print("BUILDING RAG INDEX")
        print("="*80)
        
        # Layer 1: Load documents
        documents = Layer1_DocumentLoader.load_cnn_dailymail(max_samples=max_samples)
        
        # Layer 2: Chunk documents
        chunks = Layer2_TextChunker.chunk_documents(
            documents,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap
        )
        
        # Layer 3: Generate embeddings
        chunks_with_embeddings = self.embedder.embed_documents(chunks)
        
        # Layer 4: Add to vector store
        self.vector_store.add_documents(chunks_with_embeddings)
        
        self.documents_indexed = True
        
        print("\n" + "="*80)
        print(f"✓ INDEX COMPLETE: {len(chunks)} chunks ready for retrieval")
        print("="*80)
    
    def query(self, question: str, k: int = None, show_context: bool = True) -> Dict:
        """
        Execute complete RAG query.
        
        Flow:
        Query → Embedding → Retrieval → Context Building → LLM Generation
        
        Args:
            question: User's question
            k: Number of documents to retrieve (use self.k_retrieve if None)
            show_context: Whether to print retrieved documents
            
        Returns:
            Dict with:
            - answer: Generated response
            - sources: Retrieved document IDs
            - scores: Retrieval confidence scores
            - context: Retrieved document texts
        """
        if not self.documents_indexed:
            print("⚠ Index not built. Call build_index() first.")
            return {"error": "Index not built"}
        
        k = k or self.k_retrieve
        
        print("\n" + "="*80)
        print(f"QUERY: {question}")
        print("="*80)
        
        # Step 1: Embed query
        print(f"\n[STEP 1] Embedding query...")
        query_embedding = self.embedder.embed_text(question)
        print(f"[STEP 1] ✓ Query embedding shape: {query_embedding.shape}")
        
        # Step 2: Retrieve documents
        print(f"\n[STEP 2] Retrieving top {k} documents...")
        retrieved = self.vector_store.search(query_embedding, k=k)
        
        if show_context:
            for i, (doc, score) in enumerate(retrieved, 1):
                print(f"\n[RETRIEVED {i}] Similarity: {score:.3f}")
                print(f"Document ID: {doc.id}")
                print(f"Content preview: {doc.content[:150]}...")
        
        sources = [doc.id for doc, _ in retrieved]
        scores = [score for _, score in retrieved]
        retrieved_docs = [doc for doc, _ in retrieved]
        
        # Step 3: Build prompt
        print(f"\n[STEP 3] Building prompt with context...")
        prompt = self.prompt_builder.build_rag_prompt(question, retrieved_docs)
        
        # Step 4: Generate answer
        print(f"\n[STEP 4] Generating answer...")
        answer = self.llm.generate(prompt)
        
        print("\n" + "="*80)
        print("FINAL ANSWER:")
        print("="*80)
        print(answer)
        print("="*80)
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "scores": scores,
            "context": [doc.content for doc in retrieved_docs],
        }
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the built index"""
        return {
            "num_documents": len(self.vector_store.documents),
            "embedding_model": self.embedder.embedding_model,
            "llm_model": self.llm.model_name,
            "embedding_dimension": (
                self.vector_store.embeddings.shape[1]
                if self.vector_store.embeddings is not None
                else None
            ),
        }


"""
===============================================================================
MAIN: Demonstration and Examples
===============================================================================

This section shows how to use the RAG pipeline with real examples.
"""


def main():
    """
    Complete RAG system demonstration.
    
    Requirements:
    1. OLLAMA should be running: ollama serve
    2. Model should be downloaded: ollama pull mistral (or llama2, neural-chat)
    3. Datasets library should be installed: pip install datasets
    """
    
    print("\n")
    print("X"*80)
    print("RAG (RETRIEVAL AUGMENTED GENERATION) SYSTEM - LAYER BY LAYER")
    print("X"*80)
    
    # == INITIALIZATION ==
    print("\nInitializing RAG pipeline...")
    print("This will set up:")
    print("  - Embedding model (for semantic search)")
    print("  - Vector store (for similarity retrieval)")
    print("  - LLM via OLLAMA (for answer generation)")
    
    # Create RAG pipeline
    # Note: These models are downloaded on first use
    rag = RAGPipeline(
        embedding_model="nomic-embed-text",  # ~140MB, optimized for retrieval
        llm_model="mistral",                  # ~13GB for full, ~5GB quantized
        chunk_size=256,
        chunk_overlap=50,
        k_retrieve=3
    )
    
    # == INDEX BUILDING ==
    print("\nStarting index build with CNN/DailyMail dataset...")
    print("This will:")
    print("  1. Download CNN/DailyMail dataset")
    print("  2. Split documents into chunks")
    print("  3. Generate embeddings for each chunk")
    print("  4. Store in vector database")
    print("\nWith 100 articles (~1000 chunks), this takes 2-5 minutes depending on hardware.")
    
    try:
        rag.build_index(max_samples=100)  # Use 100 articles for demo
    except Exception as e:
        print(f"Error building index: {e}")
        print("Troubleshooting:")
        print("  1. Ensure datasets library: pip install datasets")
        print("  2. Check internet connection (for dataset download)")
        print("  3. Check disk space (CNN/DailyMail is ~500MB)")
        return
    
    # == PRINT STATISTICS ==
    print("\n" + "="*80)
    print("INDEX STATISTICS")
    print("="*80)
    stats = rag.get_index_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    # == EXAMPLE QUERIES ==
    print("\n" + "="*80)
    print("EXAMPLE QUERIES")
    print("="*80)
    
    example_questions = [
        "What is climate change?",
        "How does artificial intelligence work?",
        "What are the latest developments in quantum computing?",
    ]
    
    for i, question in enumerate(example_questions, 1):
        print(f"\n[EXAMPLE {i}]")
        try:
            result = rag.query(question, show_context=True)
            
            # Print summary
            print(f"\nSummary:")
            print(f"  Question: {result['question']}")
            print(f"  Retrieved {len(result['sources'])} documents")
            print(f"  Average retrieval score: {np.mean(result['scores']):.3f}")
            print(f"  Answer length: {len(result['answer'].split())} words")
            
        except Exception as e:
            print(f"Query error: {e}")
            print("Troubleshooting:")
            print("  - Ensure OLLAMA is running: ollama serve")
            print(f"  - Ensure model is downloaded: ollama pull mistral")
            print("  - Check available models: ollama list")
    
    print("\n" + "="*80)
    print("RAG DEMO COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

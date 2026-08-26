# LLM Summarization — System Design (Short Overview)

This note captures the high-level, step-by-step system design for an LLM-based summarization application. Deeper dives into each stage (chunking strategies, evaluation, hallucination mitigation) will be added later.

---

## Pipeline at a glance

```
Document(s) → Preprocessing → Chunking (if long) → Summarization Strategy → LLM Inference → Post-processing → Output
```

---

## 1. Input Handling / Document Ingestion
- Accept input: raw text, PDF, transcript, multi-document set.
- Extract clean text (OCR/parsing if PDF, strip boilerplate/HTML).
- Detect language, length, and format up front — this decides the strategy in step 3.

## 2. Preprocessing
- Normalize text (remove noise, fix encoding issues).
- Segment into logical units (paragraphs, sections, sentences) — needed for chunking and later citation/traceability.

## 3. Chunking Decision (the core design fork)
- **If document fits in context window** → single-pass summarization (send whole doc + instruction directly).
- **If document is too long** → must choose a summarization strategy:
  - **Map-Reduce**: summarize each chunk independently ("map"), then summarize the summaries ("reduce"). Simple, parallelizable, but can lose cross-chunk connections.
  - **Refine/Iterative**: summarize chunk 1 → feed that summary + chunk 2 → refine → repeat. Preserves continuity better, but sequential (slower, harder to parallelize).
  - **Hierarchical**: cluster related chunks first (by topic/section), summarize each cluster, then summarize cluster-summaries. Better for structured long documents (reports, books).

## 4. Prompt Construction
- System prompt defines summary style: length, tone, format (bullet points vs prose), abstractive vs extractive.
- Inject the chunk(s)/summary-so-far + instruction ("Summarize preserving key facts, numbers, and names").
- Fit within context window budget (this connects directly to the chunking decision above).

## 5. LLM Inference
- Lower temperature (0.2–0.5) — summarization needs faithfulness, not creativity.
- Set max_tokens appropriate to target summary length.
- Streaming optional (less critical than in chatbots since it's often a batch/async task).

## 6. Post-processing / Guardrails
- **Faithfulness/hallucination check**: verify summary doesn't introduce facts not present in source (critical — this is the #1 failure mode of summarization).
- Deduplication of repeated points (common in map-reduce merges).
- Format enforcement (word/token limit, bullet structure, required sections).

## 7. Evaluation
- Automatic metrics: ROUGE/BERTScore for lexical/semantic overlap (weak signal alone).
- LLM-as-judge or human review for faithfulness, coverage, coherence.
- Track compression ratio (input tokens vs output tokens) as an operational metric.

## 8. Output Delivery & Logging
- Return final summary (with optional citations back to source chunks/sections).
- Log input length, chunking strategy used, latency, and cost for observability.

---

## One-line summary

Summarization system design is *ingest → decide chunking strategy based on length → summarize (single-pass, map-reduce, refine, or hierarchical) → faithfulness-checked post-processing → evaluation*, where the central engineering decision is how to handle documents longer than the context window without losing accuracy or coherence.

---

## Deep Dive: Chunking Strategies

Chunking decides *how* you split a long document into pieces once it doesn't fit in the context window. The choice affects summary quality, coherence, and cost more than almost any other step.

### 1. Fixed-Size Chunking
- Split by a fixed token/word count (e.g., every 512 tokens).
- **Pro**: simple, predictable, easy to batch/parallelize.
- **Con**: can cut sentences or ideas mid-thought, losing context at boundaries.

### 2. Fixed-Size with Overlap
- Same as above, but adjacent chunks share a small overlap (e.g., 50 tokens) so a concept split across the boundary still appears fully in at least one chunk.
- **Pro**: reduces context loss at edges.
- **Con**: slightly more tokens processed (redundant overlap), still somewhat arbitrary cut points.

### 3. Semantic / Structure-Aware Chunking
- Split at natural boundaries: paragraph breaks, section headers, sentence boundaries — not arbitrary token counts.
- **Pro**: preserves complete ideas/units of meaning; much better for coherent summaries.
- **Con**: variable chunk sizes, harder to batch uniformly, may still exceed context limit for a single huge paragraph.

### 4. Recursive Chunking
- Try splitting by largest structural unit first (e.g., sections) → if still too big, split by paragraphs → if still too big, split by sentences → down to fixed-size as last resort.
- **Pro**: adapts to document structure automatically, minimizes unnecessary splitting.
- **Con**: more complex implementation logic.

### 5. Topic/Cluster-Based Chunking
- Embed sentences/paragraphs, cluster semantically similar ones together (even if not adjacent in the raw document), then chunk by cluster.
- **Pro**: groups related content even if scattered (good for transcripts/meeting notes with interleaved topics).
- **Con**: more expensive (needs embedding + clustering step), can break original narrative order.

### 6. Sliding Window (for Refine/Iterative Strategy)
- Not exactly "chunking" the whole doc upfront — instead moves a window across the document sequentially, feeding overlapping context forward as you go (ties directly into the **Refine** summarization strategy from the pipeline).
- **Pro**: strong continuity between chunks.
- **Con**: inherently sequential, slower, harder to parallelize.

### How This Connects to the Summarization Strategy Choice

| Chunking Strategy | Pairs well with |
|---|---|
| Fixed-size / fixed-size + overlap | Map-Reduce (chunks are independent anyway) |
| Semantic / recursive | Hierarchical (natural sections → cluster summaries) |
| Sliding window | Refine/Iterative (sequential continuity matters) |
| Topic/cluster-based | Hierarchical (especially for transcripts, meeting notes) |

### Practical Guidance
- Default safe choice: **recursive chunking with overlap** — adapts to structure, minimizes broken context, works for most document types.
- Use **semantic/topic-based** chunking when document structure is messy (transcripts, chat logs) and topic coherence matters more than raw position.
- Avoid pure fixed-size chunking without overlap for summarization — it's the most likely to cause context loss and factual drift at boundaries.

### One-line summary

Chunking strategy for summarization ranges from simple fixed-size splits to overlap-based, structure-aware, recursive, and topic-clustered approaches — the right choice depends on document structure and directly determines which summarization strategy (map-reduce, refine, or hierarchical) will work best downstream.


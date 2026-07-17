# LLM Revision & Continued Training for Interview Preparation

---

## Q1. What is Continued Training?

**Beginner Understanding:**

Continued training is resuming training of a pre-trained model on new data after the original training is complete. Instead of training from scratch with random weights, you start from a checkpoint and keep optimizing.

Example:
```
Pre-training phase:
  Model trained on general web data (Wikipedia, Common Crawl, etc.)
  Saved checkpoint with learned weights

Continued training phase:
  Load checkpoint weights
  Train on new domain-specific data (medical text, code, legal documents)
  Model adapts to new data distribution
```

**Why use it:**
- **Faster convergence**: Model already learned basic patterns, only needs to adapt
- **Data efficiency**: Requires less new data to reach good performance vs. training from scratch
- **Preserve knowledge**: Retains representations learned from original large dataset
- **Cost-effective**: Less compute and time compared to full retraining

**Intermediate Understanding:**

Continued training involves several key decisions:

1. **Which layers to train?**
   - Full fine-tuning: Update all weights (most compute, most adaptation)
   - Partial fine-tuning: Freeze early layers, train later layers (balance adaptation + stability)
   - Head-only fine-tuning: Freeze backbone, train only final classification head (minimal compute)

2. **What learning rate to use?**
   - Lower than pre-training LR: Smaller steps, less disruption to learned weights
   - Typical: 1e-5 to 1e-3 (depends on task and new data size)
   - Too high: Forgets old knowledge quickly (catastrophic forgetting)
   - Too low: Converges slowly, underutilizes new data

3. **How much new data?**
   - Small new dataset (< 10K examples): Use lower LR, shorter training, regularization
   - Large new dataset (> 100K examples): Can use higher LR, train longer

**Common Scenarios:**

| Scenario | Pre-training | Continued Training | LR | Epochs |
|---|---|---|---|---|
| Domain shift (medical after general) | Web text (billions tokens) | Medical papers (100K docs) | 1e-4 | 3-5 |
| Task adaptation (classification after LLM) | General LLM (7B params) | Task-specific labels (10K) | 1e-5 | 2-3 |
| New language (after English) | English (100B tokens) | Low-resource language (1M tokens) | 5e-5 | 5-10 |

**Senior-Level Interview Answer:**

Continued training is a **transfer learning strategy** that balances adaptation and knowledge retention. The core challenge is managing the tradeoff between learning new patterns (high learning rate, many epochs) and preserving pre-trained knowledge (low learning rate, early stopping).

Key technical considerations:

1. **Catastrophic Forgetting Risk**
   - Problem: Training on new data can degrade performance on old tasks
   - Cause: Weights shift too far from pre-training optimum, old patterns are overwritten
   - Mitigation: Use lower learning rates, monitor validation on old tasks, use regularization

2. **Optimal Learning Rate Selection**
   - Related to pre-training scale and new data distribution shift
   - Heuristic: LR ~ pre-training_LR / sqrt(new_data_tokens / pretrain_tokens)
   - Example: If pre-trained on 1T tokens with LR 1e-4, continuing on 1B tokens → LR ~ 1e-5

3. **Regularization Techniques**
   - **Experience replay**: Mix old data back into training to retain knowledge
   - **Elastic weight consolidation**: Penalize changes to important pre-trained weights
   - **Knowledge distillation**: Use original model as teacher to constrain new model

4. **Evaluation Strategy**
   - Monitor loss on both new and old tasks during training
   - Early stopping based on combined validation metric (weighted average)
   - Test on holdout new data AND benchmark old tasks to measure retention

**Code Pattern (Conceptual):**

```python
# Load pre-trained model
model = load_pretrained_checkpoint("model_checkpoint.pt")

# Lower learning rate for continued training
optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Monitor both new and old task performance
for epoch in range(num_epochs):
    # Train on new data
    for batch in new_train_loader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
    
    # Validate on new data
    new_val_loss = evaluate(model, new_val_loader)
    
    # Validate on old benchmark (catch catastrophic forgetting)
    old_val_loss = evaluate(model, old_benchmark_loader)
    
    # Early stopping if old performance degrades significantly
    if old_val_loss > threshold:
        print("Stopping: Catastrophic forgetting detected")
        break
    
    scheduler.step()
```

**Common Pitfalls:**

| Mistake | Impact | Solution |
|---|---|---|
| Using pre-training LR | Forgets old knowledge rapidly | Reduce LR by 10-100x |
| Training for too many epochs | Overfits to new data, forgets old | Use early stopping on old tasks |
| No validation on old tasks | Silently degrades performance | Monitor both old + new tasks |
| Training all layers equally | Early layers unnecessarily change | Use layer-wise LR scheduling |
| Ignoring data distribution shift | Optimization becomes brittle | Analyze new data vs. pre-train dist |

**Interview One-Liner:**

Continued training adapts a pre-trained model to new domains by resuming optimization from a checkpoint with lower learning rate, carefully balancing learning speed against catastrophic forgetting of pre-trained knowledge—success requires monitoring both new and old task performance simultaneously.

**Key Interview Points:**

1. Continued training ≠ fine-tuning (fine-tuning is narrower, task-specific; continued training is broader domain/data adaptation)
2. Learning rate is the primary lever (controls knowledge retention vs. adaptation)
3. Always validate on old tasks to detect catastrophic forgetting early
4. Regularization techniques become critical with large distribution shifts
5. Continued training scales poorly to extremely different domains (may need pre-training instead)

---

## Key Takeaways

- **Continued training reuses learned representations** from pre-training, making it data-efficient and fast
- **Learning rate is critical**: Too high causes forgetting, too low causes slow adaptation
- **Monitor old + new performance**: Catching catastrophic forgetting early prevents silent failures
- **Use regularization for risky shifts**: When new data distribution differs significantly, use EWC or experience replay
- **Trade-off is unavoidable**: You cannot perfectly retain old knowledge AND fully adapt to new data with the same model capacity

---

## Q2. What is Bi-Encoder?

**High-Level Definition:**

A bi-encoder is a model architecture that **encodes two pieces of text independently** using the same or shared encoder to produce separate embeddings, then compares them using a **similarity metric** (e.g., cosine similarity, dot product).

**Architecture:**
```
Text A → Encoder → Embedding A
Text B → Encoder → Embedding B
         ↓
    Similarity = cos(A, B)
```

**Key Characteristics:**
- **Symmetric**: Both inputs use the same encoding process
- **Efficient for retrieval**: Can pre-compute embeddings for one set (e.g., documents), then quickly match against queries
- **Trade-off**: Fast inference but less expressive than cross-encoders (can't directly compare using cross-attention)

**Common Use Cases:**
- Semantic search (query vs. documents)
- Similarity matching (sentence pairs)
- Duplicate detection
- Retrieval-augmented generation (RAG)

**Example Models:**
- Sentence-BERT (SBERT)
- DPR (Dense Passage Retrieval)
- Contrastive learning models (SimCLR, CLIP)

**Interview Key Insight:**

Bi-encoders are ideal when you need **speed at scale** (pre-computed embeddings) but accept **lower accuracy** compared to cross-encoders. Cross-encoders are more accurate but slower since they compute interaction between texts during inference.

**Speed Comparison:**

| Metric | Bi-Encoder | Cross-Encoder |
|--------|---|---|
| Pre-compute embeddings | ✓ Yes (once) | ✗ No (compute at inference) |
| Query latency | O(1) similarity lookup | O(n) comparisons |
| Accuracy | Lower (independent encoding) | Higher (joint attention) |
| Use case | Large-scale retrieval | Re-ranking top-k results |

**One-liner for Interview:**

A bi-encoder independently encodes two texts into embeddings then compares them via similarity—fast for retrieval at scale but less expressive than interaction-based cross-encoders that use joint attention.

**Practical Pattern:**

```python
# Bi-encoder workflow
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Pre-compute embeddings for corpus (done once)
corpus = ["Document 1", "Document 2", "Document 3"]
corpus_embeddings = model.encode(corpus)  # Cache these

# At query time: fast retrieval
query = "Find relevant documents"
query_embedding = model.encode(query)

# Similarity scores
similarities = model.similarity(query_embedding, corpus_embeddings)
# O(1) operation for large corpus
```

**Why Bi-Encoder vs Cross-Encoder:**

**Use Bi-Encoder when:**
- Corpus is large (millions of documents)
- Need fast retrieval latency (< 100ms)
- Can afford slightly lower ranking quality
- Budget is limited (inference cost matters)

**Use Cross-Encoder when:**
- Corpus is small/medium (< 100K)
- Re-ranking top-k results from bi-encoder
- Quality is critical (fine-grained semantic matching)
- Latency is not strict constraint

**Real-World Pattern:**
```
Bi-Encoder: Fast retrieval of top-100 candidates
  ↓
Cross-Encoder: Precise re-ranking of top-10 results
  ↓
Final ranked list
```

---

## Q3. What is Cross-Encoder?

**High-Level Definition:**

A cross-encoder is a model that **jointly encodes two pieces of text together** using shared attention layers to compute a **direct relevance/similarity score**, rather than creating separate embeddings.

**Architecture:**
```
[Text A] + [Text B] → Joint Encoder → Relevance Score (0-1)
                      (with cross-attention)
```

**Key Characteristics:**
- **Joint processing**: Both texts interact through attention (can "see" each other)
- **Direct scoring**: Outputs a single score, not embeddings
- **More expressive**: Can capture complex interactions between texts
- **Slower inference**: Must process each pair at query time

**Common Use Cases:**
- Re-ranking top-k results from bi-encoder
- Semantic textual similarity with high accuracy
- Question-answer ranking
- Relevance classification

**Example Models:**
- Cross-Encoder variants from Sentence-Transformers
- ELECTRA fine-tuned for ranking
- RankNet-style models

**Bi-Encoder vs Cross-Encoder:**

| Aspect | Bi-Encoder | Cross-Encoder |
|--------|---|---|
| Processing | Independent | Joint (with interaction) |
| Output | Embeddings | Relevance score |
| Speed | Fast (pre-compute) | Slow (per-pair) |
| Accuracy | Lower | Higher |
| Best for | Large-scale retrieval | Re-ranking, precision tasks |

**One-liner for Interview:**

A cross-encoder jointly processes two texts through shared attention to produce a direct relevance score—slower but more accurate than bi-encoders, ideal for re-ranking top candidates.

**Real-World Pattern:**
```
Stage 1: Bi-encoder retrieves 100 candidates (fast)
  ↓
Stage 2: Cross-encoder re-ranks top-10 (slow but accurate)
  ↓
Final ranked list
```

**Why Two-Stage Retrieval:**

```
Bi-encoder alone (fast but imprecise):
  - Misses nuanced semantic relationships
  - Many false positives in top-100
  - Good for broad recall

Cross-encoder alone (precise but slow):
  - Would need O(n) comparisons for corpus of size n
  - Computationally prohibitive for millions of documents
  - Good for high precision on small sets

Two-stage (best of both):
  - Bi-encoder: O(1) retrieval of top-k candidates
  - Cross-encoder: O(k) precise re-ranking of top-k
  - Total cost: O(1) + O(k), where k << n
```

---

## Q4. What is Ranking in RAG/LLM?

**High-Level Definition:**

Ranking is **ordering retrieved documents by relevance to the query**, ensuring the most useful documents are presented first to the LLM.

**Why Ranking Exists:**

```
Problem without ranking:
  - Retrieved documents are arbitrary order
  - LLM sees less relevant docs first (position bias)
  - Answer quality degraded by noise

Solution with ranking:
  - Score each document's relevance
  - Sort by score (highest relevance first)
  - LLM sees best information first
  - Better answer quality
```

**Two-Stage Ranking Pattern:**

```
Stage 1 (Bi-Encoder): Fast retrieval
  Query → Embed → Search vector DB → Get top 100 candidates (O(1))
  
Stage 2 (Cross-Encoder): Precise re-ranking
  Score each of top 100 → Sort → Return top 10 (O(k))
  
Result: Most relevant documents bubble to top
```

**Visual Example:**

```
Query: "What is machine learning?"

BEFORE Ranking:
  Doc 1: "Recipe for chocolate cake" (Similarity: 0.45)
  Doc 2: "Machine Learning 101: Intro to algorithms" (Similarity: 0.92) ← Buried!
  Doc 3: "Python programming tips" (Similarity: 0.38)
  Doc 4: "Statistical methods" (Similarity: 0.71)

AFTER Ranking:
  Doc 2: "Machine Learning 101: Intro to algorithms" (Score: 0.92)
  Doc 4: "Statistical methods" (Score: 0.71)
  Doc 1: "Recipe for chocolate cake" (Score: 0.45)
  Doc 3: "Python programming tips" (Score: 0.38)
  
Result: LLM gets best context first ✓
```

**Key Insight: Position Bias**

LLMs pay disproportionate attention to early context:

```
Context order matters:
  
Good: [Most relevant] → [Medium relevant] → [Less relevant]
  LLM focuses on high-quality info first
  Answer: High quality
  
Bad: [Less relevant] → [Noise] → [Most relevant] → [More relevant]
  LLM contaminated by early noise
  Answer: Lower quality
  
Ranking solves this by putting best docs first
```

**Ranking Mechanisms:**

| Ranker Type | Method | Speed | Accuracy | Best For |
|---|---|---|---|---|
| **BM25** | Keyword/TF-IDF overlap | Fast (ms) | Low | Baseline, sparse retrieval |
| **Bi-Encoder** | Semantic similarity (pre-computed) | Fast (1-10ms) | Medium | Large-scale retrieval |
| **Cross-Encoder** | Joint relevance scoring | Slow (100-500ms) | High | Re-ranking, high precision |
| **LLM-Ranker** | LLM scores relevance | Very slow (1-5s) | Very high | Production pipelines |

**Ranking in RAG Pipeline:**

```
User Query
    ↓
Embed Query
    ↓
RETRIEVAL (Bi-encoder):
  Vector DB search → 100 candidates (fast)
    ↓
RANKING (Ranking Model):
  Score 100 candidates → Sort → Top-10
    ↓
CONTEXT BUILDING:
  Top-10 documents → Prompt construction
    ↓
LLM GENERATION:
  "Answer using this context..."
    ↓
Response (grounded in best documents)
```

**Common Pitfalls:**

| Mistake | Impact | Solution |
|---|---|---|
| No ranking at all | Noisy documents hurt LLM | Use at least bi-encoder ranking |
| Ranking only by keyword match | Misses semantic relevance | Use embedding-based ranking |
| Slow ranker (LLM for every doc) | Latency > 10 seconds | Use fast ranker (cross-encoder on top-k) |
| Rank mismatch between train/test | Ranking doesn't generalize | Validate ranking quality on hold-out set |

**Advanced Ranking Strategies:**

1. **Reciprocal Rank Fusion (RRF)**
   - Combine multiple rankers (BM25 + semantic + custom)
   - Aggregate their rankings
   - More robust than single ranker

2. **LLM-as-Ranker**
   - Use LLM itself to score relevance
   - Highest quality but expensive
   - Use only for top-k candidates

3. **Learning-to-Rank (LTR)**
   - Train ranking model on labeled (query, doc, relevance) data
   - Learns complex relevance patterns
   - Production-grade ranking

**Interview One-Liner:**

Ranking orders retrieved documents by relevance to the query using scoring models (bi-encoders or cross-encoders), ensuring the most useful information reaches the LLM first—critical for answer quality since LLMs exhibit strong position bias toward early context.

**Key Interview Points:**

1. **Position bias is real**: LLM attention weight ∝ position in context (first tokens weighted heavily)
2. **Ranking is a separate model**: Don't rely on retrieval scores alone; use dedicated ranker
3. **Two-stage ranking is standard**: Bi-encoder for speed, cross-encoder for accuracy
4. **Ranking quality directly impacts answer quality**: Bad ranking = bad answers (even with good retrieval)
5. **Ranking is often bottleneck**: Optimize carefully (latency vs accuracy trade-off)

**Production Pattern:**

```python
# Typical production RAG ranking pipeline

# Step 1: Fast retrieval (bi-encoder)
candidates = vector_db.search(query_embedding, k=100)  # Fast: O(1)

# Step 2: Accurate ranking (cross-encoder)
ranked = cross_encoder.rank(query, candidates)  # Slower: O(k) where k=100
top_10 = ranked[:10]

# Step 3: LLM generation with best context
context = format_context(top_10)
answer = llm.generate(prompt + context)

# Result: Answer grounded in best-ranked documents
```

---

## Q2. What is Text Classification in NLP?

**Definition:**

Text classification means mapping a text sequence to one or more predefined labels.

Given text $x = (w_1, w_2, \ldots, w_T)$, learn a function:

$$f_\theta: x \rightarrow y, \quad y \in \mathcal{Y}$$

where $\mathcal{Y}$ is the label set (e.g., {spam, ham}, {positive, negative, neutral}).

---

**Single-Label Multiclass Classification**

Probability for class $k$:
$$p_\theta(y=k \mid x) = \text{softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^{C} e^{z_j}}$$

Prediction:
$$\hat{y} = \arg\max_k p_\theta(y=k \mid x)$$

Cross-Entropy Loss:
$$\mathcal{L}_{CE} = -\sum_{i=1}^{N} \log p_\theta(y_i \mid x_i)$$

---

**Multi-Label Classification**

Probability for label $j$ (independent binary decisions):
$$p_\theta(y_j = 1 \mid x) = \sigma(z_j) = \frac{1}{1 + e^{-z_j}}$$

Binary Cross-Entropy Loss:
$$\mathcal{L}_{BCE} = -\sum_{i=1}^{N} \sum_{j=1}^{C} \left[ y_{ij} \log p_{ij} + (1-y_{ij}) \log(1-p_{ij}) \right]$$

---

**Intuition:** 

Model learns semantic features from text and uses them to separate classes in representation space.

**Evaluation metrics:** Accuracy, macro-F1, micro-F1, ROC-AUC.

**Interview Key Points:**

1. **Single-label assumes exactly one label per instance**; multi-label allows multiple labels
2. **Softmax enforces probabilistic constraint**: $\sum_k p_k = 1$
3. **Sigmoid for multi-label**: Each label decision is independent (probability ∈ [0,1], no constraint to sum to 1)
4. **Architecture**: Typically uses a pre-trained encoder (BERT, RoBERTa) + linear classification head
5. **Common applications**: Sentiment analysis, spam detection, topic categorization, intent classification

---

## Q3. What are Semantic Extraction, Intent Detection, Language Detection, and Topic Modeling?

**1. Semantic Extraction**

Extracts structured meaning (entities, relationships, concepts) from unstructured text.

Given text $x$, extract semantic elements:
$$\text{entities}, \text{relations}, \text{attributes} = \text{SemanticExtractor}_\theta(x)$$

Common formulation (Named Entity Recognition):
$$p_\theta(y_t \mid x, t) = \text{softmax}(W_h h_t + b)$$
where $h_t$ is the contextual representation at token $t$, $y_t$ is the entity tag.

Example: "Apple CEO Tim Cook announced Q4 earnings" → entities: {Apple, Tim Cook}, relation: {CEO-of}.

---

**2. Intent Detection**

Classifies user utterance into predefined intents (what the user wants to do).

Given utterance $u$, detect intent:
$$p_\theta(i \mid u) = \text{softmax}(W [u_{cls}; \text{pooling}(h_1, \ldots, h_T)] + b)$$

where $u_{cls}$ is the [CLS] token representation from encoder, intent $i \in \{\text{book_flight}, \text{check_weather}, \text{order_food}, \ldots\}$.

Cross-Entropy loss:
$$\mathcal{L} = -\log p_\theta(i^* \mid u)$$

Example: "Show me flights to NYC" → intent: book_flight.

---

**3. Language Detection**

Identifies which language a text is written in.

Given text $x$, detect language:
$$p_\theta(\ell \mid x) = \text{softmax}(W h_{\text{agg}} + b)$$

where $h_{\text{agg}}$ is aggregated representation (mean pooling or attention), $\ell \in \{\text{English}, \text{Spanish}, \text{Mandarin}, \ldots\}$.

Often uses character $n$-gram features or subword tokenization to capture language-specific patterns.

Example: "Bonjour, comment allez-vous?" → language: French.

---

**4. Topic Modeling**

Discovers latent topics in a corpus; each document is a mixture of topics, each topic is a distribution over words.

Latent Dirichlet Allocation (LDA) probabilistic model:

$$p(w_d \mid \theta_d, \beta) = \sum_{z=1}^{K} p(w \mid z, \beta) \cdot p(z \mid \theta_d)$$

where:
- $\theta_d$ = topic distribution for document $d$ (Dirichlet prior)
- $z$ = latent topic (1 to K)
- $\beta$ = word distribution per topic
- $w_d$ = observed words in document

Joint likelihood:
$$p(D) = \prod_{d=1}^{M} \int p(\theta_d) \prod_{n=1}^{N_d} \sum_{z=1}^{K} p(z \mid \theta_d) p(w_{d,n} \mid z) d\theta_d$$

Inference via Gibbs sampling or variational inference to estimate $\theta_d$ and $\beta$.

Example: news corpus → topics: {politics, sports, technology, entertainment}.

---

**Key Distinctions:**

| Task | Input | Output | Granularity |
|---|---|---|---|
| Semantic extraction | Text | Entities, relations | Token/phrase level |
| Intent detection | Utterance | Intent class | Utterance level |
| Language detection | Text | Language label | Document level |
| Topic modeling | Corpus | Topics, distributions | Document/corpus level |

---

## Q4. What are POS Tagging, NER, Dependency Parsing, and Constituency Parsing?

**1. Part-of-Speech (POS) Tagging**

Assigns grammatical tags to each word in a sentence.

Given sequence $x = (w_1, w_2, \ldots, w_T)$, predict tag sequence $y = (t_1, t_2, \ldots, t_T)$ where $t_i \in \{\text{NOUN}, \text{VERB}, \text{ADJ}, \text{DET}, \ldots\}$.

Formulation (sequence tagging):
$$p_\theta(y \mid x) = \prod_{t=1}^{T} p_\theta(y_t \mid x, t)$$

Per-token probability:
$$p_\theta(y_t = k \mid x, t) = \text{softmax}(W h_t + b)_k$$

where $h_t$ is contextual representation of token $t$.

Cross-Entropy loss:
$$\mathcal{L}_{POS} = -\sum_{i=1}^{N} \sum_{t=1}^{T_i} \log p_\theta(y_t^* \mid x_i, t)$$

Example: "The cat sat" → {DET, NOUN, VERB}.

---

**2. Named Entity Recognition (NER)**

Identifies entities (person, organization, location, etc.) and their boundaries in text.

Given text $x$, predict entity labels $y_t \in \{\text{O}, \text{B-PER}, \text{I-PER}, \text{B-ORG}, \text{I-ORG}, \ldots\}$ using BIO tagging scheme.

Per-token tagging:
$$p_\theta(y_t \mid x, t) = \text{softmax}(W h_t + b)$$

Sequence likelihood:
$$p_\theta(y \mid x) = \prod_{t=1}^{T} p_\theta(y_t \mid x, t)$$

Can incorporate CRF (Conditional Random Field) for transition constraints:
$$p_\theta(y \mid x) = \frac{\prod_{t=1}^{T} \exp(W_{y_{t-1}, y_t} + s_t(y_t))}{\sum_{y'} \prod_{t=1}^{T} \exp(W_{y'_{t-1}, y'_t} + s_t(y'_t))}$$

where $s_t(y_t)$ is emission score, $W_{y_{t-1}, y_t}$ is transition score.

Example: "John works at Google" → {B-PER, O, O, B-ORG}.

---

**3. Dependency Parsing**

Represents sentence structure as a directed graph where words depend on other words; produces tree with head-dependent relations.

Goal: predict directed edges (head, dependent, relation-type) forming a tree.

Given word $i$, predict its head $h(i)$ and relation $r(i)$:
$$\text{head}_i = \arg\max_j p_\theta(\text{head}=j \mid x, i)$$
$$r_i = \arg\max_r p_\theta(\text{rel}=r \mid x, i, \text{head}=j)$$

Score function (biaffine model):
$$s(i, j) = h_i^T U h_j + h_i^T v$$

where $h_i, h_j$ are representations of word $i$ and candidate head $j$.

Arc loss (maximum spanning tree):
$$\mathcal{L}_{arc} = -\sum_{i=1}^{T} \log p_\theta(\text{head}_i^* \mid x)$$

Example: "The cat sat on the mat" → edges: {sat→cat, sat→on, on→mat, cat→the, mat→the}.

Visual dependency tree:
```
        sat
       /   \
      cat   on
      |     |
     the   mat
           |
          the
```

---

**4. Constituency Parsing**

Represents sentence structure as nested phrases (constituents); produces a tree with phrase structure.

Goal: build hierarchical tree where each internal node is a syntactic category (NP, VP, PP, etc.) and leaves are words.

Chart parsing with dynamic programming:
$$s(i, j, X) = \max_{k, Y, Z} [s(i, k, Y) + s(k, j, Z) + p(X \to Y Z)]$$

where $s(i, j, X)$ is the score for span $(i, j)$ under non-terminal $X$.

Parsing loss (cross-entropy over tree structures):
$$\mathcal{L} = -\log p_\theta(\text{tree}^* \mid x)$$

Example tree for "The cat sat":
```
        S
       /|\
      NP VP
      |  |
     DET NOUN VERB
      |   |    |
     The cat  sat
```

---

**Key Distinctions:**

| Task | Structure | Output Type | Granularity |
|---|---|---|---|
| POS Tagging | Linear sequence | Syntactic labels per token | Token level |
| NER | Linear sequence | Entity boundaries + labels | Span/token level |
| Dependency Parsing | Directed tree | Head-dependent arcs + relations | Pairwise relations |
| Constituency Parsing | Phrase tree | Hierarchical phrase structure | Multi-level hierarchy |

---

**Interview Key Points:**

1. **POS & NER are local tagging problems** (sequence labeling); dependency & constituency parsing are **structural problems** (tree inference)
2. **Dependency parsing is simpler** (O(n³) or O(n²) with biaffine); constituency parsing is **more complex** (requires CKY chart parsing, O(n³|G|))
3. **Constituency trees encode phrase groups**; dependency trees encode grammatical relations
4. **CRF constraints improve tagging by modeling transitions**; CKY constraints improve parsing by enforcing tree structure
5. **Modern neural approaches use Transformers + span/arc scoring** for all four tasks

---

## Q5. What are Machine Translation, Question Answering, Summarization, and Text Generation?

**1. Machine Translation (MT)**

Translates text from source language to target language.

Given source text $x = (w_1^{src}, \ldots, w_T^{src})$, generate target text $y = (w_1^{tgt}, \ldots, w_M^{tgt})$.

Sequence-to-sequence formulation:
$$p_\theta(y \mid x) = \prod_{t=1}^{M} p_\theta(w_t^{tgt} \mid w_{<t}^{tgt}, x)$$

Encoder-decoder architecture:
$$h_{\text{enc}} = \text{Encoder}_\theta(x)$$
$$p_\theta(w_t^{tgt} \mid w_{<t}^{tgt}, x) = \text{softmax}(W \text{Decoder}_\theta(w_{<t}^{tgt}, h_{\text{enc}}))$$

Negative log-likelihood loss (cross-entropy):
$$\mathcal{L}_{MT} = -\sum_{i=1}^{N} \sum_{t=1}^{M_i} \log p_\theta(w_t^{tgt,*} \mid w_{<t}^{tgt,*}, x_i)$$

Example: English "The cat sat" → French "Le chat s'est assis".

---

**2. Question Answering (QA)**

Generates or retrieves answer to a given question, often conditioned on context document(s).

**Extractive QA**: select span from context
$$p_\theta(\text{start}=i, \text{end}=j \mid q, d) = \frac{\exp(s_{\text{start}}(i) + s_{\text{end}}(j))}{\sum_{i',j'} \exp(s_{\text{start}}(i') + s_{\text{end}}(j'))}$$

where $s_{\text{start}}(i), s_{\text{end}}(j)$ are token scores from contextualized encoder.

**Abstractive QA**: generate answer word-by-word
$$p_\theta(y \mid q, d) = \prod_{t=1}^{T} p_\theta(w_t \mid w_{<t}, q, d)$$

Loss:
$$\mathcal{L}_{QA} = -\sum_{i=1}^{N} \log p_\theta(y_i^* \mid q_i, d_i)$$

Example: Question "Where did the cat sit?", Context "The cat sat on the mat" → Answer "on the mat".

---

**3. Summarization**

Compresses a document into a shorter summary preserving key information.

Abstractive summarization (generate summary):
$$p_\theta(y \mid x) = \prod_{t=1}^{M} p_\theta(w_t \mid w_{<t}, x)$$

where $x$ is source document, $y$ is target summary, $M \ll T$ (summary is shorter).

Encoder-decoder with attention:
$$p_\theta(w_t \mid w_{<t}, x) = \text{softmax}(W [\text{decoder}_t; c_t])$$

Context vector (attention over encoder states):
$$c_t = \sum_{s=1}^{T} \alpha_{t,s} h_s^{\text{enc}}, \quad \alpha_{t,s} = \frac{\exp(e_{t,s})}{\sum_{s'} \exp(e_{t,s'})}$$

Loss (cross-entropy):
$$\mathcal{L}_{\text{summ}} = -\sum_{i=1}^{N} \sum_{t=1}^{M_i} \log p_\theta(w_t^{*} \mid w_{<t}^{*}, x_i)$$

Evaluation metrics: ROUGE-1/2/L (recall-based), BERTScore (semantic similarity).

Example: Document → Summary (shorter version capturing key points).

---

**4. Text Generation**

Generates coherent text conditioned on optional context (prompt, topic, style, etc.).

Conditional text generation:
$$p_\theta(y \mid c) = \prod_{t=1}^{T} p_\theta(w_t \mid w_{<t}, c)$$

where $c$ is conditioning context (can be empty for unconditional generation, or specific for conditional).

Per-token probability (autoregressive):
$$p_\theta(w_t \mid w_{<t}, c) = \text{softmax}(W h_t + b)$$

where $h_t = \text{Transformer}_\theta(w_{<t}, c)$.

Decoding strategies:

- **Greedy**: $w_t = \arg\max_w p_\theta(w \mid w_{<t}, c)$
- **Beam search**: keep top-$k$ hypotheses, expand greedily
- **Temperature sampling**: $w_t \sim \text{softmax}(z_t / \tau)$ where $\tau$ controls diversity
- **Top-$p$ sampling**: sample from smallest set with cumulative probability ≥ $p$

Training loss (cross-entropy):
$$\mathcal{L} = -\sum_{i=1}^{N} \sum_{t=1}^{T_i} \log p_\theta(w_t^* \mid w_{<t}^*, c_i)$$

Example: Prompt "Once upon a time" → Generated continuation.

---

**Key Distinctions:**

| Task | Input | Output | Structure | Key Challenge |
|---|---|---|---|---|
| Machine Translation | Source text | Target text (diff. lang) | Seq2seq fixed length | Handling long-range dependencies, word order |
| Question Answering | Question + Context | Answer span/text | Span selection or generation | Reasoning over multiple sentences |
| Summarization | Long document | Short summary | Variable-length compression | Preserving key info, avoiding redundancy |
| Text Generation | Optional prompt | Generated text | Open-ended | Coherence, relevance, diversity |

---

**Interview Key Points:**

1. **MT & Summarization are compression tasks**; QA & Gen are **reasoning/reasoning+generation tasks**
2. **Extractive QA is deterministic**; abstractive is **stochastic and open-ended**
3. **Beam search provides better quality but higher latency** vs greedy; temperature sampling adds **controlled diversity**
4. **ROUGE metrics are limited** for summarization (not semantic); newer metrics use embeddings or pre-trained models
5. **Teacher forcing during training vs autoregressive decoding at test time** causes **exposure bias** — all four tasks suffer from this; solutions include scheduled sampling, beam search training
6. **Seq2seq models require alignment** between input/output; Transformers handle this via cross-attention

---

## Q6. Evaluation Metrics for Sentiment Analysis, NER, and Machine Translation

**1. Sentiment Analysis Evaluation Metrics**

Sentiment analysis is a classification task. Common metrics:

**Accuracy** (overall correctness):
$$\text{Accuracy} = \frac{\text{# correct predictions}}{\text{# total samples}} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision** (correctness of positive predictions):
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall** (coverage of actual positives):
$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1-Score** (harmonic mean):
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Macro-F1** (average F1 across classes, for imbalanced data):
$$\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^{C} F1_c$$

Example: Binary sentiment (positive/negative)
```
Predicted: Pos  Neg
Actual Pos:  95   5   (TP=95, FN=5)
       Neg:  10  90   (FP=10, TN=90)

Precision = 95/(95+10) = 0.905
Recall = 95/(95+5) = 0.95
F1 = 2 * (0.905 * 0.95) / (0.905 + 0.95) = 0.927
```

---

**2. Named Entity Recognition Evaluation Metrics**

NER is sequence tagging but evaluated at **entity level** (not token level).

**Entity-level Precision**:
$$\text{Precision} = \frac{\text{# correctly predicted entities}}{\text{# predicted entities}}$$

**Entity-level Recall**:
$$\text{Recall} = \frac{\text{# correctly predicted entities}}{\text{# gold entities}}$$

**F1-Score**:
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Example:
```
Gold:      B-PER I-PER  O  B-LOC
Predicted: B-PER B-PER  O  B-LOC
           
Gold entities: [PER: John Smith (0-2)], [LOC: NYC (3-4)]
Predicted:     [PER: John (0-1)], [PER: Smith (1-2)], [LOC: NYC (3-4)]

Correct predictions: 1 (LOC: NYC)
False positives: 2 (wrong PER boundaries)
False negatives: 1 (missed full PER: John Smith)

Precision = 1/(1+2) = 0.33
Recall = 1/(1+1) = 0.5
F1 = 0.4
```

---

**3. Machine Translation Evaluation Metrics**

MT is sequence generation; metrics measure similarity between MT output and reference translation(s).

**BLEU** (Bilingual Evaluation Understudy) - precision-based, n-gram overlap:
$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

where:
- $p_n$ = precision of $n$-grams (typically $N=4$)
- $\text{BP}$ = brevity penalty (penalizes short translations):

$$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ e^{1-r/c} & \text{if } c \leq r \end{cases}$$

where $c$ = candidate length, $r$ = reference length.

**METEOR** (Metric for Evaluation of Translation with Explicit ORdering) - recall-based, includes synonyms:
$$\text{METEOR} = (1 - \gamma) \cdot F_{\text{align}} - \gamma \cdot \text{penalty}$$

where $F_{\text{align}}$ is harmonic mean of precision/recall over aligned unigrams.

**TER** (Translation Edit Rate) - edit distance:
$$\text{TER} = \frac{\text{# insertions + deletions + substitutions + shifts}}{\text{# reference words}}$$

Lower TER is better.

**BERTScore** (semantic similarity using embeddings):
$$\text{BERTScore-F} = 2 \cdot \frac{P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}$$

where $P_{\text{BERT}}, R_{\text{BERT}}$ measure cosine similarity between contextualized embeddings.

Example:
```
Reference: "The quick brown fox jumps"
MT1:       "The fast brown fox jumps"  (synonym)
MT2:       "The quick brown fox jump"  (missing 's')

BLEU(MT1) ≈ 0.67
BLEU(MT2) ≈ 0.67

BERTScore(MT1) ≈ 0.98 (semantically similar)
BERTScore(MT2) ≈ 0.95 (slight verb difference)

TER(MT1) = 0.2 (1 substitution)
TER(MT2) = 0.2 (1 deletion)
```

---

**Key Distinctions:**

| Task | Metric Type | Key Metrics | Properties |
|---|---|---|---|
| Sentiment Analysis | Classification | Accuracy, Precision, Recall, F1, Macro-F1 | Class-wise breakdown important |
| NER | Sequence Tagging | Entity F1 (exact match) | Entity-level not token-level |
| Machine Translation | Sequence Generation | BLEU, METEOR, TER, BERTScore | Multiple references, semantic aware |

---

**Interview Key Points:**

1. **Sentiment Analysis**: Macro-F1 better for imbalanced data; confusion matrix reveals class-specific errors
2. **NER**: Evaluate at **entity level** (exact match), NOT token accuracy (misleading)
3. **MT BLEU limitations**: Insensitive to synonyms; BERTScore fixes this
4. **MT metrics to use together**: BLEU + METEOR (diverse views); add BERTScore for semantic quality
5. **TER interpretable** (actual edit count) but slower than BLEU
6. **No perfect metric**: Use multiple metrics + human evaluation for validation

---

## Q7. What is Token Representation in NLP?

Token representation converts discrete tokens into numerical vectors that models can process.

Given a vocabulary $V$ with size $|V|$, each token $w_i$ is mapped to a dense vector:
$$e_i = E[w_i] \in \mathbb{R}^d$$

where $E \in \mathbb{R}^{|V| \times d}$ is the embedding matrix, $d$ is the embedding dimension.

---

**1. One-Hot Encoding (baseline, sparse)**

$$e_i \in \{0,1\}^{|V|}, \quad e_i[j] = \begin{cases} 1 & \text{if } j = \text{index}(w_i) \\ 0 & \text{otherwise} \end{cases}$$

Problem: no semantic similarity captured, dimensionality = $|V|$ (very high, sparse).

---

**2. Static Word Embeddings (Word2Vec, GloVe)**

Dense, learned vector per token, fixed regardless of context.

Word2Vec (Skip-gram) objective:
$$\mathcal{L} = -\sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j} \mid w_t)$$

$$p(w_{t+j} \mid w_t) = \frac{\exp(v'^{\top}_{w_{t+j}} v_{w_t})}{\sum_{w=1}^{|V|} \exp(v'^{\top}_{w} v_{w_t})}$$

where $v_{w_t}$ is the "input" embedding, $v'_{w}$ is "output" embedding, $c$ is context window size.

Limitation: same vector for a word regardless of context (e.g., "bank" in river vs finance context).

---

**3. Contextual Embeddings (Transformer-based)**

Token representation depends on surrounding context — same word gets different vectors in different sentences.

$$h_i = \text{Transformer}_\theta(w_1, w_2, \ldots, w_T)_i$$

where $h_i \in \mathbb{R}^d$ is the contextualized representation at position $i$, computed via self-attention over all tokens:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

Final token representation combines token embedding + positional embedding:
$$h_i^{(0)} = E[w_i] + P[i]$$

then passed through $L$ transformer layers to get contextualized $h_i^{(L)}$.

---

**Comparison:**

| Type | Context-aware? | Dimensionality | Examples |
|---|---|---|---|
| One-hot | No | $\|V\|$ (sparse) | — |
| Static embedding | No | $d$ (dense, fixed ~100-300) | Word2Vec, GloVe |
| Subword embedding | No (before context) | $d$ | BPE, WordPiece embeddings |
| Contextual embedding | Yes | $d$ (dense, ~768-4096) | BERT, GPT, LLaMA |

---

**Interview Key Points:**

1. **Subword tokenization** (BPE/WordPiece) solves out-of-vocabulary problem before embedding lookup
2. **Static embeddings suffer from polysemy**: one vector per word regardless of meaning
3. **Contextual embeddings solve polysemy** via self-attention conditioning on full sequence
4. **Positional encoding is essential** for Transformers since attention is permutation-invariant without it
5. **Embedding dimension trade-off**: higher $d$ captures more nuance but increases parameters ($|V| \times d$) and compute cost

---

## Q8. RNN Problems, How LSTM Solves Them, and Remaining LSTM Problems

**1. Vanilla RNN — The Core Problem**

RNN hidden state update:
$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

During backpropagation through time (BPTT), the gradient w.r.t. an earlier hidden state involves a **product of Jacobians** across all timesteps:
$$\frac{\partial \mathcal{L}}{\partial h_1} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{t=2}^{T} W_h^\top \, \text{diag}(\tanh'(\cdot))$$

Since $|\tanh'(\cdot)| \le 1$ and this term is multiplied $T$ times:
- If the dominant eigenvalue of $W_h$ is $< 1$ → gradients shrink exponentially → **vanishing gradient** (can't learn long-range dependencies)
- If $> 1$ → gradients grow exponentially → **exploding gradient** (unstable training, NaN losses)

Practical consequence: RNNs effectively "forget" information beyond ~10-20 timesteps.

---

**2. How LSTM Solves This**

LSTM introduces a **cell state** $C_t$ with an (mostly) **additive** update instead of purely multiplicative:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

where gates are:
$$f_t = \sigma(W_f[h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}$$
$$i_t = \sigma(W_i[h_{t-1}, x_t] + b_i) \quad \text{(input gate)}$$
$$\tilde{C}_t = \tanh(W_C[h_{t-1}, x_t] + b_C) \quad \text{(candidate)}$$
$$o_t = \sigma(W_o[h_{t-1}, x_t] + b_o), \quad h_t = o_t \odot \tanh(C_t) \quad \text{(output gate)}$$

Key insight: the gradient path through $C_t$ is:
$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

This is a **direct, additive, gated pathway** (the "constant error carousel"). If $f_t \approx 1$, gradients flow through many timesteps almost unchanged — no repeated multiplication by $W_h$ and no repeated $\tanh'$ squashing. This is why LSTM handles dependencies over 100+ steps, vs ~10-20 for vanilla RNN.

---

**3. LSTM Still Has Problems**

**a) Vanishing gradients still happen, just less severely**
If the forget gate $f_t \to 0$ (network learns to forget), the gradient path is still cut. Over very long sequences (1000+ tokens), $\prod_t f_t \to 0$ is still possible.

**b) Strictly sequential computation — no parallelization**
$$h_t = \text{LSTMCell}(h_{t-1}, x_t)$$
Each timestep depends on the previous one → cannot parallelize across the sequence dimension during training. This is $O(T)$ sequential steps, making LSTM training/inference slow on GPUs (vs Transformer's $O(1)$ sequential depth with full parallelism across tokens).

**c) Limited effective context in practice**
Even though LSTM *can* theoretically model longer dependencies than RNN, empirically effective context is still limited (~100-200 tokens) — nowhere near Transformer's 128K+ context windows.

**d) No direct token-to-token comparison**
LSTM must compress ALL history into a single fixed-size vector $h_t \in \mathbb{R}^d$ — this is an information bottleneck. Transformers instead let every token attend directly to every other token via:
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
avoiding the bottleneck entirely.

**e) Harder optimization at scale**
LSTMs don't scale as gracefully as Transformers with data/compute (no clean scaling laws), and gating complexity adds more hyperparameters and instability at very large hidden sizes.

---

**Interview Key Points:**

1. **RNN vanishing/exploding gradients**: caused by repeated multiplication of $W_h^\top$ and $\tanh'$ across $T$ timesteps in BPTT
2. **LSTM fix**: additive cell-state update with gates creates a "constant error carousel" — gradient path is $f_t$ instead of $W_h^\top \tanh'$
3. **LSTM is still sequential**: $O(T)$ steps, no parallelization across time — major bottleneck vs Transformers
4. **LSTM still has an information bottleneck**: fixed-size hidden vector must compress all history, unlike Transformer's full pairwise attention
5. **This is exactly why Transformers replaced LSTMs** for large-scale language modeling: parallel training, no compression bottleneck, longer effective context

---

## Q9. Why Multi-Head Attention in NLP?

**1. Single-head attention has a representational bottleneck**

Standard scaled dot-product attention:
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

With a single head, the model computes **one** weighted average over $V$ per query — it must compress *all* relevant relationships (syntactic, semantic, positional, coreference, etc.) into a single softmax distribution. That's a severe bottleneck: one attention pattern cannot simultaneously capture "subject-verb agreement" AND "long-range coreference" AND "local n-gram structure" for the same token.

---

**2. Multi-head formulation**

Project $Q, K, V$ into $h$ independent, lower-dimensional subspaces:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V), \quad i = 1, \ldots, h$$

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

where $W_i^Q, W_i^K \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$, and typically $d_k = d_v = d_{model}/h$ so total compute is comparable to one full-dimension head.

---

**3. Why this helps (the real reason)**

Each head learns to attend in a **different representation subspace**, so different heads can specialize:
- Head A: local syntax (adjacent word dependencies)
- Head B: long-range coreference ("it" → "the cat")
- Head C: positional/rhythm patterns
- Head D: rare-token / entity attention

Empirically (Vaswani et al. 2017, and later interpretability work like Clark et al. 2019 "What Does BERT Look At?"), heads do specialize this way — some heads attend almost entirely to the previous token, others to sentence-final tokens, others to syntactic heads/dependents.

Mathematically, this is analogous to an ensemble: instead of one softmax over $d_{model}$ dimensions, you get $h$ independent softmax distributions, each with its own $QK^\top$ geometry — increasing the **expressive capacity** of the attention mechanism without increasing total parameter count much (since $d_k = d_{model}/h$ keeps compute roughly constant).

---

**4. Trade-offs**

- Too many heads with too-small $d_k$ per head → each head has limited capacity, can become redundant (empirically, many heads *can* be pruned post-training with minimal accuracy loss — Michel et al. 2019, "Are Sixteen Heads Really Better than One?")
- Too few heads → back to the single-head bottleneck problem
- Standard choice: $h=8$ (original Transformer, $d_{model}=512$), scaling up to $h=32$–$96$ in large LLMs, keeping $d_k$ typically 64-128

---

**Interview Key Points:**

1. **Single-head bottleneck**: one softmax distribution can't capture syntax + coreference + position simultaneously
2. **Multi-head = parallel subspace attention**: each head has its own $Q,K,V$ projections, specializing in different relationship types
3. **Compute-neutral design**: $d_k = d_{model}/h$ keeps total compute roughly the same as one full-width head
4. **Heads empirically specialize**: interpretability studies show distinct attention patterns per head (local, positional, coreference, entity)
5. **Redundancy exists**: many heads can be pruned post-training with minimal accuracy loss — motivates efficient attention variants (grouped-query attention, multi-query attention)

---

## Q10. What is Label Smoothing in NLP?

**1. The problem with standard cross-entropy**

Standard training uses one-hot target distribution $y$:
$$y_k = \begin{cases} 1 & k = \text{true class} \\ 0 & \text{otherwise} \end{cases}$$

Cross-entropy loss:
$$\mathcal{L}_{CE} = -\sum_k y_k \log p_k = -\log p_{\text{true}}$$

To minimize this, the model is pushed to make $p_{\text{true}} \to 1$ and all other logits $\to -\infty$. This causes:
- **Overconfidence**: model outputs near-certain probabilities even when wrong
- **Poor calibration**: predicted probability doesn't reflect true likelihood of correctness
- **Overfitting to training labels**, especially harmful when labels are noisy

---

**2. Label smoothing formulation**

Replace the hard one-hot target with a **softened** distribution:
$$y_k^{LS} = (1-\epsilon) \cdot y_k + \frac{\epsilon}{K}$$

where $\epsilon$ is a small smoothing parameter (typically 0.1), $K$ is the number of classes.

So for the true class:
$$y_{\text{true}}^{LS} = (1-\epsilon) + \frac{\epsilon}{K}$$

For all other classes:
$$y_{\text{other}}^{LS} = \frac{\epsilon}{K}$$

Loss becomes:
$$\mathcal{L}_{LS} = -\sum_k y_k^{LS} \log p_k = (1-\epsilon)(-\log p_{\text{true}}) + \frac{\epsilon}{K}\sum_k(-\log p_k)$$

This is equivalent to a weighted combination of the original CE loss and the KL divergence to a uniform distribution $u$:
$$\mathcal{L}_{LS} = (1-\epsilon)\mathcal{L}_{CE} + \epsilon \cdot D_{KL}(u \| p)$$

---

**3. Effect on training**

- The target for the true class is now $< 1$ (e.g., 0.9 instead of 1.0 with $\epsilon=0.1$, $K$ large)
- This **caps the maximum logit gap** the model is incentivized to produce — the optimal logit for the true class becomes finite instead of $+\infty$
- Prevents the network from driving other logits arbitrarily low, which **improves calibration** (predicted confidence matches empirical accuracy better)
- Acts as a regularizer, similar in spirit to weight decay but applied to the label distribution instead of the weights

---

**4. Where it's used in NLP**

- **Machine translation** (original Transformer paper, Vaswani et al. 2017, uses $\epsilon = 0.1$) — improved BLEU despite worse perplexity (since perplexity measures exact probability match, and label smoothing intentionally reduces peak probability)
- **Sequence-to-sequence generation** (summarization, captioning) to reduce exposure to overconfident wrong tokens during beam search
- Less common in modern autoregressive LLM pretraining at massive scale (plain cross-entropy is standard there), but still used in classification heads, NER, and fine-tuning setups prone to label noise

---

**5. Trade-offs**

- **Improves calibration and generalization**, especially with noisy labels
- **Can hurt metrics that reward sharp/peaked distributions** (e.g., perplexity gets slightly worse even though downstream quality like BLEU improves)
- Choosing $\epsilon$ too high → underconfident model, slower convergence
- Interacts with **beam search**: smoother probabilities can change search dynamics (label smoothing is known to affect BLEU positively in MT specifically because it avoids single dominant hypothesis collapse)

---

**Interview Key Points:**

1. **Root cause fixed**: standard CE incentivizes $p_{\text{true}} \to 1$ and other logits $\to -\infty$, causing overconfidence
2. **Soft target formula**: $y_k^{LS} = (1-\epsilon)y_k + \epsilon/K$, equivalent to $(1-\epsilon)\mathcal{L}_{CE} + \epsilon \cdot D_{KL}(u \| p)$
3. **Improves calibration**, not raw likelihood — perplexity can get slightly worse while downstream quality (BLEU) improves
4. **Classic use case**: original Transformer MT training used $\epsilon=0.1$
5. **Trade-off**: too high $\epsilon$ → underconfident, slower convergence; interacts with beam search dynamics

---

## Q11. Sinusoidal Position Embedding vs RoPE — Why, How, and Why RoPE Came After

**1. Why position embeddings are needed at all**

Self-attention is **permutation-invariant** — it computes weighted sums over all tokens regardless of order. Without positional information, "dog bites man" and "man bites dog" would produce identical attention outputs. Since Transformers dropped recurrence/convolution (which naturally encode order via sequential processing), we need an explicit mechanism to inject sequence order into the model.

---

**2. Sinusoidal Position Embeddings (original Transformer, Vaswani et al. 2017)**

Formula:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

where $pos$ = token position, $i$ = dimension index, $d$ = embedding dimension.

**How it's used in training:**
- Precomputed once (not learned) as a fixed matrix of shape `(max_seq_len, d_model)`.
- **Added** (not concatenated) directly to token embeddings: `input = token_embedding + positional_embedding`, before the first encoder/decoder layer.
- Since it's just element-wise addition, gradients flow through backprop like any other input — the PE table itself is fixed (non-parametric), only downstream weights get updated.

**Why sinusoidal specifically:**
- Each dimension corresponds to a different frequency (geometric progression from $2\pi$ to $10000 \cdot 2\pi$), giving each position a unique "fingerprint."
- Key property: $PE_{pos+k}$ can be expressed as a **linear function** of $PE_{pos}$ (via trigonometric addition identities), so the model can theoretically learn to attend by *relative* offset, not just absolute position.
- Generalizes to sequence lengths longer than seen in training (unlike learned absolute embeddings, which have no representation for unseen positions).

**Limitation:** Despite the relative-position math property, sinusoidal PE is added **once at the input**. As it propagates through many self-attention layers, that clean relative-position signal gets diluted/entangled with content information. Attention scores $QK^T$ mix content and absolute position additively, not in a way that cleanly isolates relative distance.

---

**3. RoPE — Rotary Position Embedding (Su et al. 2021)**

**Why it was needed despite sinusoidal PE already existing:**
1. Sinusoidal/absolute PE injects position only at the input layer — relative position information is implicit and degrades through depth.
2. Learned absolute PE (BERT-style) can't extrapolate beyond max trained length at all.
3. We actually want attention scores to depend on **relative distance between tokens**, not absolute positions — more linguistically meaningful (e.g., "the word 3 tokens before me" matters more than "I'm at position 47").

**Core idea:** Instead of *adding* a position vector, RoPE **rotates** the query and key vectors in a multi-dimensional space by an angle proportional to their position, *before* computing the dot product.

For a 2D pair of dimensions, position $m$ rotates the vector by angle $m\theta$:
$$f(x, m) = \begin{pmatrix}\cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta\end{pmatrix} x$$

Applied across all dimension pairs with different frequencies $\theta_i$ (similar geometric schedule to sinusoidal).

**The key trick:** When computing $q_m^T k_n$ (attention dot product) between rotated query at position $m$ and rotated key at position $n$, the result depends **only on $(m-n)$**, the relative distance — the absolute rotation cancels out algebraically. This bakes relative position awareness directly into every attention computation, not just at the input.

**How it's applied in training:**
- No separate embedding table added to inputs. At each attention layer, Q and K vectors are rotated using precomputed sin/cos tables indexed by position, right before the $QK^T$ dot product.
- Parameter-free (like sinusoidal) but re-applied at every layer's attention, not just once at the input.

**Advantages over sinusoidal:**
- Naturally decays attention with distance (long-range tokens get less "coherent" rotation alignment), matching intuitive linguistic locality bias.
- Better length extrapolation in practice (with techniques like NTK-aware scaling / position interpolation, used in LLaMA, GPT-NeoX).
- Cleaner theoretical guarantee of relative-position dependence at every layer, not just approximately at the input.

---

**Interview Key Points:**

1. **Position embeddings exist** because self-attention is permutation-invariant and needs explicit order information.
2. **Sinusoidal PE** is added once at the input as a fixed, non-learned matrix; has a nice relative-position math property but it degrades through depth since it's injected only once.
3. **RoPE rotates Q/K vectors at every attention layer** so that the dot product depends purely on relative distance $(m-n)$, not absolute position.
4. **RoPE came after sinusoidal PE** because we wanted relative position info to persist consistently through all layers (not just at input) and to extrapolate better to longer sequences.
5. **RoPE is the default in modern LLMs** (LLaMA, Mistral, Qwen, GPT-NeoX family) due to better long-context extrapolation and cleaner relative-position modeling.

---

## Q12. What is Sliding Window Attention (SWA)?

**1. The problem it solves**

Standard full self-attention computes attention scores between **every pair of tokens** in a sequence, giving $O(n^2)$ time and memory complexity where $n$ is sequence length. For long sequences (e.g., 32K, 100K tokens), this becomes prohibitively expensive — both in compute and GPU memory (the $n \times n$ attention matrix explodes).

**2. Core idea**

Instead of letting each token attend to *all* other tokens, sliding window attention restricts each token to only attend to a fixed-size local window of nearby tokens (e.g., the previous $w$ tokens on each side).

$$\text{Attention}(q_i) \rightarrow \text{keys/values from } j \in [i-w, i+w]$$

This reduces complexity from $O(n^2)$ to $O(n \cdot w)$, since each token only computes attention scores against $w$ neighbors instead of all $n$ tokens.

**3. How it's implemented**

- A **banded/local attention mask** is applied to the $QK^T$ score matrix — positions outside the window $[i-w, i+w]$ are masked to $-\infty$ before softmax, just like causal masking but with an additional distance cutoff.
- Window size $w$ is a fixed hyperparameter (e.g., 4096 tokens in Mistral).
- Often combined with **causal masking** in decoder-only LLMs, so a token only attends to the window of *past* tokens, not future ones.

**4. Why it still works for long-range dependencies**

Even though each layer only sees a local window, stacking multiple layers effectively expands the **receptive field**, similar to how CNNs build large receptive fields from small kernels:

$$\text{receptive field at layer } L \approx L \times w$$

So after enough layers, information from distant tokens can indirectly propagate through intermediate tokens — layer 1 mixes info within a window, layer 2 mixes windows-of-windows, etc.

**5. Real-world usage**

- **Mistral 7B** uses sliding window attention (window = 4096) combined with a rolling KV-cache, enabling it to handle sequences longer than the window through this layer-stacking receptive field growth, while keeping memory bounded.
- **Longformer** combines sliding window attention with a few **global attention** tokens (e.g., `[CLS]`, question tokens in QA) that attend to everything, blending local efficiency with global context where needed.
- **BigBird** similarly combines local (sliding window) + global + random attention patterns for a sparse but effective full-context approximation.

**6. Trade-offs**

- **Big win**: linear-ish memory/compute in sequence length → enables much longer context windows in practice.
- **Risk**: pure local windows alone can lose direct long-range dependencies within a single layer — mitigated by depth (receptive field growth) or hybrid global tokens.
- Works especially well combined with **rolling/sliding KV-cache** during inference, since you don't need to store the full history, just the current window — big memory savings for generation.

---

**Interview Key Points:**

1. **Motivation**: reduces $O(n^2)$ full attention to $O(n \cdot w)$ by restricting attention to a local neighborhood.
2. **Mechanism**: banded attention mask limiting each token's attention to $[i-w, i+w]$ (or causal variant $[i-w, i]$).
3. **Long-range info still flows** via receptive field growth across stacked layers (like CNN depth).
4. **Used in production LLMs**: Mistral (pure sliding window + rolling cache), Longformer/BigBird (hybrid local + global + random).
5. **Trade-off**: efficiency vs. potential loss of direct single-hop long-range attention — addressed via depth or added global tokens.

---

## Q13. What is Multi-Query Attention (MQA)?

**1. The problem it solves**

In standard **Multi-Head Attention (MHA)**, each of the $h$ heads has its **own separate Key and Value projections**, so during autoregressive generation you must cache separate K/V tensors per head. For long sequences and large batch sizes, this **KV-cache** becomes a major memory bottleneck during inference — it's often the dominant cost, not the model weights themselves.

**2. Core idea**

Multi-Query Attention (Shazeer, 2019) keeps **multiple query heads** (as in standard MHA) but shares a **single Key and Value projection** across all heads:

$$Q_i = X W_Q^i \quad (i = 1, \dots, h \text{ separate heads})$$
$$K = X W_K, \quad V = X W_V \quad (\text{shared across all heads})$$

So instead of $h$ separate $K, V$ pairs, there's just **one** $K, V$ pair, and all $h$ query heads attend using it.

**3. Why this helps at inference time**

- **KV-cache size drops by a factor of $h$** (number of heads) — e.g., with 32 heads, cache memory shrinks ~32×.
- This directly speeds up autoregressive decoding since memory bandwidth (loading the KV-cache at every generation step) is often the bottleneck, not compute (FLOPs).
- Smaller cache also means **larger batch sizes** fit in GPU memory, improving throughput for serving.

**4. Trade-off**

- Sharing one K/V across all heads reduces representational diversity — different heads can no longer learn to attend to different "views" of keys/values, only different query projections. This can **slightly hurt model quality/accuracy** compared to full MHA, especially at larger scale.
- It's a **quality vs. inference-efficiency trade-off**, primarily motivated by production serving costs, not training-time quality.

**5. Grouped-Query Attention (GQA) — the middle ground**

Because pure MQA can hurt quality, **GQA** (used in LLaMA-2 70B, Mistral, etc.) is a compromise:
- Query heads are divided into $g$ groups, and each group shares its own K/V projection (instead of 1 shared pair for all heads, or $h$ separate pairs).
- $g=1$ recovers MQA; $g=h$ recovers standard MHA.
- GQA retains most of the inference speed benefit of MQA while recovering most of the quality of full MHA — it's now the standard choice in most modern open LLMs.

**6. Where it's used**

- **PaLM** was among the first large models to adopt MQA for efficient serving.
- **Falcon** uses MQA.
- **LLaMA-2/3, Mistral, Qwen** use **GQA** as the practical middle-ground default.

---

**Interview Key Points:**

1. **Motivation**: reduce KV-cache memory and memory-bandwidth bottleneck during autoregressive decoding.
2. **Mechanism**: multiple query heads, but a single shared K/V projection (vs. per-head K/V in standard MHA).
3. **Benefit**: ~$h\times$ smaller KV-cache → faster decoding, bigger batch sizes, cheaper serving.
4. **Cost**: reduced representational diversity across heads → potential quality drop at scale.
5. **GQA is the practical successor** — groups of heads share K/V, balancing MQA's speed with MHA's quality; used in LLaMA-2/3, Mistral, Qwen.

---

## Q14. What is Segment Embedding in BERT Architecture?

**1. Why they're needed**

BERT is trained with two objectives, one of which — **Next Sentence Prediction (NSP)** — requires processing **two sentences packed into a single input sequence** (e.g., Sentence A and Sentence B), separated by a `[SEP]` token. The model needs to know **which tokens belong to Sentence A vs. Sentence B**, since self-attention alone has no notion of "sentence boundary" beyond the `[SEP]` token itself. Segment embeddings provide this explicit signal.

**2. How it works**

BERT's final input representation for each token is the **sum of three embeddings**:

$$\text{Input}_i = \text{TokenEmbedding}_i + \text{PositionEmbedding}_i + \text{SegmentEmbedding}_i$$

- **Segment embedding** is a learned embedding from a table of just **2 rows** (for the standard 2-segment case): $E_A$ for "Sentence A" and $E_B$ for "Sentence B".
- Every token in Sentence A (including `[CLS]` and the first `[SEP]`) gets $E_A$ added; every token in Sentence B (including the final `[SEP]`) gets $E_B$ added.

**Example input construction:**
```
Tokens:   [CLS] the cat sat [SEP] it slept [SEP]
Segment:    A    A   A   A    A    B   B     B
```

**3. How it's trained**

- The segment embedding table ($E_A$, $E_B$) is a small learnable parameter matrix, just like token/position embeddings — updated via standard backprop during pretraining.
- Used jointly with the **NSP task**: given Sentence A and Sentence B, the model predicts via the `[CLS]` token's final representation whether B actually follows A in the original text, or is a random sentence. Segment embeddings let the model distinguish the two spans it must relate to each other for this task.
- Also present (and passed through, though often just set to segment A) during **single-sentence tasks** like classification, since BERT's architecture always expects a segment ID input.

**4. Relevance / why it matters**

- Enables BERT to handle **sentence-pair tasks** natively: NSP (pretraining), and downstream tasks like **Natural Language Inference (NLI)**, **Question Answering** (question = segment A, passage = segment B), **paraphrase detection**, **sentence similarity** — all of which require reasoning over *two* distinct text spans in one forward pass.
- Without segment embeddings, the model would only have `[SEP]` as a boundary marker, and no reusable dedicated signal that "these two chunks play different roles" — segment embeddings make this distinction explicit and learnable per-task.

**5. Contrast with later models**

- **RoBERTa** removed the NSP objective (found it wasn't very useful and even slightly harmful) but *kept* segment-style embeddings mainly for compatibility, later simplifying to single continuous input packing without needing a strict 2-segment split for most training.
- **GPT-style decoder-only LLMs** don't use segment embeddings at all — they're purely autoregressive over a single flat token stream; sentence/turn boundaries (e.g., in chat models) are instead encoded via special tokens (e.g., `<|user|>`, `<|assistant|>`) rather than a dedicated segment embedding table.
- **ALBERT/ELECTRA** and other BERT variants retain the segment embedding mechanism since they still use similar pretraining objectives on sentence pairs.

---

**Interview Key Points:**

1. **Purpose**: lets BERT distinguish tokens from Sentence A vs. Sentence B within a single packed input sequence.
2. **Mechanism**: a learned embedding table with 2 rows ($E_A$, $E_B$), **added** to token + position embeddings before the first transformer layer.
3. **Tied to NSP pretraining task** — critical for sentence-pair understanding (NLI, QA, paraphrase detection).
4. **Later models drop or replace it**: RoBERTa de-emphasizes NSP; GPT-style decoder LLMs use special tokens instead of segment embeddings since they're single-stream autoregressive models.
5. **Key exam trap**: segment embedding ≠ position embedding — segment marks *which sentence*, position marks *where in the sequence*; both are added together with token embeddings.

---

## Q15. Decoding Strategies for Next-Token Prediction (Greedy, Beam Search, Sampling, and More)

Once a language model outputs a probability distribution $p_\theta(w_t \mid w_{<t})$ over the vocabulary at each step, we need a **decoding strategy** to actually pick which token to emit. This choice dramatically affects output quality, diversity, and coherence — the model's probabilities alone don't determine the final text.

---

**1. Greedy Decoding**

$$w_t = \arg\max_w \, p_\theta(w \mid w_{<t})$$

**Intuition:** At every step, just pick the single most probable next token. Simple, deterministic, fast (no branching).

**Problem:** Locally optimal choices don't guarantee a globally optimal sequence. It can get stuck in repetitive loops ("I am a I am a I am a...") because once a slightly wrong token is chosen, the model just keeps confidently continuing down that path — no way to "undo" an early mistake.

---

**2. Beam Search**

**Intuition:** Instead of committing to one path, keep track of the **top-$k$ most probable partial sequences ("beams")** at each step, expand each by one token, then again keep only the top-$k$ overall.

$$\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log p_\theta(y_i \mid y_{<i})$$

- $k$ = beam width. $k=1$ reduces to greedy decoding.
- Explores more of the search space than greedy, so it finds higher-probability full sequences overall (useful for **Machine Translation**, where there's usually one "correct" answer and we want the single best output).
- **Downside**: Beams tend to converge to similar, generic, "safe" text — bad for open-ended generation (produces bland, repetitive output); also more expensive (higher latency, $k\times$ compute).
- Often combined with **length normalization** (dividing score by sequence length) since raw log-prob sums naturally favor shorter sequences.

---

**3. Temperature Sampling**

$$w_t \sim \text{softmax}(z_t / \tau)$$

**Intuition:** Instead of deterministically picking the max, **sample** from the probability distribution — but first reshape it using temperature $\tau$:
- $\tau \to 0$: distribution sharpens toward the max (approaches greedy).
- $\tau = 1$: unchanged distribution (raw model probabilities).
- $\tau > 1$: distribution flattens, more randomness/diversity, but risk of incoherence.

Adds controlled randomness so outputs aren't always identical for the same prompt — useful for creative/open-ended generation.

---

**4. Top-$k$ Sampling**

**Intuition:** Restrict sampling to only the $k$ most probable tokens (zero out the rest), then renormalize and sample.

- Prevents sampling from the "long tail" of very low-probability, nonsensical tokens.
- **Weakness**: $k$ is fixed regardless of context — sometimes the model is very confident (only 2 tokens matter) or very uncertain (50 tokens are plausible), but top-$k$ doesn't adapt to that.

---

**5. Top-$p$ / Nucleus Sampling**

**Intuition:** Instead of a fixed count $k$, take the smallest set of tokens whose **cumulative probability mass** exceeds threshold $p$ (e.g., 0.9), then sample only from that dynamic set.

$$V_p = \min \left\{ |V'| : \sum_{w \in V'} p_\theta(w) \geq p \right\}$$

- Adapts to the model's confidence: if the model is very sure, the nucleus is small (few tokens); if uncertain, it naturally grows to include more candidates.
- Generally produces more natural, less repetitive text than top-$k$ or pure temperature sampling — the current default in most production LLM chat interfaces (often combined with a moderate temperature).

---

**6. Combining Techniques**

In practice, most systems (e.g., GPT-family serving) combine **temperature + top-$k$ + top-$p$** together, plus:
- **Repetition penalty**: reduces probability of tokens already generated, to combat loops.
- **No-repeat n-gram blocking**: hard-blocks repeating an exact n-gram already seen.

---

**7. Contrastive Search (a more modern approach)**

**Intuition:** Balances model confidence with **diversity via a degeneration penalty** — it picks the next token that is likely *and* semantically different (low cosine similarity in hidden-state space) from previous tokens, explicitly discouraging repetitive/degenerate text without needing randomness at all (still deterministic).

$$w_t = \arg\max_{w \in V^{(k)}} \Big[(1-\alpha)\, p_\theta(w) - \alpha \max_{j<t} \text{sim}(h_w, h_{y_j})\Big]$$

Useful when you want high-quality, non-repetitive **deterministic** output (no sampling randomness) — an alternative to top-$p$ sampling.

---

**Interview Key Points:**

1. **Greedy** = fastest, most deterministic, but prone to repetition/looping due to no backtracking.
2. **Beam search** = keeps top-$k$ hypotheses, better for tasks with one "correct" target (MT, summarization), but produces generic/bland text for open-ended generation and is more expensive.
3. **Temperature/top-$k$/top-$p$ sampling** = introduce controlled randomness for diverse, natural, open-ended generation; top-$p$ is generally preferred since it **adapts to model confidence** dynamically, unlike fixed top-$k$.
4. **Trade-off axis**: determinism/quality-for-single-answer (greedy, beam) vs. diversity/naturalness (sampling methods) — task type dictates the right choice.
5. **Repetition penalty / n-gram blocking** are practical add-ons layered on top of any strategy to fight degenerate loops.
6. **Contrastive search** is a newer deterministic alternative that explicitly penalizes semantic similarity to prior tokens, avoiding the incoherence risk of pure sampling while still avoiding greedy's repetition problem.



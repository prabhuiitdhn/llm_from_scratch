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

---

## Q16. What is Context Length, and How Does it Affect Model Behavior?

**1. What context length is**

Context length (or context window) is the **maximum number of tokens** a model can process in a single forward pass — encompassing the prompt, any retrieved documents, conversation history, and the tokens generated so far. It's a hard architectural limit (e.g., 4K, 32K, 128K, or 1M+ tokens in newer models) beyond which the model either truncates input, fails, or degrades in quality.

$$\text{Total tokens} = \text{prompt} + \text{context/history} + \text{generated tokens} \leq L_{\max}$$

---

**2. Why context length is architecturally limited**

- **Self-attention cost**: standard attention computes $QK^T$ over all token pairs, costing $O(n^2 \cdot d)$ compute and $O(n^2)$ memory for the attention matrix. Doubling sequence length quadruples compute/memory for attention — this is the primary bottleneck.
- **Positional encoding limits**: absolute/learned position embeddings (e.g., original BERT) have a fixed-size lookup table — the model literally has no representation for position 5000 if trained only up to 512. This caps context length architecturally, not just computationally.
- **KV-cache memory at inference**: even with efficient attention, autoregressive generation needs to store Key/Value tensors for every past token, which scales linearly with context length and can dominate GPU memory for long contexts (this is why MQA/GQA and quantized KV-caches matter more as context length grows).

---

**3. How context length affects model behavior**

- **Information retention**: With longer context, the model can theoretically retain and reference more information (e.g., long documents, multi-turn conversations, large codebases) — critical for tasks like long-document summarization, RAG with many retrieved chunks, or agentic workflows with long tool-call histories.
- **"Lost in the middle" phenomenon**: Empirically, LLMs tend to attend well to information near the **beginning and end** of the context but often underuse or "forget" information buried in the **middle** of a long context, even when technically within the context window. This means simply having a large context length doesn't guarantee the model *effectively uses* all of it.
- **Quality degradation near/beyond training length**: Models trained with a max length $L$ often show degraded perplexity/coherence if evaluated on inputs near or exceeding $L$, unless explicitly designed/fine-tuned for extrapolation (e.g., via RoPE scaling techniques).
- **Latency and cost**: Since attention scales quadratically (or KV-cache scales linearly during generation), longer contexts are significantly more expensive and slower to compute — a major practical constraint for real-time applications.

---

**4. Techniques to extend context length**

- **RoPE scaling / position interpolation**: since RoPE encodes position via rotation angles, you can rescale the angle computation to "compress" a longer sequence into the range the model was originally trained on (e.g., used to extend LLaMA from 4K → 32K+).
- **NTK-aware scaling**: a refinement of position interpolation that adjusts the frequency base non-uniformly to preserve high-frequency (local) resolution while extending low-frequency (long-range) reach.
- **Sliding window attention / sparse attention** (Mistral, Longformer, BigBird): bound the $O(n^2)$ cost by only attending locally + a few global tokens, enabling much longer effective sequences within fixed compute budgets.
- **FlashAttention / memory-efficient attention kernels**: don't change complexity class but drastically reduce the constant factor and memory I/O, enabling practically longer contexts on the same hardware.
- **Continued pretraining / long-context fine-tuning**: explicitly train (or continue training) on longer sequences so the model learns to actually use distant tokens effectively, not just tolerate them numerically.

---

**5. Practical implications for system design**

- **RAG systems**: context length determines how many retrieved chunks can be included — but due to "lost in the middle," ranking/reordering retrieved chunks (most relevant near the start/end) often matters more than just maximizing chunk count.
- **Long conversation/agent memory**: systems often need summarization/compression strategies (rolling summaries, memory buffers) rather than relying purely on raw context length, since cost grows and effective attention degrades with length.
- **Trade-off in model choice**: bigger context window ≠ automatically better — must weigh cost (compute/latency), whether the model was actually trained to leverage that length effectively, and whether shorter, well-curated context outperforms a longer, noisier one.

---

**Interview Key Points:**

1. **Definition**: max tokens (prompt + history + generation) a model can process in one pass; hard architectural limit tied to positional encoding scheme and attention memory/compute.
2. **Root cause of the limit**: quadratic cost of full self-attention $O(n^2)$ plus fixed-size position representations (for absolute/learned PE) plus growing KV-cache at inference.
3. **"Lost in the middle"**: having a large context window doesn't guarantee effective use of all of it — models favor content near the start/end.
4. **Extension techniques**: RoPE scaling/NTK-aware scaling, sliding window/sparse attention, FlashAttention, and long-context continued pretraining.
5. **System design trade-off**: bigger context ≠ always better — costs more compute/latency, and quality/relevance of what's placed in context often matters more than raw length (relevant for RAG, agents, and multi-turn chat memory design).

---

## Q17. What is PagedAttention? How Does it Manage and Save Memory for KV-Caching?

**1. The problem it solves**

During autoregressive LLM inference, the **KV-cache** (Key/Value tensors for all previously generated tokens) must be stored in GPU memory to avoid recomputing attention from scratch at every step. Traditional serving systems allocate KV-cache memory as a **single large contiguous block per request**, sized for the *maximum possible sequence length* upfront. This causes two major inefficiencies:

- **Internal fragmentation**: if you reserve space for 2048 tokens but the actual output is only 200 tokens, ~90% of that reserved memory is wasted and unusable by other requests.
- **External fragmentation**: as requests of varying lengths finish and start, memory gets fragmented into unusable gaps, similar to classic OS memory fragmentation.
- Together, these effects mean GPUs waste **60-80% of KV-cache memory** in naive serving systems (per the vLLM paper), directly limiting the number of concurrent requests (batch size) a server can handle.

**2. Core idea — borrowed from OS virtual memory**

PagedAttention (introduced in **vLLM**, Kwon et al. 2023) applies the same idea as **OS paging/virtual memory** to KV-cache management:

- The KV-cache for a sequence is split into fixed-size **blocks** (e.g., 16 tokens per block), instead of one large contiguous allocation.
- These blocks can be stored **non-contiguously** anywhere in GPU memory — a **block table** (like a page table) maps each sequence's logical block index to its actual physical memory location.
- Attention computation is modified to gather K/V values across these scattered physical blocks using the block table, rather than assuming a single contiguous tensor.

$$\text{logical block } i \rightarrow \text{physical block} = \text{BlockTable}[\text{seq\_id}][i]$$

**3. How this saves memory**

- **On-demand allocation**: blocks are allocated **only as tokens are actually generated**, not pre-reserved for the worst case. A sequence that ends up short simply uses fewer blocks — no wasted reservation.
- **Near-zero internal fragmentation**: the only waste is within the *last* partially-filled block (at most block_size − 1 tokens wasted), instead of the entire unused reserved region.
- **Memory sharing across requests**: for use cases like **beam search**, **parallel sampling** (multiple completions per prompt), or **prefix sharing** (shared system prompts across requests), multiple sequences can literally point to the **same physical blocks** for shared prefixes (copy-on-write only when they diverge) — massive memory savings when many requests share a common prompt prefix.
- **Better batching / higher throughput**: because memory is used efficiently and dynamically, the server can pack far more concurrent sequences into the same GPU memory, directly increasing serving throughput (the vLLM paper reports 2-4× throughput gains over then-standard serving like HuggingFace TGI/FasterTransformer).

**4. How it's used in serving (continuous batching)**

- Works hand-in-hand with **continuous/dynamic batching**: as some sequences in a batch finish generating, their blocks are immediately freed and reallocated to new incoming requests, rather than waiting for the entire batch to finish (as in naive static batching).
- The block table indirection adds a small computational overhead (gather operations instead of simple contiguous reads) but this is vastly outweighed by the memory efficiency and throughput gains.

**5. Relevance today**

- PagedAttention is the core mechanism behind **vLLM**, now one of the most widely used open-source LLM serving engines.
- The same block-based, non-contiguous memory management idea has influenced other serving optimizations (e.g., **prefix caching** for repeated system prompts, radix-tree-based cache sharing in SGLang).

---

**Interview Key Points:**

1. **Problem**: naive KV-cache allocation reserves max-length contiguous memory per request → massive internal/external fragmentation, wasting 60-80% of memory.
2. **Core idea**: borrow OS **virtual memory paging** — split KV-cache into fixed-size blocks stored non-contiguously, tracked via a per-sequence **block table**.
3. **Memory savings**: allocate blocks on-demand (only as tokens are generated), near-zero internal fragmentation, and **shared physical blocks** for common prefixes (system prompts, beam search, parallel sampling) via copy-on-write.
4. **System impact**: enables much higher batch sizes / concurrent requests on the same GPU → 2-4× throughput improvement over naive contiguous KV-cache serving.
5. **Trade-off**: small overhead from block-table indirection/gather ops during attention, vastly outweighed by memory efficiency gains — it's the foundation of vLLM's serving engine.

---

## Q18. What is Speculative Decoding? What Problem Does it Solve?

**1. The problem it solves**

Autoregressive LLM generation is inherently **sequential and memory-bandwidth bound**: to generate each new token, you must do a full forward pass through the (often huge) model, load all its weights and KV-cache from GPU memory, and this repeats one token at a time. Even though each individual forward pass is cheap in terms of FLOPs relative to the GPU's compute capacity, the **memory bandwidth cost of loading weights per token** dominates — GPUs are underutilized doing "1 token at a time" work when they're actually capable of processing many tokens in parallel per pass. This makes decoding slow and latency-bound, especially for large models.

**2. Core idea**

Speculative decoding (Leviathan et al. 2023; Chen et al. 2023, "accelerating LLM decoding with speculative sampling") uses a **small, fast "draft" model** to speculatively generate several candidate tokens ahead, then uses the large "target" model to **verify all of them in a single parallel forward pass**, rather than generating one token at a time with the large model.

**Steps:**
1. A small draft model $M_q$ (cheap, fast) autoregressively generates $k$ candidate tokens: $\hat{y}_1, \dots, \hat{y}_k$.
2. The large target model $M_p$ (the model we actually want output from) does **one forward pass** over the whole draft sequence (all $k$ tokens at once, since verification doesn't require sequential generation), producing its own probability distribution at each position.
3. Each draft token is **accepted or rejected** using a rejection-sampling criterion that guarantees the final output distribution is **mathematically identical** to what the target model would have generated on its own — no quality loss.
4. At the first rejected token, sampling falls back to the target model's own corrected distribution at that position; everything after the rejection point is discarded and the process repeats from there.

**3. Why this speeds things up**

- If the draft model's tokens are correct roughly aligned with what the target model would generate (common in many contexts — easy/predictable tokens), then in a **single expensive target-model forward pass**, you effectively "verify" and accept multiple tokens at once, instead of needing $k$ separate expensive forward passes.
- This exploits the fact that a **parallel forward pass over $k$ tokens costs barely more than a single-token forward pass** (memory-bandwidth-bound, not compute-bound) — so verifying $k$ draft tokens together is nearly as cheap as generating just 1 token normally, but you get up to $k$ tokens "for free" when accepted.
- Speedups of **2-3× (sometimes more)** are commonly reported, with **zero degradation in output quality/distribution** — this is the key selling point: it's a **lossless** acceleration technique (unlike quantization/distillation which can hurt quality).

**4. The math behind the rejection sampling (intuition, not full derivation)**

For each proposed token $\hat{y}$, accept it with probability:
$$\min\left(1, \frac{p(\hat{y})}{q(\hat{y})}\right)$$
where $p$ = target model's probability, $q$ = draft model's probability. If rejected, sample a corrected token from the residual distribution $\text{norm}(\max(0, p - q))$. This is exactly the same math trick used in classic **rejection sampling** from statistics — it guarantees the combined accept/reject + fallback process reproduces samples from $p$ exactly, even though most tokens were actually proposed by the (different) draft distribution $q$.

**5. Practical considerations**

- **Draft model choice**: usually a much smaller version of the same model family (e.g., a distilled or smaller-parameter sibling), or sometimes the same model itself at lower precision — needs to be reasonably aligned with the target model's distribution for good acceptance rates, but cheap enough that running it repeatedly is nearly free.
- **Acceptance rate matters**: if the draft model diverges a lot from the target (e.g., very different domain/style), acceptance rate drops and the speedup shrinks — worst case, it just falls back to normal one-token-at-a-time generation with slight overhead from the wasted draft computation.
- **Batching interaction**: works best in latency-sensitive, lower-batch-size serving scenarios; at very large batch sizes the GPU is already compute-bound rather than memory-bound, so the benefit diminishes somewhat.
- **Variants**: **Medusa** and **lookahead decoding** use multiple parallel prediction heads on the target model itself (instead of a separate draft model) to generate/verify candidate continuations, reducing the need for a whole separate draft model.

---

**Interview Key Points:**

1. **Problem solved**: autoregressive decoding is memory-bandwidth-bound (one token at a time), underutilizing GPU parallel compute capacity.
2. **Core mechanism**: small draft model proposes $k$ tokens ahead; large target model verifies all $k$ in **one parallel forward pass**.
3. **Losslessness guarantee**: uses rejection sampling ($\min(1, p/q)$ acceptance + residual resampling) so output distribution is mathematically identical to the target model's own generation — no quality trade-off.
4. **Speedup source**: parallel verification of $k$ tokens costs nearly the same as generating 1 token (memory-bandwidth bound), so accepted tokens are essentially "free."
5. **Practical caveat**: speedup depends on draft-target alignment (acceptance rate) and is most beneficial in low-batch, latency-sensitive serving; variants like **Medusa**/**lookahead decoding** avoid needing a separate draft model.

---

## Q19. What is In-Context Learning (ICL) in NLP?

**1. What it is**

In-context learning is the ability of a large language model to **learn a new task at inference time purely from examples provided in the prompt**, without any gradient updates or weight changes. You show the model a few (or zero) input-output examples directly in the context window, and it infers the task pattern and applies it to a new query — all within a single forward pass.

$$p_\theta(y \mid x_{\text{query}}, \{(x_1, y_1), \dots, (x_k, y_k)\})$$

The model's parameters $\theta$ remain completely frozen — "learning" here means adapting behavior conditioned on the prompt, not updating weights.

**2. Variants**

- **Zero-shot**: only a task instruction, no examples. `"Translate to French: Hello" → "Bonjour"`
- **Few-shot**: a handful of input-output example pairs shown before the actual query.
```
English: cat → French: chat
English: dog → French: chien
English: house → French: ?
```
- **Many-shot**: dozens to hundreds of examples, feasible with today's large context windows (32K-1M tokens), often improving accuracy further compared to few-shot.

**3. Why/how it emerges (the research intuition)**

- ICL is **not explicitly trained for** in most pretraining — it *emerges* as a byproduct of next-token prediction over massive, diverse web-scale text, which naturally contains many "pattern completion" structures (Q&A pairs, lists, translations, code examples, etc.).
- One dominant hypothesis: during pretraining, the model implicitly learns to perform something akin to **Bayesian inference over latent "tasks"** — given a sequence of examples, it infers which latent task/concept is being demonstrated and continues consistent with that pattern (Xie et al., "An Explanation of In-Context Learning as Implicit Bayesian Inference").
- Another framing: some researchers show that a subset of attention heads can implement something functionally similar to a single step of **gradient descent** internally, purely through the forward pass over the demonstration examples (i.e., ICL as "implicit fine-tuning" done algorithmically inside the forward computation, not literal weight updates).
- Model **scale matters a lot**: ICL ability improves sharply with model size — smaller models often can't leverage few-shot examples effectively, while large models show strong emergent few-shot performance (a commonly cited example of an "emergent capability").

**4. Why it's useful / relevant**

- **No fine-tuning cost**: adapts a general-purpose pretrained/instruction-tuned model to a new task instantly, without collecting a fine-tuning dataset or paying for training compute.
- **Flexibility**: same frozen model can be steered for classification, translation, extraction, reasoning style, output format — just by changing the prompt's examples.
- **Foundation for prompt engineering**: techniques like **Chain-of-Thought prompting** ("Let's think step by step" + few worked examples) are a direct application of ICL — showing reasoning-style examples elicits reasoning-style outputs.
- Underlies most practical **LLM application design** today (RAG systems, agents, chatbots) — instructions + few relevant examples in the prompt often replace what used to require task-specific fine-tuned models.

**5. Limitations**

- **Sensitive to example order/formatting**: performance can vary significantly based on which examples are shown, their order, and even minor prompt phrasing changes — a known reliability issue.
- **Context length cost**: more examples = more tokens = higher inference cost and latency; also subject to the "lost in the middle" effect for many-shot settings.
- **No persistent learning**: knowledge from the prompt disappears once the context is cleared — it doesn't update the model's actual weights/knowledge for future unrelated queries (unlike fine-tuning).
- **Can still fail on genuinely novel/complex tasks** far outside the distribution the model was pretrained on, since it's fundamentally leveraging patterns already latent in pretraining, not acquiring truly new capabilities.

---

**Interview Key Points:**

1. **Definition**: task adaptation purely via prompt examples at inference time, with **zero weight updates** — contrast this clearly with fine-tuning, which does update weights.
2. **Zero/few/many-shot** are points on a spectrum, and effectiveness of few/many-shot scales strongly with model size (an emergent capability).
3. **Theoretical explanations**: implicit Bayesian task inference, and evidence that some attention heads perform something functionally like an internal gradient-descent step over the demonstrations.
4. **Practical relevance**: foundation for prompt engineering, Chain-of-Thought prompting, and most modern LLM application patterns (RAG, agents) — avoids costly fine-tuning for many use cases.
5. **Key limitations**: sensitive to example order/formatting, costs more context tokens, doesn't persist beyond the current prompt, and can still fail outside the pretraining distribution.

---

## Q20. What is the Chinchilla Scaling Law? Does it Apply to Multimodal/Vision Model Training?

**1. What it is**

The Chinchilla paper (Hoffmann et al., DeepMind 2022, "Training Compute-Optimal Large Language Models") studied how to **optimally allocate a fixed compute budget** between **model size (parameters $N$)** and **training data size (tokens $D$)** to minimize training loss.

Prior to this (e.g., GPT-3, original scaling laws from Kaplan et al. 2020), the field's assumption was: bigger models are better, so scale parameters aggressively and don't worry as much about proportionally scaling data. Chinchilla showed this was **compute-suboptimal**.

**Key finding:** For a compute budget $C$ (roughly $C \approx 6ND$ FLOPs), loss is minimized when model size $N$ and data size $D$ are scaled **roughly equally**:

$$N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}$$

Practically: **for every doubling of model parameters, you should also roughly double the number of training tokens.** The rule of thumb that emerged: train on **~20 tokens per parameter** (e.g., a 70B parameter model should be trained on ~1.4 trillion tokens).

**2. The evidence — Chinchilla vs. Gopher**

DeepMind trained **Chinchilla (70B params, 1.4T tokens)** using the *same compute budget* as **Gopher (280B params, 300B tokens)** — a much bigger model trained on much less data. Chinchilla, despite being **4× smaller**, outperformed Gopher on downstream benchmarks. This was the empirical proof that most large models before this point were **undertrained relative to their size** — wasting compute on parameters instead of data.

**3. Why this matters / effects**

- **Reshaped LLM training strategy industry-wide**: after Chinchilla, most labs shifted toward "compute-optimal" ratios — e.g., LLaMA (Meta) explicitly cited Chinchilla and trained smaller models (7B-65B) on much larger token counts (1-1.4T+ tokens) than GPT-3-era conventions would have suggested, achieving strong performance with smaller, cheaper-to-serve models.
- **Inference cost consideration**: a smaller, Chinchilla-optimal model that's cheaper to *train* is also cheaper to *serve* at inference time — this is a major practical win, since inference cost (serving millions of queries) often dominates total lifetime cost more than one-time training cost.
- **Overtraining beyond Chinchilla-optimal is now common in practice**: many production models (e.g., LLaMA-2/3) are deliberately trained on **more tokens than "compute-optimal"** predicts, because they prioritize a smaller, cheaper-to-serve model even if it costs more (non-optimal) compute to train — trading extra training compute for better inference economics. This is a deliberate deviation from pure Chinchilla-optimality, showing the law informs decisions but isn't a rigid mandate.

**4. Is it "important to follow"? — Nuanced answer**

- **Yes, as a guiding principle**, not a strict law: it corrected a systematic bias (over-investing in parameters, under-investing in data) and is essential context for **compute budgeting decisions** — if you're compute-constrained and want the best possible loss for a fixed budget, Chinchilla ratios are the right starting point.
- **No, not as a rigid constraint** in all cases: real-world deployment considerations (inference latency/cost, hardware memory limits, downstream task requirements) often justify deviating from strict compute-optimality — e.g., **deliberately overtraining smaller models** past the "optimal" point because a smaller model saves far more in serving costs than it loses in training efficiency.
- Also, Chinchilla's scaling exponents were fit from experiments primarily on **text-only dense transformer LLMs** — extrapolating exact numbers to other regimes needs care.

**5. Multi-modal and Vision Model Training — does Chinchilla-style scaling apply?**

- **Core principle transfers**: the general insight — "compute-optimal training requires jointly scaling model size and data size, not parameters alone" — is broadly applicable and has been studied for vision transformers (ViT) and multimodal models too (e.g., Google's scaling studies on ViT, and multimodal scaling law work from various labs).
- **But exact ratios differ by modality/architecture**: 
  - **Vision models** often have different data availability/redundancy characteristics (image data has different information density than text), and architectures (ViT vs. CNN vs. hybrid) have different compute-to-parameter-efficiency profiles — so the "~20 tokens per parameter" heuristic doesn't directly transfer numerically; each modality/architecture combo needs its own scaling law fit.
  - **Multimodal models** (vision-language models like CLIP, Flamingo, GPT-4V-style) add complexity: you're balancing compute across potentially multiple encoders/modalities plus a fusion mechanism, and data across differently-sized modality-specific datasets (e.g., image-text pairs are far scarcer/noisier than pure text corpora) — so compute-optimal allocation becomes a **higher-dimensional optimization problem** (how much compute per modality, not just total model size vs. total data size).
  - Data **quality and diversity** matter more in low-data-availability regimes (e.g., paired image-text data is much scarcer than raw text), so pure token-count scaling laws are a less complete picture for multimodal settings — deduplication, filtering, and synthetic data quality often dominate over raw scaling ratios.

---

**Interview Key Points:**

1. **Chinchilla's core finding**: compute-optimal training scales model size and data size roughly equally ($N \propto C^{0.5}$, $D \propto C^{0.5}$); rule of thumb ≈ 20 tokens per parameter.
2. **Empirical proof**: Chinchilla (70B, 1.4T tokens) beat Gopher (280B, 300B tokens) at equal compute — showed earlier large models were undertrained relative to their size.
3. **Industry impact**: shifted practice toward smaller, more heavily-trained models (LLaMA family explicitly followed this), reducing both training and **inference serving costs**.
4. **Not a rigid law**: production models often deliberately **overtrain smaller models beyond Chinchilla-optimal** because inference cost savings outweigh training compute inefficiency — a well-known intentional deviation.
5. **Multimodal/vision extension**: the *principle* (balance model size and data, don't just scale parameters) transfers, but exact scaling exponents/ratios are modality- and architecture-specific, and multimodal settings add extra complexity around cross-modal compute allocation and data scarcity/quality — an active area of ongoing scaling-law research, not a solved, universal formula.

---

## Q21. What is Zero Redundancy Optimizer (ZeRO) in AI Model Training?

**1. The problem it solves**

Training large models with standard **data parallelism** requires every GPU to hold a **full, redundant copy** of:
- Model parameters
- Gradients
- Optimizer states (e.g., Adam's momentum and variance terms — which for mixed-precision Adam training can be **~2-3× the size of the parameters themselves**, since Adam stores fp32 master weights + momentum + variance)

For a large model (e.g., 175B parameters), this memory footprint per GPU becomes enormous — often exceeding available GPU memory well before you even get to activations. The redundancy is wasteful: **every single GPU stores an identical copy** of parameters/gradients/optimizer states, when in a distributed setting this data could instead be split (sharded) across GPUs.

**2. Core idea**

ZeRO (Rajbhandari et al., Microsoft DeepSpeed, 2019) eliminates this memory redundancy by **partitioning (sharding) the training state across data-parallel GPUs**, instead of replicating it fully on each one. Each GPU only stores a **slice** of the parameters/gradients/optimizer states, and GPUs communicate (via all-gather/reduce-scatter operations) to reconstruct what's needed just-in-time during the forward/backward pass.

ZeRO has **three progressive stages**, each sharding more of the training state:

- **ZeRO Stage 1**: shard only the **optimizer states** (Adam moments) across GPUs. Biggest memory hog (optimizer states) is no longer replicated — ~4× memory reduction for optimizer state alone, with minimal extra communication.
- **ZeRO Stage 2**: additionally shard the **gradients** across GPUs (each GPU only keeps gradients for its parameter shard) — further memory savings.
- **ZeRO Stage 3**: additionally shard the **model parameters themselves** — each GPU holds only $1/N$ of the actual parameters at rest, and gathers the full parameter set (via all-gather) **just before** it's needed for a given layer's forward/backward computation, then discards it again afterward. This achieves the **maximum memory reduction** (roughly linear in the number of GPUs, $N$), at the cost of the most communication overhead.

$$\text{Memory per GPU} \approx \frac{\text{Params} + \text{Gradients} + \text{Optimizer States}}{N_{\text{GPUs}}} + \text{Activations}$$

(compared to full replication of params+gradients+optimizer-states per GPU in vanilla data parallelism)

**3. How training proceeds with ZeRO Stage 3 (the most aggressive stage)**

- Each GPU stores only its shard of parameters at rest.
- Just before a layer needs its full weights for the forward pass, an **all-gather** operation reconstructs the complete weight tensor for that layer across all GPUs (temporarily).
- After using it, the full weights are discarded again, freeing memory — same pattern repeats for backward pass gradients.
- This trades **extra communication bandwidth** for **massive memory savings**, allowing models far larger than any single GPU's memory to be trained via standard data parallelism (rather than needing complex model/pipeline parallelism alone).

**4. Why this matters**

- Enables training of very large models on GPU clusters **without needing complex model-parallel sharding schemes** as the primary/only solution — it's compatible with, and often combined with, tensor/pipeline parallelism for even larger scale.
- Democratized large-model training/fine-tuning — DeepSpeed's ZeRO (and the related open-source Hugging Face **Accelerate**/**FSDP** integration) is why practitioners can fine-tune multi-billion parameter models on modest GPU clusters that otherwise couldn't fit even a single replica's optimizer states.
- **PyTorch's FSDP (Fully Sharded Data Parallel)** is essentially the native PyTorch implementation of the same ZeRO Stage 3 idea, now a standard tool in the training ecosystem.

**5. Trade-offs**

- **Communication overhead increases** with more aggressive sharding (Stage 3 > Stage 2 > Stage 1) since more all-gather/reduce-scatter operations are needed — network bandwidth between GPUs becomes a critical factor (this is why fast interconnects like NVLink/InfiniBand matter a lot for ZeRO-3 efficiency).
- **ZeRO-Offload/ZeRO-Infinity** extensions push this further by offloading shards to **CPU RAM or even NVMe disk**, trading even more speed for the ability to train models that don't fit in aggregate GPU memory at all.
- Choosing the right stage is a **memory vs. communication speed** trade-off: Stage 1/2 for moderate memory pressure with less overhead, Stage 3 when memory is the hard constraint and you can tolerate more communication cost.

---

**Interview Key Points:**

1. **Problem solved**: standard data parallelism fully replicates parameters + gradients + optimizer states on every GPU — extremely memory-wasteful for large models.
2. **Core idea**: shard (partition) this training state across GPUs instead of replicating it, reconstructing full tensors on-demand via all-gather when needed.
3. **Three stages**: Stage 1 shards optimizer states, Stage 2 adds gradient sharding, Stage 3 adds full parameter sharding (max memory savings, most communication).
4. **Real-world equivalent**: PyTorch's **FSDP** implements the same idea natively; used alongside DeepSpeed for large-scale LLM training/fine-tuning.
5. **Trade-off**: memory savings come at the cost of increased inter-GPU communication — network bandwidth (NVLink/InfiniBand) becomes critical, especially at Stage 3; further extended by ZeRO-Offload/Infinity to spill shards to CPU/NVMe for even larger models.

---

## Q22. Data Parallelism vs. Model Parallelism in AI Training — Problems Solved and Techniques

**1. The core problem both solve**

Training large models faces two distinct resource constraints:
1. **Not enough compute throughput** to process a huge dataset in reasonable time (even if the model fits on one GPU).
2. **Not enough memory** on a single GPU to even hold the model (parameters + gradients + optimizer states + activations), regardless of dataset size.

**Data parallelism** solves problem #1 (throughput/speed). **Model parallelism** solves problem #2 (memory capacity). They're often combined, since large-scale LLM training usually hits both constraints simultaneously.

---

**2. Data Parallelism (DP)**

**Idea:** Replicate the **entire model** on every GPU; split the **training batch** across GPUs so each GPU processes a different subset of data in parallel.

$$\text{GPU}_i \text{ computes gradients on mini-batch}_i, \quad \text{gradients averaged/synced across all GPUs}$$

**Mechanism:**
1. Each GPU holds a full copy of the model.
2. Each GPU does a forward + backward pass on its own shard of the batch, producing local gradients.
3. Gradients are **synchronized/averaged** across all GPUs (via **all-reduce** communication) before the optimizer step, so every GPU ends up with identical updated weights.

**Benefit:** Linear-ish speedup with more GPUs (more data processed per step), simple to implement.

**Limitation:** Doesn't help if the model itself doesn't fit on a single GPU — every GPU still needs the full model + full optimizer states (this is exactly the memory-redundancy problem that **ZeRO/FSDP** solve, by sharding the redundant copies while keeping the data-parallel training pattern).

---

**3. Model Parallelism**

**Idea:** Split the **model itself** across multiple GPUs, since it's too large to fit on one device, regardless of batch size concerns. Two main flavors:

**a) Tensor Parallelism (intra-layer parallelism)**

Splits **individual weight matrices/operations within a layer** across GPUs. E.g., for a large matrix multiplication $Y = XW$, split $W$ column-wise or row-wise across GPUs, each computing a partial result, then combine (via all-reduce/all-gather) to get the full output.

$$W = [W_1 | W_2 | \dots | W_k] \quad \Rightarrow \quad Y = X[W_1|W_2|\dots|W_k] = [XW_1 | XW_2 | \dots]$$

- Used heavily in **Megatron-LM** style training, splitting attention heads and MLP weight matrices across GPUs within a node (requires fast interconnects like NVLink since communication happens *within* every layer's forward/backward pass).
- **Trade-off**: high communication frequency (every layer needs a sync), so it's typically limited to GPUs on the same node with very fast interconnect.

**b) Pipeline Parallelism (inter-layer parallelism)**

Splits the model **by layers** across GPUs — e.g., GPU 1 holds layers 1-10, GPU 2 holds layers 11-20, etc. Data flows through the "pipeline" of GPUs sequentially.

$$\text{GPU}_1(\text{layers } 1\text{-}10) \rightarrow \text{GPU}_2(\text{layers } 11\text{-}20) \rightarrow \dots$$

- **Naive version** causes "bubble" idle time (GPU 2 waits for GPU 1 to finish before starting).
- **Micro-batching / pipeline scheduling** (e.g., **GPipe**, **PipeDream**, **1F1B scheduling**) splits the batch into smaller micro-batches and overlaps forward/backward computation across stages to reduce idle "bubble" time and improve GPU utilization.
- **Trade-off**: less communication-intensive than tensor parallelism (only activation tensors pass between pipeline stages), so it works reasonably well across nodes (higher-latency network), but pipeline bubbles reduce efficiency if not well-scheduled.

---

**4. Combining Techniques — 3D Parallelism**

Real-world large-scale LLM training (e.g., GPT-3/4-scale, LLaMA, Megatron-Turing NLG) combines **all three dimensions simultaneously**, often called **3D parallelism**:

- **Tensor parallelism**: within a node (fast NVLink) to split individual layers.
- **Pipeline parallelism**: across nodes to split the model into stages.
- **Data parallelism**: across replicated pipeline groups to increase throughput, often combined with **ZeRO sharding** to reduce optimizer/gradient memory redundancy within each data-parallel replica.

This lets training scale to thousands of GPUs, balancing memory capacity (model/pipeline parallelism) and throughput (data parallelism).

---

**5. Techniques/Frameworks that implement these**

- **PyTorch DDP (DistributedDataParallel)**: standard data parallelism implementation.
- **DeepSpeed ZeRO / PyTorch FSDP**: sharded data parallelism (memory-efficient DP, discussed in Q21).
- **Megatron-LM**: tensor parallelism implementation for Transformers (splits attention/MLP matrices).
- **GPipe / PipeDream / DeepSpeed pipeline module**: pipeline parallelism with micro-batch scheduling to reduce bubble overhead.
- **DeepSpeed 3D parallelism / Megatron-DeepSpeed**: combines tensor + pipeline + data parallelism (+ ZeRO) into one unified training stack — the standard approach for training frontier-scale LLMs.

---

**Interview Key Points:**

1. **Data parallelism** solves the **throughput** problem: replicate the full model, split the *batch* across GPUs, sync gradients via all-reduce — doesn't help if the model itself doesn't fit on one GPU.
2. **Model parallelism** solves the **memory capacity** problem: split the *model* itself across GPUs — either **tensor parallelism** (split individual matrix ops within a layer, high communication, needs fast intra-node interconnect) or **pipeline parallelism** (split by layers across GPUs, lower communication, but risks pipeline "bubble" idle time).
3. **Micro-batching/scheduling** (GPipe, 1F1B) is essential to make pipeline parallelism efficient by overlapping stages.
4. **ZeRO/FSDP bridges the two**: shards the redundant optimizer/gradient/parameter state within data parallelism, reducing memory pressure without needing full model parallelism.
5. **Real large-scale training combines all of them** ("3D parallelism": tensor + pipeline + data/ZeRO) — Megatron-LM and DeepSpeed are the standard reference implementations for this combined approach.



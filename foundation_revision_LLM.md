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



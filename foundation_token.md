# Tokenisation Basics for NLP and LLM Pipelines

This note is for interview preparation and practical engineering understanding. It focuses on how tokenisation affects training, validation, and inference quality.

---

## 1. What tokenisation is

Tokenisation is the process of converting raw text into discrete units (tokens) that a model can process.

Common token units:
- Character-level tokens
- Word-level tokens
- Subword tokens (most common in modern LLMs)

Why tokenisation matters:
- Models read token IDs, not raw strings.
- Cost is measured in tokens.
- Context length is a token budget.
- Bad tokenisation choices can hurt accuracy, latency, and memory.

---

## 2. Where tokenisation fits in the pipeline

### Training
- Normalize text consistently.
- Build or select a tokenizer/vocabulary.
- Encode input-output pairs to token IDs.
- Add special tokens such as BOS/EOS/PAD.
- Apply truncation and padding policies.
- Produce attention masks and (if needed) label masks.

Normalization techniques (beginner to advanced):
- Lowercasing: Convert text to lowercase for consistency when casing is not task-critical.
- Unicode normalization (NFC/NFKC): Standardize visually similar text forms into a consistent representation.
- Whitespace normalization: Collapse repeated spaces/newlines and trim edges.
- Punctuation normalization: Standardize quotes, dashes, ellipses, and repeated punctuation patterns.
- Number/date normalization: Convert variable patterns to stable forms when task allows.
- URL/email/handle placeholders: Map noisy identifiers to placeholders such as <url> or <email>.
- Emoji/symbol policy: Keep, map, or remove emojis and symbols based on task needs.
- Accent/diacritic handling: Normalize or preserve accents depending on language and accuracy goals.
- Domain normalization: Standardize units, abbreviations, and terminology in domain-specific corpora.
- Versioned preprocessing contracts: Freeze and reuse the exact same normalization rules in train/val/inference.

Why line "Normalize text consistently" matters:
- It prevents train-inference mismatch in token distributions.
- It reduces vocabulary noise and unstable tokenization behavior.
- It improves reproducibility, debugging, and model quality monitoring.

### Validation
- Use the same tokenizer and same preprocessing as training.
- Do not rebuild vocabulary from validation set.
- Track OOV [Out of vocabulary] /UNK [ unknown ] behavior and sequence truncation rate.

out the vocabulary: means which is not being defined as words or in dictionary i.e can be ignored by model and consider it as UNK [ Unknown ]

### Inference
- Use the exact same tokenizer version used in training.
- Encode incoming user prompt and context.
- Respect max sequence limits and truncation rules.
- Decode model outputs back to text.

---

## 3. Key concepts you must know

- Vocabulary: mapping from token string to token ID.
- Special tokens: PAD [ Padding ], UNK [Unknown ], BOS [Beginning of sentence], EOS [end of sentence], SEP [ Separator ], etc.
- Attention mask: 1 for real tokens, 0 for padding tokens.
- Truncation: cutting sequences longer than max length.
- Padding: extending sequences to fixed size for batching.
- OOV (out-of-vocabulary): text pieces not in vocabulary.
- Tokenizer drift: mismatch between expected and actual tokenization behavior in production.

---

## 4. Why subword tokenisation is standard in LLMs

Subword methods (BPE/WordPiece/Unigram) are used because they:
- Handle rare and unseen words better than pure word-level methods.
- Keep vocabulary size manageable.
- Work across domains and many languages.
- Reduce unknown-token failures.

Tradeoff:
- More tokens for some words can increase compute cost.

BPE: Byte pair Encoding
example: text = "low lower"
split: ['l', 'o', 'w'], ['l', 'o', 'w', 'e', 'r']
merge: ['l', 'o'] -> ['lo', 'w'], ['lo', 'w', 'e', 'r']
result: ['lo', 'w', 'lo', 'w', 'e', 'r']

Subword/wordpiece : mostly used for handling misspelling or multilingual data
example: unfriendly
Un, friend, ly

Explanation:
WordPiece breaks `"unfriendly"` into subword pieces that already exist in the vocabulary:
`"unfriendly"` → `["un", "friend", "ly"]`
Each piece is a meaningful morpheme (prefix, root, suffix) that the tokenizer learned from training data.

Why this handles misspelling:
- `"unfrendly"` (misspelled) → `["un", "fre", "nd", "ly"]`
- Word-level tokenizer: entire word → `<UNK>` (total failure)
- Subword tokenizer: splits into known pieces → model still gets partial signal from `"un"` and `"ly"`
- The model can still infer meaning from recognizable subwords even when the full word is misspelled.

Why this handles multilingual data:
- `"spielen"` (German: to play) → `["spiel", "en"]`
- `"spielplatz"` → `["spiel", "platz"]`
- Shared subwords across languages let the model transfer knowledge without needing every word form in every language as a separate vocabulary entry.

How WordPiece decides where to split:
1. Start with the full word. If it's in vocabulary → keep as one token.
2. If not, try the longest prefix in vocabulary, then tokenize the remainder.
3. Continuation pieces are often marked with `##` (in BERT-style): `["un", "##friend", "##ly"]`

Detailed explanation of longest-prefix matching:
The algorithm processes a word left-to-right, always taking the longest possible match from the vocabulary first, then repeating on what's left.

Step-by-step example:
Suppose vocabulary contains: `["un", "friend", "friendly", "ly", "f", "r", "i", "e", "n", "d"]`

Word: `"unfriendly"`
- Step 1: Try full word "unfriendly" → NOT in vocab
- Step 2: Try longest prefix: "unfriendl" → no, "unfriend" → no, ... "un" → YES ✓ → take "un"
- Remainder: "friendly"
- Step 3: Try "friendly" → YES ✓ (it's in vocab!) → take "friendly"
- Remainder: "" (done)
- Result: ["un", "friendly"]

Another example where "friendly" is NOT in vocab:
Vocabulary: `["un", "friend", "ly", "f", "r"]`
Word: `"unfriendly"`
- Step 1: "unfriendly" → not in vocab
- Step 2: Longest prefix match → "un" ✓. Remainder: "friendly"
- Step 3: "friendly" → not in vocab. Longest prefix: "friend" ✓. Remainder: "ly"
- Step 4: "ly" → YES ✓
- Result: ["un", "friend", "ly"]

Why "longest prefix first" matters:

| Strategy | Result for "friendly" | Quality |
|---|---|---|
| Longest prefix first | `["friend", "ly"]` | Semantically meaningful pieces |
| Shortest prefix first | `["f", "r", "i", "e", "n", "d", "ly"]` | Too fragmented, loses meaning |

Longest-match preserves the most semantic information per token, reducing sequence length and keeping meaningful units together.

What happens when no prefix matches:
If even single characters aren't in vocabulary → fall back to `<UNK>`. But in practice, byte-level or character-level fallbacks ensure this rarely happens in modern tokenizers.

Senior one-liner:
WordPiece uses greedy longest-prefix matching to maximize semantic density per token — each split captures the largest known subword, then recurses on the remainder until the full word is consumed.

Key insight for interviews:
Subword tokenization gives graceful degradation — unknown or malformed words don't collapse to a single useless `<UNK>` token. The model always gets some signal from recognizable pieces, which is why it works well for noisy real-world text, typos, and languages the tokenizer wasn't primarily trained on.

### 4.1 Tokenizer techniques from beginner to advanced

Beginner level:
- Whitespace tokenization: split by spaces; easy but fragile.
- Rule-based tokenization: regex or punctuation-aware splitting.
- Character-level tokenization: robust to unseen words but sequences become long.
- Word-level tokenization: intuitive, but suffers from large vocabulary and OOV issues.

Intermediate level:
- Subword BPE (Byte Pair Encoding): merges frequent symbol pairs to build useful subwords.
- WordPiece: selects merges by likelihood-based criteria, widely used in encoder models.
- Unigram language model tokenization: starts with larger vocab and prunes to maximize likelihood.
- Byte-level tokenization: operates on bytes, improves robustness to arbitrary text formats.

Advanced and research-facing level:
- SentencePiece pipelines: language-agnostic training directly from raw text (BPE or Unigram variants).
- Domain-adaptive tokenizer training: retrain or extend tokenizers for legal, biomedical, code, or multilingual domains.
- Vocabulary expansion and remapping strategies: add domain terms while controlling embedding initialization risk.
- Tokenizer-free or latent/patch tokenization research: alternatives that reduce dependence on handcrafted token boundaries.
- Dynamic or task-aware segmentation research: adapt token boundaries by task distribution and inference constraints.

---

## 5. Common engineering mistakes

- Training with one tokenizer version and serving with another.
- Building vocabulary using validation/test data (data leakage).
- Ignoring truncation, causing silent information loss.
- Incorrect padding masks, hurting attention behavior.
- Not tracking token length distributions before deployment.

---

## 6. Senior-level interview Q&A

### Q1. What is tokenisation and why does it matter for LLM systems?

Tokenisation is conversion of text into token IDs for model processing. It matters because it directly affects model input representation, sequence length, latency, cost, and quality. In production, tokenization consistency is a reliability requirement, not just preprocessing.

### Q2. Why should tokenizer fitting be done on training data only?

Because using validation or test data during tokenizer fitting introduces leakage. This can inflate evaluation quality and hide real-world generalization issues.

**Detailed Explanation:**

**The Core Issue: Data Leakage**

When you fit a tokenizer (build its vocabulary) on validation or test data, you're allowing information from those datasets to influence how text gets converted to tokens. This creates a subtle but severe form of **data leakage**.

**Why This Matters: Step-by-Step**

Scenario 1: The Problem (Incorrect Approach)
```
Training data:    ["cat", "dog", "bird"]
Validation data:  ["elephant", "giraffe", "zebra"]
Test data:        ["tiger", "lion", "bear"]

❌ WRONG: Fit tokenizer on ALL three sets
  Vocabulary built: {cat, dog, bird, elephant, giraffe, zebra, tiger, lion, bear}
  
Problem:
  - Validation sees specialized tokens for "elephant", "giraffe", "zebra"
  - These tokens only exist BECAUSE validation data was used
  - Model learns to rely on these dedicated tokens
  - At test time, if "elephant" appears differently (or similar unseen animal),
    the model fails because it never learned to handle that pattern
  - Your validation/test metrics are artificially inflated
```

Scenario 2: The Correct Approach
```
Training data:    ["cat", "dog", "bird"]
Validation data:  ["elephant", "giraffe", "zebra"]
Test data:        ["tiger", "lion", "bear"]

✓ CORRECT: Fit tokenizer on TRAINING data only
  Vocabulary built: {cat, dog, bird, <unk>}
  
What happens:
  - During validation: "elephant" → splits into subwords or <unk>
    Model must handle unseen words
  - During testing: "tiger" → splits into subwords or <unk>
    Model must generalize
  - Model learned to work with token patterns from training
  - Validation and test metrics reflect REAL generalization
```

**The Leakage Mechanism (Why It's Silent & Dangerous)**

1. **Vocabulary Bias**: If you use validation data to build vocab, rare or domain-specific words in validation get their own tokens
   - Example: If validation has many medical terms, they get vocabulary entries
   - Training never saw these patterns during optimization
   - Model metrics look good in validation, but fail on real data with different vocabulary distribution

2. **Statistical Mismatch**: The model's weights are optimized for the training token distribution
   - If validation tokens don't exist in that distribution, metrics are artificially high
   - Real inference has different token patterns → performance drops

3. **Information Seepage**: Even subtle biases compound
   - Tokenizer fitting on all data means validation data slightly influences token ID assignments
   - Token IDs affect embeddings, which affects all downstream computation
   - This bias is never caught during evaluation

**Concrete Example: Why Metrics Become Misleading**

Domain Shift Scenario:
```
Training corpus: General English Wikipedia
  Vocab built from: Common words, BPE merges on frequent patterns
  Vocabulary: 50K tokens optimized for general text

Validation + Test corpus: Medical research papers
  Medical terms: "carcinoma", "metastasis", "biopsy", "hemoglobin"

❌ IF you fit tokenizer on all data:
  Vocabulary: 50K tokens INCLUDING dedicated medical tokens
  Validation accuracy: 94% (model sees dedicated tokens for medical terms)
  Test accuracy: 94% (test also has medical tokens)
  
  Reality check: Deploy to general English → accuracy drops to 72%
  Why? The model optimized for those dedicated medical tokens in validation,
  but never learned robust representations in training

✓ IF you fit tokenizer on training only:
  Vocabulary: 50K tokens from Wikipedia (general English)
  During validation: Medical words → broken into subwords or <unk>
  Validation accuracy: 78% (model struggles, as expected)
  Test accuracy: 79% (consistent struggle)
  
  Reality check: Deploy to general English → accuracy stays ~80%
  Why? Metrics were honest. Model learned robust general patterns.
```

**The Compounding Effect (Multi-Token Problem)**

With subword tokenization, the effect scales:
```
Word: "carcinoma" (unseen in training, present in validation data used for vocab fitting)

❌ If vocab includes this word:
  "carcinoma" → [token_ID: 31045]  (single dedicated token)
  Model learns an embedding for this specific token during training?
    → NO! It never saw this token in training
    → Embedding initialized randomly
    → Validation metrics fool you into thinking model understands this word

✓ If vocab does NOT include this word (training-only fitting):
  "carcinoma" → [token_IDs: 12034, 5019, 8892]  (broken into subword pieces)
  Model learns embeddings for these subword pieces during training
  When it sees "carcinoma" in validation, it composes understanding from known pieces
  Metrics are honest: lower, but representative of real generalization
```

**Why It's Hard to Detect**

This leakage is **silent** because:
1. No error messages
2. Metrics look good in validation
3. Only appears later in production or on truly held-out data
4. The bias is encoded in the tokenizer, not the model weights themselves

**The Fix: Three Separate Tokenization Paths (Best Practice)**

```python
# CORRECT workflow:

# 1. Build vocabulary from training data ONLY
tokenizer = BasicTokenizer()
tokenizer.fit(training_texts)  # ← ONLY training

# 2. Encode validation with the same tokenizer
validation_encoded = tokenizer.encode(validation_texts)

# 3. Encode test with the same tokenizer
test_encoded = tokenizer.encode(test_texts)

# Result: 
#   - Same vocabulary across all three splits
#   - No data from validation/test influenced the vocabulary
#   - If validation/test have unseen words → they map to <unk> or subwords
#   - Metrics reflect true generalization
```

**Why This Parallels Train/Test Split Philosophy**

General ML principle:
  Train/validation/test MUST be separate
  → Prevents overfitting to validation selection

Tokenizer fitting principle:
  Vocabulary is built from training data ONLY
  → Prevents leakage into token distribution
  → Ensures consistent encoding across splits
  → Metrics reflect real-world token patterns the model will encounter

**Senior-Level Interpretation**

At scale, tokenizer fitting is a **statistical decision** encoded in the preprocessing layer:

- **Training tokenizer fit on ALL data**: You're saying "assume validation/test tokens are representative of production. This is wrong."
- **Training tokenizer fit on training data only**: You're saying "production data will have unseen patterns. Validation/test should measure how model handles that."

The second is correct because **real production always has distributional shift** compared to training.

| Aspect | Train-Only Fitting | All-Data Fitting |
|--------|---|---|
| Leakage risk | None | High |
| Validation metrics | Honest, conservative | Inflated, optimistic |
| Unseen words handled by | Model's learned subword patterns | Dedicated vocabulary entries |
| Production performance | Matches validation closely | Degrades significantly |
| Token distribution match | Train ↔ Val ↔ Test aligned | Train misaligned from Val/Test |
| Debug difficulty | Easy to spot drift | Silent failures |

### Q3. What breaks if training and inference use different tokenizer versions?

The model receives different token ID patterns than it learned during training, which can degrade instruction following, retrieval grounding, formatting compliance, and overall quality.

### Q4. How does tokenisation impact context length?

Context length is measured in tokens, not words. A tokenization scheme that produces more tokens per input reduces effective usable context and increases inference cost.

### Q5. Why do we need attention masks with padding?

Without attention masks, models may attend to padding tokens as if they were real content, which introduces noise into training and inference.

### Q6. What is a robust truncation policy?

A robust policy preserves the most important information for the task, tracks truncation rate as a metric, and is aligned between training and inference. It should be explicit, not accidental.

**Detailed Explanation:**

**What Truncation Is & Why It Matters**

Truncation is cutting off text when it exceeds your maximum sequence length. Unlike padding (which adds filler tokens), truncation permanently removes information. This is a lossy operation—once text is truncated, the model never sees what was cut off.

Example:
```
Original text (200 tokens):
"The patient presented with chest pain on Monday. 
The doctor ran tests and found... [important diagnosis info] ... 
Treatment plan includes..."

With max_length=128:
Truncated to: "The patient presented with chest pain on Monday. 
The doctor ran tests and found..."

What the model never sees:
"...important diagnosis info... Treatment plan includes..."

Result: Critical medical information is lost silently.
```

**What Makes a Policy "Robust"?**

A robust truncation policy has three core properties:

1. **Task-Aware Preservation**: Truncate in a way that keeps the most task-critical information
   - Classification: Keep the main content, truncate metadata/timestamps
   - QA: Keep the question and evidence paragraphs, truncate distractors
   - Summarization: Keep full text as much as possible, cut only if forced
   - Chat: Keep recent user message and assistant context, truncate old history

2. **Tracking & Visibility**: Monitor truncation as a metric
   - What % of samples get truncated?
   - How much text is being lost on average?
   - Does truncation rate vary by data slice (language, user segment, prompt type)?

3. **Train-Inference Alignment**: Same truncation rule everywhere
   - If training truncates from the **end** of text, inference must too
   - If training keeps the **first 90% + last 10%**, inference must match
   - Mismatch between train and inference creates distribution shift

**Why "Explicit, Not Accidental" Matters**

Accidental truncation:
```python
# ❌ BAD: Implicit truncation (accidental, not tracked)
text = "very long text..."
encoded = tokenizer(text)  # Quietly truncates if too long
# You don't know how much was cut or if it matters
# Different backends may truncate differently
```

Explicit truncation:
```python
# ✓ GOOD: Explicit policy (intentional, tracked, reproducible)
def truncate_for_task(text, max_length, strategy="preserve_end"):
    """
    Truncate text with a known strategy.
    
    Args:
        strategy: "preserve_end" (keep conclusion)
                  "preserve_start" (keep context)
                  "preserve_middle" (keep key section)
    """
    tokens = tokenizer.encode(text)
    
    if len(tokens) <= max_length:
        return tokens, False  # Not truncated
    
    truncated_tokens = truncate_strategy(tokens, max_length, strategy)
    return truncated_tokens, True  # Was truncated
```

**Concrete Example: Instruction Tuning**

```
Input: [INST] Summarize this article: {article_text} [/INST]

Article is 5000 tokens. Max sequence length = 2048.

❌ BAD truncation policy:
  - Just cut it off at 2048
  - Lose the conclusion/summary targets
  - Model can't learn what you want

✓ GOOD truncation policy:
  - Reserve tokens: [INST] + instruction = 50 tokens
  - Reserved for [/INST] + response = 256 tokens
  - Available for article: 2048 - 50 - 256 = 1742 tokens
  - Truncate article to 1742, preserving first paragraphs
  - Rationale: Article beginning typically has key context
```

**Why Silent Truncation Is Dangerous**

```
Training phase:
- Sample: "User query: [50 tokens] | Context: [1000 tokens]"
- After truncation: "User query: [50 tokens] | Context: [950 tokens]"
- Truncation rate: 5% (tracked? usually no)
- Model learns with truncated context

Inference phase:
- Same user query: [50 tokens] | Same context: [1000 tokens]
- If truncation policy differs or isn't tracked:
  - Different backend might cut at [900 tokens]
  - Or context might be preserved fully
  - Model receives different distribution than training
  - Quality drops mysteriously

Root cause: Truncation was implicit/accidental
```

**How to Implement a Robust Policy**

```python
class RobustTruncationPolicy:
    def __init__(self, max_length, strategy="preserve_important"):
        self.max_length = max_length
        self.strategy = strategy
        self.truncation_stats = {"total": 0, "truncated": 0}
    
    def truncate(self, tokens, task_type="general"):
        """Apply task-aware truncation with explicit tracking."""
        
        # Track baseline
        original_length = len(tokens)
        self.truncation_stats["total"] += 1
        
        if len(tokens) <= self.max_length:
            return tokens, False  # Not truncated
        
        # Apply strategy
        if task_type == "qa":
            # For QA: preserve question (first part)
            truncated = self._preserve_question(tokens)
        elif task_type == "summarization":
            # For summarization: preserve document start
            truncated = self._preserve_start(tokens)
        else:
            # Default: preserve end (often has conclusion)
            truncated = self._preserve_end(tokens)
        
        self.truncation_stats["truncated"] += 1
        loss_pct = (1 - len(truncated) / original_length) * 100
        
        # Log for monitoring
        print(f"Truncated: {original_length} → {len(truncated)} ({loss_pct:.1f}% loss)")
        
        return truncated, True
    
    def _preserve_end(self, tokens):
        """Keep last N tokens (conclusion/summary)."""
        return tokens[-self.max_length:]
    
    def _preserve_start(self, tokens):
        """Keep first N tokens (context/setup)."""
        return tokens[:self.max_length]
    
    def truncation_rate(self):
        """Return % of samples that were truncated."""
        if self.truncation_stats["total"] == 0:
            return 0
        return (self.truncation_stats["truncated"] / 
                self.truncation_stats["total"]) * 100

# Usage
policy = RobustTruncationPolicy(max_length=2048, strategy="preserve_important")

# Training
for batch in training_data:
    for text in batch:
        tokens = tokenizer.encode(text)
        truncated_tokens, was_truncated = policy.truncate(tokens, task_type="qa")
        # Train on truncated_tokens

# Inference
same_policy = RobustTruncationPolicy(max_length=2048, strategy="preserve_important")
for user_input in inference_data:
    tokens = tokenizer.encode(user_input)
    truncated_tokens, was_truncated = same_policy.truncate(tokens, task_type="qa")
    # Inference with truncated_tokens (same policy!)

# Monitoring
print(f"Truncation rate in production: {same_policy.truncation_rate():.2f}%")
```

**Red Flags (What NOT to Do)**

| Anti-Pattern | Why It's Bad |
|---|---|
| Truncate without tracking | Don't know if it's a problem |
| Different truncation in train vs. inference | Distribution shift → quality drop |
| Truncate randomly from middle | Lose both context and conclusion |
| Assume model handles truncation gracefully | Silently fails on real data |
| Don't version truncation policy | Debugging becomes impossible |
| Ignore truncation rate growth | Can creep up as data changes |

**Senior-Level Insight**

Truncation is a **compression constraint encoded in preprocessing**—not a model design choice. A robust policy treats it as:
- A first-class metric (tracked, monitored, versioned)
- Task-aware (not one-size-fits-all)
- Deterministic (reproducible across train/val/inference)
- Documented (why this strategy, why this max_length)

The policy becomes part of your model's **input contract**—just like vocabulary or normalization. Breaking the contract (different truncation rules) breaks the model.

### Q7. How do you evaluate tokenization quality in a production NLP pipeline?

Track token length distributions, truncation rates, OOV/UNK rates (if relevant), task metrics by length bucket, and failure modes caused by malformed or multilingual inputs. Combine these with human review on edge cases.

### Q8. What is tokenizer drift?

Tokenizer drift is production behavior shift where real inputs tokenize differently than expected due to new domains, languages, formats, or version mismatch. It can reduce quality even if model weights are unchanged.

### Q9. Why are subword tokenizers preferred over word-level tokenizers in LLMs?

They generalize better to unseen words and multilingual text while keeping vocabulary manageable. This improves robustness and reduces hard OOV failures.

### Q10. What is the senior-level mindset for tokenisation?

Treat tokenization as part of model design and serving architecture. Version it, test it, monitor it, and align it across data preparation, evaluation, and production inference.

### Q11. What are attention mask and label mask from simple to senior understanding?

Simple:
- Attention mask tells the model which tokens are real and which are just padding.
- Label mask tells training which token positions should contribute to loss.

Intermediate:
- Attention mask affects the forward pass (what the model can attend to).
- Label mask affects the backward pass (where gradients come from).
- In many training setups, ignored labels are set to -100 so loss is skipped at those positions.

Senior-level interview answer:
Attention mask controls information flow during attention, preventing padded or invalid positions from influencing contextual representations. Label mask controls optimization scope by selecting which token positions contribute to objective computation. In chat SFT, a common pattern is to attend to full context (system, user, assistant) while applying loss primarily to assistant tokens. Correct mask design is a high-leverage reliability concern because mask errors silently distort both context utilization and gradient signal.

Memory hook:
- Attention mask = what the model can look at.
- Label mask = what the model learns from.

### Q12. What are OOV/UNK behavior and sequence truncation rate?

OOV/UNK (unknown) behavior:
- OOV means out-of-vocabulary text pieces that are not directly represented by the tokenizer vocabulary.
- UNK behavior describes how often those pieces become unknown tokens (for tokenizers that use an UNK token) and how that affects model quality.
- High UNK usage can indicate domain mismatch, language mismatch, noisy input formats, or tokenizer/version problems.

Sequence truncation rate:
- This is the percentage of samples whose tokenized length exceeds max sequence length and are therefore cut.
- Truncation can remove important instructions, evidence, or output targets, causing silent quality degradation.

Useful formulas:

$$
\text{UNK Rate} = \frac{\text{\\#UNK tokens}}{\text{\\#total tokens}}
$$

$$
\text{Truncation Rate} = \frac{\text{\\#truncated samples}}{\text{\\#total samples}}
$$

Senior-level interpretation:
- UNK rate is a representation-quality signal.
- Truncation rate is a context-loss signal.
- Both should be monitored by data slice (language, product flow, user segment, and prompt type), not only as global averages.
- If either metric rises, teams should check tokenizer alignment, input cleaning, context packing strategy, and max-length policy before assuming model-weight issues.

### Q13. What tokenization techniques should I know from beginner to advanced for NLP research and GenAI engineering?

Beginner:
- Whitespace, rule-based, character-level, and word-level tokenization to understand core tradeoffs.

Production baseline:
- Subword methods (BPE, WordPiece, Unigram) and byte-level variants, because these are standard for modern LLM pipelines.

Senior-level engineering:
- Choosing tokenizer method by task, language mix, domain, latency budget, and context budget.
- Measuring token efficiency (tokens per sample), truncation impact, OOV/UNK behavior, and downstream quality impact.
- Maintaining strict tokenizer versioning across training, validation, and inference.

Research-level awareness:
- Domain-adaptive tokenizer retraining, vocabulary expansion strategies, multilingual balancing, and tokenizer-free modeling directions.
- Understanding that tokenization is a modeling choice that affects scaling cost, alignment behavior, and evaluation fairness.

### Q14. What are BPE, WordPiece, and Unigram in NLP, and how do you explain them from beginner to senior level?

Beginner understanding:
- All three are subword tokenization methods.
- They break words into smaller pieces so models can handle rare or unseen words better.
- Example idea: instead of failing on a rare word, the tokenizer can split it into known pieces.

Intermediate understanding:
- BPE (Byte Pair Encoding): starts from small units and repeatedly merges the most frequent token pairs.
- WordPiece: also builds subwords, but chooses merges using a likelihood-oriented scoring objective rather than only raw frequency.
- Unigram language model: starts with a large candidate vocabulary and removes tokens that contribute less to corpus likelihood.

Practical differences:
- BPE is simple and efficient, widely used.
- WordPiece is common in many encoder-style NLP systems.
- Unigram is often used via SentencePiece and can be strong for multilingual and noisy text.

Senior-level interview answer:
BPE, WordPiece, and Unigram are subword vocabulary optimization strategies with different training objectives. BPE is frequency-merge driven, WordPiece is objective-scored merge driven, and Unigram is probabilistic pruning driven. In production, selection is not just academic: it affects token efficiency, context utilization, robustness on rare/domain-specific terms, multilingual behavior, and total training/inference cost. A senior engineer validates tokenizer choice with downstream metrics, length distributions, truncation impact, and domain-slice quality, then freezes tokenizer versioning as part of release governance.

### Q15. What does "Normalize text consistently" mean from beginner to senior level?

Beginner meaning:
- Apply the same text cleaning rules to all data so similar text looks the same before tokenization.

Intermediate meaning:
- Normalization stabilizes token IDs by reducing avoidable text variation (case, spacing, punctuation, unicode forms).
- The same normalization must be used in training, validation, and inference.

Senior-level interview answer:
"Normalize text consistently" means establishing a task-aware preprocessing contract and enforcing it identically across the entire ML lifecycle. At senior level, this is a reliability and governance concern, not only a preprocessing step. I version normalization rules, measure their impact on token distributions, OOV/UNK behavior, truncation rate, and downstream quality, and avoid over-normalization that removes task-critical signal (for example casing in NER or symbols in code/biomedical text).

Memory hook:
- Inconsistent normalization creates silent distribution shift.
- Consistent normalization improves token stability and production reliability.

### Q16. Does punctuation get added into vocabulary? How do you decide vocabulary size? Does this mean all words in the world should be in vocabulary?

Yes, punctuation is included in the vocabulary. Tokens like `.`, `,`, `?`, `!`, `:` are separate vocabulary entries because they carry semantic and structural meaning — a question mark changes intent, a period signals sentence boundaries, and commas affect parsing.

You do not need every word in the world. Modern NLP uses subword tokenization (BPE, WordPiece, SentencePiece), which solves this:
1. Start with characters (a-z, digits, punctuation) — these cover anything.
2. Merge frequent character pairs iteratively into subword tokens.
3. Stop merging when you hit a target vocabulary size.

Common vocabulary sizes: GPT-2 ~50K, LLaMA ~32K, GPT-4/cl100k ~100K.

Rare/unseen words get split into known subwords. Example: `"unhappiness"` → `["un", "happi", "ness"]`. This gives open vocabulary coverage with a finite token set.

Tradeoffs when choosing size:

| Smaller vocab | Larger vocab |
|---|---|
| More tokens per sentence (longer sequences) | Fewer tokens per sentence (shorter sequences) |
| Better generalization to rare words | More direct token-to-word mapping |
| Smaller embedding matrix | Larger embedding matrix (more parameters) |

Interview-ready summary:
Vocabulary size is a compression tradeoff — subword tokenization lets you represent any text with a fixed-size vocabulary (typically 32K–100K), where common words are single tokens and rare words are composed from subword pieces, including punctuation as explicit tokens for structural fidelity.

### Q17. What is Unicode normalization (NFC/NFKC) and why does it matter in NLP?

The same visible character can be stored differently in memory. For example, the letter é can be:
- Precomposed: single code point `U+00E9` (é as one unit)
- Decomposed: two code points `U+0065` (e) + `U+0301` (combining accent ´)

Both look identical on screen but are different byte sequences. Without normalization, a tokenizer treats them as different inputs → different token IDs → inconsistent model behavior.

The four Unicode normalization forms:

| Form | What it does |
|---|---|
| NFD | Decomposes into base + combining characters |
| NFC | Composes into single precomposed characters (most compact) |
| NFKD | Decomposes + replaces compatibility variants (e.g., `ﬁ` → `fi`) |
| NFKC | Composes + replaces compatibility variants |

NFC vs NFKC (the two common in NLP):
- NFC: canonical composition. `e + ´` → `é`. Preserves meaning, picks one representation.
- NFKC: does everything NFC does plus flattens compatibility characters: `ﬁ` → `fi`, `①` → `1`, `Ｈｅｌｌｏ` → `Hello`.

Why it matters:
1. Without normalization, the same word can get different token IDs depending on how it was typed or copied.
2. Search/retrieval can miss exact matches.
3. Deduplication fails on visually identical but byte-different strings.

Interview-ready summary:
Unicode normalization (typically NFKC in NLP) collapses equivalent character representations into a canonical form so that visually identical text always produces the same token IDs, preventing silent distribution mismatch between training and inference.

### Q18. How is number/date normalization done in NLP?

The problem: the same number or date can appear in many text forms, creating unnecessary token variation.
- `"12/05/2026"` vs `"May 12, 2026"` vs `"2026-05-12"` — each tokenizes differently.
- `"1,000"` vs `"1000"` vs `"1.000"` (European) — same value, different tokens.

Common normalization strategies:

| Strategy | Example | When to use |
|---|---|---|
| Canonical date format | All dates → `YYYY-MM-DD` | When exact date matters but format doesn't |
| Placeholder replacement | `"May 12, 2026"` → `<DATE>` | When the date value itself doesn't matter (e.g., sentiment analysis) |
| Number bucketing | `3847` → `<NUM>` | Classification tasks where exact number is noise |
| Comma/separator removal | `1,000,000` → `1000000` | Standardize numeric representation |
| Unit standardization | `"3.5M"` → `"3500000"` | When downstream needs consistent magnitude |

Practical implementation (Python):
- Regex replacement: `re.sub(r'\b\d+[\d,\.]*\b', '<NUM>', text)`
- Date parsing: `dateutil.parser.parse("May 12, 2026").strftime("%Y-%m-%d")`
- Separator removal: `re.sub(r'(\d),(\d)', r'\1\2', text)`

When NOT to normalize:
- Math/finance tasks — exact numbers are the answer.
- NER — you need to extract the original date/number.
- Code generation — literals must be preserved.
- Medical dosages — `"500mg"` vs `"0.5g"` distinction matters.

Interview-ready summary:
Number/date normalization reduces surface-form variation by mapping equivalent representations to a canonical form or placeholder, improving token stability and generalization — but only when the task objective doesn't depend on the original format or exact value.

### Q19. How does a vocabulary look for a large document/corpus used to train an LLM?

A vocabulary for a large training corpus is a lookup table: token string → integer ID.

Example vocabulary file (simplified, ~50K entries):
```json
{
  "the": 0, "of": 1, "and": 2, "to": 3, "a": 4, "in": 5,
  "Ġthe": 100, "Ġis": 101,
  "un": 5023, "friend": 5024, "ly": 5025, "Ġhello": 8901,
  ".": 13, ",": 14, "?": 15, "!": 16,
  "Ġneur": 31002, "olog": 31003, "ical": 31004,
  "<PAD>": 50255, "<UNK>": 50256, "<BOS>": 50257
}
```
(`Ġ` = space prefix, meaning "this token starts a new word")

Structure of a real vocabulary (e.g., GPT-2 with 50,257 tokens):

| Range | What's in it | Examples |
|---|---|---|
| Top ~1000 | Most common full words and subwords | `the`, `is`, `of`, `ing`, `tion` |
| ~1000–10000 | Common words + frequent subwords | `hello`, `friend`, `comput`, `ment` |
| ~10000–30000 | Less common subwords + domain terms | `neur`, `olog`, `quant`, `ization` |
| ~30000–50000 | Rare subwords + single characters + special | `ñ`, `♦`, byte fallbacks |
| Special tokens | Control tokens | `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>` |

How it's built from a large corpus:
1. Collect all training text (could be terabytes — web, books, code, etc.)
2. Run BPE/WordPiece/Unigram algorithm on it:
   - Start with base characters (~256 bytes or unicode chars)
   - Iteratively merge most frequent pairs
   - Stop at target size (e.g., 32K, 50K, 100K)
3. Result: a fixed mapping file (`vocab.json` + `merges.txt`)

Real files you'd see in a tokenizer directory:
- `vocab.json` — token_string → token_id mapping
- `merges.txt` — BPE merge rules in order (for BPE-based)
- `tokenizer_config.json` — special tokens, max length, etc.
- `special_tokens_map.json`

Detailed explanation of vocab.json and merges.txt:

These two files together define a complete BPE tokenizer.

`vocab.json` — The dictionary:
A simple JSON mapping of every token string to its unique integer ID. This is what the model actually sees — it never sees text, only these integer IDs. When you feed `"hello"`, the tokenizer looks up each piece in this file and returns the corresponding IDs.

`merges.txt` — The splitting recipe:
An ordered list of merge rules that tells the tokenizer how to build subwords from characters. Order matters — earlier rules = more frequent pairs = applied first.

Example merges.txt (BPE merge rules, applied in order):
```
Ġ t
i n
Ġ a
h e
r e
Ġt he
i ng
```

How they work together at tokenization time:
Input: `"hello"`
- Step 1: Split into characters → `['h', 'e', 'l', 'l', 'o']`
- Step 2: Apply merge rules in order from merges.txt:
  - Rule "h e" → `['he', 'l', 'l', 'o']`
  - Rule "he ll" → `['hell', 'o']`
  - Rule "hell o" → `['hello']`
- Step 3: Look up each merged token in vocab.json → `'hello' → 8901`
- Result: `[8901]`

Why two files instead of one:

| File | Purpose |
|---|---|
| `vocab.json` | Maps token strings → IDs (the **what**) |
| `merges.txt` | Defines how to split/merge text into those tokens (the **how**) |

You need both because:
- `vocab.json` alone doesn't tell you how to break `"unfriendly"` into subwords
- `merges.txt` alone doesn't tell you what ID each piece maps to
- Together they form a deterministic, reproducible tokenization pipeline

Quick analogy:
- `vocab.json` = a phone book (name → number)
- `merges.txt` = the rules for how to spell/combine names before looking them up

Senior one-liner:
`vocab.json` defines the token-to-ID mapping and `merges.txt` defines the ordered merge operations that deterministically convert raw text into those tokens — both are required to reproduce identical tokenization across training and inference.

### Q20. How does merges.txt look in code, and how are the rules defined?

You don't write `merges.txt` manually — it's automatically generated by the BPE training algorithm.

The BPE training algorithm (simplified Python):
```python
from collections import Counter

# Step 1: Start with character-level splits of your corpus
corpus = {
    ('l', 'o', 'w'): 5,            # "low" appears 5 times
    ('l', 'o', 'w', 'e', 'r'): 2,  # "lower" appears 2 times
    ('n', 'e', 'w'): 6,            # "new" appears 6 times
    ('n', 'e', 'w', 'e', 'r'): 3,  # "newer" appears 3 times
}

merges = []  # This becomes merges.txt

for i in range(num_merges):  # e.g., 10000 iterations
    # Step 2: Count all adjacent pairs across corpus
    pair_counts = Counter()
    for word, freq in corpus.items():
        for j in range(len(word) - 1):
            pair = (word[j], word[j+1])
            pair_counts[pair] += freq

    # Step 3: Find the most frequent pair
    best_pair = pair_counts.most_common(1)[0][0]
    # e.g., ('n', 'e') with count 9 (6 from "new" + 3 from "newer")

    # Step 4: Record this merge rule
    merges.append(best_pair)
    # merges.txt gets: "n e"

    # Step 5: Apply merge — replace all occurrences in corpus
    new_corpus = {}
    for word, freq in corpus.items():
        new_word = merge_pair(word, best_pair)  # ('n','e','w') → ('ne','w')
        new_corpus[new_word] = freq
    corpus = new_corpus

# Step 6: Save
# merges.txt = ordered list of merge rules
# vocab.json = all unique tokens seen after all merges + base chars
```

What merges.txt actually looks like (real file from GPT-2):
```
#version: 0.2
Ġ t
Ġ a
h e
i n
r e
o n
Ġt he
e r
Ġ s
a t
...
(~50,000 lines total)
```
Each line = one merge rule: `"token_A token_B"` means "merge A+B into AB".
Rules are in priority order — line 1 is applied before line 2, etc.

How rules are applied at tokenization time:
```python
# Input word: "newer"
tokens = ['n', 'e', 'w', 'e', 'r']

# Apply merges in order:
# Rule 1: "n e" → merge
tokens = ['ne', 'w', 'e', 'r']
# Rule 5: "e r" → merge
tokens = ['ne', 'w', 'er']
# Rule 42: "ne w" → merge (if exists)
tokens = ['new', 'er']
# Rule 89: "new er" → merge (if exists)
tokens = ['newer']
# Stop when no more applicable rules
```

Using HuggingFace tokenizers library (production way):
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=50000,
    min_frequency=2,
    special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
)

# Train on corpus — generates vocab + merges automatically
tokenizer.train(files=["corpus.txt"], trainer=trainer)
tokenizer.save("my_tokenizer.json")
```

Key summary:

| What | How |
|---|---|
| Who writes merges.txt? | The BPE training algorithm, not humans |
| What decides the rules? | Frequency — most common adjacent pairs get merged first |
| How many rules? | `vocab_size - base_characters` (e.g., 50000 - 256 ≈ 49744 rules) |
| Order matters? | Yes — earlier rules = higher frequency = applied first |
| When is it frozen? | After training — never changes during model training or inference |

Interview-ready summary:
`merges.txt` is the ordered output of BPE's greedy frequency-based pair-merging algorithm — each rule says "combine these two tokens into one," applied in strict priority order during tokenization, and the entire file is generated once from the training corpus then frozen for the model's lifetime.

Key points for interview:
1. Vocabulary is corpus-derived — frequent patterns get their own token; rare ones stay split.
2. Size is fixed before training — once built, it never changes during training or inference.
3. Frequency-driven — "the" is one token, but "neurological" might be `["neur", "olog", "ical"]` because it's rarer.
4. Covers everything — even unseen text is representable via subword fallback to characters/bytes.

Interview-ready summary:
A vocabulary is a fixed-size frequency-optimized mapping from subword strings to integer IDs, built once from the training corpus, where common patterns get dedicated entries and rare text decomposes into smaller known pieces — ensuring full coverage without infinite size.

---

## 7. Practical checklist

- Freeze tokenizer version before large training runs.
- Log token stats for train/val/inference traffic.
- Monitor truncation and length drift weekly.
- Validate that special token IDs match model config.
- Keep decode/encode round-trip tests in CI.
- Include multilingual and noisy-text test cases.

---

## 8. Quick revision lines

- Tokenisation maps text to IDs; models consume IDs.
- Token budget drives cost, latency, and context usage.
- Keep tokenizer consistent across training, validation, and inference.
- Subword tokenization improves robustness in real-world inputs.
- Senior engineers monitor tokenization metrics in production.

---

## 9. Transformer Architecture Q&A (Encoder vs Decoder)

### Q21. What is the difference between encoder-only and decoder-only transformer?

**Beginner Understanding:**

Encoder-only and decoder-only are two different transformer architecture designs for different tasks.

| Aspect | Encoder-Only | Decoder-Only |
|--------|---|---|
| **Attention** | Bidirectional (sees all tokens) | Causal (left-to-right only) |
| **Primary Task** | Understanding/classification | Generation/next-token prediction |
| **Examples** | BERT, RoBERTa, ELECTRA | GPT-2, GPT-3, Llama, Llama-2 |
| **Training Method** | Masked language modeling (MLM) | Causal language modeling (CLM) |
| **Input** | Full context available at once | Tokens generated left-to-right |

**Intermediate Understanding:**

Encoder-only (bidirectional):
- All positions can attend to all other positions (past, present, future)
- Training task: "The cat sat on the [MASK]" → predict the masked word using full context
- Use cases: classification, semantic search, NER, QA (retrieve answer from passage)
- Advantage: Rich contextual understanding from both directions
- Drawback: Cannot generate text (doesn't know how to produce left-to-right sequences)

Decoder-only (causal):
- Each position can only attend to previous positions (left-to-right)
- Training task: "The cat sat on the" → predict "mat", then "The cat sat on the mat" → predict next token
- Use cases: text generation, instruction tuning, chat, code generation
- Advantage: Naturally supports autoregressive generation (inference matches training)
- Drawback: Cannot "see ahead" even though full context is available at training time

**Senior-Level Interview Answer:**

Encoder-only vs decoder-only is a **fundamental design tradeoff** rooted in the task objective and inference constraint.

Encoder-only (BERT):
- Training: Bidirectional attention enables full context optimization—efficient (15% of tokens masked, one forward pass predicts them all)
- Inference: No autoregressive generation needed; suitable for classification/ranking/retrieval tasks
- Architecture: Pure transformer encoder stack, position-dependent but not causally-masked

Decoder-only (GPT):
- Training: Causal masking applied even though full sequence exists, ensuring training distribution matches inference
- Inference: Tokens generated one-at-a-time with autoregressive sampling; each token consumes only prior context
- Architecture: Pure transformer decoder stack with causal self-attention and language modeling head

The key insight: **Causal masking during training is NOT a limitation but a requirement** for decoder-only models. It prevents exposure bias (train-inference distribution mismatch) where training used unrealistic full context but inference only has partial context.

Encoder-decoder hybrid (T5, BART) uses both: encoder processes full input bidirectionally, decoder generates output token-by-token with causal masking. Best of both worlds at 2x parameter cost.

### Q22. What is causal masking in decoder-only architecture?

**Beginner Understanding:**

Causal masking is a technique that **prevents the model from looking at future tokens** when processing the current position. It ensures the model can only see past tokens (left-to-right), simulating how generation works.

Example:
```
Sequence: "I love NLP"
Position:  0 1   2 3

Position 0 ("I"):    can see [0]              (only itself)
Position 1 ("love"): can see [0, 1]           (past + self)
Position 2 ("NLP"):  can see [0, 1, 2]        (all past + self)

With causal masking, position 1 cannot see position 2.
Without it (bidirectional), position 1 could peek at "NLP" (cheating).
```

**Intermediate Understanding:**

Causal masking is implemented as an **upper triangular attention mask**:

```
Query-Key attention matrix for seq_len=4:
Position:  0  1  2  3
         ┌──────────────┐
    0    │ ✓  ✗  ✗  ✗  │  ← Query 0 can only attend to position 0
    1    │ ✓  ✓  ✗  ✗  │  ← Query 1 can attend to positions 0,1
    2    │ ✓  ✓  ✓  ✗  │  ← Query 2 can attend to positions 0,1,2
    3    │ ✓  ✓  ✓  ✓  │  ← Query 3 can attend to all positions
         └──────────────┘

✓ = allowed to attend
✗ = blocked (masked)
```

Implementation (PyTorch):
```python
# Create upper triangular boolean mask (True = blocked)
mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

# Apply to attention scores
scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(d_head)
scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)
# Positions marked as True get -1e9 → softmax(-1e9) ≈ 0 weight

weights = F.softmax(scores, dim=-1)  # Future tokens get ~0 attention
```

**Senior-Level Interview Answer:**

Causal masking is a **training-inference alignment mechanism**. Without it, decoder-only models exhibit severe exposure bias.

Why it's essential:

Training vs inference mismatch (without causal masking):
```
Training (no causal mask, bidirectional):
  Position 2 ("NLP") can attend to [0, 1, 2, 3, 4, 5, ...]
  Model sees all future context — easy, unrealistic task
  Learns to rely on lookahead information

Inference (autoregressive generation):
  Position 2 can only see [0, 1, 2] (previously generated tokens)
  Positions 3+ don't exist yet
  Model receives different context distribution → quality crashes
```

With causal masking:
```
Training (with causal mask, enforced left-to-right):
  Position 2 can attend to [0, 1, 2] only
  Matches inference-time context distribution
  Gradients optimize for realistic left-to-right generation

Inference:
  Same causal-masked attention applied
  No distribution mismatch — model performs as trained
```

Mechanically, causal masking works by:
1. Computing full attention scores between all positions
2. Setting future-position scores to -1e9 (large negative value)
3. Softmax converts -1e9 → ≈0 weight
4. Model effectively ignores future tokens (zero gradient contribution)

Key insight for interviews:
Causal masking is **not** a computational limitation or a convenience. It's a **correctness requirement** to ensure training reflects inference constraints. Removing it would train a fundamentally different distribution-matching task, causing catastrophic failure in production.

| Aspect | With Causal Masking | Without Causal Masking |
|--------|---|---|
| Training context | Position i sees [0...i] | Position i sees [0...seq_len] |
| Inference context | Position i sees [0...i] | Position i sees [0...i] |
| Distribution match | ✓ Aligned | ✗ Misaligned |
| Exposure bias risk | Low (training ≈ inference) | High (different tasks) |
| Production quality | Stable | Degrades significantly |

The fundamental principle: **Always train what you will deploy.** Causal masking ensures this principle by making training and inference use identical context distributions.

### Q23. How does element-wise addition of position embeddings help in learning attention?

**The Problem You're Identifying:**

With concatenation, position embeddings form a **separate, structured signal**:

```python
# CONCATENATION (problematic)
combined = [token_emb, pos_emb]  # Shape: [batch, seq, 2*d_model]
                 ↑        ↑
            random      highly structured
           (semantic)   (sequential 0,1,2,3...)

Problem:
- Position channel is predictable/deterministic
- Token channel is random (learned from data)
- Attention could "cheat" by learning to rank based on position structure
- Position info dominates because it's cleaner/more organized
```

With concatenation, the model could learn position shortcuts during attention:
```
Example attention ranking (bad):
  Token at pos 0: rank = 0.3
  Token at pos 1: rank = 0.5  ← Higher because position channel stronger
  Token at pos 2: rank = 0.7  ← Even higher
  
Result: Ranking by position alone, ignoring semantic similarity
```

**Why Addition Solves This:**

With addition, the two signals are **fused together**:

```python
# ADDITION (correct)
combined = token_emb + pos_emb  # Shape: [batch, seq, d_model]
              ↑          ↑
          random    structured
          mixed into one representation

Result:
- Position info is "hidden" inside the combined vector
- Model must learn to disentangle both signals
- Can't just rank by position alone
- Forces proper semantic + positional reasoning
```

**Example of embedding fusion:**
```
Token emb for token 10:   [0.5, 0.2, -0.1, 0.3, 0.1, -0.4, 0.2, 0.0]
Position emb for pos 0:   [0.1, -0.2, 0.3, -0.1, 0.2, 0.0, -0.3, 0.1]
Combined (addition):      [0.6, 0.0, 0.2, 0.2, 0.3, -0.4, -0.1, 0.1]
                           ↑
                     Can't separate them!
                     Requires learning their interaction
```

**Why Addition Forces Better Attention Learning:**

During forward pass:
```python
# Token 10 at position 0:
combined_emb = token_emb + pos_emb  # Blended signal

# Attention computation uses this blended representation
q = linear_q(combined_emb)
k_other = linear_k(other_combined_emb)
scores = q @ k_other.T  # Must account for BOTH semantics AND position

# Attention learns: "Attend more to semantically similar tokens 
#                    that are nearby (not too far away)"
```

During backward pass (gradient flow):
```
Loss gradient flows back through:
  1. Attention weights
  2. Combined embedding
  3. Both token_emb AND pos_emb simultaneously

The model learns:
- How much position contributes to this attention decision
- How much token semantics contribute
- They're learned together as an inseparable team
- Can't optimize one while ignoring the other
```

**Concrete Ranking Behavior Comparison:**

With concatenation (problematic):
```
Query at position 2:
  combined = [token_feat_1, ..., token_feat_4, pos_feat_1, ..., pos_feat_4]
                         token part               position part

Attention scores:
  Token 1 at pos 0: high_token_sim + low_pos_penalty  = 0.5
  Token 2 at pos 1: low_token_sim  + low_pos_penalty  = 0.3  ← Position bonus!
  Token 3 at pos 2: mid_token_sim  + zero_pos_penalty = 0.5
  Token 4 at pos 3: high_token_sim + high_pos_penalty = 0.2

Result: Position 2 and 0 tie, even though pos 0 semantically different
Problem: Position structure too visible, enables ranking shortcuts
```

With addition (correct):
```
Query at position 2:
  combined = [0.6, 0.0, 0.2, 0.2, 0.3, ...]  ← Mixed signal, can't separate

Attention scores:
  Token 1 at pos 0: 0.4  (semantically similar, but far)
  Token 2 at pos 1: 0.2  (not similar, close)
  Token 3 at pos 2: 0.8  (semantically similar, AND local context)
  Token 4 at pos 3: 0.3  (not similar, close)

Result: Position 2 wins clearly because semantic + positional signals align
Learning: "Position 2 is semantically similar AND nearby — worth attending"
Benefit: Must use both signals, can't shortcut via position alone
```

**Why This Matters for Attention Learning:**

| Aspect | Concatenation | Addition |
|--------|---|---|
| **Position signal visibility** | Separate, clean channel | Fused, implicit |
| **Attention ranking risk** | Can rank by position alone | Must use both signals |
| **Gradient distribution** | Pos and token gradients separate | Gradients mix and interact |
| **Learned patterns** | May learn position shortcuts | Learns true semantic+positional reasoning |
| **Generalization** | Can overfit to position patterns | More robust to position shifts |

**Code Example: Attention Learning with Addition**

```python
# ADDITION (correct learning)
token_emb = torch.tensor([0.5, 0.2, -0.1, 0.3])
pos_emb = torch.tensor([0.1, -0.2, 0.3, -0.1])

combined = token_emb + pos_emb  # [0.6, 0.0, 0.2, 0.2]

# During attention:
q = W_q @ combined  # Linear projection mixes both signals
k = W_k @ other_combined

# Backprop:
loss.backward()
# Gradients flow to BOTH token_emb and pos_emb
# They're optimized together as an inseparable team
# Neither can dominate the ranking independently

---

# CONCATENATION (ranking problem)
combined = torch.cat([token_emb, pos_emb])  # [0.5, 0.2, -0.1, 0.3, 0.1, -0.2, 0.3, -0.1]

# During attention:
q = W_q @ combined  # W_q has separate "position detector" rows
                    # Could learn to use pos features directly for ranking

# The model might learn:
# "If pos feature 5 is high → rank high"  (BAD, position shortcut)
# Instead of:
# "If semantic features + pos features align → rank high"  (GOOD)
```

**Key Insight:**

Addition forces **entanglement** of signals → model can't separate or prioritize position artificially. It must learn their **true interaction**: how position constrains meaningful semantic relationships.

Concatenation allows **separation** → attention could shortcut by learning position patterns directly, ignoring semantic signal.

**Senior-Level Interview Answer:**

Element-wise addition of position embeddings prevents the model from learning position-based ranking shortcuts in attention. By fusing token and position signals into a single vector (rather than concatenating them as separate channels), the model is forced to learn their interaction during gradient flow. This ensures attention weights reflect semantic similarity constrained by positional context, not position patterns alone. The mechanism is fundamental to why addition is preferred over concatenation in transformer architectures — it aligns the optimization objective with the desired learning outcome: position-aware semantic attention rather than position-dominant ranking.

---

## 10. Representation Learning in NLP

### 10.1 What representation learning is

Representation learning is the process of automatically learning **numerical vector encodings** of language (tokens, words, sentences, documents) such that the geometry of the vector space captures meaningful properties — semantic similarity, syntax, relationships, and context.

Instead of hand-engineering features (TF-IDF counts, POS tags, n-gram indicators), the model learns *what to represent and how* directly from data, optimized end-to-end for a task or a self-supervised objective.

```
"king" - "man" + "woman" ≈ "queen"   (classic embedding-arithmetic example)
```

The embedding table you build during tokenisation (Section 3, "Vocabulary") is the entry point into representation learning: token IDs are meaningless integers until the embedding layer turns them into vectors that training can shape.

### 10.2 Levels of representation in NLP

| Level | What it captures | Example |
|---|---|---|
| Token/subword embedding | One fixed vector per token ID | Word2Vec, GloVe, embedding table lookup |
| Contextual embedding | Same token, different vector depending on surrounding context | BERT, ELMo, GPT hidden states |
| Sentence/document embedding | Pooled or [CLS]-based vector for a whole span of text | Sentence-BERT, Nomic Embed (used in RAG chunking) |
| Cross-modal embedding | Text and another modality (image, audio) aligned in one shared space | CLIP, LLaVA vision-language embeddings |

### 10.3 Historical progression

1. **Static embeddings** (Word2Vec, GloVe) — one fixed vector per word regardless of context. "Bank" (river) and "bank" (money) get the same vector.
2. **Contextual embeddings** (ELMo, BERT) — the vector for a word depends on the sentence it appears in, solving the polysemy problem above.
3. **Transformer-based representations** — self-attention builds representations by letting every token attend to every other token; this is the backbone of modern LLMs and why causal masking (Section 9, Q22) and position-embedding fusion (Section 9, Q23) directly shape representation quality.
4. **Contrastive representation learning** (SimCLR, CLIP, SimCSE) — representations are learned by pulling similar pairs together and pushing dissimilar pairs apart in embedding space, rather than predicting the next token or a masked token.

### 10.4 Why it matters for the tokenisation-to-inference pipeline

- Tokenisation decides *what* discrete units exist; representation learning decides *what those units mean* as vectors.
- Good representations transfer across tasks (pretraining → fine-tuning) — this is the entire premise of transfer learning in NLP.
- Representation collapse (embeddings converging to near-identical vectors, losing discriminative power) is a real production failure mode, monitored the same way tokenizer drift is monitored (Section 3, "Tokenizer drift").
- Bi-encoder vs cross-encoder retrieval (used in RAG re-ranking) is a direct representation-learning design choice: independent representations (fast, less expressive) vs jointly-computed representations (slow, more expressive).

### 10.5 Interview-ready summary

Representation learning is the process of learning vector encodings of language such that vector-space geometry reflects semantic and syntactic meaning — progressing historically from static embeddings, to contextual embeddings, to transformer self-attention representations, to contrastive-learned embeddings — and it is the foundation layer that tokenisation feeds into and every downstream NLP task depends on.

Key points for interview:
1. Representation learning replaces hand-engineered features with learned, task-optimized vectors.
2. Static embeddings (Word2Vec/GloVe) give one vector per word; contextual embeddings (BERT/GPT) give a different vector per occurrence based on context.
3. Contrastive learning (CLIP, SimCSE) shapes representation space by similarity/dissimilarity rather than next-token or masked-token prediction.
4. Representation collapse and embedding drift are production risks that require the same monitoring discipline as tokenizer drift.
5. Retrieval architectures (bi-encoder vs cross-encoder) are a practical embodiment of representation-learning trade-offs: precompute-and-compare speed vs joint-attention accuracy.

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

### Q3. What breaks if training and inference use different tokenizer versions?

The model receives different token ID patterns than it learned during training, which can degrade instruction following, retrieval grounding, formatting compliance, and overall quality.

### Q4. How does tokenisation impact context length?

Context length is measured in tokens, not words. A tokenization scheme that produces more tokens per input reduces effective usable context and increases inference cost.

### Q5. Why do we need attention masks with padding?

Without attention masks, models may attend to padding tokens as if they were real content, which introduces noise into training and inference.

### Q6. What is a robust truncation policy?

A robust policy preserves the most important information for the task, tracks truncation rate as a metric, and is aligned between training and inference. It should be explicit, not accidental.

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
$\text{UNK Rate} = \frac{\#\text{UNK tokens}}{\#\text{total tokens}}$

$\text{Truncation Rate} = \frac{\#\text{truncated samples}}{\#\text{total samples}}$

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

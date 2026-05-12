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

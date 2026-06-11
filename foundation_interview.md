# Foundation Interview Prep — LLM + AI + Computer Vision + Vision-Language Fusion

> **Goal:** Foundational concept mastery for AI Researcher interviews with a focus on LLM, CV, and image-text fusion.
> **Format:** Question → (discuss) → Best Answer
> **Date created:** 2026-06-11

---

## Question Set 1 — Core Foundations

### Q1. Pretraining vs Fine-Tuning vs Alignment
Explain the difference between pretraining, supervised fine-tuning, and alignment in modern LLM pipelines. Why are all three often needed?

---

### Q2. Self-Attention & Scaled Dot-Product Attention
What is self-attention, and why is scaled dot-product attention used instead of plain dot-product attention?

---

### Q3. Transformer Block Components
In a Transformer block, what are the roles of:
1. Multi-head attention
2. Feed-forward network
3. Residual connections
4. Layer normalization

---

### Q4. Tokenization & Its Impact
What is tokenization, and how does tokenization quality impact downstream model performance and inference cost?

---

### Q5. Context Length in LLMs
Why does context length matter in LLMs, and what are practical failure modes when sequence length grows too large?

---

### Q6. Encoder-Only vs Decoder-Only vs Encoder-Decoder
Compare encoder-only, decoder-only, and encoder-decoder architectures. Give one strong use case for each.

---

### Q7. Hallucinations in LLMs
What are hallucinations in LLMs? Distinguish between factual hallucination, reasoning error, and instruction-following failure.

---

### Q8. RAG vs Fine-Tuning
What is retrieval-augmented generation (RAG), and when is RAG better than fine-tuning?

---

### Q9. Evaluation Metrics — Precision, Recall, F1, Accuracy
Explain precision, recall, F1, and accuracy. In which situations is accuracy a misleading metric?

---

### Q10. Train/Val/Test Splits & Data Leakage
What are train/validation/test splits, and what is data leakage? Give one real example of leakage in multimodal datasets.

---

### Q11. CV Task Hierarchy — Classification, Detection, Segmentation
In computer vision, what are key differences between classification, detection, and segmentation tasks?

---

### Q12. CNN Inductive Biases vs Vision Transformers
What inductive biases do CNNs provide, and why did Vision Transformers still become successful?

---

### Q13. Contrastive Learning & CLIP-style Training
Explain contrastive learning at a high level. How does CLIP-style training align image and text embeddings?

---

### Q14. Image-Text Fusion Strategies
In image-text fusion systems, what is the difference between:
1. Early fusion
2. Late fusion
3. Cross-attention fusion

What are the trade-offs of each?

---

### Q15. Embedding Space Alignment & Cosine Similarity
What is embedding space alignment, and why does cosine similarity commonly work for cross-modal retrieval?

---

### Q16. Zero-Shot vs Few-Shot vs Fine-Tuned Transfer
Explain zero-shot, few-shot, and fine-tuned transfer in vision-language models with practical examples.

---

### Q17. Bias & Safety Risks in Vision-Language Models
What are common biases and safety risks in vision-language models (e.g., social bias, shortcut learning, spurious correlations)?

---

### Q18. Evaluation Suite for Image-Text Models
How would you design a basic evaluation suite for an image-text model used in production? Include both quality and reliability checks.

---

### Q19. Quantization for LLM / VLM Inference
What is quantization, and what are typical trade-offs between latency, memory, and quality for LLM/VLM inference?

---

### Q20. Debugging a Weak Image+Text Model
If you had to improve a weak image+text model, what would you inspect first:
1. Data
2. Architecture
3. Training objective
4. Evaluation

Why in that order?

---

## Answers Section
*(Answers will be added question-by-question during discussion)*

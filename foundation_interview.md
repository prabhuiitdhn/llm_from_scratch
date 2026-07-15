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

---

### A1. Pretraining vs Supervised Fine-Tuning vs Alignment

#### 1. Pretraining — *"Learn the world"*
The model is trained on **massive, raw, unlabeled text** (trillions of tokens — web, books, code) using a self-supervised objective, typically **next-token prediction** (causal LM) or **masked token prediction** (BERT-style).

**What it learns:** Grammar, syntax, facts, reasoning patterns, world knowledge — general-purpose language representations.

**Analogy:** A student who has read every textbook, article, and novel — knows a lot, but has no specific skill or manners yet.

**Cost:** Extremely expensive (millions of $ in compute). Done once or rarely.

---

#### 2. Supervised Fine-Tuning (SFT) — *"Learn the task"*
The pretrained model is further trained on **curated, labeled (instruction, response) pairs** — humans write ideal outputs for given prompts.

**What it learns:** How to follow instructions, how to format answers, task-specific behavior (summarization, QA, coding, etc.).

**Analogy:** The same student now takes a structured course with a teacher who shows them exactly how to answer exam questions.

**Key insight:** SFT alone can still produce a model that is unhelpful, sycophantic, or unsafe because the labels are limited in coverage.

---

#### 3. Alignment — *"Learn human values & safety"*
After SFT, alignment methods like **RLHF** (Reinforcement Learning from Human Feedback) or **DPO** (Direct Preference Optimization) teach the model to **prefer outputs humans rate as better** — helpful, honest, harmless.

**What it learns:** Rank good answers over bad ones (from human preference data), refuse harmful requests, be calibrated and not overconfident.

**Analogy:** The student now learns professional etiquette, ethics, and how to handle edge cases — things no textbook explicitly taught.

---

#### Why all three are needed?

| Stage | Without it, model is... |
|---|---|
| Pretraining | Has no knowledge of language or world |
| SFT | Knows language but can't follow instructions reliably |
| Alignment | Follows instructions but may be unsafe, biased, or sycophantic |

They address **different failure modes** — knowledge, task formatting, and value alignment.

> **Key interview insight:** Pretraining is about **capability**. SFT is about **behavior**. Alignment is about **values**.

---

#### Follow-up Questions for Deeper Understanding

**FU1 — Pretraining mechanics:**
During pretraining, the model predicts the next token. What exactly is the loss function, and what does minimizing it actually teach the model?

**FU2 — SFT data quality:**
If you have 1 million SFT examples of mediocre quality vs 10,000 extremely high-quality examples — which would you choose and why? What does this tell you about SFT?

**FU3 — Alignment depth:**
RLHF has a reward model in it. Where does that reward model come from, and what is the fundamental risk of training against it?

**FU4 — Ordering insight:**
Why can't you do alignment *before* SFT? What breaks if you reverse the order?

**FU5 — Vision-Language bridge:**
In a vision-language model like LLaVA or CLIP, which of the three stages applies to the vision encoder? Does a vision encoder go through all three stages the same way a language model does?

---

#### Follow-up Answers — Q1 Deep Dive

**FU1 — What does minimizing next-token prediction loss actually teach?**

The loss is **cross-entropy** between the predicted token distribution and the actual next token:

$$\mathcal{L} = -\sum_{t} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

To minimize this, the model is forced to build **internal world models** — it must understand grammar, facts, cause-effect, discourse structure, and even basic reasoning, because all of these help predict what word comes next. It's not just memorization; a model that memorizes can't generalize to unseen sequences.

> **Key insight:** Next-token prediction is a deceptively powerful proxy. Good prediction requires genuine understanding of language and knowledge.

---

**FU2 — 1M mediocre vs 10K high-quality SFT examples?**

**Choose 10K high-quality.** Research (e.g., LIMA paper — "Less Is More for Alignment") showed that ~1,000 carefully curated examples can produce strong instruction-following behavior. Why?

- The pretrained model already has capability. SFT is just **steering**, not teaching from scratch.
- Noisy, mediocre data introduces **conflicting gradients** — the model learns inconsistent formats, wrong tone, or bad reasoning patterns.
- Quality here means: diverse coverage, correct reasoning, clean formatting, appropriate length.

> **Key insight:** SFT is a **signal injection**, not a volume game. Data quality dominates data quantity in fine-tuning.

---

**FU3 — Where does the RLHF reward model come from, and what is the risk?**

The reward model (RM) is trained on **human preference pairs** — humans compare two model outputs and pick the better one. The RM learns to score outputs as humans would.

**The fundamental risk: reward hacking (Goodhart's Law)**

> "When a measure becomes a target, it ceases to be a good measure."

The LLM being trained by RL will find outputs that **score high on the reward model** but are not actually good — e.g., very verbose, confident-sounding but wrong answers. The RM is an imperfect proxy for true human preference, and the policy exploits its blind spots.

This is why **KL divergence penalty** is added — to keep the aligned model from drifting too far from the SFT base model.

---

**FU4 — Why can't you do alignment before SFT?**

Because **alignment needs a behaviorally coherent base to refine**.

- A raw pretrained model doesn't follow instructions at all — it just continues text. Asking humans to compare two raw pretrained completions for "helpfulness" is meaningless noise.
- The reward model trained on such comparisons would have no useful signal.
- RLHF needs a model that already **attempts** the task (from SFT) before it can learn to do it *better*.

Order matters: capability → task behavior → value refinement. Skipping SFT means alignment has nothing to align.

---

**FU5 — Does the vision encoder go through all three stages like an LLM?**

Not in the same way. In vision-language models:

| Stage | Language Model | Vision Encoder |
|---|---|---|
| Pretraining | Next-token prediction on text | Contrastive (CLIP), masked patch prediction (MAE), or supervised classification (ImageNet) |
| SFT | Instruction-following on text | Usually **frozen** or lightly fine-tuned — vision features are projected into language space via an adapter/projector |
| Alignment | RLHF/DPO on (prompt, response) pairs | Rarely aligned independently — alignment is applied to the **whole system** via the LLM head |

In **LLaVA**, the vision encoder (CLIP ViT) is frozen during instruction tuning — only the **projection layer** and LLM are trained. This preserves strong visual representations from contrastive pretraining.

> **Key insight for vision domain:** Vision encoders are pretrained separately, then **frozen and bridged** to the language model. The alignment stage operates on the combined system, not the vision encoder alone.

---

### FU5 Deep Dive — How Vision Encoders Train & Interact with Language

#### Part 1: Vision Encoder Pretraining Strategies

**Strategy A — Supervised Classification (ImageNet)**
Train on labeled images with cross-entropy loss. Learns object-level semantics.
Limitation: closed-vocabulary, coarse labels — can't generalize to open-ended descriptions.

**Strategy B — Contrastive Learning (CLIP-style) — Most important for VLMs**

Training data: **(image, text caption) pairs** scraped from the internet (~400M pairs for CLIP, 5B for LAION).

```
Image: [photo of a golden retriever on a beach]
Text:  "a dog playing on the sand near the ocean"
```

Two encoders trained in parallel:
- **Image encoder** (ViT or ResNet) → image embedding $v$
- **Text encoder** (Transformer) → text embedding $t$

Loss (InfoNCE / contrastive):

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_{j} \exp(\text{sim}(v_i, t_j)/\tau)}$$

Pulls matching pairs together, pushes non-matching pairs apart. $\tau$ is a learned temperature.

**What it learns:** Semantic alignment — "image of a dog" and "a dog" land near each other in shared vector space. Enables zero-shot generalization via open-vocabulary concepts.

**Strategy C — Masked Image Modeling (MAE-style)**
Randomly mask ~75% of image patches. Train encoder to reconstruct missing pixels.
Learns low/mid-level structure — texture, shape, spatial layout. Used in BEiT, MAE, EVA.

---

#### Part 2: How Vision Encoder Connects to a Language Model

A ViT-L/14 on a 224×224 image produces **256 patch tokens × 1024-dim**. The LLM expects tokens in its own embedding space (e.g., 4096-dim for a 7B LLM). A **Projector** bridges this gap:

```
Image (224×224)
    ↓
Vision Encoder (ViT)            → [256 patch tokens × 1024-dim]
    ↓
Projector (MLP or Cross-Attn)   → [256 tokens × 4096-dim]
    ↓
Concatenate with text tokens
    ↓
LLM (e.g. LLaMA)                → generates response
```

**Projector designs:**

| Type | How it works | Used in |
|---|---|---|
| Linear projection | Matrix multiply to change dimension | CLIP zero-shot |
| MLP projector | 2-layer MLP with nonlinearity | LLaVA-1.5 |
| Cross-attention | Image tokens attend to language tokens | Flamingo, Qwen-VL |

---

#### Part 3: Training Data at Each Stage

**Stage 1 — Vision encoder pretraining (CLIP-style)**
```json
{ "image": "beach.jpg", "caption": "a woman walking her dog in a park" }
```
Source: CC12M, LAION-5B — web-scraped alt-text and captions.

**Stage 2 — Projector alignment pretraining**
Goal: teach projector to translate visual tokens into LLM's language space.
Vision encoder frozen. Only projector weights update.
```json
{
  "image": "cat_on_table.jpg",
  "conversations": [
    {"role": "human", "text": "Describe the image."},
    {"role": "assistant", "text": "A cat is sitting on a wooden table near a window."}
  ]
}
```
Source: LLaVA-Pretrain 558K (CC3M filtered image-caption pairs).

**Stage 3 — Visual instruction tuning (SFT)**
Goal: follow complex visual instructions — VQA, reasoning, OCR, charts.
Projector + LLM fine-tuned. Vision encoder remains frozen.
```json
{
  "image": "graph.png",
  "conversations": [
    {"role": "human", "text": "What is the peak value in this bar chart and in which year?"},
    {"role": "assistant", "text": "The peak value is 4.2 million in 2019, shown by the tallest bar."}
  ]
}
```
Source: LLaVA-Instruct-150K (GPT-4 generated), VQAv2, TextVQA, GQA.

---

#### Part 4: Full VLM Training Pipeline

```
Phase 0:  Vision encoder pretrained (CLIP/MAE)            → frozen after this
Phase 1:  Projector pretrained on image-caption pairs     → only projector trains
Phase 2:  Full SFT on visual instruction data             → projector + LLM train
Phase 3:  Optional alignment (RLHF/DPO)                  → whole system
```

---

#### Why Freeze the Vision Encoder?

1. Already has strong representations from contrastive pretraining on 400M–5B pairs.
2. Fine-tuning risks **catastrophic forgetting** — visual features degrade.
3. **Computational cost** — ViT-L has ~300M params; freezing saves GPU memory.
4. The bottleneck is **language alignment**, not vision quality — the projector is the critical bridge.

---

#### Key Interview Takeaways

> 1. Vision encoders are **pretrained independently** using contrastive or self-supervised objectives.
> 2. The **projector/adapter** is the critical bridge mapping visual patch tokens into the LLM's embedding space.
> 3. Training happens in **phases** — encoder frozen, projector aligned first, then LLM fine-tuned.
> 4. Training data evolves: raw captions → simple descriptions → complex instruction-following conversations.
> 5. Richer projector architecture (cross-attention > MLP > linear) allows more dynamic visual-language fusion.

---

### Cross-Attention Fusion & Positional Encoding in ViT — Deep Dive

---

#### Part 1: How Cross-Attention Fusion Works

**What is Attention in General?**
Attention answers: "How much should token A look at token B when building its representation?"
- Self-attention: every token attends to every other token in the **same sequence**
- Cross-attention: tokens from **sequence A** attend to tokens from **sequence B**

**Cross-Attention Mechanism — Step by Step**

In vision-language cross-attention, **language tokens ask questions** and **image tokens provide answers**.

| Matrix | Source | Meaning |
|---|---|---|
| **Query (Q)** | Language tokens | "What am I looking for?" |
| **Key (K)** | Image patch tokens | "What do I contain?" |
| **Value (V)** | Image patch tokens | "What information do I provide?" |

$$\text{CrossAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Steps:
1. Each language token generates a Query vector
2. Each image patch token generates a Key vector
3. $QK^T$ scores: "how relevant is this image patch to this language token?"
4. Softmax normalizes into attention weights (sum to 1)
5. Weighted sum over Value vectors → each language token gets a **vision-informed representation**

**Visual Intuition:**
```
Text query: "What color is the car?"

Language tokens:  [What] [color] [is]  [the]  [car] [?]
                    Q      Q     Q      Q      Q    Q
                     \      \     \      \     /    /
                      ──────── cross-attention ───────
                     /      /     /      /     \    \
                    K,V    K,V   K,V    K,V   K,V  K,V
Image patches:  [sky] [road] [red_car] [tree] [building] ...

Result: "car" token now strongly attends to [red_car] patch → outputs red-aware representation
→ Model generates: "The car is red."
```

**Cross-Attention vs MLP Projector:**

| | MLP Projector | Cross-Attention |
|---|---|---|
| Image tokens fed as | Prefix tokens concatenated to text | Dynamic lookup via Q/K/V |
| Language can choose what to look at? | No — all patches treated equally | Yes — input-dependent attention weights |
| Computation | Cheaper | More expensive but more expressive |
| Used in | LLaVA, InstructBLIP | Flamingo, Qwen-VL, BLIP-2 |

> **Key insight:** Cross-attention lets the language model **selectively attend** to the most relevant image regions given the current text context. MLP projection treats all patches uniformly — one-time transformation, not dynamic.

---

#### Part 2: Positional Encoding in ViT Patches

**The Problem — ViT Has No Built-in Spatial Structure**
A ViT treats an image as a flat sequence of patches. Without positional encoding, the model cannot tell whether a patch is top-left or bottom-right. Shuffling patches randomly would give the same output — the model is **permutation invariant**.

**How ViT Patches are Created**
A 224×224 image with patch size 16×16:

$$\text{Number of patches} = \frac{224}{16} \times \frac{224}{16} = 14 \times 14 = 196 \text{ patches}$$

Each 16×16 patch is flattened → linear projection → patch embedding of dim $d$ (e.g., 768).
A **[CLS] token** is prepended → total sequence = 197 tokens.

**Positional Encoding Types:**

**Type 1 — Learned 1D Positional Embedding (Original ViT)**
Each position (0 to 196) gets a learnable vector of size $d$, added to the patch embedding:
$$z_i = \text{PatchEmbed}(x_i) + \text{PE}(i)$$
Simple and effective. Limitation: fixed to training resolution — doesn't generalize to higher-res inputs.

**Type 2 — 2D Sinusoidal / Learned 2D Positional Embedding**
Encode (row, column) separately:
$$\text{PE}(r, c) = [\text{PE}_{\sin}(r) \;||\; \text{PE}_{\sin}(c)]$$
Better for dense tasks (segmentation, detection) where spatial layout matters explicitly.

**Type 3 — Interpolated Positional Encoding (for resolution generalization)**
When using a pretrained ViT at higher resolution than trained on — bilinearly interpolate the learned 2D position grid to the new size.
Example: LLaVA-1.5 uses 448×448 input with CLIP's 336×336 ViT → PE interpolated from 196 → 784 positions.

**Type 4 — RoPE (Rotary Position Embedding) for Vision**
Newer models (InternViT, EVA) use 2D RoPE — encodes relative positions via rotation in the attention computation rather than additive shifts. Better length generalization — handles variable resolution without retraining.

**Why Positional Encoding Matters for VLMs:**

| Task | Why position matters |
|---|---|
| VQA: "What is on the left?" | Must know which patches are spatially left |
| OCR / document understanding | Word order is spatial — left-to-right, top-to-bottom |
| Counting objects | Must distinguish separate spatial regions |
| Grounding: "Where is the cat?" | Output bounding box requires spatial patch identity |

> **Key insight:** Positional encoding converts a ViT from a **bag-of-patches** model into a spatially-aware encoder. Higher-resolution inputs with interpolated or 2D PE dramatically improve fine-grained visual tasks like OCR and dense grounding.

---

### Top CV Applications Solved Using Vision Transformers (ViT)

---

#### 1. Image Classification
The origin task. ViT (2020) matched then surpassed CNNs on ImageNet when pretrained on large datasets (JFT-300M).
Key models: ViT, DeiT (data-efficient), Swin Transformer
**Why ViT wins at scale:** global receptive field from token 1 — CNNs need many layers to see the whole image.

---

#### 2. Object Detection
**DETR (Detection Transformer, 2020) — landmark paper:**
- CNN backbone extracts features → flattened as tokens → Transformer encoder-decoder
- Decoder uses **learned object queries** that cross-attend to image features
- Outputs bounding boxes + class directly — **no NMS (non-max suppression) needed**

Follow-ups: Deformable DETR, DINO, DETA — all ViT-based.

---

#### 3. Semantic & Instance Segmentation
Models: Mask2Former, SegFormer, **SAM (Segment Anything Model)**
- SegFormer: hierarchical ViT backbone + lightweight MLP decoder → fast dense prediction
- **SAM (Meta, 2023):** ViT-H pretrained on 1B masks — prompts (point, box, text) → any object segmented zero-shot
- SAM 2: extends to **video segmentation** with temporal attention

**Why ViT helps:** Global context — can segment objects spanning the full image, which CNNs struggle with.

---

#### 4. Visual Question Answering (VQA)
Image + natural language question → predicted answer.
Models: BLIP-2, LLaVA, InstructBLIP
Architecture: ViT encoder → projector → LLM decoder
Requires both spatial understanding (ViT) and language reasoning (LLM) — a pure fusion task.

---

#### 5. Image Captioning
ViT encodes image → LLM/decoder generates description.
Models: BLIP, GIT (Generative Image-to-Text), Flamingo
Key challenge: **hallucination** — model generates plausible but visually unsupported descriptions. Linked to projector alignment quality.

---

#### 6. Visual Grounding & Referring Expression Comprehension
Given: "the man wearing a blue hat on the left" → output bounding box.
Models: GLIP, Grounding DINO, OFA
Uses cross-attention between text tokens and image patches to localize the referred region.
**Positional encoding quality directly impacts accuracy here.**

---

#### 7. Document Understanding / OCR
Reading text in complex layouts — receipts, forms, papers, charts.
Models: TrOCR (ViT + BERT), Donut (no OCR engine), LayoutLMv3
**Why ViT over CNN:** fine-grained patch attention reads small text; 2D PE captures row/column layout; high-res ViT (448×448+) needed.

---

#### 8. Medical Imaging
Radiology (X-ray, CT, MRI), pathology, ophthalmology.
Models: MedSAM, BioViL, CheXzero (CLIP for chest X-rays)
- Long-range dependencies matter — lesion in one region changes interpretation of another
- Contrastive pretraining on (image, radiology report) pairs → zero-shot diagnosis
- SAM adapted to 3D volumes for CT/MRI segmentation

---

#### 9. Video Understanding
Action recognition, video QA, dense captioning.
Models: TimeSformer, VideoMAE, Video-LLaMA, InternVideo

**Two strategies for time:**
- **Divided space-time attention:** attend to spatial patches first, then temporal dimension
- **Tubelet embedding:** 3D patch tokens spanning $t$ frames simultaneously

---

#### 10. 3D Vision & Point Clouds
LiDAR / depth data for autonomous driving, robotics.
Models: Point Transformer, 3DETR, Uni3D
Patches become **3D point clusters** — same attention mechanism over spatial point neighborhoods.

---

#### Summary Map

```
Vision Transformer Applications
│
├── 2D Image Tasks
│   ├── Classification (ViT, DeiT, Swin)
│   ├── Detection (DETR, Grounding DINO)
│   ├── Segmentation (SegFormer, SAM, Mask2Former)
│   └── Document/OCR (Donut, TrOCR, LayoutLM)
│
├── Vision-Language Tasks
│   ├── VQA (LLaVA, BLIP-2)
│   ├── Captioning (Flamingo, GIT)
│   └── Grounding (GLIP, OFA)
│
├── Specialized Domains
│   ├── Medical (MedSAM, CheXzero)
│   └── 3D / Point Cloud (Point Transformer)
│
└── Video (TimeSformer, VideoMAE, Video-LLaMA)
```

> **Interview angle for vision+language profile:** Know DETR (detection), SAM (segmentation), and LLaVA/BLIP-2 (VLMs) deeply. These three cover the full spectrum an interviewer would probe for an image+text fusion researcher role.

---

## Cross-Attention for Multi-Modal (Video + Text) AI

### Q. How does cross-attention work for multi-modal video and text AI?

**Beginner Understanding:**

Cross-attention is a mechanism that lets one modality (e.g., text) "look at" and learn from another modality (e.g., video frames). Unlike self-attention where tokens only attend to other tokens of the same modality, cross-attention creates bridges between different modalities.

Example:
```
Video frames: [frame_0, frame_1, frame_2, ...]  (visual embeddings)
Text query:  "What is happening?"  (text embeddings)

Cross-attention:
  Text query attends to video frames
  → Text learns "which frames are relevant"
  → Combines visual information into text representation
  → Model understands video content through text question
```

**Intermediate Understanding:**

Cross-attention has three components: Query (Q), Key (K), Value (V)

```
Self-attention (within one modality):
  Q from text, K from text, V from text
  → Text attends to text

Cross-attention (between modalities):
  Q from text, K from video, V from video
  → Text queries attend to video keys/values
  → Text learns which video regions are relevant
```

**Attention Score Computation:**
```
scores = (Q @ K.T) / sqrt(d_model)
weights = softmax(scores)  # Attention weights
output = weights @ V
```

In multi-modal:
```
Q shape: [batch, seq_len_text, d_model]
K shape: [batch, num_frames, d_model]  ← Video
V shape: [batch, num_frames, d_model]  ← Video

scores shape: [batch, seq_len_text, num_frames]
              ↑ Each text token gets weights over frames

output shape: [batch, seq_len_text, d_model]
              ↑ Each text token is enriched with video info
```

**Common Multi-Modal Architectures:**

| Architecture | Flow | Use Case |
|---|---|---|
| **Video Captioning** | Video → Encoder → Cross-Attention ← Text Decoder | Generate text describing video |
| **Visual QA** | Video → Encoder, Question → Cross-Attention | Answer questions about video |
| **Video Retrieval** | Query text & Video → Cross-Attention alignment | Find videos matching text |
| **Video Understanding** | Video frames → Cross-Attention ← Text prompts | Classify/segment with text guidance |

**Concrete Example: Video Captioning**

```
Input: Video with frames [f0, f1, f2, f3]
Task: Generate caption

Architecture:
  1. Video Encoder: Extract visual embeddings per frame
     f0 → v_emb_0 [shape: d_model]
     f1 → v_emb_1 [shape: d_model]
     f2 → v_emb_2 [shape: d_model]
     f3 → v_emb_3 [shape: d_model]
  
  2. Text Decoder (with cross-attention to video)
     
     Start with BOS token → text_emb
     
     FIRST DECODING STEP:
       Query: text_emb = "BOS" embedding [d_model]
       Key/Value: [v_emb_0, v_emb_1, v_emb_2, v_emb_3] (video)
       
       Attention scores:
         score_0 = text_emb @ v_emb_0 / sqrt(d_model) = 0.3
         score_1 = text_emb @ v_emb_1 / sqrt(d_model) = 0.5  ← Highest
         score_2 = text_emb @ v_emb_2 / sqrt(d_model) = 0.1
         score_3 = text_emb @ v_emb_3 / sqrt(d_model) = 0.1
       
       Attention weights (after softmax):
         w_0 = 0.15,  w_1 = 0.35,  w_2 = 0.25,  w_3 = 0.25
       
       Output: enriched_emb = w_0*v_emb_0 + w_1*v_emb_1 + w_2*v_emb_2 + w_3*v_emb_3
       
       Interpretation: Model attended most to frame 1 (tennis player hitting ball)
     
     Then generate token: "A" (from enriched_emb + self-attention history)
     
     SECOND DECODING STEP:
       Query: "A" token embedding (now enriched with video context)
       Cross-attend to video again:
       
       Output: enriched_emb_2 (new video context)
       Generate next token: "person"
     
     Continue until EOS token
     
  Result: "A person playing tennis"
```

**Senior-Level Technical Details:**

**1. Multi-Head Cross-Attention**

```python
# Pseudo-code
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        self.W_q = nn.Linear(d_model, d_model)  # Query projection (from text)
        self.W_k = nn.Linear(d_model, d_model)  # Key projection (from video)
        self.W_v = nn.Linear(d_model, d_model)  # Value projection (from video)
        self.num_heads = num_heads
    
    def forward(self, text_emb, video_emb):
        # text_emb: [batch, seq_len_text, d_model]
        # video_emb: [batch, num_frames, d_model]
        
        Q = self.W_q(text_emb)  # [batch, seq_len_text, d_model]
        K = self.W_k(video_emb)  # [batch, num_frames, d_model]
        V = self.W_v(video_emb)  # [batch, num_frames, d_model]
        
        # Split into multiple heads
        Q = Q.view(batch, seq_len_text, num_heads, d_head).transpose(1, 2)
        K = K.view(batch, num_frames, num_heads, d_head).transpose(1, 2)
        V = V.view(batch, num_frames, num_heads, d_head).transpose(1, 2)
        
        # Compute attention per head
        scores = Q @ K.transpose(-2, -1) / sqrt(d_head)
        # Shape: [batch, num_heads, seq_len_text, num_frames]
        
        weights = softmax(scores, dim=-1)
        output = weights @ V
        # Shape: [batch, num_heads, seq_len_text, d_head]
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch, seq_len_text, d_model)
        return output
```

**2. Sequential vs Parallel Cross-Attention**

Sequential (Decoder-style for generation):
```
Step 1: Generate token 1
  - Cross-attend to all video frames
  - Use previously generated tokens (self-attention)
  - Decode token 1

Step 2: Generate token 2
  - Cross-attend to all video frames (same video)
  - Use token 1 + previous history (self-attention)
  - Decode token 2

Risk: Compounding errors if early tokens are wrong
```

Parallel (Encoder-style, for understanding):
```
Process all video frames + entire text together:
  - All text tokens cross-attend to all video frames simultaneously
  - Symmetric bidirectional alignment
  - Better for retrieval/matching tasks

Advantage: No error accumulation, captures full alignment
```

**3. Video Representation Strategies**

```
Strategy 1: Frame-level cross-attention
  Video: [frame_0, frame_1, frame_2, ...]  (each ~2D image)
  Query: text tokens
  
  Problem: Too many frames (30fps × 60s = 1800 frames)
  Solution: Subsample to keyframes or temporal pooling

Strategy 2: Temporal pooling before cross-attention
  Video: Aggregate frames temporally → [segment_0, segment_1, segment_2]
  Each segment = temporal average of 5-10 frames
  
  Benefit: Fewer video "tokens" for cross-attention
  Tradeoff: Loss of temporal fine-grain detail

Strategy 3: Spatial-temporal features
  Video: Extract 3D CNN features
  Shape: [num_segments, spatial_patches, d_model]
  Cross-attend over both spatial and temporal dimensions
```

**4. Common Video + Text Architectures**

| Model | Architecture | Cross-Attention Role |
|---|---|---|
| **ViLBERT** | Separate ViT (video) + BERT (text) + cross-attention layers | Bidirectional alignment between modalities |
| **CLIP** | Vision encoder + Text encoder + cosine similarity matching | Align video & text in shared embedding space |
| **BLIP** | Vision transformer encoder + Text decoder with cross-attention | Unified understanding + generation |
| **Flamingo** | Vision tokens + frozen LLM + gated cross-attention | Inject visual grounding into LLM |
| **LLaVA** | Vision encoder → Linear projection → LLM input (implicit cross-attn) | Visual features feed into LLM text stream |

**Key Interview Points:**

1. **Cross-attention is directional**: Query modality "asks" about Key/Value modality (not symmetric unless applied both ways)

2. **Sequence length matters**: Video has 100s-1000s of frames; text has 10s-100s tokens. Cross-attention costs O(seq_text × seq_video × d_model)

3. **Temporal dynamics**: Video inherently has temporal structure; must preserve it (not just treat frames as set of images)

4. **Alignment challenge**: Without supervision, cross-attention learns alignment from scratch (noisy in early training). Helps to have paired video-text data.

5. **Cascading errors in generation**: When text is generated sequentially (decoder style), early mistakes propagate. Cross-attention keeps video as anchor to recover.

**Interview One-Liner:**

Cross-attention enables multi-modal understanding by letting one modality (e.g., text queries or generated tokens) dynamically select and aggregate features from another modality (e.g., video frames) through a learned weighted combination—the attention weights reveal which frames/regions are semantically relevant to each text element.

**Practical Implementation Pattern:**

```python
# Simplified video QA with cross-attention

# Extract video frame features
video_features = video_encoder(video)  # [batch, num_frames, d_model]

# Encode question
question_tokens = text_tokenizer(question)  # [batch, seq_len, d_model]

# Cross-attention decoder
for step in range(max_answer_length):
    # Current answer token (or BOS for first step)
    current_token_emb = get_token_embedding(answer_tokens[-1])
    
    # Cross-attend to video
    video_context = cross_attention(
        query=current_token_emb,
        key=video_features,
        value=video_features
    )  # [d_model]
    
    # Combine with question context (self-attention on question history)
    question_context = self_attention(question_tokens, history)
    
    # Fuse all contexts
    combined = fusion(video_context + question_context)
    
    # Generate next answer token
    next_token = answer_decoder(combined)
    answer_tokens.append(next_token)
```

**Common Pitfalls:**

1. **Ignoring temporal order**: Treating video frames as unordered set (loses motion/causality)
2. **No frame preprocessing**: Raw pixels to attention is inefficient; always use visual encoder first
3. **Insufficient masking**: Without proper masking, model can attend to "future" frames (cheating)
4. **Over-compression**: Pooling too aggressively loses details needed for fine-grained video understanding
5. **Training instability**: Cross-attention can be slow to converge; needs careful initialization and learning rate scheduling


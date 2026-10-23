Important points:
1. multimodel projector
2. cross-modal alignment
3. perceiver style visual resampling
4. gates cross attention
5. Q-former, multimodal alignment
6. connectors, or fusion module
7. Cross attention, co-attention, adapters, samplers, modality specific merger modules
8. Early fusion, intermediate fusion, late fusion
9. Gated cross attention

# 25 In-Depth Interview Questions: Small VLM, VLM and Vision-AI (On-Device / Edge)

## A. VLM Architecture and Internals
1. **Architecture trade-offs:** Compare the design of a small VLM such as SmolVLM, PaliGemma, MobileVLM, Qwen2-VL-2B, InternVL-1B or Florence-2. How do the vision encoder (SigLIP vs CLIP vs ViT), the projector (linear, MLP, Q-Former, Perceiver resampler, pixel-shuffle) and the LLM backbone each affect accuracy, latency and memory? If you had a 2B-parameter budget on a phone, how would you split it across the three components, and why?

2. **Visual token explosion:** A 1080p image can produce thousands of visual tokens. How do approaches like AnyRes/tiling, dynamic resolution (Qwen2-VL's naive dynamic resolution, M-RoPE), pixel-shuffle and token merging or pruning (FastV, ToMe, SparseVLM) reduce this? Walk through how the visual token count affects prefill latency, KV-cache size and time-to-first-token. How would you pick the token budget for a given NPU?

3. **Attention and positional encoding:** Explain how positional information is handled for image patches in the encoder (learned absolute vs 2D RoPE) and in the LLM (1D RoPE vs M-RoPE vs 2D positions). What goes wrong when you run at a resolution different from training? How would you interpolate position embeddings, and what is the accuracy impact?

4. **Contrastive vs generative pretraining:** Why does SigLIP (sigmoid loss) scale better than CLIP (softmax contrastive loss) at small batch sizes? What does each give you for VLM downstream quality? When would you prefer a vision encoder trained generatively (e.g. a captioning objective, AIMv2) over a contrastive one for a small VLM?

5. **Cross-attention vs early-fusion (prefix) designs:** Compare Flamingo/Llama-3.2-Vision style cross-attention with LLaVA-style token concatenation. What are the implications for KV-cache, memory, batching, streaming video and on-device runtime support?

## B. Training, Fine-tuning and Data
6. **Multi-stage VLM training recipe:** Describe a production recipe (projector alignment → full or LoRA instruction tuning → optional preference tuning). Which components do you freeze or unfreeze at each stage, and why? What failure modes appear if the vision encoder is unfrozen too early (catastrophic drift, loss of OCR or fine-grained capability)?

7. **Fine-tuning a VLM with limited data and compute:** You have 20k domain-specific image-text pairs, such as appliance UI screens or home-robot scenes. Compare LoRA, QLoRA, DoRA, prefix tuning and projector-only tuning. How do you choose the target modules and ranks, and how do you stop the model forgetting general capabilities? How do you detect and measure forgetting?

8. **Data engineering for VLMs:** How would you build a high-quality instruction dataset for a niche domain, such as a smart-home or robot perception assistant? Cover synthetic data from larger teachers, caption re-writing, hard-negative mining, dedup (perceptual hashes), contamination checks against benchmarks, privacy and PII (faces, license plates), and label noise. How do you validate the synthetic data?

9. **Hallucination in VLMs:** What causes object, attribute and relation hallucination (language-prior dominance, weak grounding, low-res features)? Compare mitigations: contrastive decoding (VCD), OPERA, RLHF-V / DPO with image-grounded preference data, grounding-aware training and higher-res encoders. How do you measure it (POPE, CHAIR, HallusionBench), and how would you set up a regression gate in CI?

10. **Distillation for small VLMs:** How would you distill a 70B+ teacher into a ≤2B student? Cover logit distillation vs feature or attention-map distillation vs sequence-level distillation, and distilling the vision tower. How do you handle tokenizer or vocabulary mismatch between teacher and student? What are the limits of distillation relative to training small models on more tokens?

## C. On-Device Optimization and Deployment
11. **Quantization in depth:** Compare PTQ vs QAT, weight-only (W4A16, GPTQ, AWQ) vs weight+activation (W8A8, SmoothQuant) and KV-cache quantization (INT8/INT4/FP8). Why are vision encoders and projectors usually more sensitive to quantization than the LLM? How do activation outliers affect INT8 NPUs? How would you diagnose a layer-wise accuracy collapse and fix it with mixed precision?

12. **Latency anatomy of a VLM on-device:** Break down end-to-end latency (image decode → preprocess → vision encoder → projector → prefill → decode). Which stages are compute-bound vs memory-bandwidth-bound on a mobile SoC? How do you measure and profile each stage, and which optimizations apply to which stage?

13. **Deployment stack choices:** Compare ExecuTorch, ONNX Runtime (with QNN or NNAPI EPs), TFLite/LiteRT, llama.cpp/GGUF, MLC-LLM, TensorRT-LLM and the vendor SDKs (Qualcomm QNN/SNPE, Samsung Exynos NPU / ONE toolchain). How would you export a VLM (dynamic shapes, multiple graphs for the encoder and decoder, KV-cache as I/O)? What breaks during export (unsupported ops, dynamic control flow, rotary embeddings), and how do you work around it?

14. **Memory budgeting:** The device has 12 GB RAM shared with the OS and other apps, and your VLM gets about 2–3 GB. Compute the memory for weights (INT4), KV-cache (layers × heads × dim × seq × bytes), vision activations and the image buffer. How would you use mmap/lazy loading, KV-cache eviction, sliding-window or sink attention, and GQA/MQA to fit? What are the OOM and thermal risks, and how do you handle them gracefully?

15. **Speculative decoding, batching and scheduling on-device:** Does speculative decoding help on a memory-bound mobile decoder? When would it hurt (draft cost, NPU kernel support)? How do you do continuous batching, prefix caching for repeated system prompts or repeated images, and chunked prefill when the device also runs camera and ISP pipelines concurrently?

16. **Thermal, power and QoS:** How do you design for sustained performance under thermal throttling and a battery budget (J/token, mJ/inference)? What are the strategies: adaptive resolution, early exit, model cascading (tiny classifier → small VLM → cloud), DVFS-aware scheduling, and heterogeneous CPU/GPU/NPU partitioning? How do you benchmark reproducibly?

17. **Hybrid on-device and cloud:** Design the routing logic that decides when a query is handled by the on-device VLM vs a larger cloud model. Cover confidence calibration, uncertainty estimation, privacy constraints, latency SLOs, offline fallback and consistency of behaviour between the two.

## D. Vision AI and Sensing AI
18. **Efficient vision backbones:** Compare MobileNetV3/V4, EfficientNet, MobileViT, FastViT, EfficientFormer and RepViT, and distilled ViTs (DeiT-Tiny, TinySAM, EdgeSAM). What operators are NPU-friendly (e.g. depthwise convs, GELU or LayerNorm costs, attention quantization)? How do structural re-parameterization and NAS help on a specific target device?

19. **Dense perception on-device:** How would you deploy open-vocabulary detection or segmentation (Grounding-DINO, OWL-ViT, YOLO-World, SAM-lite, CLIPSeg) on-device? Cover real-time constraints, tracking-by-detection and keeping temporal consistency across frames. How do you combine these with a VLM for grounded reasoning (box or point outputs, visual prompts)?

20. **Multimodal sensor fusion:** You have RGB, depth, LiDAR and IMU. Compare early, mid and late fusion. How would you feed depth or point clouds into a VLM (depth as a pseudo-image, 3D tokens like 3D-LLM / PointLLM / LLaVA-3D, BEV features)? Cover time synchronization, calibration (intrinsics, extrinsics, camera-LiDAR, camera-IMU), missing or degraded sensors and sensor dropout during training.

21. **Depth, LiDAR and IMU processing details:** Explain monocular depth estimation (Depth Anything, scale ambiguity) vs metric depth from ToF/LiDAR and how you deal with noise, holes and multi-path artifacts. Describe LiDAR point-cloud processing (voxelization, PointNet++, PointPillars, sparse convs) and IMU pre-integration and drift. How does visual-inertial odometry combine them, and how would you use the result as VLM context?

22. **Video and temporal understanding on-device:** How do you do streaming video understanding with a small VLM? Cover frame sampling strategies, temporal token compression, memory banks, KV-cache reuse across frames and event-triggered inference to save power. What are the trade-offs against a lightweight temporal model (TSM, MoViNet)?

23. **Simulation and sim-to-real:** How would you use Gazebo or MuJoCo to generate RGB, depth, LiDAR and IMU data for training or evaluating a perception or VLM-driven agent? Cover sensor noise models, domain randomization, the sim-to-real gap for vision, closed-loop evaluation of VLM-based policies (VLA models, e.g. RT-2 / OpenVLA / π0-style) and how to make results reproducible.

## E. Evaluation, Production and Systems Thinking
24. **Evaluation beyond benchmarks:** MMMU, MMBench, TextVQA, DocVQA, ChartQA, OCRBench and POPE don't reflect device reality. How would you build an evaluation suite for a product feature (e.g. on-device visual search or scene understanding on a phone, TV or robot)? Cover task-specific metrics, LLM-as-judge for VLMs and its biases, slice-based analysis (low light, motion blur, rotated or occluded images, multilingual OCR, e.g. Hindi/Indic scripts), A/B testing, regression detection, and quantifying the accuracy loss from each optimization step (quantization, pruning, token reduction).

25. **End-to-end production case study:** *"Design an on-device VLM assistant for a Samsung home-robot (Ballie-type) or Galaxy phone camera that answers questions about the surroundings in under 1 second, offline, with privacy guarantees."* Walk through requirements → model selection → data and training → compression → runtime and hardware mapping → latency and memory budget → safety and privacy (on-device processing, face blur, guardrails against prompt injection via images or text in the scene) → monitoring, OTA model updates, rollback and A/B rollout. Where would you expect the biggest surprises when moving from lab to production?

---

# Q1: Small-VLM architecture trade-offs (encoder, projector, LLM)

**Question:** Compare the design of a small VLM such as SmolVLM, PaliGemma, MobileVLM, Qwen2-VL-2B, InternVL-1B or Florence-2. How do the vision encoder (SigLIP vs CLIP vs ViT), the projector (linear, MLP, Q-Former, Perceiver resampler, pixel-shuffle) and the LLM backbone each affect accuracy, latency and memory? If you had a 2B-parameter budget on a phone, how would you split it across the three components, and why?

## 1. Basic: the three-block mental model
Almost every VLM is three parts:

```
Image → [Vision Encoder] → patch features → [Projector] → visual tokens → ┐
                                                                          ├→ [LLM decoder] → text
Text  → [Tokenizer + Embedding] → text tokens ─────────────────────────────┘
```

| Block | Job | Typical size |
|---|---|---|
| Vision encoder | Turns pixels into semantic patch embeddings | 86M to 400M (SigLIP-B / SO400M) |
| Projector | Maps vision-feature space into the LLM embedding space, and can reduce token count | 1M to 50M |
| LLM | Reasoning and language generation | 0.5B to 3B in small VLMs |

**Key idea:** the visual tokens are fed to the LLM as if they were words. The LLM never sees pixels.

## 2. Component choices

### Vision encoder
- **CLIP ViT:** softmax contrastive loss. It is strong on semantics but weaker on fine detail and OCR.
- **SigLIP / SigLIP2:** sigmoid loss, which handles small batches better. Most modern small VLMs use it (PaliGemma, SmolVLM, many Qwen-class models).
- **Native-resolution ViT (Qwen2-VL):** keeps aspect ratio and makes a variable token count. It is better for documents but harder to deploy because the shapes are dynamic.
- **Conv or hybrid encoders (FastViT, ConvNeXt, MobileCLIP):** NPU-friendlier and lower latency. Detail and OCR are often weaker.

### Projector

| Type | Pros | Cons |
|---|---|---|
| Linear / 2-layer MLP (LLaVA) | Simple, preserves all information, trains easily | No token reduction, so many visual tokens |
| Pixel-shuffle / space-to-depth (SmolVLM, InternVL) | Cuts the token count 4x to 9x by folding spatial neighbors into channels | Some fine detail is lost |
| Q-Former / Perceiver resampler (BLIP-2, Flamingo) | Fixed small token count (e.g. 32 to 64) | Harder to train, loses spatial detail, weak at OCR and grounding |
| Convolutional or pooling-based (MobileVLM LDP) | Cheap, keeps local structure | Fixed pooling can hurt |

**Practical trend:** MLP plus pixel-shuffle won over Q-Former. It is simpler and more accurate at the same token budget, especially for text-heavy images.

### LLM backbone
Choices are Qwen2.5 (0.5B/1.5B/3B), Gemma, SmolLM2, Phi, Llama-3.2-1B/3B. What matters for edge:
- GQA, which shrinks the KV-cache.
- Tied embeddings and vocabulary size. A 150k vocabulary costs a large share of a 0.5B model's parameters.
- Whether the runtime supports its ops.
- Multilingual support, including Indic scripts for Samsung India.

## 3. Where the cost actually goes
This is the key insight for latency and memory.

**Compute:** ViT cost grows with patches², since attention is quadratic in patches. A 400M encoder at 384px with 729 patches costs a lot of FLOPs once per image. The LLM prefill cost grows with (text + visual tokens) × params.

**Memory:**
- Weights = parameters × bytes. For example, 2B at INT4 is about 1 GB.
- KV-cache = 2 × layers × kv_heads × head_dim × seq_len × bytes. This grows with the number of visual tokens.

**Rule of thumb:**
- The vision encoder dominates time-to-first-token for a single image.
- Visual tokens dominate LLM prefill and KV-cache.
- Decode speed depends on LLM size and memory bandwidth, and is roughly independent of the image.

So the projector is the lever that connects the two. Fewer tokens cut LLM prefill and KV-cache.

## 4. Parameter split for a ~2B budget
There is no single correct answer, so state your assumptions. A reasonable starting point:

| Component | Params | Rationale |
|---|---|---|
| Vision encoder | ~90 to 400M | SigLIP-B (86M) at 384 or 512px if latency matters, SO400M if accuracy matters |
| Projector | ~10 to 30M | MLP plus pixel-shuffle 3x3 to 4x |
| LLM | ~1.5B | Reasoning and language quality. The LLM size drives the quality of answers more than the encoder |

How I would decide:
1. **Define the tasks.** OCR and document tasks need higher resolution and a strong encoder. General scene Q&A tolerates a small encoder.
2. **Set the latency budget.** For example, TTFT under 1 s means the encoder plus prefill must fit in about 600 ms on the target NPU.
3. **Run ablations.** Hold the total budget fixed and sweep the encoder size and resolution against the LLM size. Plot accuracy against latency and memory, and pick the Pareto point.
4. **Check the empirical results.** From the SmolVLM and Idefics3 work, a small encoder (93M) paired with small LLMs (135M to 2B) works well, because a large encoder with a tiny LLM wastes capacity. More resolution beats a bigger encoder for text-heavy tasks.

## 5. Real-world traps
- **Quantization asymmetry:** the vision encoder and projector are more sensitive to INT4/INT8 than the LLM, so keep them at higher precision (FP16 or INT8) and quantize the LLM aggressively.
- **Dynamic shapes:** variable-resolution encoders (Qwen2-VL) are hard to export to fixed-shape NPU graphs. Use tile-based fixed shapes.
- **Operator support:** LayerNorm, GELU and attention in the ViT may not map to the NPU and can fall back to the CPU.
- **Text-in-image:** low resolution plus a lossy projector destroys OCR accuracy.
- **Weak LLM:** a strong encoder cannot fix a weak LLM, because errors then come from reasoning, not perception.

## 6. Strong interview answer
> "I'd start from the product tasks and latency budget, then treat it as a Pareto search over three knobs: encoder size and resolution, projector token budget, and LLM size. For a ~2B on-device budget I'd likely choose a SigLIP-base-class encoder, a pixel-shuffle MLP projector to cap visual tokens at roughly 64 to 256 per tile, and a ~1.5B GQA LLM. I'd keep the encoder and projector at INT8/FP16 and quantize the LLM to INT4. Then I'd validate with ablations on on-device latency, memory, and task accuracy, not just parameter count."

## 7. Follow-up questions to prepare for
1. Why did the field move from Q-Former to MLP projectors?
2. How does pixel-shuffle reduce tokens without a learned module?
3. Why is the vision encoder usually trained contrastively and not from scratch with the LLM?
4. If OCR accuracy drops after you reduce tokens, what do you change first?
5. How does a 150k vocabulary affect a 0.5B LLM's size and the choice of backbone?

---

# Q1 deep dive: the five follow-ups, with numbers

The numbers below are approximate and meant for reasoning in an interview. Note: for small GQA models the KV-cache is usually not the main cost (see section 0), which corrects the emphasis in Q1 section 3.

## 0. Worked numbers: what actually costs time and memory

**KV-cache per token** (Qwen2.5-1.5B: 28 layers, 2 KV heads, head_dim 128, FP16):

```
2 (K,V) × 28 layers × 2 kv_heads × 128 × 2 bytes ≈ 28 KB/token
→ 1,000 visual tokens ≈ 28 MB
```

GQA makes the KV-cache small here. Without GQA (12 KV heads) it would be 6x larger. The KV-cache only becomes a problem with long video or many images.

**Compute** (FLOPs ≈ 2 × params × tokens, plus attention):

| Stage | Setup | Approx. FLOPs |
|---|---|---|
| SigLIP-B/16 encoder | 512px → 1024 patches, 86M params | ~0.2 TFLOP per image/tile |
| SigLIP-SO400M/14 encoder | 384px → 729 patches, 400M params | ~0.65 TFLOP per image/tile |
| LLM prefill, 1.5B (1.3B non-embedding) | 81 visual + ~50 text tokens | ~0.34 TFLOP |
| LLM prefill, same model | 1,000 tokens | ~2.6 TFLOP |

**Takeaways:**
- With a big encoder and few visual tokens, the encoder dominates TTFT.
- With many tiles and a high token count, LLM prefill dominates.
- Tiling multiplies the encoder cost: N tiles means N encoder passes.
- Decode is memory-bandwidth-bound and independent of the image.

## 1. Why did the field move from Q-Former to MLP projectors?

**What Q-Former does:** it has about 32 learned query tokens that cross-attend to the frozen ViT features. The output is a fixed-size summary of 32 tokens regardless of image size (BLIP-2).

**Why it lost:**
1. **Information bottleneck.** 32 queries must summarize 500+ patches. Fine details such as text, small objects and positions are the first to go. This is why it is weak on OCR, docs, charts and grounding.
2. **Spatial structure is destroyed.** Queries attend globally, so there is no guarantee that "top-left" survives. An MLP projector keeps a 1:1 patch-to-token mapping with the 2D layout intact.
3. **Harder to train.** It adds its own pretraining stage and needs lots of data to converge. An MLP learns quickly from a modest alignment dataset, as LLaVA showed.
4. **Evidence from ablations.** Apple's MM1 study and the Idefics2/3 and Cambrian-1 work found that image resolution and the number of visual tokens matter far more than the connector type. A simple connector with enough tokens matches or beats complex ones.
5. **Data moved the bottleneck.** With high-quality instruction data, the simple design wins.

**Where Q-Former and Perceiver still make sense:**
- Many images or long video, where you need a hard token cap per frame.
- Strict token budgets where a lossy summary is acceptable (captioning, retrieval).
- Cross-attention VLMs (Flamingo) that need a fixed-size visual memory.

## 2. How does pixel-shuffle reduce tokens without a learned module?

**Pixel-shuffle (space-to-depth)** is a reshape. Group each r×r block of neighboring patch tokens into one token by concatenating their channels:

```
(H, W, C)  →  (H/r, W/r, C·r²)       tokens reduced by r²
```

Example with SigLIP-SO400M at 384px with patch 14 (27×27 = 729 patches, C=1152), r=3:
- Tokens: 729 → 81.
- Channels: 1152 → 10,368.
- The MLP then maps 10,368 → LLM hidden size (e.g. 2048), which is about 21M parameters.

**Why this is better than average pooling:**
- Pooling averages neighbors, which is lossy and blurs fine detail.
- Pixel-shuffle is lossless as a reshape. All the information is still there, now in the channels.
- The learned MLP then decides how to compress it, so the learning is moved into a cheap place.

**Costs and limits:**
- The first MLP layer gets large (C·r² inputs). It is cheap in FLOPs but adds parameters.
- Higher r means the MLP must decompress more in one step. Quality drops for text-dense images when r is too aggressive (r=4 and above is risky for OCR).
- Idefics3 and SmolVLM use r=3 to 4 with tiling. For example, a 512px tile with patch 16 gives 1024 patches, and r=4 gives 64 tokens per tile.

**Edge angle:** reshapes are free on an NPU, and the matmul is regular, so this is deployment-friendly. There is no cross-attention and no dynamic shape.

## 3. Why is the vision encoder usually pretrained contrastively and not trained from scratch with the LLM?

1. **Data scale and cost.** Contrastive pretraining uses billions of noisy web image-text pairs, which are cheap. VLM instruction data is millions of expensive, curated samples. You reuse the encoder's semantic knowledge instead of relearning it.
2. **Already language-aligned.** The features were trained to match text, so the projector only has to translate, not discover semantics.
3. **Stability and compute.** Training an LLM and a ViT jointly from scratch is expensive and unstable (early-fusion models like Chameleon and Fuyu need huge compute).
4. **Efficient staged training.** Freeze the encoder first, train the projector, then unfreeze with a low learning rate.

**Limits of the contrastive objective:**
- The loss only needs a global embedding to match a caption. Local detail, counting, spatial relations and OCR are not rewarded, so they are weak.
- Training at low resolution (224 to 384px) hurts text-heavy tasks.

**How the field addresses this:**
- SigLIP2 adds a captioning decoder, self-distillation and masked prediction to improve dense and localization features.
- AIMv2 pretrains autoregressively on image and text tokens.
- Native-resolution and aspect-ratio-preserving variants (NaFlex, Qwen2-VL's encoder) help with documents.
- Unfreezing the encoder during instruction tuning, with a low LR, recovers task-specific detail.

## 4. OCR accuracy dropped after reducing visual tokens. What do you change first?

**Principle:** first find where the information is lost, then fix that stage. Don't guess.

**Step 1: Locate the loss.**
- Check the effective text size. After resizing to the encoder input, text should be at least ~8 to 10 px tall. If it is smaller, the encoder can't see it, and no projector fix helps.
- Run a linear probe or a quick OCR head on the encoder features. If that works, the loss is in the projector or the LLM. If it fails, the loss is in the encoder input.
- Rule out quantization by running the same eval in FP16.

**Step 2: Fix in order of cost (cheapest first):**
1. **Reduce the pixel-shuffle ratio** (r=4 → 3 → 2). This restores tokens at only a modest cost.
2. **Make tiling adaptive.** Use a text-density or complexity detector, so you spend tokens only on text-heavy images and keep a low-token path for natural scenes. This keeps average latency low.
3. **Raise the resolution** with tiles plus a global thumbnail.
4. **Fix the data.** Add OCR-heavy data: documents, screenshots, UI, synthetic rendered text, multilingual and Indic scripts. Unfreeze the projector and encoder (low LR) for this stage.
5. **Change the architecture last.** Use a stronger or higher-res encoder, or a text-aware branch (a lightweight OCR module whose output is passed to the LLM as text).

**Pragmatic production option:** run a dedicated on-device OCR engine and give its text to the VLM. It is often cheaper and more accurate than forcing a small VLM to read pixels.

## 5. How does a 150k vocabulary affect a 0.5B LLM's size and backbone choice?

**Embedding parameters = vocab × hidden size.**

| Model | Vocab | Hidden | Embedding params | Share of total |
|---|---|---|---|---|
| Qwen2.5-0.5B | ~152k | 896 | ~136M | ~27% of ~0.49B |
| Qwen2.5-1.5B | ~152k | 1536 | ~233M | ~15% of ~1.54B |
| Llama-3.2-1B | ~128k | 2048 | ~263M | ~21% of ~1.24B |
| SmolLM2-135M | ~49k | 576 | ~28M | ~21% of 135M |

These use tied input and output embeddings, so they are counted once.

**Consequences on-device:**
1. **"Parameter count" is misleading.** A "0.5B" model has only ~0.36B non-embedding parameters in its transformer. Compare non-embedding parameters for reasoning capacity, and total for memory.
2. **The LM head is a real decode cost.** Every decoded token does a hidden × vocab matmul. For Qwen2.5-0.5B this is ~136M MACs out of ~0.5B, so roughly a quarter of per-token compute and weight traffic. Decode is bandwidth-bound, so this directly cuts tokens per second.
3. **Embedding lookup vs LM head differ in cost.** The input lookup only touches a few rows, so it can be memory-mapped or kept in low precision cheaply. The LM head reads the whole matrix every step.
4. **Quantization sensitivity.** The embedding and LM head are often kept at higher precision (INT8/FP16) because INT4 hurts quality, which reduces the savings.
5. **Benefit of a large vocab:** better tokenization efficiency, meaning fewer tokens for the same text, especially for multilingual and code. Fewer tokens means fewer decode steps. For Indic languages a small English-centric vocab can inflate token counts 3 to 5x, which hurts latency more than the LM head does.

**Mitigations for edge:**
- **Vocabulary pruning:** keep only the tokens you need (e.g. English plus Hindi), shrinking V to 32k to 64k. This requires remapping the embeddings and a short recovery fine-tune.
- **Restrict the output to a smaller token subset** at decode time for constrained tasks.
- **Pick the backbone by deployment constraints,** not only benchmarks: non-embedding size, vocab, GQA, runtime op support and language coverage.

## 6. Interview-ready one-liners
- "Pixel-shuffle is a lossless reshape, so the learning is moved to the MLP. That's why it beats pooling and Q-Former at the same token count."
- "With GQA, KV-cache for small LLMs is only ~28 KB/token, so prefill compute and the encoder, not KV memory, dominate TTFT."
- "For OCR regressions, I first locate where the information is lost (input resolution, encoder, projector or LLM) before changing the architecture."
- "I compare small LLMs by non-embedding parameters and look at the LM-head cost, because decode is bandwidth-bound."

---

# Q2: Visual token explosion

**Question:** A 1080p image can produce thousands of visual tokens. How do approaches like AnyRes/tiling, dynamic resolution (Qwen2-VL's naive dynamic resolution, M-RoPE), pixel-shuffle and token merging or pruning (FastV, ToMe, SparseVLM) reduce this? Walk through how the visual token count affects prefill latency, KV-cache size and time-to-first-token. How would you pick the token budget for a given NPU?

All FLOP and latency numbers are illustrative. The NPU throughput is an assumption, not a measurement.

## 1. Why tokens explode

```
tokens = (H / patch) × (W / patch)
1920×1080, patch 14 → 137 × 77 ≈ 10.5k tokens
```

Three costs scale with this number N:
1. **Vision encoder.** Linear in N for the MLP and projections, and quadratic in N for self-attention, which hurts at native resolution.
2. **LLM prefill.** Roughly 2 × params × N FLOPs, plus a quadratic attention term.
3. **KV-cache.** N × bytes per token, and it also consumes the context window. On-device contexts are often only 2k to 4k.

The goal is to give the LLM as few tokens as possible while keeping the information the task needs.

## 2. The families of techniques

### A. Tiling / AnyRes (LLaVA-NeXT, InternVL dynamic tiling, Idefics3 splitting)
- Pick a tile grid that matches the image aspect ratio (e.g. 2×3). Resize each tile to the encoder's native size (336/384/448/512), then add one global thumbnail tile.
- `tokens = (tiles + 1) × tokens_per_tile`.
- **Pros:**
  - The encoder sees fixed shapes, so it exports well to NPUs and can be batched.
  - It reuses encoders pretrained at a fixed resolution.
  - Text-heavy images get real detail.
- **Cons:**
  - Encoder cost is multiplied by the tile count.
  - No cross-tile attention in the encoder, so objects on tile boundaries are split.
  - The LLM needs tile position cues (row/column separator tokens, which add overhead).
  - Token count grows linearly with tiles, so a naive grid can eat the context window.

#### Tiling explained step by step

**The problem it solves.** A vision encoder such as SigLIP-384 was trained on exactly **384×384** inputs. Its position embeddings are learned for a 27×27 grid (384/14). If you feed it a 1920×1080 photo, you have two bad options:
- **Squash the whole image to 384×384.** The image is downscaled about 5x, the aspect ratio is distorted, and small text and details become unreadable.
- **Feed it at full size.** The encoder was not trained for that, and the cost is huge (quadratic attention).

Tiling is the workaround: cut the big image into several pieces that are each exactly the encoder's native size, then run the encoder on each piece.

**Concrete example.** A **1200×800** photo (aspect ratio 3:2) and an encoder with native size **384**.

1. **Choose a grid that matches the aspect ratio.** Candidates are grids such as 1×1, 2×1, 1×2, 2×2, 3×2, 2×3, up to a maximum number of tiles. Pick the grid whose aspect ratio (columns ÷ rows) is closest to the image's. Image: 1200/800 = 1.5. Grid 3 columns × 2 rows = 1.5, a perfect match.
2. **Resize the image to the grid's canvas.** Canvas = (3 × 384) × (2 × 384) = **1152 × 768**. You scale the image to this size, keeping the aspect ratio. If it doesn't match exactly, models either pad the borders (LLaVA-NeXT) or stretch slightly (InternVL).
3. **Cut into tiles.** Each tile is exactly 384×384, the "native size". This is what "resize each tile to the encoder's native size" means. In practice you resize the whole canvas so the tiles come out at native size.

```
Canvas 1152×768                      Tiles (each 384×384)
┌─────────┬─────────┬─────────┐      T1  T2  T3
│   T1    │   T2    │   T3    │      T4  T5  T6
├─────────┼─────────┼─────────┤
│   T4    │   T5    │   T6    │
└─────────┴─────────┴─────────┘
```

4. **Add one global thumbnail.** Take the whole original image, squash it down to 384×384, and treat it as a 7th tile.
5. **Encode each tile independently.** The encoder runs 7 times, each on a 384×384 input.
6. **Project and concatenate.** All 7 sets of tokens (after the projector) are concatenated into one sequence for the LLM.

**Why the thumbnail is needed.**
- Each tile is encoded independently, with no attention between tiles. A tile showing "part of a face" or "half of a sign" doesn't know the full scene.
- The thumbnail gives the LLM the global layout and context at low resolution. The tiles give the local detail at high resolution.
- The LLM combines both: it knows from the thumbnail that this is a street scene, and it reads the sign text from tile T3.
- Some models put the thumbnail first (LLaVA-NeXT), others last (InternVL). The order only needs to be consistent with training.

**Effective resolution gained.**

| Approach | Pixels the encoder sees | Result |
|---|---|---|
| Squash to a single tile | 384×384 = 147k | Small text and details lost |
| 3×2 tiles + thumbnail | 6 × 147k + 147k ≈ 1.03M | About 6x more detail; text is readable |

**Token count for this example** (SigLIP-384, patch 14, 27×27 = 729 patches per tile):
- Without reduction: 7 × 729 = **5,103 tokens**.
- With pixel-shuffle r=3: 7 × 81 = **567 tokens**.

This is why tiling is almost always combined with pixel-shuffle or a similar reduction.

**How the LLM knows which tile is where.** Tiles are encoded independently, so the LLM gets no spatial layout unless you add cues:
- **Separator or row tokens**, e.g. Idefics3 inserts text markers like `<row_1_col_2>` before each tile's tokens.
- **Newline tokens** at the end of each row of the feature map (LLaVA-NeXT's `image_newline`).

These cost a few extra tokens per tile but help the model reason about positions ("the object in the top-right").

**Costs and failure modes.**
- **Encoder cost is multiplied.** 7 tiles means 7 encoder passes, so encoder latency and energy go up about 7x.
- **Boundary problem.** An object that straddles a tile edge is split across two tiles. Neither tile sees it whole. The thumbnail partly compensates.
- **Padding waste.** If the aspect ratio doesn't match any grid, padded areas still cost tokens.
- **Token growth.** More tiles consume the context window quickly. Models cap the maximum number of tiles (e.g. 6 to 12).
- **Strength on-device:** every tile has the same fixed shape, so the encoder graph is static and NPU-friendly, and tiles can be batched.

**One-sentence summary:** tiling splits a large image into a grid of encoder-native-size crops chosen to match the aspect ratio, adds a downscaled whole-image thumbnail for global context, encodes each crop independently, and concatenates all their tokens for the LLM. It gives high effective resolution with a fixed-shape encoder, at the cost of an encoder pass per tile.

### B. Native dynamic resolution (Qwen2-VL, Qwen2.5-VL)
- The ViT uses 2D RoPE and is trained on variable sizes. Tokens come from the actual image size, clamped by `min_pixels`/`max_pixels`.
- A 2×2 patch merge in the projector gives a 4x reduction. With patch 14, each token covers 28×28 pixels, so 1080p is about 1920×1080/784 ≈ **2.6k tokens**.
- **Pros:**
  - Keeps the aspect ratio with no tiling artifacts.
  - Full-image attention in the encoder.
  - A continuous quality/cost knob via `max_pixels`.
- **Cons:**
  - Dynamic shapes are hard for static-graph NPUs, so you export a few resolution buckets.
  - Global attention in the encoder is quadratic. Qwen2.5-VL uses windowed attention in most ViT layers to control this.

**M-RoPE (multimodal rotary embedding):**
- The rotary dimensions are split into three groups for (temporal, height, width) position IDs. Text tokens use the same ID in all three, so M-RoPE reduces to ordinary 1D RoPE for text.
- An image's tokens get (t = constant, h, w) from the patch grid. Text after the image continues from max(h, w) + 1, not from h×w.
- **Benefit:** the image occupies a small positional range, which helps long context and extrapolation to larger images. For video, t increments per frame.
- This is covered in more depth in Q3.

### C. Pixel-shuffle / space-to-depth
- A lossless reshape that cuts tokens by r². The MLP then learns to unpack the information (see the Q1 deep dive).
- It is usually combined with tiling (Idefics3, SmolVLM, InternVL).
- Static shape and a regular matmul, so it is very NPU-friendly.

### D. Token merging and pruning (mostly training-free)

| Method | Where | Idea | Notes |
|---|---|---|---|
| **ToMe** | Inside the ViT, every layer | Bipartite soft matching merges the most similar tokens, tracked with a size-weighted attention | Speeds up the encoder itself. Merging is irreversible, and dense tasks suffer |
| **FastV** | Inside the LLM, after an early layer K | Prunes visual tokens that get the least attention (from the last token) | Plug-and-play. Needs attention scores (conflicts with FlashAttention-style fused kernels). Query-dependent |
| **SparseVLM** | Inside the LLM, progressive | Uses text tokens to rate visual-token relevance, prunes layer by layer, and recycles pruned tokens into compact ones | Text-guided and query-aware. More complex |
| **VisionZip / PruMerge** | Before the LLM, on encoder outputs | Keeps dominant tokens (by CLS or attention score) and merges the rest | Query-agnostic, so it cuts all of prefill |

Three distinctions matter in practice:
1. **Before the LLM or inside it?** Pruning before the LLM saves the whole prefill. Pruning at layer K saves only the layers after K. The first K layers still process all N tokens.
2. **Query-agnostic or query-aware?** Query-agnostic pruning lets you cache the image's KV for multi-turn questions. Query-aware pruning gets better accuracy per token but must be recomputed for every question.
3. **Hardware fit:** methods that need dynamic token counts, or attention scores in the middle of the network, don't fit static NPU graphs. You would have to pad or bucket the surviving token count.

### E. Learned compression and adaptive budgets
- Q-Former/Perceiver resamplers (fixed count, lossy, covered in Q1).
- Matryoshka-style multimodal models (e.g. M3): train with nested token granularities, so one model can run at 9, 36, 144 or 576 tokens at inference. This is a good fit for a runtime budget.
- Dynamic or learned selection that depends on image complexity.

## 3. Effect on latency, KV-cache and TTFT

```
TTFT ≈ decode_image + preprocess + T_encoder(res, tiles) + T_projector
       + T_prefill(N_visual + N_text) + first-token step
```

- **Prefill is compute-bound** and roughly linear in N. Doubling tokens roughly doubles prefill time, with extra attention cost at long lengths.
- **Decode is memory-bound** and mostly independent of N. N only grows the KV read each step.
- **KV-cache = N × bytes/token.** This varies enormously with architecture:

| Model | Bytes per token | 2,880 tokens (5 tiles × 576) |
|---|---|---|
| LLaVA-1.5-7B (MHA, 32 layers × 32 heads × 128, FP16) | ~512 KB | ~1.5 GB |
| Qwen2.5-1.5B (GQA, 2 KV heads, 28 layers, FP16) | ~28 KB | ~80 MB |

  For small GQA models the KV-cache is not the binding constraint. Prefill time and context length are.
- **Context window:** a 4k on-device context with 2.9k image tokens leaves little room for the prompt, history and answer.

### Worked example for 1080p (illustrative)
Assume a 1.5B LLM with 1.3B non-embedding parameters, and an effective 10 TOPS for the NPU (real utilization is often 20 to 40% of the peak spec).

| Strategy | Visual tokens | LLM prefill FLOPs | Prefill at 10 TOPS | Notes |
|---|---|---|---|---|
| Naive, patch 14, no reduction | ~10.5k | ~27 TFLOP | ~2.7 s | Fails a 1 s budget. The encoder with N² attention is also heavy |
| Qwen2-VL style (2×2 merge) | ~2.6k | ~6.8 TFLOP | ~0.7 s | Plus the encoder cost at that resolution |
| Tiling, 384px tiles, r=3 (about 15 tiles + 1 thumbnail) | ~1.3k | ~3.4 TFLOP | ~0.35 s | The encoder runs 16 times: ~16 × 0.65 = ~10 TFLOP for SO400M, ~16 × 0.2 = ~3 TFLOP for SigLIP-B |
| Downscale to a single 384 tile, r=3 | 81 | ~0.2 TFLOP | ~0.02 s | Fast, but small text and fine detail are lost |

**Takeaways:**
- Tiling lowers LLM tokens, but it moves the cost to the encoder.
- Always add encoder and LLM costs. Don't optimize one stage in isolation.
- A single downscaled tile is the fastest option, but it can't read small text.

## 4. How to pick the token budget for a given NPU

1. **Fix the SLO and the worst case.** For example, TTFT ≤ 1 s. Budget for the thermally throttled state (e.g. 60 to 70% of peak), not the cold-start benchmark.
2. **Subtract the fixed costs:** image decode and preprocessing, text tokens (system prompt plus question), projector, and the first decode step. Leave margin.
3. **Measure on the real device, not from the spec sheet:**
   - Encoder latency at several resolutions or tile counts.
   - LLM prefill throughput (tokens/s) at several lengths, including the chunk size you will use.
   - Decode tokens/s.
4. **Solve for N:** `T_encoder + N / prefill_rate + T_fixed ≤ SLO`.
5. **Get the accuracy-vs-tokens curve** for each task you ship (scene Q&A, OCR, counting, UI). Choose the **knee** of the curve, where extra tokens stop helping.
6. **Offer tiers, not one number.** Compile a few static graphs (e.g. 64 / 144 / 256 / 576 tokens), and select one at runtime based on:
   - task type and text density,
   - thermal and battery state,
   - latency target (interactive vs background).
7. **Respect NPU constraints:**
   - Static shapes mean padded buckets, so token counts should be multiples of the prefill chunk size (e.g. 128). The padding waste counts toward the budget.
   - Check memory: weights + activations + KV at the chosen N, within the 2 to 3 GB budget.
8. **Validate end to end** under sustained load with the camera and ISP running, because the NPU is usually shared.

## 5. Real-world traps
- **Hidden multipliers:** tiling multiplies encoder cost. Each tile is a full encoder pass.
- **Separator and thumbnail tokens** are overhead that adds up with many tiles.
- **Dynamic token counts** (FastV, native resolution) break static-graph compilation unless you bucket them.
- **Attention-score-based pruning** conflicts with fused attention kernels, which don't expose scores.
- **Multi-turn caching:** query-agnostic pruning lets you reuse the image KV across questions. Query-aware pruning doesn't.
- **Accuracy is task-dependent:** natural-scene questions tolerate 4 to 8x token reduction, while OCR, charts and small-object tasks do not. Evaluate per task.
- **Context budget:** image tokens compete with chat history and the answer.

## 6. Strong interview answer
> "Visual tokens scale as (H/p)×(W/p), so a 1080p image is ~10k tokens at patch 14, which is far too many for an on-device 1 s TTFT. I'd attack it at three stages. First, structurally: pixel-shuffle or a 2×2 merge for 4 to 9x reduction, and either tiling for fixed-shape NPU graphs or native resolution when the runtime supports it. Second, adaptively: pick the resolution and tile count per task, since text-heavy images need tokens that scene Q&A doesn't. Third, optionally prune before the LLM with a query-agnostic method so I can cache the image KV. Then I'd measure encoder latency and prefill tokens per second on the actual NPU under thermal throttling, solve for the token budget that meets the SLO, pick the knee of the accuracy-vs-tokens curve per task, and ship a few static-shape tiers selected at runtime. I'd remember that tiling moves cost from the LLM to the encoder, so I'd always budget the two together."

## 7. Follow-up questions to prepare for
1. Why does token pruning inside the LLM (FastV) save less than pruning before it, and why does it conflict with FlashAttention?
2. How does M-RoPE keep the positional range small, and why does that help long context?
3. How would you export a native-resolution ViT to a fixed-shape NPU graph? (Buckets, padding, masks.)
4. Tiling vs native resolution: when would you choose each for a Samsung phone camera feature?
5. How do you reuse the image's KV-cache across multi-turn questions, and what breaks it?
6. How does video change the token budget, and how would you adapt it per frame?

---

# Q3: Positional encoding for image patches (encoder vs LLM)

**Question:** Explain how positional information is handled for image patches in the encoder (learned absolute vs 2D RoPE) and in the LLM (1D RoPE vs M-RoPE vs 2D positions). What goes wrong when you run at a resolution different from training? How would you interpolate position embeddings, and what is the accuracy impact?

Where accuracy effects are given, they describe the direction and the mechanism. Exact magnitudes depend on the model and task, so measure them on your own eval.

## 0. Why positions are needed
Self-attention is permutation-equivariant. Without position information, an image is a **bag of patches** and "top-left" is indistinguishable from "bottom-right". Position information is what lets a model read text, count objects, understand "left of" or localize a box.

There are two separate places where positions get injected:
1. **Inside the vision encoder**, to order patches in 2D.
2. **Inside the LLM**, to order visual tokens relative to text tokens (and to each other).

## 1. Positions in the vision encoder

| Scheme | How it works | Used by | Resolution flexibility |
|---|---|---|---|
| **Learned absolute** | A table of shape (N_train × D) is added to the patch embeddings at the input. For SigLIP-so400m-384 that is 27×27 = 729 rows | ViT, CLIP, SigLIP | Poor. Tied to the training grid |
| **Fixed 2D sin-cos** | Same idea, but the table is computed, not learned | MAE | Better, but still absolute and still an input shift |
| **2D RoPE** | The head dimension is split in two halves. One half is rotated by the patch's **row** index and the other by its **column** index. Applied to q and k in every attention layer | Qwen2-VL ViT, Pixtral, EVA-02, DINOv3 | Good. Relative, no table, works with any grid shape |
| **Relative position bias** | A learned bias added to attention logits by relative offset | Swin, T5-style | Tied to window sizes |
| **Conditional / conv position encoding** | A depthwise conv over the feature map injects local position | CPVT, some mobile ViTs | Very good, and NPU-friendly |

**Learned absolute:** one add at the input. It is trivial on an NPU, but the model only knows the exact grid it was trained on.

**2D RoPE:** every attention layer sees the relative row and column offset between patches. There is no fixed table, so variable sizes and aspect ratios work. Pixtral also inserts line-break tokens between rows. SigLIP2's NaFlex variant keeps learned embeddings but resizes them to the image's patch grid and pads with an attention mask.

## 2. Positions in the LLM

### Recap: 1D RoPE
For each pair of channels (x_2i, x_2i+1) of q and k at position m, rotate by the angle `m · θ_i`, where `θ_i = base^(-2i/d)`. Then `q_m · k_n` depends only on the offset `m − n`. Text-only LLMs use this.

### Naive approach: flatten the image into 1D (LLaVA-style, InternVL)
Visual tokens are laid out in raster order and given consecutive 1D positions, like words.

Problems:
1. **Distances are distorted.** A token's vertical neighbor is `W` positions away, and `W` changes with image size. Horizontal neighbors are 1 apart.
2. **2D structure is implicit.** The model must infer rows from content, or from explicit newline or separator tokens.
3. **Large positional range.** A 27×27 image consumes 729 positions, and larger images consume far more.
4. **Distance to the question.** RoPE attention generally decays with distance, so visual tokens at the start of a long image are "far" from the question text.

### M-RoPE (Qwen2-VL)
Split the rotary dimensions into three sections, for **(temporal, height, width)**. For example, Qwen2-VL uses sections of [16, 24, 24] (in rotary pairs) for head_dim 128. Each token has three position IDs:

| Token type | (t, h, w) |
|---|---|
| Text | (p, p, p), all equal. This is identical to ordinary 1D RoPE |
| Image patch | (p0, p0 + row, p0 + col). t is constant, h and w come from the grid |
| Video frame | t increments per frame; h and w from the grid |

After an image, the next text position continues from `p0 + max(H', W')`, **not** from `p0 + H'·W'`.

Why this matters:
- True 2D relative structure inside the LLM. Vertical and horizontal neighbors are close.
- The image uses a small positional range (about max(H', W') instead of H'·W'). This helps long context and helps with images larger than anything seen in training.
- Text still behaves exactly as in the base LLM, so language ability is preserved.
- Qwen2.5-VL goes further and ties the temporal ID to real timestamps for video.

### Other designs
- **Cross-attention VLMs (Llama-3.2-Vision, Flamingo):** visual tokens are not in the LLM's sequence, so the LLM needs no image positions. Position handling lives only in the vision encoder.
- **Prefix-LM (PaliGemma, Gemma 3):** image tokens attend **bidirectionally** among themselves, while text stays causal. This is an attention-mask choice, and it interacts with how positions are used.

## 3. What goes wrong at a different resolution than training

### Encoder with learned absolute positions
- **Shape mismatch.** The table has N_train rows. A different grid either errors out or silently uses the wrong rows.
- **Even with interpolation, there is a distribution shift:**
  - The interpolated embeddings are smoother than the trained ones and lose position-specific detail.
  - Objects appear larger or smaller in patch units, so the learned attention patterns and receptive fields no longer match.
  - The network was never trained with these embedding values.
- **Aspect-ratio distortion** if you squash a non-square image into a square grid.
- **Quadratic attention cost** with more patches.

### Encoder with RoPE
- It extrapolates better, but rotation angles beyond the training range are still out of distribution.
- Mitigations: train with multi-resolution or random-scale augmentation, rescale the position IDs (for example normalize coordinates), or use NTK- or YaRN-style frequency scaling.

### LLM side
- More visual tokens than the trained context can break RoPE extrapolation.
- With 1D flattening, large images burn through positional range. M-RoPE reduces this.
- The projector was trained with a certain token statistic. A different resolution changes object scale per token, which causes a mild train-test mismatch.

### How it shows up in practice
- **Natural-scene captioning often degrades gracefully.**
- **OCR, grounding, counting and spatial questions degrade first.** These need precise positions.

## 4. How to interpolate position embeddings

**Recipe (learned absolute table, e.g. timm's `resample_abs_pos_embed`):**
1. Separate any CLS or register-token embeddings. Don't interpolate those.
2. Reshape the patch table from `(h0·w0, D)` to `(1, D, h0, w0)`.
3. Resize with **bicubic** (or bilinear) interpolation to `(h1, w1)`. Use antialiasing when downsizing.
4. Reshape back to `(h1·w1, D)` and add the CLS embedding back.
5. For non-square images interpolate to `(h1, w1)` directly, so the aspect ratio is preserved.
6. **Fine-tune briefly at the new resolution** with a low learning rate. This usually recovers most of the loss, because the network adapts to the interpolated embeddings.

**Related techniques:**
- **FlexiViT:** to change the patch size, resample the patch-embedding weights (PI-resize) rather than the position table. This keeps the token count and is useful for trading accuracy for compute.
- **SigLIP2 NaFlex:** resize the table per image to its patch grid, pad to a maximum sequence length, and mask the padding.

**Accuracy impact:**
- A small change in resolution (for example 1.2 to 1.5x per side) with interpolation costs little on classification and captioning, but more on OCR, dense and grounding tasks.
- A large change (2x or more) without fine-tuning can cause large drops.
- Fine-tuning at the new resolution typically recovers most of it.
- Always measure a **resolution-vs-accuracy curve per task** before choosing a shipping resolution.

## 5. On-device deployment implications
- **Interpolate offline, not at runtime.** For a fixed-shape NPU graph, precompute the interpolated table for each supported resolution bucket and bake it in as a constant. The position add then costs nothing.
- **RoPE tables:** precompute cos/sin per bucket. Compute them in FP32 and store as FP16 or FP32, because low precision hurts at large angles. Don't quantize the tables.
- **M-RoPE needs 3×N position IDs as an input,** and runtime support varies. Check that your runtime (ExecuTorch, llama.cpp, vendor SDK) implements it. If it doesn't, you pass precomputed cos/sin for each request from the CPU.
- **Decode positions with M-RoPE.** After an image the next text position is not the KV-cache length. It is `cache_length + rope_delta`, where `rope_delta` accounts for the compression of the image's positional range. Qwen2-VL tracks this explicitly. Forgetting it gives wrong positions during decode and degraded output.
- **Prefix caching:** text positions after an image depend on that image's grid size, so the cache after an image can't be reused for a different image. A system-prompt prefix before the image can be.

## 6. Debugging position bugs
- **Symptom:** captioning looks fine but OCR, counting and spatial answers collapse. This suggests the model is treating the image as a bag of patches.
- **Checks:**
  - Shuffle the patches, or zero the position embeddings, and compare. A healthy model should get noticeably worse.
  - Row-major vs column-major flatten mismatch between training and inference.
  - Off-by-one in row or newline tokens.
  - Aspect-ratio distortion in preprocessing.
  - Padding tokens not masked out.
  - Wrong `rope_delta` handling during decode.

## 7. Strong interview answer
> "Attention is permutation-equivariant, so positions have to be injected in two places. In the encoder, classic ViTs, CLIP and SigLIP add a learned absolute table at the input, which ties them to one grid. 2D RoPE splits the head dimension into row and column halves, so it is relative and handles variable sizes. In the LLM, flattening patches into 1D RoPE distorts vertical distances and burns positional range. M-RoPE (Qwen2-VL) splits the rotary dimensions into temporal, height and width, reduces to ordinary RoPE for text, and compresses an image's positional span to about max(H, W). At a different resolution than training, learned absolute tables mismatch. Interpolating them bicubically in 2D, preserving the aspect ratio, helps, but the embeddings are still out of distribution, so I'd fine-tune briefly at the target resolution. OCR and grounding degrade first. On-device, I'd interpolate offline per resolution bucket and bake the table in as a constant, compute RoPE tables in high precision, pass M-RoPE position IDs explicitly, and handle `rope_delta` during decode."

## 8. Follow-up questions to prepare for
1. Why is 2D RoPE relative while a learned absolute table is not, and how does that change generalization to new aspect ratios?
2. In M-RoPE, why do text tokens get the same ID in all three sections, and why does this preserve the base LLM's behavior?
3. Why is the next text position after an image `max(H', W')` and not `H'·W'`? What are the trade-offs?
4. How would you extend a ViT trained at 384 to 768 without full retraining?
5. What breaks in a runtime that doesn't support 3D position IDs, and how would you work around it?
6. Why does bidirectional attention among image tokens (prefix-LM) matter for how positions are used?

---

# Q4: Contrastive vs generative pretraining for the vision encoder

**Question:** Why does SigLIP (sigmoid loss) scale better than CLIP (softmax contrastive loss) at small batch sizes? What does each give you for VLM downstream quality? When would you prefer a vision encoder trained generatively (e.g. a captioning objective, AIMv2) over a contrastive one for a small VLM?

Where the answer describes why something works, mechanisms are marked as likely explanations. The papers mostly report empirical results.

## 0. The idea in one picture
Both families start from image-text pairs, but they train the image encoder with different supervision:

```
Contrastive (CLIP, SigLIP)           Generative (CapPa, LocCa, AIMv2)
image → ViT → 1 global vector ─┐     image → ViT → patch features ─┐
text  → text encoder → vector ─┴→ match?    decoder (cross-attn or causal) → predict the caption tokens
```

- **Contrastive:** "does this image match this text?" Roughly one global signal per pair.
- **Generative:** "predict every token of the caption (or even the image patches)". Many dense signals per pair.

**Terminology note:** both CLIP and SigLIP are contrastive in the broad sense. They train on image-text pairs, pull matching pairs together, push non-matching pairs apart, and learn a shared embedding space. They differ in the loss: **CLIP is softmax contrastive** (InfoNCE-style), while **SigLIP is sigmoid contrastive** (an independent binary loss per pair). **SigLIP2 is a hybrid**: it keeps the sigmoid contrastive loss and adds captioning/localization decoder losses, self-distillation and masked prediction.

## 1. CLIP loss vs SigLIP loss

### CLIP: softmax over the batch
For a batch of B pairs, compute the B×B similarity matrix `S = t · (img @ txt.T)`. Apply softmax cross-entropy per row (image → text) and per column (text → image), with the diagonal as the correct answer.

```python
logits = t * img @ txt.T
loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
```

- Each positive competes against **B−1 negatives**, so the task changes with B.
- The softmax normalizer needs the **whole row**. Multi-device training must all-gather embeddings and hold the full B×B matrix.

### SigLIP: independent sigmoid per pair
Treat each of the B×B (image, text) pairs as its own **binary classification**: +1 on the diagonal, −1 elsewhere.

```python
logits = t * img @ txt.T + b                 # learnable temperature t and bias b
labels = 2 * torch.eye(B) - 1                # +1 diagonal, -1 off-diagonal
loss = -F.logsigmoid(labels * logits).sum() / B
```

- Each pair's loss term does not depend on any other pair through a normalizer.
- The learnable **bias b** is initialized very negative (the paper uses about −10), with temperature initialized to `log 10`. This counters the huge initial imbalance of B−1 negatives per positive, so early training isn't dominated by negatives.

### Why SigLIP does better at small batch sizes
What the paper reports: the sigmoid loss outperforms the softmax loss at **small and medium batch sizes**, the gap shrinks as the batch grows, and both saturate at roughly a 32k batch. SigLIP can also be run with very large batches cheaply.

Likely reasons (mechanisms, not proven facts):
1. **Decoupled from batch size.** In softmax, the difficulty of the task (rank among B items) and the right temperature depend on B. With a small B, each positive has few negatives, so the problem is easy and the signal is weak. In sigmoid, each pair is a calibrated binary decision whose meaning doesn't change with B.
2. **No global normalization.** There is no coupling of every pair in a row through the denominator, which gives better-behaved gradients when the batch is small.
3. **Explicit bias term** handles class imbalance directly.

Engineering benefit:
- **Memory-efficient chunked implementation.** Each device computes the loss for its local block, then texts are rotated around a ring of devices. Memory is O(b_local²) per device, not O(B²). There is no all-gather of the full matrix.

**Practical consequence:** if you continue pretraining or adapt an encoder on a domain's image-text pairs with limited GPUs (small batch), sigmoid loss is the better default.

## 2. What each family gives a VLM

### Contrastive encoders: CLIP (softmax contrastive) and SigLIP (sigmoid contrastive)
Strengths:
- **Semantic, language-aligned features.** The projector and LLM start close to a good solution, so alignment is cheap.
- **Zero-shot classification and retrieval** out of the box, which also supports on-device semantic image search.
- Trained on billions of noisy web pairs, and robust to noisy captions.

Weaknesses:
- The loss only needs a good **global** match, so local detail, OCR, counting, spatial relations and fine-grained attributes are not directly rewarded.
- **Bag-of-words behavior:** weak at compositionality and attribute binding ("horse eating grass" vs "grass eating horse", as in Winoground and ARO).
- **"CLIP-blind pairs"** (the Eyes Wide Shut / MMVP findings): visually different images that CLIP embeds almost identically. VLMs built on such encoders inherit the blind spots.

A practical detail: VLMs usually take **patch tokens from the penultimate layer** (or the final layer before the pooling head), not the pooled embedding. The last layers are specialized for the contrastive objective.

### Generative encoders
**Captioning-based pretraining (CapPa, LocCa):** the encoder feeds a text decoder that predicts the caption.
- **Dense signal:** every caption token supervises the image features, which is far richer than one match/mismatch bit.
- LocCa adds location-aware tasks (referring expressions, grounding, detection), which improves localization.
- Better on fine-grained, compositional and OCR-like tasks.

**AIMv2 (Apple):** a vision encoder with a multimodal decoder that autoregressively predicts both image patches and text tokens. It is simple, scales well, and is reported to give strong features for dense and multimodal tasks, with good performance when plugged into a VLM.

**SigLIP2 is a hybrid:** it combines the sigmoid contrastive loss with a captioning-and-localization decoder objective, self-distillation, masked prediction, multilingual data, and NaFlex variable-resolution variants. This improved dense features and localization while keeping zero-shot ability.

**Self-supervised (DINOv2):** no language, but very strong spatial and dense features. It is often combined with a language-supervised encoder (mixture of encoders in Cambrian-1-style work) to fix CLIP's blind spots.

### Side-by-side

| Aspect | Contrastive | Generative (captioning / AIMv2-style) |
|---|---|---|
| Supervision per pair | About one global signal | Many token-level signals |
| Features | Global, semantic | Denser, more local detail |
| OCR, grounding, counting | Weaker | Stronger |
| Zero-shot classification / retrieval | Yes | Not by default (needs extra work) |
| Compositionality | Weak | Better |
| Sensitivity to caption noise | More tolerant | More sensitive, so recaptioning helps |
| Training cost | Cheaper (no decoder) | Extra decoder compute during pretraining (dropped afterwards) |
| Batch-size dependence | Yes (softmax) or low (sigmoid) | None |
| Alignment to LLM | Easy (already language-aligned) | Also good, and often better on dense tasks |

## 3. When to prefer a generative or dense-feature encoder for a small VLM
1. **The product needs OCR, documents, UI screens, grounding, counting or spatial reasoning.** These are exactly the contrastive weak spots.
2. **The LLM is small.** A 0.5 to 1.5B LLM has little capacity to "reason around" weak visual features. Detail lost in the encoder can't be recovered later.
3. **Token budgets are tight.** With few visual tokens, each token must carry more information. Dense, generatively trained features tend to carry more per token.
4. **You will fine-tune the encoder anyway.** A better initialization helps, especially with limited data.
5. **Compositional and fine-grained attribute tasks** (smart-home objects, appliance states).

When contrastive-only is still the right choice:
- You need **retrieval or zero-shot embeddings** (on-device gallery search, semantic image search). A dual-encoder embedding is needed for that.
- Scene-level understanding is the main task.
- You want the smallest and cheapest encoder.

**Practical default today:** a SigLIP2-class encoder, since it is hybrid. If OCR and grounding dominate, evaluate AIMv2 or add a second dense encoder. One encoder can serve both the embedding use case and the VLM, so you encode once.

## 4. How to choose in practice
You rarely pretrain the encoder yourself. You pick among released ones and measure:
1. **Probe test:** attentive or linear probes on dense tasks relevant to the product (OCR crops, grounding, counting).
2. **Plug-in test:** train a small LLaVA-style alignment (frozen LLM, cheap data) with each candidate encoder, then compare on your task evals. This tells you more than ImageNet zero-shot.
3. **Deployment fit:** parameter count, token count at the target resolution, NPU operator support, quantization sensitivity (measure INT8/INT4 effects), and license.
4. **Combine if needed:** a second small dense encoder, or distill the dense features into the main one.

## 5. Strong interview answer
> "CLIP uses a softmax over the whole batch, so each positive competes with B−1 negatives and the loss needs a global view of the similarity matrix. That ties the task and the temperature to batch size and needs an all-gather. SigLIP is also contrastive, but it turns every image-text pair into an independent binary problem with a learnable bias, so the loss doesn't depend on the batch. The paper reports it does better at small and medium batches, with the gap closing and both saturating around 32k, and it can be implemented in a chunked, memory-efficient way. For VLMs, contrastive encoders give semantic, language-aligned features, zero-shot and retrieval, but only a global signal, so they're weak on OCR, counting, spatial relations and compositionality. Generative objectives like captioning, LocCa or AIMv2 supervise every caption token, giving denser features. For a small VLM, where the LLM can't compensate for weak visual features and token budgets are tight, and where the product needs OCR or grounding, I'd prefer a generative or hybrid encoder such as SigLIP2 or AIMv2. I'd still keep a contrastive embedding when I need retrieval. I'd decide by plugging candidates into a small alignment run and comparing on the product's tasks, plus checking quantization and NPU fit."

## 6. Follow-up questions to prepare for
1. Why does the SigLIP bias term need to be initialized to a large negative value?
2. Why do VLMs use penultimate-layer features instead of the pooled embedding?
3. What are CLIP-blind pairs, and what does a mixture of encoders fix?
4. Why can a caption-predicting decoder be thrown away after pretraining, and what does the encoder keep?
5. How does the quality of captions (alt-text vs synthetic recaptioning) affect contrastive vs generative training?
6. How would you share one encoder between on-device image search and the VLM, and what's the scheduling or memory benefit?

---

# Q5: Cross-attention vs token-concatenation ("early-fusion") VLM designs

**Question:** Compare Flamingo/Llama-3.2-Vision style cross-attention with LLaVA-style token concatenation. What are the implications for KV-cache, memory, batching, streaming video and on-device runtime support?

All numbers below are illustrative, based on public model configs from memory. Verify them against the actual model cards before quoting them in an interview.

## 0. Terminology: "early fusion" is overloaded
There are really three designs:

| Design | How vision enters the LLM | Examples |
|---|---|---|
| **1. Deep fusion via cross-attention** | New cross-attention layers inserted between LLM layers. Text hidden states are the queries, visual features are the keys and values | Flamingo, OpenFlamingo, Llama-3.2-Vision, NVLM-X |
| **2. Input-level token concatenation** (what the question calls "early fusion") | Visual features are projected to the LLM embedding space and placed **in the input sequence**. The LLM's ordinary self-attention handles both | LLaVA, Qwen2-VL, InternVL, PaliGemma, SmolVLM, Idefics3 |
| **3. Native early fusion** | No separate pretrained vision encoder. Image tokens or patches go straight into one model trained multimodally from scratch | Chameleon, Fuyu, Emu3, Llama 4 |

This answer compares 1 and 2, with a short note on 3 at the end. In an interview, state this distinction first. Calling LLaVA "early fusion" is common but loose.

## 1. The two mechanisms

```
Token concatenation (LLaVA-style)              Cross-attention (Flamingo / Llama-3.2-Vision)

[vis tok][vis tok]...[text tok]...             text tokens only in the LLM sequence
        │                                                 │
   ┌────▼────┐   every layer                   ┌─────────▼────────┐
   │ Self-   │   attends over ALL tokens       │ LLM self-attn    │ (text only)
   │ attn    │   (visual + text)               ├──────────────────┤
   └─────────┘                                 │ Gated x-attn     │◄─ K,V from visual features
        ...                                    │ (every k-th layer)│
                                               ├──────────────────┤
                                               │ LLM self-attn ...│
                                               └──────────────────┘
```

**Cross-attention details:**
- New cross-attention (+FFN) blocks are inserted every few layers (Llama-3.2-Vision 11B: 8 cross-attention layers added to 32 text layers, so 40 total).
- **Gating:** Flamingo multiplies the cross-attention output by `tanh(α)` with α initialized to 0, so at the start the model **is** the original LLM. This gives a stable start.
- The base LLM weights can stay **frozen**. Llama-3.2-Vision keeps the text weights frozen, so its text-only behavior matches the base model.

**Concatenation details:**
- Only a small projector (MLP) is new. The LLM is usually fine-tuned (fully or with LoRA) so it learns to read visual tokens.
- Visual tokens pass through **every** LLM layer, so they are refined layer by layer together with the text.

## 2. Implications, dimension by dimension

### 2.1 Compute and prefill

| | Concatenation | Cross-attention |
|---|---|---|
| Where visual tokens go | Through all LLM layers (attention + FFN) | Only through the vision encoder, then the K/V projections of the cross-attention layers |
| Prefill cost vs N_vis | ∝ N_vis × full LLM size | Tiny. Visual tokens never run the LLM's FFN stack |
| Tolerance for many visual tokens | Low. You need token reduction (pixel-shuffle, merging) | High. You can feed thousands of tokens per image |

**Illustrative example.** Take an 8B-class LLM (about 7B non-embedding parameters, 32 layers, 8 KV heads, head_dim 128, FP16) and one image with 1,600 visual tokens. The vision encoder cost is the same in both designs and is excluded.

| | Concatenation | Cross-attention (8 cross layers) |
|---|---|---|
| Prefill FLOPs for the visual tokens | 2 × 7B × 1,600 ≈ **22 TFLOP** | Only the K/V projections: ≈ **0.2 TFLOP** |
| Visual KV memory per token | 2 × 32 layers × 8 × 128 × 2 B ≈ **128 KiB** | 2 × 8 layers × 8 × 128 × 2 B ≈ **32 KiB** |
| Visual KV for 1,600 tokens | ≈ **200 MiB** | ≈ **50 MiB** |

So, for the same visual-token count, cross-attention is about 100x cheaper to prefill and about 4x smaller in cache, because only the cross-attention layers store visual K/V.

### 2.2 KV-cache behavior

- **Concatenation:**
  - Visual tokens live in the **self-attention KV-cache of every layer**.
  - They consume the text **context window**. On-device contexts are small (2k to 4k), so this is a real constraint.
  - Every decode step attends to them, so decode attention work grows with N_vis.
- **Cross-attention:**
  - Visual K/V is computed **once** per image per cross layer and stored in a separate, **static, read-only cache**.
  - It does **not** grow as text is generated, and does not use the text context window.
  - The text self-attention KV-cache stays small.

For small GQA LLMs the self-attention KV is already small (about 28 KB per token for Qwen2.5-1.5B), so the cache-size advantage is real but not decisive. The prefill-compute advantage and the context-window advantage matter more.

### 2.3 Memory and parameters

- **Concatenation:** adds only the projector, at a few to tens of millions of parameters. The LLM stays the same size.
- **Cross-attention:** adds whole attention + FFN blocks. Llama-3.2-Vision goes from an 8B text model to 11B, and from 70B to 90B. The extra parameters include the vision encoder and the cross-attention adapter.
- **For small on-device VLMs this is a big negative.** A 1.5B LLM plus eight extra cross-attention blocks and a vision encoder is a large relative increase in a tight 2 to 3 GB budget.

### 2.4 Training and capability preservation

- **Cross-attention:**
  - A frozen LLM keeps its text ability exactly. There is no catastrophic forgetting of language skills.
  - Training is modular and cheaper. You train the cross-attention layers (and the encoder).
  - Gate initialization to zero makes training stable.
- **Concatenation:**
  - The LLM must be tuned, so you risk degrading text ability. You mix in text-only data or use LoRA to mitigate this.
  - Richer interaction: visual tokens are contextualized by every layer, instead of being a read-only memory.

**Empirical comparisons (as reported, verify the details):**
- Idefics2's study ("What matters when building vision-language models") reported that cross-attention did better when the unimodal backbones were **frozen**, but a fully autoregressive (concatenation) design did better when the backbones were trained (e.g. with LoRA).
- NVIDIA's NVLM study compared decoder-only (NVLM-D), cross-attention (NVLM-X) and a hybrid (NVLM-H). It reported that cross-attention was more computationally efficient for high-resolution images, while decoder-only did better on OCR-related and reasoning tasks.
- The field has largely converged on concatenation for small and mid-size VLMs, and Llama 4 moved to native early fusion.

### 2.4b Attention pattern difference
- In concatenation, visual tokens are updated by the LLM layers. In cross-attention, visual features are **static** encoder outputs that the text reads at certain layers. They are never refined by the LLM.
- Causal masking matters too. In a decoder-only model, visual tokens placed before the question can't see the question. Prefix-LM variants (PaliGemma, Gemma 3) allow bidirectional attention among image tokens.

### 2.5 Batching and serving

- **Concatenation:**
  - Visual tokens are just input embeddings (`inputs_embeds`). Standard LLM serving engines (continuous batching, chunked prefill, prefix caching) work with little change.
  - Prefix caching by image hash is possible.
  - Variable token counts (different tile counts) make ragged sequences, handled by padding or packing.
- **Cross-attention:**
  - You need a second cache type, per-request cross-attention masks (which image each text token may see, e.g. Flamingo restricts text to the most recent image), and padding of the visual memory.
  - Mixed batches of text-only and image requests complicate scheduling, since text-only requests skip the cross layers.
  - Serving support is often patchier. Features like prefix caching and chunked prefill were limited early on for cross-attention models in several engines (verify for your engine).

### 2.6 Streaming video and always-on camera

- **Cross-attention:**
  - Designed for interleaved, continuous inputs. Frame features go into a **visual memory** (Flamingo compresses each frame to a fixed 64 tokens with a Perceiver resampler).
  - Frames cost the encoder plus the K/V projections, **not** a full LLM prefill. This suits an always-on camera with only occasional questions.
  - Text context doesn't grow with the number of frames. You can window the visual memory (last K frames) to bound cost.
  - Needs explicit temporal embeddings or masks.
- **Concatenation:**
  - Frames become tokens in the sequence. The KV-cache grows linearly with time, so you need eviction, sliding windows, token merging or sink tokens (StreamingLLM-style).
  - Causality helps: earlier frames' KV stays valid, so you can append incrementally.
  - A common trick for sparse queries: encode frames continuously, **store the embeddings**, and run the LLM prefill lazily only when a question arrives.

### 2.7 On-device runtime support

- **Concatenation:**
  - Fits the common stack: a decoder-only LLM graph with `inputs_embeds` input, plus a separate vision-encoder + projector graph.
  - One static KV-cache, standard quantization (INT4 weights), existing runtime support (llama.cpp, MLC, ExecuTorch, vendor NPU toolchains). Verify M-RoPE support if you use Qwen-style models (see Q3).
- **Cross-attention:**
  - Needs non-standard layer types, a second read-only cache, masks and skip logic for text-only requests. Support in mobile runtimes and NPU compilers is thinner.
  - A plus for static-graph NPUs: the visual memory can have a **fixed maximum size** (padded and masked), and K/V is written once and only read afterwards.
  - Another plus: the encoder and K/V-projection stage can run on a different compute unit and be pipelined with text decoding.

## 3. Summary table

| Aspect | Concatenation (LLaVA-style) | Cross-attention (Flamingo / Llama-3.2-Vision) |
|---|---|---|
| Added parameters | Small (projector) | Large (cross-attn blocks + encoder) |
| Prefill cost vs visual tokens | High, ∝ N_vis × LLM size | Low |
| Max practical visual tokens | Limited, needs reduction | High |
| Visual KV memory | All layers, in the self-attn cache | Cross layers only, separate static cache |
| Uses text context window | Yes | No |
| Text ability | Must be protected (LoRA, mixed data) | Preserved if the LLM is frozen |
| Accuracy on OCR / detail | Generally better when the LLM is tuned | Competitive when frozen, usually behind when concatenation is tuned |
| Serving / batching | Standard LLM engines | Needs special support |
| Streaming / always-on | Context grows, needs eviction or lazy prefill | Natural visual memory, cheap per frame |
| On-device runtime support | Widely supported | Patchy |
| Best for | Small and mid-size VLMs, OCR, most products | Frozen LLM reuse, many tokens per image, streaming with sparse queries |

## 4. Note on native early fusion (design 3)
- **Chameleon / Emu3:** images are discretized into tokens in a shared vocabulary and the model is trained from scratch. **Fuyu:** a linear patch projection, with no vision encoder at all. **Llama 4:** native multimodality with early fusion.
- **Pros:** no encoder bottleneck, any-to-any generation, simpler pipeline.
- **Cons:** very large training compute, training instability (Chameleon needed QK-norm among other fixes), no reuse of strong pretrained vision encoders.
- **Edge relevance:** removing the separate ViT can cut latency, but it demands big pretraining budgets. It is not a practical choice for a small on-device VLM unless you inherit a pretrained model.

## 5. What I'd pick for an on-device small VLM
**Default: concatenation** with pixel-shuffle token reduction. Reasons:
- Smallest added parameters (critical for a 2 to 3 GB budget).
- Works with the standard LLM runtime, quantization and NPU toolchains.
- Better OCR and fine-detail accuracy once the LLM is tuned.

**Consider cross-attention when:**
1. You must **reuse a frozen text LLM** exactly, for example the same on-device assistant LLM weights serve both text and vision requests, and you can't afford a second tuned copy.
2. You need **many visual tokens** (high-resolution or many images) and can't afford their LLM prefill.
3. The workload is **always-on streaming** with sparse queries, so frames should cost only the encoder and K/V projections.

Mitigation to keep concatenation's text ability: LoRA adapters that are applied only for vision requests, plus a text-data mix during tuning.

## 6. Strong interview answer
> "These are different fusion points. In LLaVA-style concatenation, the projector turns visual features into LLM input embeddings, so visual tokens pass through every layer and use the self-attention KV-cache and context window. In Flamingo or Llama-3.2-Vision, the text sequence stays text-only, and gated cross-attention layers inserted every few layers read the visual features as K/V from a separate static cache. The big consequences are: cross-attention doesn't pay the LLM's prefill cost for visual tokens, so it tolerates thousands of tokens, uses a smaller cache since only cross layers store visual K/V, and keeps text ability if the LLM is frozen. The price is extra parameters (Llama-3.2-Vision adds about 3B to an 8B model), special serving and runtime support, and in reported comparisons concatenation tends to win on OCR and detail once the LLM is tuned. For a small on-device VLM, I'd choose concatenation with pixel-shuffle: minimal added parameters, standard runtime and NPU support, and better detail accuracy. I'd consider cross-attention for a frozen shared text LLM, very high token counts, or always-on streaming with sparse queries, where frames should cost only the encoder, not an LLM prefill. I'd also mention native early fusion, like Chameleon or Llama 4, as a third design, but too compute-hungry for small edge models."

## 7. Follow-up questions to prepare for
1. Why does tanh gating initialized at zero make cross-attention training stable and preserve the base LLM?
2. Why does cross-attention avoid LLM prefill cost for visual tokens, and what compute remains?
3. How does the mask in Flamingo handle interleaved images and text?
4. Why did Idefics2 find different winners for frozen vs trained backbones?
5. How would you implement lazy prefill for an always-on camera in a concatenation design, and what is the latency trade-off?
6. What would you need to add to an on-device runtime to support cross-attention layers (cache types, masks, graph partitioning)?
7. Why can a shared frozen LLM be an advantage for memory on a phone?

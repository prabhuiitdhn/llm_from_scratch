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

---

# Q6: Multi-stage VLM training recipe

**Question:** Describe a production recipe (projector alignment → full or LoRA instruction tuning → optional preference tuning). Which components do you freeze or unfreeze at each stage, and why? What failure modes appear if the vision encoder is unfrozen too early (catastrophic drift, loss of OCR or fine-grained capability)?

Hyperparameters below are typical starting points from the LLaVA, Qwen2-VL and related papers (recalled from memory). Treat them as defaults to tune, not fixed rules.

## 0. The big picture

```
Pretrained ViT (contrastive/hybrid)     Pretrained LLM (text-only)        Random-init projector
          └───────────────┬────────────────────┘                                 │
                          ▼                                                      │
Stage 1  Projector alignment      train: projector            freeze: ViT, LLM ◄─┘
Stage 1.5 (optional) mid-training train: all (or ViT+proj+LLM) high-quality knowledge/OCR/doc data
Stage 2  Visual instruction SFT   train: projector + LLM (full or LoRA); ViT frozen or low-LR
Stage 3  Preference tuning (opt.) train: LLM (LoRA or small LR); ViT frozen; DPO-style
Stage 4  Compression-aware (opt.) QAT / distillation for the on-device target (see Q10, Q11)
```

**Why stages at all?** A VLM glues together two independently pretrained models through a randomly initialized projector. Each stage is there to avoid destroying what the pretrained parts already know while the new glue is still bad.

## 1. Stage-by-stage recipe

| Stage | Trainable | Frozen | Data | Typical LR | Purpose |
|---|---|---|---|---|---|
| **1. Alignment** | Projector only | ViT, LLM | ~0.5M to several M image-caption pairs (clean, short) | ~1e-3 (MLP projector), 1 epoch | Map visual features into the LLM's embedding space |
| **1.5 Mid-training** (optional) | All, or ViT + projector + LLM | Nothing, or ViT partly | Millions of higher-quality samples: detailed captions, OCR, documents, charts, knowledge | ~2e-5 LLM, ViT 10x lower | Inject visual knowledge, OCR and fine-grained skills |
| **2. Visual instruction SFT** | Projector + LLM (full or LoRA); ViT frozen or low-LR | ViT (or most of it) | ~0.5M to several M instruction samples: VQA, OCR, docs, charts, grounding, reasoning, multi-turn, plus text-only data | ~2e-5 (full), ViT ~2e-6 if unfrozen, cosine, 1 to 3 epochs | Teach instruction following, formats, reasoning over images |
| **3. Preference tuning** (optional) | LLM via LoRA or very small LR | ViT, projector (usually) | Image-grounded preference pairs (hallucination-focused) | ~5e-7 to 5e-6 (full), β ≈ 0.1 | Reduce hallucination, improve style and safety |
| **4. Compression-aware** (optional) | Depends (QAT, distillation) | — | Mix of earlier data | Low | Preserve accuracy after quantization |

Real-world examples:
- **LLaVA-1.5:** stage 1 trains the projector on about 558K caption pairs, stage 2 trains projector + LLM on about 665K instruction samples. The vision encoder stays frozen.
- **LLaVA-NeXT / OneVision:** adds a high-quality "knowledge" stage and unfreezes the ViT at a learning rate about 10x lower than the LLM's.
- **Qwen2-VL (as reported):** stage 1 trains only the ViT on image-text data, stage 2 unfreezes everything on a large mixed multimodal corpus, stage 3 freezes the ViT and instruction-tunes the LLM.

## 2. Why each choice

### Stage 1: why freeze both backbones and train only the projector
- The projector starts random. Its outputs are garbage at first, so the loss is high and gradients are large and noisy.
- If those gradients flow into the pretrained LLM or ViT, they overwrite good weights to compensate for a bad projector. The backbones **co-adapt to a broken connector** instead of the connector learning to speak their language.
- Training only the projector is cheap (a few million parameters), stable and fast. A high learning rate is safe.
- The data is simple captions. The result is mostly alignment, little instruction following.

**Nuance:** some studies (e.g. Prismatic VLMs) reported that a separate stage 1 barely helps and a single-stage run saves compute. So treat the two-stage split as a safe default, not a law, and validate it on your setup.

### Stage 2: why unfreeze the LLM
- The LLM must learn to follow instructions about images, produce the right formats (short answers, JSON, boxes), and reason over visual tokens.
- **Full fine-tuning vs LoRA:**
  - Full FT has more capacity and is the better choice with millions of samples. For a 0.5 to 2B LLM, it is affordable.
  - LoRA is lighter on memory, protects text ability (the base weights stay put), and suits small data or limited compute.
- Keep **text-only data** (about 10 to 20% of the mix) in the blend to protect language ability.
- Mask the loss: compute it only on assistant tokens, not on the prompt or visual-token positions.

### Whether to unfreeze the ViT in stage 2
The evidence is mixed:
- Some studies found a **frozen** encoder is better or equal (Prismatic, with their data).
- Others (Cambrian-1, LLaVA-NeXT) found unfreezing at a **lower LR** improves results, especially for OCR and fine-grained tasks, once there is enough data.
- The practical rule: freeze early, unfreeze late, at about 10x lower LR than the LLM, with enough data, and with monitoring (see section 3).

### Stage 3: why preference tuning
SFT'd VLMs lean on language priors and hallucinate objects and attributes. Preference data teaches "grounded beats plausible".
- **Data:**
  - RLHF-V (human fine-grained corrections).
  - RLAIF-V (AI feedback from open models).
  - POVID (create rejected answers by injecting errors).
  - mDPO (adds an image-conditional preference: a corrupted image makes the original answer "rejected", forcing the model to actually use the image).
  - MPO / mixed preference optimization.
- **Settings:** LoRA or a very small LR on the LLM, ViT frozen, β around 0.1. Add an SFT term to avoid likelihood displacement, where the chosen answers' likelihood drops too.
- **Watch for:** verbosity and length bias, image-agnostic preferences, general-capability regression.
- **Optional extension:** RL with verifiable rewards (GRPO) for grounding (IoU reward), counting or math. Note that this tends to lengthen outputs, which raises on-device latency.

## 3. Failure modes of unfreezing the vision encoder too early

**What "too early" means:** unfreezing the ViT in stage 1, before the projector is aligned, or at the full LLM learning rate, or on small or narrow data.

1. **Garbage-gradient drift.** A random projector produces large noisy gradients that flow back through the LLM into the ViT. The ViT features distort to compensate for the bad projector, and the representation learned from billions of pairs is overwritten by a few hundred thousand samples.
2. **Loss of OCR and fine-grained detail ("capability collapse").**
   - Early-stage data is typically low-resolution, short captions.
   - Captions rarely mention small text, small objects or exact counts. The encoder is rewarded only for the gist, so it **discards** the fine detail.
   - OCR, charts and small objects then degrade, often without any obvious loss-curve warning. The training loss looks fine while the capability is gone.
3. **Catastrophic forgetting of semantics and robustness.** Zero-shot recognition, rare concepts, multilingual text and out-of-distribution robustness degrade.
4. **Instability.** Loss spikes, growing activation norms or attention entropy collapse, especially with bf16 and high LR.
5. **Position-embedding drift** if you also change the resolution while training.
6. **Cost.** Backpropagating through a high-resolution ViT multiplies activation memory and time, so you need gradient checkpointing.

**How to detect it:**
- **Feature drift:** compare unfrozen ViT features against the frozen reference on a probe set (cosine similarity or CKA) per layer.
- **Probes:** linear or attentive probes on OCR crops and dense tasks, and zero-shot accuracy if the contrastive head is kept.
- **Per-stage eval gates:** run TextVQA, DocVQA, OCRBench and your product tasks after each stage.
- **Per-module gradient norms** to spot an exploding connector.

**How to mitigate it:**
- Always run stage 1 (projector warm-up) first.
- Unfreeze with a **ViT learning rate about 10x lower** than the LLM's.
- Use **layer-wise LR decay** (lowest LR for early ViT layers), or unfreeze **only the last N blocks**, or use LoRA on the ViT.
- Add a **feature-distillation or L2-SP loss** toward the original encoder, or use EMA.
- Keep **OCR and dense data** in the mix, and train at the target resolution.

## 4. Other failure modes across the stages

| Stage | Failure | Cause and fix |
|---|---|---|
| 1 | Under-trained projector, so hallucination later | Too few steps or poor data. Check alignment quality on held-out captions |
| 2 | Text ability regression | LLM drift. Add text data, use LoRA, lower LR, or interpolate weights with the base model (WiSE-FT style) |
| 2 | Overfitting | Too many epochs. Use 1 to 3 epochs and check validation loss |
| 2 | Wrong or missing label masks, wrong chat template, missing image tokens | Pipeline bugs that look like "model quality" problems. Unit-test the data collator |
| 2 | Data imbalance | One domain dominates. Use balanced or weighted sampling |
| 2 | Packing cross-contamination | Attention leaking across packed samples. Use proper masks |
| 2 | Benchmark contamination | Dedup against eval sets |
| 3 | Likelihood displacement, verbosity | Add an SFT term, length control, tune β |
| 3 | Image-agnostic preferences | Use image-conditional pairs such as mDPO |

## 5. Production considerations for a small on-device VLM
- **Train at the deployed configuration.** The projector is tied to the encoder, the LLM, the pixel-shuffle ratio and the tile/token budget. Fix these before stage 1, and don't change the token budget after alignment without retraining the projector.
- **Resolution curriculum.** Align at low resolution (cheap), then move to the target resolution and tiling in later stages.
- **Don't redo stage 1 unnecessarily.** For domain adaptation (appliance UIs, home-robot scenes), start from an already aligned and instruction-tuned small VLM and run a stage-2-style domain SFT with LoRA. Redo stage 1 only if you swap the encoder or the LLM, or change the projector (this connects to Q7).
- **Gate each stage with evals,** including a text-only regression check, an OCR/dense-task check and a hallucination check (POPE/CHAIR). Version the data and seeds for reproducibility.
- **Plan for quantization early.** Run PTQ evaluation after each stage, and use QAT in the last stage if INT4/INT8 hurts the vision tower or projector (Q11).
- **Compute:** stage 1 fits on one GPU. Stage 2 usually needs FSDP or ZeRO, bf16, gradient checkpointing and packing. As a reference point, the LLaVA-1.5 paper reports roughly a day on 8×A100 for the 13B model.

## 6. Strong interview answer
> "A VLM glues a pretrained ViT and a pretrained LLM through a random projector, so I stage training to protect both backbones. Stage 1 freezes the ViT and LLM and trains only the projector on clean captions at a high LR, so bad early gradients don't overwrite pretrained knowledge. An optional mid-training stage unfreezes more on high-quality OCR, document and knowledge data. Stage 2 is visual instruction tuning: projector plus LLM, full fine-tuning when I have millions of samples, LoRA when data or compute is small, with text-only data mixed in to protect language ability, and the loss masked to assistant tokens. I keep the ViT frozen, or unfreeze it at about 10x lower LR once I have enough data. Stage 3, optional, is DPO-style preference tuning on image-grounded pairs, with the ViT frozen, to cut hallucination. Unfreezing the ViT too early is dangerous: a random projector sends noisy gradients that drift the features away from their pretrained state, and narrow low-res caption data teaches the encoder to drop detail that captions never mention, so OCR and fine-grained capability collapse, often without a visible loss warning. I'd detect that with feature-drift measurements, probes and per-stage eval gates, and prevent it with projector warm-up, low or layer-wise ViT LR, partial unfreezing, and distillation toward the original encoder. For an on-device product I'd fix the token budget and resolution before alignment and start domain work from an already aligned model rather than redoing stage 1."

## 7. Follow-up questions to prepare for
1. Why does a random projector send harmful gradients to the backbones, and why does a high LR on the projector alone not hurt?
2. How does narrow caption data cause the encoder to lose OCR ability, and why doesn't the training loss show it?
3. Full FT vs LoRA in stage 2: how do you decide for a 1.5B LLM and 200k samples?
4. How would you measure and prevent text-ability regression in stage 2?
5. Why is image-conditional preference data (mDPO) needed for DPO on VLMs?
6. When would you skip stage 1, and what does that cost?
7. How do you pick the order of the resolution curriculum, and what changes when you move to tiling?

---

# Q7: Fine-tuning a VLM with limited data and compute

**Question:** You have 20k domain-specific image-text pairs, such as appliance UI screens or home-robot scenes. Compare LoRA, QLoRA, DoRA, prefix tuning and projector-only tuning. How do you choose the target modules and ranks, and how do you stop the model forgetting general capabilities? How do you detect and measure forgetting?

Hyperparameters are common starting points (recalled from the LoRA, QLoRA and DoRA papers and practice). Tune them on your own validation data.

## 0. Frame the problem before choosing a method

**20k pairs is small, and the base model matters.** The right starting point is an already aligned and instruction-tuned small VLM (SmolVLM, Qwen2-VL-2B and similar), not a raw encoder + LLM. You are doing domain adaptation, not stage 1 or 2 from Q6.

**First, find out what you are teaching.** The adaptation needed decides the method:

| What fails | Where the fix lives | Cheapest method |
|---|---|---|
| Output format, style, domain vocabulary (e.g. strict JSON for UI states) | LLM | LoRA on the LLM, small rank |
| New reasoning over domain concepts | LLM | LoRA, medium rank |
| The encoder can't "see" the domain (tiny icons, UI widgets, rare appliance parts) | Vision encoder, projector, resolution | ViT LoRA or partial unfreeze, projector, higher resolution or tiles |
| Visual style shift only (screenshots vs photos) | Projector | Projector-only |

**Diagnostic: oracle-caption test.** Give the LLM a correct text description of the image instead of the image. If it then answers correctly, the failures are **perception**. If it still fails, they are **language, reasoning or format**. This tells you where to spend your trainable parameters.

**Data shape gotcha.** "Image-text pairs" are often captions. Training only on bare captions teaches the model to caption everything and erodes instruction-following. Wrap the data in diverse instruction templates (caption, VQA, structured extraction) and mix in general instruction data.

## 1. The methods compared

### Full fine-tuning
Highest capacity, but it needs optimizer state for every parameter, overfits easily with 20k samples, and forgets the most. Not the first choice here.

### LoRA
Freeze W and learn a low-rank update: `W' = W + (α/r)·B·A`, with `A` (r × d_in), `B` (d_out × r). `B` is initialized to zero, so training starts from the base model.
- Trainable parameters per matrix: `r · (d_in + d_out)`.
- **Mergeable:** after training you fold `BA` into `W`, so there is **no inference overhead**. This matters on-device.
- A study ("LoRA Learns Less and Forgets Less") reported that LoRA underperforms full fine-tuning on the target domain for hard tasks, but forgets less and keeps more diverse outputs. That is the trade-off you want with 20k samples.

### QLoRA
LoRA on top of a **4-bit (NF4) quantized base**, with double quantization and paged optimizers. The adapters are trained in BF16.
- Cuts training memory a lot, which matters for 7B and larger models on a single GPU.
- **For a 2B VLM it's rarely needed.** 2B in BF16 is about 4 GB, so plain LoRA fits comfortably on a 16 to 24 GB GPU.
- Costs: slower steps (dequantization), a small quality gap, and **merging into a quantized base isn't lossless**. For deployment you merge into the BF16 base and re-quantize, so evaluate after that step.
- Quantize only the LLM. The vision encoder and projector are more quantization-sensitive (Q11).

### DoRA
Decomposes each weight into **magnitude and direction**. LoRA updates only the direction, and a separate magnitude vector is learned.
- Reported to behave closer to full fine-tuning, and to be more robust at low rank.
- Mergeable, so there is no inference overhead. Training is slightly slower and uses more memory.
- Gains are usually modest. Try it when LoRA at a reasonable rank underfits.

### Other LoRA knobs worth knowing
rsLoRA (scales by α/√r for stability at high ranks), LoRA+ (different learning rates for A and B), PiSSA (initializes from principal singular components), VeRA (tiny, shared random matrices), AdaLoRA (adaptive rank allocation).

### Prefix tuning / prompt tuning
Train **virtual tokens**: soft prompt embeddings at the input, or key/value prefixes at each layer.
- Very few trainable parameters and no weight changes.
- Drawbacks: the virtual tokens **consume context and KV positions at inference** and can't be merged into weights. They are less expressive and more sensitive to initialization and learning rate, and weaker on small models (prompt tuning improves with scale).
- Rarely preferred over LoRA for a 1 to 2B model.

### Projector-only tuning
Train the few million projector parameters and freeze everything else.
- Very cheap, no text-ability forgetting, and a strong **first baseline**.
- Fixes alignment shifts: a different visual style, a changed token budget or resolution.
- Cannot teach new output formats, vocabulary or reasoning, and cannot fix encoder blind spots.

### Comparison

| Method | Trainable % | Training memory | Forgetting risk | Domain capacity | Inference overhead | Best when |
|---|---|---|---|---|---|---|
| Full FT | 100% | Highest | Highest | Highest | None | Lots of data, big budget |
| **LoRA** | ~1 to 2% | Low | Low to medium | Medium to high | None after merge | **Default** |
| QLoRA | ~1 to 2% | Lowest | Low to medium | Medium | None after merge (re-quantize) | ≥7B models on small GPUs |
| DoRA | ~1 to 2% | Low+ | Low to medium | Medium to high | None after merge | LoRA underfits at low rank |
| Prefix / prompt | <0.1% | Very low | Very low | Low | Extra tokens/KV | Rare for small models |
| Projector-only | ~0.2 to 1% | Very low | Very low (text-wise) | Low (alignment only) | None | Baseline, style shift, token-budget change |

## 2. Choosing target modules and ranks

### Target modules
- **LLM: apply LoRA to all linear layers** (q, k, v, o and the MLP gate, up and down projections). The QLoRA paper found that LoRA on all linear layers matters more than the rank, and that attention-only LoRA does not match full fine-tuning. The MLP layers store much of the "knowledge", so they matter for domain facts. For format-only or style-only tasks, attention-only is often enough.
- **Projector: train it fully.** It is small. Add it to the modules that are saved and trained (`modules_to_save` in common libraries).
- **Vision encoder:** add LoRA to the ViT attention and MLP layers (often just the last third of the layers) **only if the oracle-caption test shows perception errors**. Use a lower learning rate.
- **Embeddings and LM head:** leave them alone, unless you add new special tokens (for example box-coordinate or UI-action tokens). Then train those new rows via `modules_to_save`.
- **Norm layers:** leave frozen.

### Rank, alpha and learning rate
- **Start with r = 16** on all linear layers, α = 16 to 32 (scale α/r of 1 to 2). Sweep {8, 16, 32, 64}.
- Rank tracks **task complexity**, not data size. Format and style tasks need r of roughly 4 to 16. New perception or knowledge needs r of 32 to 64 or more. Beyond that you see diminishing returns, and higher rank means more learning and more forgetting.
- **Worked example (Qwen2.5-1.5B-sized LLM, 28 layers, d = 1536, MLP 8960, GQA with kv_dim 256), r = 16, all linear:**
  - Per layer: q ≈ 49k, k ≈ 29k, v ≈ 29k, o ≈ 49k, gate ≈ 168k, up ≈ 168k, down ≈ 168k → about 659k.
  - Across 28 layers → about 18.5M trainable parameters, around **1.2% of 1.5B**.
- **Learning rate:** LoRA uses a higher LR than full fine-tuning, typically 1e-4 to 2e-4 for adapters, cosine schedule, 3 to 5% warmup, dropout about 0.05. If you also train the projector, use about 1e-5 to 1e-4. For ViT LoRA, use about 1e-5 to 5e-5.
- **Epochs:** 2 to 3 for 20k samples, with early stopping on validation. Effective batch size 32 to 64.

### Compute budget (illustrative)
- 20k samples × roughly 600 tokens (image + text) ≈ 12M tokens per epoch. On a single A100-class GPU with a ~2B model this is on the order of tens of minutes to a couple of hours per epoch.
- Memory is dominated by **activations from image tokens**, not weights. Use gradient checkpointing, cap the number of tiles or `max_pixels` during training, small micro-batches with gradient accumulation, bf16, FlashAttention and an 8-bit or paged optimizer if needed.

## 3. Preventing forgetting

1. **Limit what can change.** Low rank, fewer modules, frozen encoder. This is the cheapest protection.
2. **Replay / data mixing.** Mix about 10 to 30% general data: general VQA, OCR, instruction data and **text-only** data.
   - If you don't have the original SFT data, use **self-generated replay**: sample the base model's answers on general images and prompts, and train on them. This works as self-distillation.
3. **Lower LR and fewer epochs.** Stop at the knee of the domain-vs-general curve.
4. **Regularize toward the base model.** Add a KL or logit-distillation loss on replay data. L2-SP and EWC exist but are rarely used in practice.
5. **Weight-space interpolation (free knob).** After training, scale the adapter by λ < 1 when merging (or interpolate base and fine-tuned weights, as in WiSE-FT). Sweep λ ∈ {0.3 … 1.0} and pick the point on the domain-vs-general Pareto curve.
6. **Separate adapters.** Keep a domain adapter apart from the base and load it only for domain requests. It costs only tens of MB. On static-graph NPUs it is usually merged instead, since runtime support for hot-swapping adapters is limited.
7. **Keep prompt and chat-template formats consistent,** and vary the system prompts you train with.

## 4. Detecting and measuring forgetting

**Build two evaluation sets before you train:**
1. **Domain set:** held-out samples plus a small human-curated golden set.
2. **General regression suite:** fixed subsets of MMBench, MMMU, TextVQA, DocVQA, ChartQA, OCRBench, POPE, MME, plus **text-only** checks (an MMLU subset, IFEval, GSM8K), safety/refusal behavior, and multilingual checks (e.g. Hindi).

**What to measure:**
- **Forgetting = acc_base − acc_finetuned** per benchmark. Report the average drop and the worst-case drop. Set product-driven gates, for example "general average drop ≤ 1 to 2 points and no single benchmark drops more than 3 to 5 points".
- **Track at checkpoints** (every N steps) and choose the checkpoint at the knee.
- **Cheap continuous proxy:** the KL divergence or perplexity relative to the base model's outputs on general prompts.
- **Behavior drift:** output length and style, language switching, hallucination rate (POPE), refusal behavior, calibration.
- **If the ViT was tuned:** feature drift versus the frozen encoder (CKA or cosine similarity) on a probe set.

**Statistical care:** small eval sets are noisy. Use several hundred samples per slice, bootstrap confidence intervals, greedy decoding, and multiple seeds for the key decisions.

## 5. The data side (where 20k pairs usually go wrong)
- **UI screenshots and robot scenes are extremely repetitive,** so the effective number of samples is far below 20k. Deduplicate near-duplicates (perceptual hashes, embeddings).
- **Split by app, device, scene or session, not randomly.** A random split leaks near-identical screens into validation and gives inflated scores.
- **Augment:** image augmentations (crop, color, brightness, resolution jitter), question paraphrases, and several task formulations from the same image (caption, VQA, grounding, state classification). Add verified synthetic Q&A from a larger teacher, plus hard negatives.
- **Label quality over quantity.** Audit a few hundred samples and enforce one consistent output schema (for example JSON).
- Watch class imbalance and the long tail.

## 6. Recommended recipe (decision flow)
1. **Baseline:** zero-shot and few-shot with the base VLM. Run error analysis with the oracle-caption test.
2. **Run 1:** projector + LLM LoRA (r = 16, all linear), LR 2e-4, 2 to 3 epochs, 20 to 30% replay. Evaluate on the domain set and the general suite.
3. **If perception-bound:** add ViT LoRA (last third, low LR), or raise the resolution or tile budget (then re-check the token budget on-device).
4. **If it underfits:** try r = 32 or 64, or DoRA. **If it overfits or forgets:** drop to r = 8, add replay, cut epochs, or use adapter scaling λ < 1.
5. **Pareto sweep** of the merge scale λ. Pick the best trade-off and gate it with the general suite.
6. **Deploy:** merge the LoRA into the BF16 base, then quantize (INT4 weights), and **re-run both domain and general evals after quantization**.

**Production gotcha: quantization can erase LoRA gains.** A LoRA delta has a small magnitude and can be of the same order as INT4 rounding noise, so part of the fine-tuning benefit disappears after quantization. Mitigations: train against the quantized base (QLoRA-style, or QA-LoRA / LoftQ initialization), use QAT in a final stage, or keep the adapter separate in higher precision (costs extra latency and needs runtime support).

## 7. Strong interview answer
> "With 20k domain pairs I'd start from an already aligned and instruction-tuned small VLM, and first find what actually fails. The oracle-caption test separates perception errors from language and format errors. Then I'd pick the cheapest method that covers that failure. Projector-only is the baseline: no text forgetting, but it can only fix alignment shifts. LoRA on all linear layers of the LLM plus a fully trained projector is my default, since it merges with no inference overhead, and at 'learns less, forgets less' it suits small data. QLoRA is for 7B and larger, because a 2B model fits in BF16, and it makes merging and deployment quantization more awkward. DoRA is a drop-in for when low-rank LoRA underfits. Prefix tuning adds tokens at inference and is weaker on small models. I'd use r around 16 on all linear layers with LR around 2e-4, sweep 8 to 64, and add ViT LoRA only if the errors are perceptual. To limit forgetting I'd mix 10 to 30% general and text-only replay, keep a KL term to the base model, stop early, and sweep an adapter scale at merge time to pick the Pareto point. I'd measure forgetting with a fixed regression suite covering general VQA, OCR, text-only, safety and multilingual behavior, gating on average and worst-case drop with confidence intervals. The data pitfalls are near-duplicate screens and leaked splits, so I'd deduplicate and split by app or scene. Finally, I'd re-evaluate after merging and quantizing, because INT4 can wipe out small LoRA deltas."

## 8. Follow-up questions to prepare for
1. Why is LoRA on all linear layers better than higher rank on attention only?
2. Why does LoRA "learn less and forget less", and when is that the wrong trade-off?
3. Why can INT4 quantization erase LoRA gains, and what are your options?
4. How does DoRA's magnitude/direction decomposition change training dynamics?
5. How would you build replay data when you don't have the original SFT set?
6. How would you tell a perception failure from a reasoning failure in a failing VQA sample?
7. Why does a random split of UI screenshots overestimate accuracy, and how do you fix the split?
8. When would you hot-swap LoRA adapters on-device, and what does the NPU runtime need?

---

# Q8: Data engineering for VLMs

**Question:** How would you build a high-quality instruction dataset for a niche domain, such as a smart-home or robot perception assistant? Cover synthetic data from larger teachers, caption re-writing, hard-negative mining, dedup (perceptual hashes), contamination checks against benchmarks, privacy and PII (faces, license plates), and label noise. How do you validate the synthetic data?

Thresholds and numbers below are typical starting points, not universal values. Calibrate them on your own data.

## 0. Principles
- **Data is the main lever** for a small VLM. The architecture is mostly solved, so quality, coverage and correctness decide the result.
- **Build the evaluation set first.** A human-labeled golden set, never used for generation or training, defines "good".
- **Every sample needs provenance.** Source, license, consent, how its label was produced (human, teacher model and version, or program) and its quality scores.
- **Verify before you scale.** Synthetic data scales errors as easily as it scales volume.

```
Spec + golden eval → collect raw → quality/PII filter → dedup → decontaminate → balance/select
   → generate (human seeds, programmatic, teacher) → verify (auto + consensus) → human audit
   → hard negatives / preference pairs → version + ablate → iterate
```

## 1. Start from a spec, not from data
Write down what the assistant must do, then derive the data from that.

| Dimension | Examples for a smart-home / robot assistant |
|---|---|
| Task types | Object and state recognition ("is the stove on?"), reading displays (OCR), spatial questions ("what's left of the sofa?"), counting, hazard detection (wet floor, open flame, child or pet near a hazard), find-me-X, multi-turn follow-ups, **unanswerable** questions |
| Output formats | Short answers, JSON state, boxes or points, refusals with a reason |
| Conditions | Low light, IR night mode, motion blur, fisheye or wide-angle, low camera height (robot egocentric), clutter, occlusion, reflections |
| Languages | English, Hindi and other target languages, in prompts and in on-screen text |
| Sensors | RGB-only vs RGB + depth + LiDAR + IMU (see Q20) |

Build a **coverage matrix** (scenes × objects × lighting × viewpoint × question type × language) and track fill rates per cell. Gaps in the matrix are where the model will fail in the field.

## 2. Sourcing images

| Source | Pros | Cons |
|---|---|---|
| Real captured (product cameras, consented) | Matches the deployment distribution | Privacy, collection cost, slow |
| Public datasets (COCO, Open Images, ScanNet, ARKitScenes, HM3D, Replica, Ego4D-style) | Cheap, diverse | License terms, domain gap, benchmark overlap |
| Simulation (Gazebo, Habitat, Isaac, MuJoCo) | **Free exact ground truth** (depth, segmentation, poses, object states) | Sim-to-real gap (Q23) |
| Web scrape | Scale | License risk, noise, PII |
| Generated images (diffusion) | Rare-case coverage | Domain gap, artifacts, often unrealistic details |

Track the **license and provenance** of every source. A dataset with unclear rights can block a product launch.

## 3. Producing labels

### 3.1 Human annotation
- Use for **gold seeds** (about 500 to 2,000 samples) and for the **golden eval set**.
- Write clear guidelines with examples and edge cases. Measure inter-annotator agreement (Cohen's κ or Krippendorff's α) and revise the guidelines until it is acceptable.

### 3.2 Programmatic labels from structured ground truth
- From simulation or detectors, generate QA with **exact answers**: counts, distances (from depth), relations (from boxes and geometry), states (from the simulator).
- Cheap and verifiable, but the language is templated. Paraphrase with an LLM, and keep the programmatic answer as the ground truth so paraphrasing can't change it.

### 3.3 Synthetic data from larger teacher models
- **Give the teacher more than the image:** pass detections, OCR output, depth or a scene graph as metadata. This anchors its answers.
- **Use a rubric prompt:** question types, required output format, and an explicit instruction to say "not visible" when unsure.
- **Two-step generation** (caption or description first, then Q&A) is often more grounded than direct Q&A.
- **Patterns from public work:** LLaVA (text-only teacher given captions and boxes), ShareGPT4V and ALLaVA (detailed captions), Cambrian-style mixtures.
- **License check:** some teacher models' terms restrict using their outputs to train other models. For a commercial product, verify the terms, or use a permissively licensed open teacher.
- **Privacy rule:** do not send raw user or sensitive images to third-party APIs. Scrub or anonymize **before** any external call.
- **Cost control:** cascade a cheap model first and escalate hard samples to a stronger teacher.

### 3.4 Caption re-writing (recaptioning)
- Raw alt-text is short, noisy and often unrelated to the image. Recaption with a VLM that writes detailed, factual descriptions of objects, states, positions, counts and visible text.
- **Mix raw and synthetic captions** instead of fully replacing. Synthetic captions can be stylistically homogeneous, miss long-tail details and add hallucinations.
- Use a domain-specific caption template that always mentions state, position, count and on-screen text.

### 3.5 Hard-negative mining
Hard negatives teach the model to discriminate near-misses:
- **Text negatives:** swap an attribute, object or relation ("door open" → "door closed", "left" → "right"), or change a count.
- **Image-side negatives:** pairs of images that look alike (high CLIP/DINO similarity) but differ in the labeled attribute (the "CLIP-blind pair" idea).
- **Absent-object questions:** "Is there a cat?" when there isn't one, to reduce hallucination. Keep the yes/no ratio balanced.
- **Model-driven mining:** samples where the current model is confidently wrong, or where it disagrees with the teacher or with itself across paraphrases.
- **Preference pairs for DPO:** chosen = grounded answer, rejected = plausible but hallucinated answer (Q6, Q9).

## 4. Deduplication

| Level | Method | Notes |
|---|---|---|
| Exact | Hash of decoded pixels (e.g. SHA-256 after normalization) | Catches byte-level and re-encoded copies |
| Near-duplicate (visual) | **Perceptual hashes** (pHash, dHash, aHash, wHash) with a Hamming-distance threshold (e.g. a small distance on a 64-bit hash) | Fast, but fragile to crops, rotations and color shifts |
| Semantic duplicate | **Embedding similarity** (CLIP, SigLIP, DINOv2, or SSCD for copy detection) with a high cosine threshold, using ANN search (FAISS or HNSW); **SemDeDup**-style clustering to prune within tight clusters | Catches what hashes miss |
| Video frames | Sample on scene change or embedding novelty, not at a fixed FPS | Consecutive frames are near-duplicates |
| Text | MinHash-LSH on questions and answers; cap repeated templates | Avoids template collapse |

Practical points:
- **Group IDs.** Give each sample an app, device, scene or session ID. Cap the samples per group, and **split train/val/test by group**. Without this, near-identical screens end up on both sides of the split and inflate validation scores (Q7).
- **Cluster-balanced sampling** reduces the head of the distribution and lifts the long tail.
- **Image quality filters:** minimum resolution, blur (e.g. variance of the Laplacian), over- or under-exposure, corruption, extreme aspect ratios. For web data, an image-text alignment score (CLIPScore or SigLIP score) removes unrelated pairs. These scores are weak on fine detail, so don't rely on them alone.

## 5. Contamination checks against benchmarks
Contamination makes the benchmark numbers meaningless and can hide real regressions.

- **Image level:** hash and embedding nearest-neighbor search between all training images and the test images of every benchmark you report. Many datasets share source images (COCO, Visual Genome, web crawls), so overlap is common.
- **Text level:** n-gram overlap (e.g. 13-gram) and fuzzy matching on questions and answers.
- **Pair level:** (image, question) combinations that match benchmark items.
- **Teacher contamination:** a teacher that memorized a benchmark can reproduce its items when generating "new" data. Check synthetic data too.
- **Process:** decontaminate first, then train. Report results with and without the overlapping items. Keep an internal golden set that is never used in prompts, seeds or generation, and version it. Use time-based splits where possible.

## 6. Privacy and PII
Sensitive items: faces, license plates, screens showing notifications, names, emails, phone numbers or addresses, documents, children, home interiors (bedrooms and bathrooms), GPS in EXIF metadata, and any audio.

**Pipeline:**
1. **Consent and legal basis first.** Data minimization, opt-in collection, retention limits, and awareness of the relevant regimes (e.g. GDPR, India's DPDP Act, Korea's PIPA). Involve legal and privacy teams early.
2. **Strip metadata** (EXIF, GPS, device IDs).
3. **Detect and handle faces and plates** (face and plate detectors): blur, pixelate, inpaint, or drop the sample.
4. **OCR the image text** and run PII detection (regexes plus NER, e.g. Presidio-style). Mask the matching regions in the image and redact the corresponding text.
5. **Scrub generated text** too. Teacher models can write real-looking names or leak details seen in the image.
6. **Exclude sensitive scenes** entirely (e.g. bathrooms) unless there is a specific, consented need.
7. **Access control, audit logs, deletion workflows.** If a user withdraws consent or a source's license is revoked, you must be able to remove their data and retrain, so keep lineage.
8. **Memorization risk:** models can regurgitate training data. Deduplication, PII scrubbing and extraction tests reduce this.

**Trade-off to call out:** blurred faces change the training distribution. If the product handles real faces on-device (e.g. person-related queries), models trained only on blurred faces may behave differently. Options: use consented data with real faces for those tasks, or use realistic anonymization (face replacement) rather than blur, and test on real faces.

## 7. Label noise

**Sources:** ambiguous questions, annotator errors, teacher hallucinations, template bugs, OCR errors, subjective attributes ("clean", "messy").

**Detect:**
- **Model-based:** confident learning / cleanlab-style detection using out-of-fold predictions, or high-loss samples after a first training pass.
- **Consensus:** several teachers or sampled answers. Keep samples where, say, at least 2 of 3 agree.
- **Consistency:** a paraphrased question should give the same answer, and the same question in a different format should too.
- **Programmatic verification:** counts vs detector output, distances vs depth, text vs OCR engine output.
- **Human spot audits** with stratified sampling (section 8).

**Handle:** relabel, drop, down-weight, or add an "uncertain" class.

**Caution:** don't over-filter. Removing every hard example biases the dataset toward easy cases. Keep **hard but correct** samples, and separate "hard" from "wrong".

## 8. How to validate synthetic data

Think of four layers of checks, from cheap to authoritative.

### 8.1 Automatic verification
- **Grounding checks:** extract the objects and attributes the generated text claims, and verify them with a detector or an independent verifier VLM using decomposed yes/no questions (TIFA/DSG/VQAScore- and CHAIR-style checks).
- **Cross-teacher agreement and self-consistency:** sample several answers and keep the majority.
- **Format validators:** JSON schema, answer type, length, language ID, boilerplate detection ("As an AI language model…").
- **Geometry and state checks:** counts, spatial relations and distances against boxes, depth or simulator state.
- **Blind test:** answer the question with a **text-only** model. If it is right without the image, the question leaks through language priors and has little visual value. Also try a **shuffled-image test**: if the answer doesn't change with a wrong image, the image isn't needed.

### 8.2 Distribution checks
- Diversity: n-gram and embedding-cluster coverage, question-type distribution, answer-length distribution, template-collapse rate.
- Balance: yes/no ratio, class balance, per-cell fill in the coverage matrix.
- Real-vs-synthetic gap: compare embedding distributions (FID-like for images, MAUVE-like for text). A synthetic set that is far from real data is a warning sign.

### 8.3 Human audit
- **Stratified sample per batch** (e.g. 300 to 1,000 items). With n = 400, the 95% confidence interval on an error rate is about ±5 percentage points.
- Use an **error taxonomy** (hallucination, wrong count, vague, unsafe, wrong format), and set acceptance thresholds per slice (e.g. ≥95% correct).
- If a batch fails, fix the **teacher prompt or pipeline** and regenerate, instead of patching individual samples.

### 8.4 Downstream ablation (the authoritative test)
- Train a small proxy model (or a short LoRA run) **with and without** the synthetic subset, and compare on the **real, human-labeled golden set**. The test data must never be synthetic.
- Track hallucination metrics (POPE, CHAIR) too. Synthetic data can raise hallucination even while raising accuracy.
- Run data-mixture ablations and scaling curves. Use influence or clustering analysis to find subsets that cause regressions.
- **Model-collapse guard:** a student can't exceed its teacher, and repeated training on model outputs degrades diversity. Keep a fixed fraction of real data, use multiple teachers, and keep verifiers independent of the generator.

## 9. Infrastructure
- **Schema per sample:** image URI, content hash, group ID, source, license, consent flag, PII-scan result, label provenance (human / program / teacher + model version + prompt hash), quality scores, split.
- **Versioning and lineage:** DVC, lakeFS or Delta-style versioning and manifests, so you can reproduce any training run and remove a source on request.
- **Storage:** sharded formats (WebDataset, Parquet).
- **CI for data:** automated checks on every dataset change: schema, leakage, duplicate rate, class balance, PII scan, and a small eval run.
- **Data card** documenting sources, known biases, filters and intended use.

## 10. Domain-specific notes for a smart-home / robot assistant
- Use **simulation (Gazebo, Habitat, Isaac, MuJoCo)** to produce RGB, depth and segmentation with exact answers ("how far is the sofa?" with a metric answer), then mix with real images and evaluate on real images.
- Capture **egocentric, low-height viewpoints** and camera artifacts (fisheye, rolling shutter, IR night mode).
- Include **unanswerable and occluded** cases, and train calibrated answers like "I can't tell from this view", so the model doesn't guess.
- Include **safety-relevant** cases (stove, wet floor, child or pet near hazards) with careful, reviewed labels.
- Cover **multilingual prompts and on-screen text** (Hindi, Korean and others in your market).
- Add **multi-turn and temporal** samples (video clips) when the product needs them.

## 11. Illustrative pipeline numbers
Expect heavy attrition: for example 300k raw images → about 100k after quality, PII, dedup and decontamination filters (a 30 to 50% yield is common) → generation and verification may drop a further 10 to 30% of samples. Plan collection volume accordingly.

## 12. Strong interview answer
> "I'd start with the product spec and a human-labeled golden eval set, and a coverage matrix over scenes, conditions, question types and languages. For sources I'd mix consented real captures, public datasets and simulation, where Gazebo or Habitat gives exact depth, segmentation and state for programmatic QA. For labels I'd use human gold seeds, programmatic QA from structured ground truth with LLM paraphrasing, and teacher-generated data where the teacher gets the image plus detections, OCR and depth, a rubric, and permission to say 'not visible', after checking the teacher's license terms and never sending unscrubbed private data to an external API. I'd recaption noisy captions but mix them with raw ones, and mine hard negatives from attribute swaps, absent-object questions, near-duplicate image pairs and model failures. I'd dedup with exact hashes, perceptual hashes and embedding or SemDeDup clustering, split by scene or app group to avoid leakage, and decontaminate against every reported benchmark at image, text and pair level. For privacy I'd strip EXIF, blur or replace faces and plates, OCR and redact PII, scrub generated text, and keep lineage so a deletion request is possible. For label noise I'd use multi-teacher consensus, consistency and programmatic checks, and cleanlab-style detection, while keeping hard-but-correct examples. Synthetic data is validated in four layers: automatic grounding and format checks including a text-only blind test, distribution checks, a stratified human audit with a confidence-interval-based acceptance threshold, and, finally, a downstream ablation on a real human-labeled test set, with hallucination metrics tracked."

## 13. Follow-up questions to prepare for
1. Why can perceptual hashes miss duplicates that embeddings catch, and what are the failure modes of each?
2. How do you decide the similarity threshold for embedding-based dedup without deleting useful variation?
3. What is the text-only blind test for VQA data, and what does a high blind-test accuracy tell you?
4. How would you detect that a teacher model memorized a benchmark?
5. What changes in the pipeline if the teacher's license forbids training on its outputs?
6. Face blur vs face replacement: how does each affect a model that must handle real faces on-device?
7. How would you estimate the error rate of a 200k synthetic set from a 400-sample audit, and what confidence do you get?
8. What are the risks of training repeatedly on model-generated data (model collapse), and how do you mitigate them?

---

# Q9: Hallucination in VLMs

**Question:** What causes object, attribute and relation hallucination (language-prior dominance, weak grounding, low-res features)? Compare mitigations: contrastive decoding (VCD), OPERA, RLHF-V / DPO with image-grounded preference data, grounding-aware training and higher-res encoders. How do you measure it (POPE, CHAIR, HallusionBench), and how would you set up a regression gate in CI?

Method descriptions are from the papers as I remember them. Treat any numbers as approximate and check the original papers before quoting them.

## 0. What "hallucination" means for a VLM
The model produces content that is **not supported by the image** (faithfulness errors). Types:

| Type | Example |
|---|---|
| **Object** | Mentions a "microwave" that isn't in the picture |
| **Attribute** | Wrong color, state ("door open"), material, wrong count |
| **Relation** | Wrong spatial relation ("left of") or action ("holding a cup") |
| **Text / OCR** | Invents or misreads on-screen text |
| **Presupposition** | Accepts a false premise ("why is the cat sleeping?" when there is no cat) |
| **Temporal (video)** | Wrong event order |

Note the difference between **faithfulness** (is it supported by the image?) and **factuality** (is it true in the world?). This question is about faithfulness.

## 1. Causes

### 1.1 Language-prior dominance
- The LLM is much stronger than the visual pathway. It has strong **co-occurrence priors** ("kitchen → microwave, fork, knife").
- CHAIR-style analyses find that hallucinated objects are usually those that frequently co-occur with objects that really are in the image.
- **Text inertia grows during generation.** As the answer gets longer, the share of attention on image tokens tends to shrink and the model conditions more on its own previous words. An early hallucination then "snowballs" into later claims. OPERA ties hallucination to "summary token" attention patterns where the model over-trusts earlier text.
- Yes/no questions show a **yes-bias** in many models.

### 1.2 Weak grounding and alignment
- **Information bottleneck:** a lossy projector or a heavy token reduction (Q1, Q2).
- **Encoder limits:** contrastive features are global and bag-of-words-like (CLIP-blind pairs, Q4), so attributes and relations are poorly encoded.
- **Training data teaches fabrication:** captions and instruction data that contain details the model cannot actually perceive. SFT on rich teacher-written descriptions can teach a weaker student to "fill in" plausible detail. Synthetic data with its own hallucinations adds more (Q8).
- **Objective mismatch:** next-token likelihood doesn't penalize unsupported claims, and preference tuning can reward verbosity or agreement with the user.

### 1.3 Low-resolution features
- Small objects and text vanish after downscaling or aggressive pooling, and the LLM fills the gap with a plausible guess.
- Tile boundaries can split objects (Q2).

### 1.4 Decoding
- Greedy or beam decoding favors the highest-probability, language-plausible continuation. Long outputs give more chances to drift.

### 1.5 Small-VLM and on-device specifics
- A small model has less capacity to both see and verify, and the token budget is tight.
- **Quantization or token pruning can raise hallucination** (for example INT4 on the vision tower, or pruning visual tokens). Measure it on the deployed artifact.

## 2. Mitigations compared

### 2.1 Decoding-time (training-free)

**VCD (Visual Contrastive Decoding):**
- Run the model twice per step, once on the original image and once on a **distorted image** (Gaussian noise added). Combine logits as `(1 + α)·logit(original) − α·logit(distorted)`, with an adaptive plausibility constraint that keeps only tokens already likely under the original image.
- Intuition: language-prior-driven tokens stay likely even when the image is corrupted, so subtracting the distorted distribution removes them.
- **Cost:** about **2x decode compute and a second KV-cache**. Extra hyperparameters (α, noise level). It can hurt fluency or correct tokens.

**OPERA:**
- Beam search with an **over-trust penalty** on candidates showing a "knowledge aggregation" attention pattern, plus a retrospection step that rolls back to the summary token and re-selects.
- **Cost:** beam search, needs access to attention maps (conflicts with fused attention kernels like FlashAttention), and is considerably slower.

**Others:** PAI (amplify attention on image tokens at inference), DoLa-style layer contrast, M3ID, ICD, and post-hoc correctors such as Woodpecker and LURE (detector + LLM revision).

**Simple controls:** lower temperature or greedy decoding, shorter answers, prompts that say "answer only from what is visible, say not visible if unsure", closed-vocabulary or schema-constrained outputs.

### 2.2 Training with image-grounded preference data
- **RLHF-V:** fine-grained, segment-level human corrections, with a dense DPO that weights the corrected segments more. Works with a small amount of high-quality data.
- **RLAIF-V:** open-source AI feedback. Responses are split into atomic claims that are verified, and the loop can be run iteratively.
- **POVID:** automatically builds rejected answers by injecting hallucinations, and uses noisy images.
- **mDPO:** adds an **image-conditional** preference, where the same answer with a corrupted image is the "rejected" side, so the model is forced to use the image. It also adds an anchor so the chosen answer's likelihood doesn't drop.
- Also: LRV-Instruction (negative instructions for robustness), HA-DPO, MPO.
- **Cost at inference: zero.** This is why it fits on-device.
- **Pitfalls:**
  - Length bias: a model can lower its hallucination rate just by saying less. Always check recall or informativeness too.
  - Likelihood displacement (add an SFT term).
  - Off-policy data mismatch. On-policy samples (from the current model, labeled by a verifier) tend to work better.
  - Preferences that don't actually depend on the image.

### 2.3 Grounding-aware training
- Train on **grounding tasks**: referring expressions, detection, region captioning (Kosmos-2, Shikra, Ferret, GLaMM-style data). This ties words to regions.
- **Look-then-answer:** make the model output boxes or evidence first, then the answer. This forces evidence for object claims.
- **Negative and abstention data:** absent-object questions, unanswerable and occluded cases, answers like "I can't see that" (Q8).
- **Faithful SFT data:** don't train on details beyond what the model can perceive. Verify synthetic captions with grounding checks (Q8).

### 2.4 Higher-resolution and better encoders
- Higher resolution and tiling reduce small-object and OCR hallucination (Q2).
- A mixture of encoders (e.g. DINOv2 + SigLIP) or a generatively trained encoder fixes CLIP-blind weaknesses (Q4).
- **Cost:** more encoder FLOPs and more visual tokens on-device. Treat the resolution-vs-hallucination curve as part of your token-budget decision.

### 2.5 System-level and post-hoc
- **Verification pass:** check the generated claims with a detector, an OCR engine or a verifier VLM, and filter or rewrite unsupported claims.
- **Uncertainty and abstention:** token-level entropy, self-consistency, calibrated confidence. Low confidence → abstain, ask for a better view, or route to the cloud (Q17).
- **Product design:** closed vocabularies for detectable objects, showing evidence boxes in the UI, restricting what the assistant is allowed to claim.

### Comparison

| Method | Where | Training needed | Inference overhead | Strength | Weakness | On-device fit |
|---|---|---|---|---|---|---|
| VCD | Decoding | No | ~2x decode + 2nd KV-cache | Plug-and-play | Cost, fluency, hyperparameters | Poor |
| OPERA | Decoding | No | Beam search, attention access | No retraining | Slow, kernel conflicts | Poor |
| Preference tuning (RLHF-V, mDPO, RLAIF-V) | Training | Yes | **None** | Persistent, cheap at inference | Length bias, data quality | **Good** |
| Grounding-aware training | Training | Yes | None (or short evidence tokens) | Evidence-based answers | Needs grounded data | Good |
| Higher resolution / better encoder | Architecture | Yes | More compute and tokens | Fixes root cause for small things | Latency, memory | Depends on budget |
| Post-hoc verifier / abstention | System | Optional | Extra pass or detector | Catches residual errors | Extra latency and components | Good if lightweight |

## 3. What I'd do on-device (priority order)
1. **Fix data first:** grounded, verified captions and instructions, negative and abstention samples (Q8).
2. **Preference tuning** with on-policy, image-conditional pairs, plus an SFT term, checking informativeness alongside hallucination.
3. **Right resolution** for the slices that matter (small objects, text), chosen from the resolution-vs-hallucination curve.
4. **Lightweight verification and uncertainty-based abstention or routing.**
5. **Avoid contrastive decoding** unless the latency budget allows 2x decode.

## 4. How to measure

### POPE (Polling-based Object Probing Evaluation)
- Yes/no questions: "Is there a {object} in the image?", built from COCO-style annotations (also versions on other datasets).
- Balanced positives and negatives. The negatives come from three sampling strategies: **random**, **popular** (frequent objects) and **adversarial** (objects that co-occur with present ones).
- Metrics: accuracy, precision, recall, F1, and the **"yes" ratio** (to expose yes-bias).
- Limits: only existence, limited vocabulary, and it saturates for strong models.

### CHAIR (Caption Hallucination Assessment with Image Relevance)
- On generated captions: `CHAIR_i` = hallucinated object instances ÷ all mentioned instances, `CHAIR_s` = captions with at least one hallucinated object ÷ all captions.
- Uses COCO object annotations and synonym lists.
- Limits: restricted to COCO categories, synonym handling, and it **rewards saying less**, so report coverage or recall alongside it.

### HallusionBench
- Image-question pairs that separate **language hallucination** from **visual illusion**, with edited or paired images and control questions. It reports consistency-style metrics (e.g. question-level and figure-level accuracy), which expose models that answer from priors.

### Others
- **MMHal-Bench** (LLM-judged, open-ended, several question types), **AMBER** (generative + discriminative, existence, attribute, relation), **MMVP** (CLIP-blind pairs), **GAVIE**, **HaloQuest**, and others.
- **Build product-specific probes.** If you have ground truth (a simulator scene graph, annotated home images), you can auto-generate POPE-style probes: positives from the ground-truth objects, negatives from absent objects chosen at random, by popularity, and by co-occurrence. Do the same for attributes, relations, counts and OCR (string match).

Metric weaknesses to remember: yes/no probes miss long-form hallucination, CHAIR misses attributes and relations, and LLM-judge scores have variance and bias.

## 5. CI regression gate

**Principles:**
- **Fixed, versioned eval sets** and **deterministic decoding** (greedy, temperature 0, fixed prompt templates, fixed image preprocessing and seeds).
- **Evaluate the deployment artifact**: the quantized model with the real preprocessing, token budget and runtime. Hallucination can change after quantization or token pruning, so an FP16 checkpoint result is not enough.
- **Gate on several metrics together**, so no single metric can be gamed:
  - POPE (adversarial split) F1, with the yes-ratio kept in a band (e.g. roughly 0.4 to 0.6).
  - CHAIR_s and CHAIR_i must not rise by more than x points, **and coverage/recall must not fall by more than y** (this catches "say less" cheating).
  - HallusionBench question- and figure-level accuracy.
  - MMHal-style judge score (with a pinned judge version).
  - **Product slices:** UI states, hazard detection, small objects, low light, multilingual OCR, counting, and the **abstention rate** on unanswerable cases (it must not collapse to zero or to always-refuse).
- **Golden failures set:** every past incident or bug becomes a permanent test case.

**Statistical rigor:**
- Compute confidence intervals by bootstrap. Compare candidate and baseline on the **same items** with a paired test (e.g. McNemar for binary correctness) and gate on a significant regression, not raw noise.
- For LLM-judge metrics, pin the judge model, and calibrate it against human labels periodically.
- If you sample, use several seeds.

**Tiering:**
- **PR smoke test:** about 500 to 1,000 items, minutes.
- **Nightly:** the full suite.
- **Release gate:** full suite + on-device run + a human audit of about 200 long-form outputs.

**Production monitoring after release:** thumbs-down rate, abstention rate, confidence-distribution drift, shadow evaluation and A/B tests, with the release criteria stating which trade-offs are acceptable (hallucination vs informativeness).

**Sketch of a gate check:**
```python
def gate(cand, base):
    # cand/base: dicts with per-item correctness arrays and aggregate metrics
    assert 0.4 <= cand["pope_yes_ratio"] <= 0.6
    assert cand["pope_adv_f1"] >= base["pope_adv_f1"] - 0.01
    assert cand["chair_s"] <= base["chair_s"] + 0.01
    assert cand["coverage"] >= base["coverage"] - 0.02   # blocks the "say less" trick
    assert mcnemar_p(cand["correct"], base["correct"]) > 0.05 or \
           cand["correct"].mean() >= base["correct"].mean()   # no significant regression
```

## 6. Strong interview answer
> "VLM hallucination, meaning object, attribute and relation claims not supported by the image, has three main causes: the LLM's language priors dominate, especially as the answer gets longer and attention to image tokens fades; grounding is weak because of lossy connectors, global contrastive features and training data with details the model can't perceive; and low resolution loses small objects and text, which the LLM then fills in plausibly. Mitigations split by where they act. Decoding-time methods like VCD, which contrasts original and distorted-image logits, and OPERA, which penalizes over-trust patterns in beam search, need no training but cost roughly 2x decode or slow beam search with attention access, so they are poor fits for on-device. Training-time methods are better: preference tuning with image-grounded pairs like RLHF-V, RLAIF-V and mDPO, where mDPO adds image-conditional preferences, grounding-aware training with boxes and negative or abstention data, and higher resolution or a better encoder where small objects matter. Those add no inference cost, apart from resolution's compute. I'd also add lightweight verification and uncertainty-based abstention or cloud routing. For measurement I'd use POPE (with the yes-ratio), CHAIR with coverage, HallusionBench and product-specific probes auto-generated from ground truth. The CI gate runs deterministic decoding on the quantized deployment artifact, uses paired significance tests with confidence intervals, and gates on several metrics together so 'saying less' can't pass, with a permanent golden-failure set and a tiered pipeline."

## 7. Follow-up questions to prepare for
1. Why do hallucinated objects tend to be the ones that co-occur with the true objects?
2. Why does VCD's distorted-image logit help, and when does subtracting it remove correct tokens?
3. Why can DPO reduce hallucination just by making outputs shorter, and how do you detect that?
4. What does mDPO's image-conditional term add over plain DPO?
5. Why can quantization or token pruning raise the hallucination rate, and how would you localize which?
6. How would you auto-generate POPE-style probes for attributes and relations from a simulator's scene graph?
7. Why do CHAIR and POPE miss some hallucinations, and what would you add for long-form outputs?
8. How would you design abstention so the model says "can't tell" without becoming useless?

---

# Q10: Distillation for small VLMs

**Question:** How would you distill a 70B+ teacher into a ≤2B student? Cover logit distillation vs feature or attention-map distillation vs sequence-level distillation, and distilling the vision tower. How do you handle tokenizer or vocabulary mismatch between teacher and student? What are the limits of distillation relative to training small models on more tokens?

Cost and compute figures are order-of-magnitude illustrations. Paper details (ULD, MiniLLM, GKD, Distillation Scaling Laws and others) are from memory, so check them before quoting.

## 0. Framing
- **Teacher:** a large VLM such as Qwen2.5-VL-72B, InternVL-78B or Llama-3.2-90B-Vision. It may also be a text-only LLM plus a separate vision teacher.
- **Student (≤2B):** a small vision encoder (SigLIP-B-class, about 90M), a projector (pixel-shuffle MLP), and a 0.5 to 1.5B LLM.
- **What you are transferring:** reasoning style, OCR and grounding skill, output format, calibration. It is **not** unlimited world knowledge (see section 5).
- There are three places to distill: the **LLM's outputs** (sequence or logits), the **internal representations** (features, attention) and the **vision tower**.

## 1. The three levels of LLM distillation

### 1.1 Sequence-level (black-box) distillation
The teacher generates responses (captions, QA, rationales) for images and prompts. The student is trained on them with ordinary cross-entropy.
- **Pros:**
  - Works with any teacher, even an API model or one with a different tokenizer.
  - Simple and scales well.
  - Most open small VLMs are built mainly this way (LLaVA-style GPT-generated data, ShareGPT4V, Cambrian, and others).
- **Cons:**
  - One sample per prompt, so it loses the teacher's full distribution ("dark knowledge").
  - It inherits teacher errors and hallucinations.
  - License limits on teacher outputs (Q8).
- **Improvements:**
  - Rationale (CoT) distillation, but keep the length bounded because long outputs cost latency on-device.
  - Rejection sampling with a verifier, and multi-teacher consensus (Q8).

### 1.2 Logit (token-level, white-box) distillation
Match the student's next-token distribution to the teacher's at every position.

```python
def kd_loss(student_logits, teacher_topk_idx, teacher_topk_logp, labels, tau=1.0, alpha=0.5):
    # student_logits: (B, T, V); teacher_*: top-k cached from the teacher, shape (B, T, k)
    logp_s = torch.log_softmax(student_logits / tau, dim=-1)
    logp_s_k = logp_s.gather(-1, teacher_topk_idx)                  # student log-probs on teacher's top-k
    p_t = torch.softmax(teacher_topk_logp / tau, dim=-1)            # renormalized over top-k
    kl = (p_t * (torch.log(p_t + 1e-9) - logp_s_k)).sum(-1).mean() * tau**2
    ce = F.cross_entropy(student_logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
    return alpha * ce + (1 - alpha) * kl
```

- **Why it is stronger:** the full distribution carries many more bits per token than one sampled token. It encodes near-ties ("red" vs "orange") and uncertainty. This makes it far more **sample-efficient**.
- **Costs:**
  - You need a teacher forward pass for every training token.
  - Storing full 150k-vocabulary logits is impractical, so people cache **top-k** (e.g. k = 20 to 100) probabilities and renormalize, or bucket the tail.
- **Which divergence:**
  - **Forward KL** (teacher || student) is mean-seeking. A small student then spreads probability over many modes, which can produce odd in-between outputs.
  - **Reverse KL** (MiniLLM) is mode-seeking and often better for generation with limited capacity.
  - JSD and skew-KL variants (DistiLLM) trade off between them.
  - **GKD (generalized KD)** trains on **student-generated** sequences scored by the teacher (on-policy). This fixes the train-test mismatch (exposure bias) of offline data.
- **Temperature:** τ softens both distributions, with a τ² factor on the KL term. Typical τ is 1 to 2.
- **Offline vs online:** offline (cached teacher outputs on a fixed dataset) is cheaper but off-policy. Online teacher scoring is better but costs more compute.

### 1.3 Feature and attention-map distillation
Match intermediate representations: hidden states (MSE or cosine, through a learned linear map because the dimensions differ) or attention distributions and value relations (MiniLM / TinyBERT style).
- **Layer mapping:** the student has fewer layers, so map each student layer i to a teacher layer g(i) (uniform spacing or last layers only).
- **Dimension-agnostic option:** MiniLM-style distillation matches attention distributions and value-relation matrices, which don't depend on hidden size or head count.
- **Pros:** dense supervision and faster convergence.
- **Cons:** extra design choices (layer mapping, projections, loss weights), more memory (you must keep hidden states), it can over-constrain the student, and gains at LLM scale are usually modest compared with logit distillation.

### Comparison

| Level | Needs same tokenizer? | Teacher access | Signal richness | Cost | Best use |
|---|---|---|---|---|---|
| Sequence-level | No | Any (even an API) | Low | Teacher generation | Default starting point, cross-family teachers |
| Logit KD | **Yes** (or alignment) | Weights / logits | High | Teacher forward per token (cacheable as top-k) | Same-family teacher, data-limited regimes |
| Feature / attention KD | No (but needs layer and dim mapping) | Weights | Dense, structural | Memory, tuning | Vision tower, early convergence |

## 2. Distilling the vision tower

**Why do it:** a small student encoder (SigLIP-B, MobileCLIP, FastViT, EfficientViT) can inherit capabilities of much bigger encoders (InternViT-6B, EVA-CLIP, SigLIP-SO400M, DINOv2-g, SAM) using only **unlabeled images**, which is cheap and label-free.

**Methods:**
- **Patch-token feature distillation:** match student patch tokens to teacher patch tokens (MSE, cosine or smooth-L1) after aligning the spatial grid, with a linear head for dimension mismatch.
- **Global and affinity distillation:** match the image-text similarity matrix or embeddings (CLIP-KD, TinyCLIP's affinity mimicking with weight inheritance).
- **MobileCLIP "reinforced" data:** precompute and store teacher embeddings and synthetic captions once in the dataset, then reuse them for every training run. This amortizes the teacher cost.
- **Agglomerative multi-teacher distillation (e.g. AM-RADIO):** one small encoder learns from several teachers at once (CLIP-type for semantics, DINOv2 for dense features, SAM for segmentation).
- **Task-specific distillation:** EfficientSAM, TinySAM and EdgeSAM distilled from SAM.
- **Token alignment:** if the student sees fewer tokens or lower resolution, pool the teacher's feature map to the student's grid (adaptive average pooling).

**End-to-end output distillation also reaches the vision tower:** training the student VLM against teacher logits sends gradients into the student's encoder. The common sequence is:
1. Feature-distill the vision tower from the teacher encoder(s) on unlabeled images.
2. Align the projector (Q6 stage 1).
3. Run end-to-end logit or sequence distillation.

**Resolution trap:** if the teacher sees a high-resolution image and the student sees a low-resolution one, the student is asked to produce details it cannot see. It learns to **guess**, which becomes hallucination (Q9). Keep the input gap small, or only keep teacher answers that can be verified at the student's resolution.

## 3. Tokenizer and vocabulary mismatch

**The problem:** logit KL needs the same sequence of tokens and the same vocabulary. For example, a Qwen teacher has about 152k tokens and a student like SmolLM2 has about 49k.

**Options, from most to least practical:**
1. **Choose a student in the same family and tokenizer as the teacher.** This is the best practical fix.
   - Example: Qwen2.5-VL-72B → a Qwen2.5-0.5B/1.5B-based student.
   - Meta reported that the Llama 3.2 1B/3B models used logits from larger Llama 3.1 models during pretraining, and Google reported distilling Gemma 2's smaller models from a larger teacher.
2. **Sequence-level only.** Decode the teacher's text and re-tokenize it with the student's tokenizer. This is tokenizer-agnostic.
3. **Cross-tokenizer logit alignment:**
   - **Universal Logit Distillation (ULD)** uses an optimal-transport-style loss between probability distributions, so no token alignment is needed.
   - **Minimum-edit-distance alignment** (as in FuseLLM-style work) aligns tokens across tokenizers and then fuses distributions.
   - **Dual-space KD (DSKD)** projects hidden states into each other's output spaces.
   - A practical fallback is to align at **word or character boundaries** and match the probability of each word, falling back to sequence-level when alignment fails.
4. **Vocabulary transplant on the student:** swap in the teacher's vocabulary and initialize the new embeddings (for example, from the mean embedding of each token's old sub-tokens). This works, but you must retrain, and a bigger vocabulary inflates the student's embedding and LM head (see Q1 deep dive, section 5).
5. **Restrict to a shared vocabulary subset:** if you prune the vocabulary for target languages (also an edge optimization), restrict the teacher's distribution to the retained tokens and renormalize.

**Multimodal-specific details to align:**
- **Special tokens:** image placeholders, chat-template tokens.
- **Coordinate and box formats:** location tokens or normalized ranges (e.g. 0 to 1000) must match, or you convert at the sequence level.
- **Multiple teachers with different tokenizers:** use sequence-level data or ULD-style losses.

## 4. A recipe for a 70B+ teacher → ≤2B student
1. **Choose the student:** vision tower (SigLIP-B-class), same-family LLM, pixel-shuffle projector, with the token budget and resolution fixed up front (Q2, Q6).
2. **Vision tower distillation (optional):** feature and affinity losses from the teacher encoder(s) on unlabeled images.
3. **Alignment:** projector training as in Q6 stage 1.
4. **Main distillation:** diverse prompts (image + instruction) → teacher responses, filtered with verifiers (Q8). Train with CE, plus **top-k logit KL** if the tokenizers match. Cache teacher top-k offline.
5. **On-policy refinement:** sample from the student, then either get teacher logits on the student's sequences (GKD) or use the teacher as a judge to rank student outputs for DPO-style training (Q9). Mine hard examples (Q8).
6. **Compression-aware stage:** quantization-aware training, with the FP16 student as the "teacher" (self-distillation) for the quantized student (Q11).
7. **Evaluate per capability:** OCR, grounding, reasoning, hallucination, calibration, and teacher-student agreement, with a text-only regression check.

**Combined loss:** `L = λ1·CE + λ2·τ²·KL(logits) + λ3·L_feat(vision) + λ4·L_hidden`, with the weights ramped or scheduled (for example, feature loss early and fading out).

### Compute accounting (illustrative)
- Teacher forward per token ≈ 2 × 70B = 140 GFLOP. Student training per token ≈ 6 × 1.5B = 9 GFLOP. The **teacher's forward pass is about 15x the student's training step per token.**
- For 10B tokens: teacher forward ≈ 1.4e21 FLOPs, on the order of **1,000 H100-hours** at realistic utilization.
- Storing top-20 (index + probability) for 10B tokens ≈ 1.2 TB.
- This is why people use **offline top-k caching**, **sequence-level data** and **reused teacher artifacts** (the MobileCLIP idea).

## 5. Limits of distillation vs training small models on more tokens

1. **Capacity gap.** A student can't mimic a teacher that is far beyond its capacity, and a much stronger teacher doesn't always give a better student. A "teacher assistant" of intermediate size can help.
2. **Distillation doesn't add parameters.** Long-tail world knowledge, rare facts and fine-grained recognition scale with parameter count. Distillation transfers skills, style, formats and calibration much better than raw knowledge.
3. **The student can't exceed the teacher,** and inherits its hallucinations, biases, refusal behavior and mistakes.
4. **Information gap between the teacher's and the student's input** (resolution, token budget). The student can't learn what it cannot see (section 2).
5. **Scaling evidence.** The "Distillation Scaling Laws" work reported that distillation beats supervised training only under certain conditions: the teacher already exists (so its cost is amortized) and the compute budget is not huge. With enough tokens and compute, plain supervised training of the small model catches up or wins. Count the teacher's inference and data-generation compute honestly.
6. **The alternative is over-training small models.** Small models like Qwen2.5-0.5B and SmolLM2 were trained on trillions of tokens, far beyond Chinchilla-optimal, because inference cost is what matters. Distillation improves **data efficiency**, and the two are complementary: Gemma 2 and Llama 3.2 used distillation inside large-token training runs.
7. **Reasoning distillation can inflate latency.** Long CoT traces cost decode time on-device. Prefer short rationales, or distill the final answers plus compact reasoning.
8. **Always run the ablation** "same tokens with ground-truth labels vs distilled", and measure the gain per unit of total compute including the teacher.

## 6. Strong interview answer
> "I'd treat it as three layers. For the LLM, sequence-level distillation is the default: the teacher generates verified responses, which works across tokenizers and even for API teachers, but it loses the full distribution. If the student shares the teacher's tokenizer, I'd add token-level logit KL, cached offline as top-k, because the full distribution carries far more signal per token. I'd consider reverse KL or GKD-style on-policy distillation to fix exposure bias, since a small student can't cover all the teacher's modes. For the vision tower, I'd feature-distill a small encoder from the big teacher encoders on unlabeled images, possibly multi-teacher in the AM-RADIO style, then align the projector and finish with end-to-end distillation. I'd keep the teacher and student inputs matched, because a student can't learn details it cannot see, and it will hallucinate instead. For tokenizer mismatch, the cleanest fix is a same-family student, like Qwen-VL-72B to a Qwen-0.5B/1.5B student. Otherwise I'd use sequence-level data, ULD or edit-distance alignment, or vocabulary transplant at some retraining cost, and I'd watch special tokens and box formats. The limits are real: the capacity gap, the fact that distillation moves skills better than long-tail knowledge, the student inheriting teacher errors, and the cost accounting. The teacher's forward pass is roughly 15x the student's training step per token, and distillation scaling work says that with a big enough budget plain supervised training of the small model can match it. So I'd use distillation to improve data efficiency within an already token-rich recipe, and I'd always ablate against ground-truth-label training on the same tokens."

## 7. Follow-up questions to prepare for
1. Why is forward KL mean-seeking and reverse KL mode-seeking, and which is better for a small student?
2. What is exposure bias in offline distillation, and how does GKD fix it?
3. How does top-k truncation of teacher logits bias the student, and how do you reduce that bias?
4. Why do MiniLM-style attention-relation losses not need matching hidden sizes?
5. How do you distill from a teacher with a different tokenizer when you cannot change the student's tokenizer?
6. Why does a much stronger teacher sometimes produce a worse student?
7. How would you measure whether distillation helped beyond just training on the same tokens with ground truth?
8. How do you distill a vision encoder so that the student works at a lower resolution than the teacher without learning to hallucinate?

---

# Q11: Quantization in depth

**Question:** Compare PTQ vs QAT, weight-only (W4A16, GPTQ, AWQ) vs weight+activation (W8A8, SmoothQuant) and KV-cache quantization (INT8/INT4/FP8). Why are vision encoders and projectors usually more sensitive to quantization than the LLM? How do activation outliers affect INT8 NPUs? How would you diagnose a layer-wise accuracy collapse and fix it with mixed precision?

Method descriptions are from the papers as I remember them. Hardware support (INT4, FP8, per-token scales) varies by NPU and toolchain, so verify against your target's documentation.

## 0. The basics you need to state first

**Affine quantization:** `q = clamp(round(x / s) + z)`, and `x̂ = s · (q − z)`.

**Choices that define a scheme:**

| Choice | Options |
|---|---|
| **What** is quantized | Weights, activations, KV-cache, embeddings and LM head |
| **Granularity** | Per-tensor, per-channel (weights), per-group (e.g. 32/64/128 values), per-token (activations), per-block (microscaling formats) |
| **Symmetry** | Symmetric (z = 0) or asymmetric |
| **Activation scales** | Static (calibrated offline) or dynamic (computed at runtime) |
| **Format** | INT8, INT4, FP8, NF4, MX formats |

**Why quantize, and what it actually speeds up:**
- **Decode is memory-bandwidth-bound.** Each token reads all the weights, so fewer bytes per weight means faster decode. This is why **weight-only** INT4 helps decode.
- **Prefill and the vision encoder are compute-bound.** Weight-only quantization does not reduce their compute, because the weights are dequantized back to a higher precision before the matmul. To use the integer MAC arrays you need **integer activations too (W8A8)**.
- So: weight-only quantization targets decode and memory, and W8A8 targets prefill and the encoder.

**One hardware rule worth memorizing:** quantization scales can only vary along dimensions that are **not** reduced in the matmul. That means per-output-channel for weights and per-token (row) for activations. A per-input-channel activation scale cannot be factored out of the accumulation sum, so integer hardware can't do it efficiently. This is why activation outliers across channels are so hard.

## 1. PTQ vs QAT

### PTQ (post-training quantization)
- No or minimal training. You run a small **calibration set** (about 128 to 1,024 samples) to find activation ranges and, in some methods, to correct the weights.
- **Calibration of ranges:** min/max, percentile, MSE-minimizing, or entropy (KL) based.
- **Learned PTQ:** AdaRound and BRECQ (learned rounding), OmniQuant (learned clipping and equivalent transforms).
- **Pros:** fast, cheap, no training pipeline.
- **Cons:** can collapse at ≤4 bits or with strong outliers. Small models usually have less redundancy and are generally more sensitive than large ones.

### QAT (quantization-aware training)
- Simulate quantization in the forward pass (**fake-quant**), and use the straight-through estimator for the backward pass. The weights, and often the scales (LSQ-style learnable step sizes), adapt to the quantization noise.
- **For LLMs and VLMs:** LLM-QAT (data-free, uses self-generated data), EfficientQAT, and distillation-based QAT where the FP16 model is the teacher (Q10).
- **Pros:** recovers accuracy at 4-bit weights and W4A8.
- **Cons:** needs data and compute. For a ≤2B model this is affordable.

### Decision rule
Start with PTQ. If the gates fail (accuracy, OCR, hallucination), escalate in steps: better calibration → equivalent transforms or rotations → learned clipping → targeted QAT on the offending parts. For a small model aimed at INT4 or W4A8, plan for QAT in the last stage (Q6 stage 4).

## 2. Weight-only quantization (W4A16 family)

- **RTN with groups:** round-to-nearest with per-group scales (group size 32 to 128) is the baseline. Each group stores a scale, and optionally a zero-point.
- **GPTQ:** layer-wise, second-order. It quantizes weights column by column and updates the remaining columns to minimize the layer's output error `‖WX − ŴX‖²`, using the Hessian estimated from calibration activations. Strong at 3 to 4 bits. **Risk:** it can overfit to the calibration distribution, so calibrate a VLM with multimodal data.
- **AWQ:** activation-aware. A small fraction (~1%) of weight channels are "salient", determined by activation magnitude. AWQ scales those channels before quantization so they lose less precision (`W·diag(s)` and `X/diag(s)`, an equivalent transform). No backprop, kernel-friendly (no mixed precision inside a layer), and often generalizes better than reconstruction-based methods.
- **Others:** SqueezeLLM (non-uniform codebooks plus sparse outliers), QuIP# and AQLM (vector quantization, down to about 2 bits), HQQ (data-free), NF4 from bitsandbytes (QLoRA), GGUF k-quants for llama.cpp.

**Size arithmetic (1.5B parameters):** FP16 ≈ 3 GB, INT8 ≈ 1.5 GB, INT4 with group 128 ≈ 4.25 to 4.5 bits per weight including scales (and zeros), about 0.8 GB.

**Limits:** weight-only doesn't speed up the compute-bound parts. On NPUs, check kernel support. In some stacks "W4A16" means INT4 weights with **INT16** activations (Qualcomm-style), not FP16.

## 3. Weight + activation quantization (W8A8 and beyond)

**W8A8:** INT8 weights (per output channel) and INT8 activations (per-tensor static, or per-token dynamic). It is the standard NPU path because INT8×INT8 uses the integer MAC arrays.

**The activation-outlier problem:**
- Transformer activations have **a few channels with magnitudes 10 to 100x larger** than the rest (systematic outlier channels, and "massive activations" on certain tokens).
- With per-tensor INT8, `s = max|x| / 127`. One outlier forces a large step size, so the typical values round to only a few levels. The **effective precision of normal channels falls to 2 to 3 bits**.
- This tends to appear at the inputs to `down_proj`/`fc2`, the residual stream, LayerNorm outputs and some attention outputs.

**SmoothQuant:** migrate difficulty from activations to weights. With a per-channel factor `s_j = max|X_j|^α / max|W_j|^(1−α)` (α ≈ 0.5), compute `X' = X / s`, `W' = s · W`. It is mathematically equivalent and the division is folded into the preceding LayerNorm or linear weights, so there is **no runtime overhead**.

**Rotation-based methods (QuaRot, SpinQuant):** multiply by (random or learned) Hadamard-like rotations that spread outliers evenly over channels, so activations become easier to quantize. Rotations can be fused into the weights, and this makes W4A8 and even W4A4 more feasible. Other tools: OmniQuant (learnable clipping and transforms), PrefixQuant-style isolation of massive-activation tokens into a fixed prefix.

**Static vs dynamic scales:**
- **Static per-tensor** is the most NPU-friendly but the least accurate. Unseen inputs (night images, unusual UIs) can **saturate** it (clipping).
- **Per-token dynamic** is more accurate but needs runtime reductions, and not every NPU supports it.

**FP8:** E4M3/E5M2 are an option on recent GPUs. Most mobile NPUs don't support FP8, so verify. Microscaling (MX) formats are emerging.

## 4. KV-cache quantization

- **What:** store K and V in INT8, FP8 or INT4 instead of FP16.
- **INT8** is usually close to lossless (per-token or per-head scales).
- **Lower bits:**
  - **KIVI:** keys have outlier channels, so quantize K per-channel and V per-token, and keep a recent window in full precision.
  - **KVQuant:** per-channel keys quantized before RoPE, non-uniform codebooks, and dense-and-sparse handling of outliers.
- **When it matters:** the benefit is largest for long context, many visual tokens, video, or non-GQA models. For a small GQA LLM at short context, the cache is small (about 28 KB per token for Qwen2.5-1.5B, Q5 deep dive), so weights and activations matter more.
- **Risks:**
  - Errors accumulate across long contexts.
  - Visual tokens have different K/V statistics from text tokens, so test on long multimodal tasks.
  - Keep attention-sink tokens and the most recent tokens in higher precision if you go to INT4.

## 5. Why vision encoders and projectors are more sensitive

These are commonly observed effects with plausible mechanisms. Verify them on your own model.

1. **Activation structure in ViTs.** ViTs show high-norm outlier tokens (the "artifacts" that motivated register tokens), large inter-channel variance at LayerNorm inputs, post-GELU asymmetry, and **post-softmax attention maps with long-tailed values in [0, 1]**. Uniform INT8 wastes range on these (FQ-ViT-style work uses log2 quantizers and special handling for LayerNorm and softmax).
2. **Continuous outputs with no discrete "absorber."** The LLM ends in a softmax over the vocabulary, where small perturbations often don't change the chosen token. Vision features and projector outputs are continuous vectors consumed directly downstream, so noise passes straight through, and accumulates over 12 to 27 layers.
3. **Less redundancy.** A 90M to 400M encoder has far fewer parameters than a 1.5B LLM to absorb noise.
4. **Fine detail is fragile.** OCR, small objects and counting depend on subtle feature differences. Quantization noise acts like blur on features, so fine detail disappears first.
5. **The projector's output must match what the LLM expects.** Projector outputs are injected where word embeddings would be. The LLM's embedding table is a well-conditioned lookup, but projector outputs are continuous, so a small scale or distribution shift can disrupt the LLM. The pixel-shuffle projector's first layer also has a very large fan-in (for example 10k inputs), which gives a wide accumulation range.
6. **Poor memory trade.** The projector is only a few million parameters, so quantizing it saves almost nothing. The encoder is small in memory too, but heavy in compute. This creates a tension: you want integer activations for encoder speed, which is exactly where the outliers hurt.
7. **Calibration mismatch.** The calibration images must represent the product (text-heavy, low-light, screens). And for the **LLM**, visual tokens inside the LLM have different activation statistics from text tokens, so calibrate the LLM with **multimodal** data that includes visual tokens.

## 6. Activation outliers on INT8 NPUs, in detail

1. **Per-tensor step size problem** (section 3): one outlier channel wrecks the resolution for all the others.
2. **The hardware constraint:** activation scales must be per-tensor or per-token. A per-channel activation scale can't be factored out of the reduction, so the hardware can't do it. SmoothQuant and rotations work around exactly this limitation.
3. **Mixed precision partitioning:** many NPUs run LayerNorm/RMSNorm, softmax, GELU/SiLU and RoPE in FP16/INT16, or fall back to the CPU/DSP. Each precision boundary adds quantize/dequantize operations and memory traffic.
4. **INT16 activations (A16)** are an alternative for sensitive layers: much better accuracy, at 2x the activation memory traffic.
5. **Static calibration risk:** a range that was fine on calibration data may clip on a new input. Clipping errors are often catastrophic (saturated attention or garbage features), so monitor the saturation rate.
6. **Mitigations:** SmoothQuant folding, Hadamard rotations, percentile or MSE-based clipping (trading clipping error against rounding error), keeping outlier-prone layers at higher precision, per-token dynamic scales where supported, learned clipping in QAT, and training-time techniques that avoid outliers ("quantizable transformers" ideas such as clipped softmax and gated attention).

## 7. Diagnosing a layer-wise accuracy collapse

**Step 1: Fix a baseline.** Run the FP16 model on a golden set, and define both task metrics (accuracy, OCR, POPE) and numerical metrics (cosine similarity, SQNR or MSE of intermediate activations, KL of the final logits against FP16).

**Step 2: Bisect by component.** Quantize only one component at a time (vision encoder, projector, LLM, embedding and LM head), and also do the reverse (everything quantized except one). This locates the collapse quickly.

**Step 3: Layer and op sensitivity sweep.** Quantize one layer or op type at a time, with everything else at FP16, and measure the change in metric. You can also use a Hessian-based sensitivity score (HAWQ-style). Rank layers by sensitivity.

```python
def sensitivity_sweep(model, layers, eval_fn, quantize_layer):
    base = eval_fn(model)                       # FP16 baseline metric (higher is better)
    scores = {}
    for name in layers:
        with quantize_layer(model, name):       # quantize only this layer, restore on exit
            scores[name] = base - eval_fn(model)
    return sorted(scores.items(), key=lambda kv: -kv[1])   # most sensitive first
```

**Step 4: Look at activation statistics.** Per-layer histograms, max/median ratio, kurtosis, per-channel ranges. A large outlier ratio at a layer's input is a strong hint. Focus on `down_proj`/`fc2` inputs, the residual stream, LayerNorm outputs, softmax outputs and the projector's input.

**Step 5: Track error propagation.** Run FP16 and quantized models side by side with hooks, and plot the cosine similarity of hidden states against depth. A sharp drop at one layer means a local problem. A gradual slide means compounding error.

**Step 6: Check calibration.** Is the calibration set representative (visual tokens, text-dense images, low-light)? Measure the saturation rate on held-out inputs.

**Step 7: Rule out non-quantization bugs.**
- Wrong scale folding or per-channel axis.
- Fused LayerNorm/BN mistakes.
- Residual adds of two tensors with different scales that need rescaling.
- **Attention mask values** (-1e9 can't be represented in FP16 or INT ranges, so use a representable large negative, e.g. -65504 for FP16 or a safe fixed value).
- **FP16 overflow:** massive activations above 65,504 become inf.
- RoPE tables accidentally quantized.
- Padding tokens not masked.

**Step 8: Fix, from cheapest to most expensive.**
1. Better calibration (percentile or MSE clipping, more diverse data, per-channel weights).
2. Equivalent transforms: SmoothQuant or AWQ scaling, rotations.
3. Finer granularity: group size 64 or 32, per-token dynamic scales.
4. **Mixed precision:** keep the most sensitive layers or ops at higher precision (INT8, INT16 activations or FP16), under a size and latency budget. Treat it as a budgeted optimization: maximize accuracy subject to size ≤ M and latency ≤ T, using a greedy "sensitivity per byte" ordering or an ILP/search (HAQ/HAWQ-style).
5. Targeted QAT or learned clipping with distillation from FP16, on only the offending parts.
6. Architecture or training changes to reduce outliers (registers, clipped softmax, replacing awkward activations with NPU-friendly ones).

**Step 9: Validate on the device.** NPU numerics (fixed-point requantization, lookup tables for nonlinear ops, rounding) can differ from the simulator. Dump intermediate tensors from the device and compare layer by layer against the simulator and FP16.

## 8. Example mixed-precision plan for a ~2B on-device VLM

| Component | Precision | Reason |
|---|---|---|
| LLM linear layers | W4 (group 128, AWQ/GPTQ, or QAT), activations A8 with SmoothQuant, or A16 on NPUs that prefer it | Decode is bandwidth-bound |
| `down_proj` in the first and last 2 layers | W8 | Most sensitive |
| Vision encoder (SigLIP-B class) | W8A8, per-channel weights, LayerNorm/softmax/GELU in FP16 or INT16, patch-embed and last block FP16 | Compute-bound, outlier-prone |
| Projector | FP16 or INT8 | Tiny, sensitive, no memory benefit from lower bits |
| Embedding table / LM head (tied) | INT8 | Large vocabulary (Q1 deep dive, section 5), sensitive |
| KV-cache | INT8 (per-token / per-head scales). If INT4, keep sink and recent tokens in FP16 | Matters for long context and video |
| RoPE tables, norms | FP32/FP16 | Precision-sensitive |

**Size estimate (illustrative):** LLM non-embedding 1.3B × ~4.5 bits ≈ 0.74 GB, embedding/LM head 233M × 1 byte ≈ 0.23 GB, encoder 90M × 1 byte ≈ 0.09 GB, projector in FP16 ≈ 0.04 GB. Total ≈ **1.1 GB**, versus about 3.8 GB in FP16.

## 9. Evaluation notes
- Perplexity is not enough. Evaluate **task accuracy, OCR, grounding and hallucination (POPE/CHAIR, Q9) on the deployed artifact**, with slices for low light and small text.
- Measure **latency on the device**, not just size. INT4 can be slower in prefill if the runtime dequantizes inefficiently.
- Quantization can erase small LoRA deltas (Q7), so re-evaluate fine-tuned models after quantizing.

## 10. Strong interview answer
> "I start with what each scheme speeds up. Decode is memory-bound, so weight-only INT4 with group scales (AWQ, GPTQ) cuts weight traffic. Prefill and the vision encoder are compute-bound, so they need W8A8 to use integer MACs. PTQ is the first step: calibrate, and use AWQ or GPTQ for weights. For activations, outlier channels make per-tensor INT8 collapse, since one outlier sets the step size and leaves the other channels with 2 to 3 effective bits. NPUs can only scale per-token or per-output-channel because per-input-channel scales don't factor out of the reduction, so I'd use SmoothQuant folding or Hadamard rotations, clipping, and higher precision for the outlier-prone ops. If PTQ gates still fail, I'd run QAT with the FP16 model as a distillation teacher. For KV-cache, INT8 is usually safe, INT4 needs KIVI-style handling and an FP window, and it matters mainly for long context and video. Vision encoders and projectors are more sensitive because of ViT outlier tokens and long-tailed softmax and LayerNorm statistics, continuous outputs with no discrete absorber, fewer parameters, fragile fine detail for OCR, and because the projector's output must match what the LLM expects. They also give little memory saving, so I keep them at INT8 or FP16. For a layer-wise collapse, I'd bisect by component, run a single-layer sensitivity sweep, inspect activation outlier ratios, track cosine similarity against depth, check calibration and non-quantization bugs like mask values and FP16 overflow, then fix with calibration, smoothing or rotation, finer granularity, budgeted mixed precision and targeted QAT. I'd confirm on the device, because NPU numerics differ from the simulator, and evaluate hallucination and OCR on the deployed artifact."

## 11. Follow-up questions to prepare for
1. Why can't an integer matmul use per-input-channel activation scales?
2. Why does weight-only INT4 speed up decode but not prefill?
3. How does SmoothQuant's α trade off between activation and weight quantization error?
4. How do Hadamard rotations help, and why can they be fused into the weights?
5. Why does GPTQ risk overfitting to its calibration data, and how does AWQ avoid it?
6. Why is the KV-cache of a visual token different to quantize than a text token's?
7. How do you pick which layers get higher precision under a memory budget?
8. What differences between an NPU's numerics and a simulator could hide a bug?

---

# Q12: Latency anatomy of a VLM on-device

**Question:** Break down end-to-end latency (image decode → preprocess → vision encoder → projector → prefill → decode). Which stages are compute-bound vs memory-bandwidth-bound on a mobile SoC? How do you measure and profile each stage, and which optimizations apply to which stage?

All hardware numbers below are **hypothetical, flagship-class assumptions** (a 40 TOPS INT8 NPU and LPDDR5X at about 68 GB/s peak). They exist to show the reasoning. Replace them with measurements from your target device.

## 0. The pipeline and the two headline metrics

```
camera / file → [1 image decode] → [2 preprocess: resize, tile, normalize]
   → [3 vision encoder] → [4 projector] → [5 merge embeddings + positions]
   → [6 LLM prefill] → first token ─── TTFT
   → [7 decode loop: 1 token per step, sampling, detokenize, stream] ─── TPOT per token
```

- **TTFT** (time to first token) ≈ stages 1 to 6, plus the first sampling step.
- **TPOT** (time per output token) = one decode step.
- **E2E** = `TTFT + (n_out − 1) × TPOT`.

**Key consequence:** for short answers, TTFT dominates (encoder + prefill). For answers longer than roughly 10 to 20 tokens, **decode dominates**, so output-length control is often the biggest lever on E2E latency.

## 1. Stage by stage: resource, bound type, levers

| # | Stage | Runs on | Bound by | Notes |
|---|---|---|---|---|
| 1 | Image decode (JPEG/HEIC, or YUV from the ISP) | HW decoder, CPU, ISP | Memory / fixed-function | A full-resolution photo is expensive to decode on CPU. YUV preview frames skip this |
| 2 | Preprocess (resize, crop/tile, normalize, HWC→CHW, dtype) | CPU (NEON), GPU, ISP scaler, or fused into the NPU graph | **Memory-bound** (elementwise passes over pixels) | Often overlooked. Multiplied by the number of tiles |
| 3 | Vision encoder (ViT) | NPU (INT8/FP16) | **Compute-bound** for the matmuls. **Memory-bound** for unfused elementwise ops and softmax | Cost ∝ tiles × FLOPs per tile |
| 4 | Projector (MLP, pixel-shuffle) | NPU | Negligible compute | Real cost is **graph breaks, layout transposes, dtype conversions, CPU round-trips** |
| 5 | Embedding merge, M-RoPE position IDs | CPU / NPU | Memory (copies) | Small, but avoid extra copies |
| 6 | LLM prefill | NPU | **Compute-bound** when the chunk is large enough, memory-bound for small chunks | Cost ∝ N_total × params |
| 7 | Decode | NPU (or GPU/CPU in some stacks) | **Memory-bandwidth-bound** | Reads all weights (+ KV) every token |
| 8 | Sampling, detokenization, UI streaming | CPU | Latency / sync overhead | Top-k/top-p over a 150k vocabulary on CPU costs a few ms per token if not fused |

### Roofline reasoning (why prefill is compute-bound and decode is not)
For a weight matmul with M tokens in flight:
- FLOPs = `2 · M · params`.
- Bytes read ≈ `params × bytes_per_weight`.
- Arithmetic intensity = `2M / bytes_per_weight` ops per byte (INT8: 2M, INT4: 4M).
- Ridge point = peak ops ÷ effective bandwidth. With 40 TOPS and about 50 GB/s of effective bandwidth, that is about **800 ops/byte**.

```python
peak_tops, eff_bw = 40e12, 50e9
ridge = peak_tops / eff_bw                       # ~800 ops/byte
for bytes_per_w, name in [(1.0, "INT8"), (0.5, "INT4")]:
    m_needed = ridge * bytes_per_w / 2           # tokens needed to become compute-bound
    print(name, round(m_needed))                 # INT8 ~400, INT4 ~200
```

- **Prefill and the vision encoder** have M in the hundreds to thousands, so they sit **above the ridge** (compute-bound). Chunked prefill with chunks smaller than about 200 to 400 tokens leaves the NPU under-used.
- **Decode** has M = 1. Its intensity is about 4 ops/byte for INT4, roughly **200x below the ridge**, so utilization is under 1% and speed is set purely by bytes read per second.
- **Elementwise ops** (LayerNorm, GELU/SiLU, softmax, residual adds, quantize/dequantize) are memory-bound and can take a large share of encoder time if they are not fused.
- Real utilization is typically **20 to 40% of the peak** even in the compute-bound stages.

### Decode upper bound
`tokens/s ≈ effective_bandwidth / bytes_read_per_token`.

For a 1.5B-class LLM at INT4 (about 0.74 GB of transformer weights) plus an INT8 LM head (about 0.23 GB), that is about 1.0 GB per token. At about 40 GB/s of effective bandwidth (shared with the CPU, GPU, ISP and display), the ceiling is about **40 tokens/s**. Real systems are lower after dispatch overhead, dequantization, KV reads and sampling, so figure 15 to 30 tokens/s.

## 2. Worked example (illustrative)
Assumptions: a SigLIP-B/16 encoder at 512 px (about 0.2 TFLOP per tile), 4 tiles + 1 thumbnail, pixel-shuffle to 64 tokens per tile, a 1.5B LLM (1.3B non-embedding), an effective 10 TOPS on the NPU, and decode at 25 tokens/s.

| Stage | Estimate | Reasoning |
|---|---|---|
| Image decode (HW, 12 MP) | ~25 ms | Fixed-function decoder |
| Preprocess (5 tiles) | ~10 ms | Memory-bound resize and normalize |
| Vision encoder | ~100 ms | 5 tiles × 0.2 TFLOP = 1 TFLOP at 10 TOPS |
| Projector + merge | ~3 ms | Small, if there are no graph breaks |
| Prefill | ~100 ms | ~380 tokens × 2 × 1.3B ≈ 1 TFLOP at 10 TOPS |
| First-token step | ~40 ms | One decode step |
| **TTFT** | **~280 ms** | |
| Decode of 29 more tokens | ~1.16 s | 29 × 40 ms |
| **E2E for a 30-token answer** | **~1.4 s** | Decode is about 80% of it |

**Takeaways:**
- With a sub-second E2E target (Q25), a 30-token answer does not fit at this decode speed. You must **shorten answers**, speed up decode (INT4, speculative decoding, smaller vocabulary), or stream the first tokens early to improve perceived latency.
- For video or many tiles, the encoder becomes the dominant term.
- Fit your own numbers with the microbenchmarks in section 3.

## 3. How to measure and profile

### 3.1 Metrics to collect
- TTFT, TPOT (inter-token latency), E2E, and **per-stage times**.
- **Percentiles (p50, p95, p99)**, not only the mean. Tail spikes come from DVFS, thermal throttling, GC, background load and memory pressure.
- **Cold vs warm:** model load, NPU graph compilation or initialization, page faults on memory-mapped weights, and clock ramp-up make the first inference much slower.
- **Sustained behaviour:** run for several minutes to see the throttled steady state (Q16).
- **Energy** per inference (power rails or fuel gauge), **peak memory** (RSS/PSS, DMA buffers), and **accuracy** measured in the same run.

### 3.2 Tools
- **System tracing:** Perfetto and systrace (with custom trace markers around each stage), simpleperf, Android Studio profiler. These show CPU threads, scheduling gaps and memory.
- **GPU/NPU vendor tools:** Qualcomm QNN profiling and AI Hub, Snapdragon Profiler, Arm Streamline, MediaTek NeuroPilot, Samsung's Exynos NPU tooling (check your SDK's profiler), Apple Instruments for Core ML/ANE.
- **Framework profilers:** ExecuTorch ETDump/Inspector, TFLite `benchmark_model` with op profiling, ONNX Runtime profiling, `llama-bench` (llama.cpp, reports prompt-processing and token-generation rates), MLC-LLM benchmarks.
- **Roofline check:** compare achieved TOPS and GB/s (from hardware counters) against the peak to see which stage is far from its limit.

### 3.3 Methodology (this is where most mistakes happen)
1. **Instrument stage boundaries** with monotonic timestamps or trace events.
2. **Synchronize with the accelerator.** NPU and GPU calls are often asynchronous. If you don't wait on the fence, you measure enqueue time, not execution time.
3. **Warm up** before timing, and measure cold start separately.
4. **Control the environment for microbenchmarks:** fixed clocks where possible, a known starting temperature, background apps controlled. Then repeat under **default DVFS and realistic load** (camera and ISP running) for the realistic numbers.
5. **Run at least 30 to 100 iterations** and report percentiles.
6. **Microbenchmark each stage in isolation:**
   - Encoder alone, with random input, at each supported resolution and tile count.
   - Prefill sweep over N = 64 to 2048 tokens (and with the chunk size you ship).
   - Decode sweep over context lengths 0 to 4k, to see the KV-read effect.
7. **Fit simple latency models** to feed the token-budget solver from Q2:
   - `T_prefill(N) ≈ a + b·N (+ c·N²)`
   - `T_decode(ctx) ≈ w + k·ctx`
8. **Trace end to end and look for gaps:** idle NPU time between stages (scheduling or copy overhead), **CPU fallbacks** from unsupported ops (one fallback inside the ViT can ruin the whole encoder), host-to-device copies, layout transposes, and contention with the camera pipeline.
9. **Use real prompts and images.** E2E depends on the output-length distribution, so benchmark with a representative workload.

## 4. Optimizations by stage

| Stage | Levers |
|---|---|
| **Image decode** | Hardware decoder. **Decode-time downscaling** (JPEG scaled decode at 1/2, 1/4, 1/8 is much cheaper than full decode then resize). Take a lower-resolution YUV stream from the camera/ISP instead of the full still. Zero-copy buffers (dmabuf / AHardwareBuffer). Overlap decode with model warm-up |
| **Preprocess** | Fuse resize, normalize and layout conversion into one pass. Use GPU, ISP scaler or NEON. Feed uint8 into the graph with the normalization folded into the first layer's weights. Pre-allocate buffers and avoid Java/Python copies. Tile on the GPU |
| **Vision encoder** | Fewer or adaptive tiles and lower resolution where the task allows (Q2). W8A8 on NPU kernels (Q11). Fused attention, windowed attention for large N. Replace unsupported ops to avoid CPU fallback. Static shape buckets. Token merging inside the ViT. A smaller or distilled encoder (Q10). **Cache encoder outputs** across turns, and skip near-identical video frames. **Pipeline** the encoder with prefill (see below) |
| **Projector** | Fuse into the tail of the encoder graph or the head of the LLM graph. Avoid CPU round-trips, extra layout transposes and dtype flips |
| **Prefill** | Fewer visual tokens (pixel-shuffle, pruning, Q2). **Chunk sizes matched to the compute-bound threshold** (a few hundred tokens). **Prefix caching** of the system prompt's KV. W8A8 so the integer MACs are used. Fused attention. KV reuse across turns. Overlap with the encoder |
| **Decode** | INT4 weights (fewer bytes per token). GQA/MQA and KV quantization (fewer KV bytes). **Smaller vocabulary or cheaper LM head** (Q1 deep dive). **Speculative decoding**, which raises arithmetic intensity by verifying several tokens per weight read (Q15). One fused decode graph per token to avoid per-op dispatch. **On-device argmax/sampling** to avoid CPU-NPU synchronization per token. Static in-place KV buffers. Weights resident and pre-faulted (no page faults) |
| **Output control** | Short, structured answers, constrained decoding, early stopping, **streaming the first tokens to the UI** for perceived latency |
| **System** | Precompiled NPU context binaries (no on-device JIT). Keep the model warm and resident (and plan for the OS low-memory killer). Lazy-load only what the feature needs. Performance hints or sustained-performance modes to avoid slow DVFS ramp-up. Control memory-bandwidth contention from the camera and display |

### Pipelining ideas
- **Encoder → prefill pipelining:** with a causal LLM, you can start prefilling visual tokens from tile 1 while the encoder still processes tile 2. This hides part of the encoder time.
- **Parallel work on other units:** tokenization, text embedding and position-ID computation on the CPU while the NPU runs the encoder.
- **Heterogeneous split:** some stacks run prefill on the NPU and decode on the GPU or CPU. Whether that wins depends on your runtime and memory bandwidth, so measure it.

## 5. Cold start and load time
- Loading a ~1.1 GB model from flash is on the order of a few hundred milliseconds to seconds. Add NPU context creation, and a possible compile step if there is no precompiled binary.
- **Mitigations:** precompiled context binaries, memory-mapped weights, loading the LLM and the vision tower lazily or in parallel, warm-up at app launch, and a plan for the OS killing background processes (then you pay the cold start again).
- Cold-start cost is a separate SLO from TTFT. Measure and report it separately.

## 6. Common pitfalls
- **Measuring async enqueue time** instead of execution time.
- **Mean-only reporting** that hides p99.
- **Marketing TOPS vs achieved TOPS.** INT4 doesn't give 2x over INT8 if the dequantization runs on a slower vector unit.
- **Silent CPU fallbacks** from a single unsupported op.
- **Non-zero-copy transfers and layout conversions** (NCHW/NHWC).
- **Padding waste** from static-shape buckets.
- **Latency variance from tile count.** Cap the number of tiles to bound tail latency.
- **Dev-board vs production-device differences** (thermal policy, memory configuration).
- **Benchmarking only the model,** and forgetting decode, preprocess and sampling.

## 7. Using the breakdown to set budgets
Start from the SLO (say E2E ≤ 1 s) and allocate: capture + preprocess about 50 ms, encoder about 150 ms, prefill about 150 ms, first token about 40 ms, which leaves about 600 ms of decode. At 40 ms per token that is about 15 tokens. If answers need to be longer, you must raise decode speed, cut the earlier stages, or stream and accept a longer total. This is how the numbers connect to the token budget in Q2 and the case study in Q25.

## 8. Strong interview answer
> "I split the pipeline into image decode, preprocess, vision encoder, projector, prefill and decode, and I report TTFT, TPOT and E2E with percentiles, separately for cold and warm runs. By a roofline argument, with a ridge point of about 800 ops per byte on a flagship SoC, prefill and the encoder have hundreds of tokens in flight, so they are compute-bound, while decode with one token per step is around 200x below the ridge and is memory-bandwidth-bound. Decode speed is roughly bandwidth divided by bytes read per token, so INT4 weights, GQA and KV quantization help directly. Preprocessing and unfused elementwise ops are memory-bound and often forgotten, and the projector's real cost is graph breaks and copies, not FLOPs. For a short answer, TTFT is dominated by the encoder and prefill. For anything longer than about 10 to 20 tokens, decode dominates, so answer-length control is a major lever. To measure, I'd instrument stage boundaries with trace markers, synchronize with the accelerator so I don't time enqueue, warm up, control or at least record clocks and temperature, take 30 to 100 runs for percentiles, microbenchmark each stage with sweeps over tiles, prompt length and context length, fit linear latency models, and look at Perfetto traces for NPU idle gaps and CPU fallbacks. Optimizations map by stage: scaled decode and zero-copy for image decode, fused preprocessing, fewer tiles and W8A8 with fused attention for the encoder, chunked prefill with prefix caching, INT4, fused decode graphs, on-device sampling and speculative decoding for decode, plus precompiled NPU binaries and warm models for the system. I'd budget stages from the SLO and verify everything on the real device under thermal throttling."

## 9. Follow-up questions to prepare for
1. Why is decode memory-bandwidth-bound while prefill is compute-bound? Show it with arithmetic intensity.
2. What chunk size would you pick for prefill on an NPU with a ridge point of about 800 ops/byte, and what limits it from above?
3. How can the projector take a significant share of time even though it has few FLOPs?
4. How do you make sure you are not measuring asynchronous enqueue instead of execution time?
5. How would you detect and eliminate a CPU fallback inside the vision encoder?
6. Why can per-token CPU sampling hurt decode speed, and how do you fix it?
7. How would you pipeline the vision encoder with LLM prefill, and what does causality allow?
8. How do you separate cold-start latency from steady-state latency in your reporting?

---

# Q13: Deployment stack choices

**Question:** Compare ExecuTorch, ONNX Runtime (with QNN or NNAPI EPs), TFLite/LiteRT, llama.cpp/GGUF, MLC-LLM, TensorRT-LLM and the vendor SDKs (Qualcomm QNN/SNPE, Samsung Exynos NPU / ONE toolchain). How would you export a VLM (dynamic shapes, multiple graphs for the encoder and decoder, KV-cache as I/O)? What breaks during export (unsupported ops, dynamic control flow, rotary embeddings), and how do you work around it?

This space changes fast. The descriptions below are from memory and represent the stacks roughly as of 2025. Anything I mark "(verify)" should be checked against the current docs before you rely on it in an interview.

## 0. How to think about the choice
A deployment stack answers four questions:
1. **Which hardware does it reach** (CPU, GPU, NPU, which vendor)?
2. **How does the model get in** (export to a graph, or run eagerly)?
3. **Does it handle the LLM loop** (KV-cache, sampling, chunked prefill) and multimodal inputs?
4. **How much control and maintenance** does it need (static shapes, quantization recipe, operator coverage)?

NPUs give the best latency and energy, but they need static shapes, constrained operators and vendor toolchains. CPUs and GPUs are flexible but slower and use more power. Production systems usually need **both**: an NPU fast path and a fallback.

## 1. The stacks

### 1.1 ExecuTorch (PyTorch Edge)
- **Flow:** `torch.export` → edge dialect → lower to **delegates** → a `.pte` file → a small C++ runtime.
- **Backends:** XNNPACK (CPU), Qualcomm QNN, MediaTek, Core ML / MPS (Apple), Vulkan (GPU), Arm Ethos-U, and a Samsung Exynos delegate (verify current status).
- **LLM and VLM:** has Llama-family examples with KV-cache, 4-bit group-wise quantization (torchao), QAT/SpinQuant recipes, and a LLaVA example.
- **Pros:** PyTorch-native end to end, ahead-of-time memory planning, a small runtime, and custom-op and partitioner hooks. A good fit if you want one export path to several backends.
- **Cons:** newer, delegate coverage varies, and `torch.export` is strict about dynamic control flow and data-dependent shapes.

### 1.2 ONNX Runtime (ORT)
- **Flow:** export to ONNX (`torch.onnx.export`), then run with an execution provider (EP).
- **EPs:** QNN EP (Qualcomm HTP), NNAPI (Android, being phased out in favor of other paths, verify), XNNPACK, Core ML, DirectML, CUDA/TensorRT, OpenVINO.
- **LLM:** the ORT GenAI library handles the generation loop and KV-cache, with multimodal support for some models (e.g. Phi vision models). `MatMulNBits` provides 4-bit weights, and Olive automates optimization.
- **Pros:** portable, mature tooling, strong on Windows and cross-platform.
- **Cons:** the QNN EP wants **static shapes and QDQ-quantized models**. Unsupported ops cause graph partitions and CPU fallbacks. The 2 GB protobuf limit needs external-data files.

### 1.3 TFLite / LiteRT
- **Flow:** convert to TFLite (for PyTorch, `ai_edge_torch` via `torch.export`/StableHLO). Google renamed TFLite to **LiteRT**.
- **Delegates:** XNNPACK, GPU (OpenCL/OpenGL/Metal), vendor NPU delegates (Qualcomm, MediaTek, and historically Samsung's), NNAPI (deprecated).
- **LLM and VLM:** `ai_edge_torch.generative` builds LLMs with the KV-cache as inputs/outputs and separate prefill and decode **signatures**. MediaPipe LLM Inference API / LiteRT-LM serve Gemma-family models, including multimodal ones (verify).
- **Pros:** huge Android footprint, a good GPU delegate, Google-maintained.
- **Cons:** conversion pain for new architectures, op coverage gaps, and the FlatBuffers 2 GB limit for large models.

### 1.4 llama.cpp / GGUF
- **Flow:** no export. A C/C++ runtime executes the model eagerly from a **GGUF** file (k-quants such as Q4_K_M, i-quants).
- **Backends:** CPU (NEON, i8mm, SVE), Metal, Vulkan, OpenCL (with an Adreno-optimized path), CUDA. NPU support is limited and experimental, including a Hexagon backend (verify).
- **Multimodal:** vision encoders run through the `mtmd`/`mmproj` path, with many VLMs supported (Qwen-VL family, Gemma, SmolVLM, InternVL and others).
- **Pros:** the fastest way to run a new model, natively dynamic shapes, mmap loading, speculative decoding, grammar-constrained sampling, prefix caching, MIT license.
- **Cons:** mostly CPU/GPU, so higher power than an NPU. The vision encoder is less optimized. Not a static-graph path.

### 1.5 MLC-LLM (Apache TVM)
- **Flow:** compile the model with TVM into model-specific kernels (auto-tuned) for Vulkan, OpenCL, Metal, CUDA and WebGPU. Packages Android and iOS apps.
- **Pros:** strong GPU performance on phones (Adreno, Mali), cross-platform, paged KV-cache, quantization formats like `q4f16`.
- **Cons:** a complex compile pipeline, a smaller ecosystem, and **no mainstream NPU path**.

### 1.6 TensorRT-LLM
- **Flow:** build TensorRT engines (a visual engine plus an LLM engine for VLMs), with in-flight batching, paged KV, FP8/INT4 (AWQ, GPTQ) and speculative decoding.
- **Targets:** **NVIDIA GPUs only**, including Jetson-class edge modules and the cloud.
- **Fit:** robots or devices built on Jetson, and the **server-side tier** for cloud fallback (Q17).
- **Cons:** vendor lock-in, engines are rebuilt per GPU and TensorRT version, and it is not for phone SoCs.

### 1.7 Vendor SDKs

**Qualcomm (QNN / Genie / AI Hub; SNPE is the legacy predecessor):**
- QNN (AI Engine Direct) targets the **HTP** (Hexagon NPU), GPU and CPU. You convert models, quantize (PTQ, plus **AIMET** for advanced PTQ and QAT), and compile offline to a **context binary**.
- Qualcomm AI Hub compiles and profiles on real devices. **Genie** is an LLM runtime on top of QNN that handles the KV-cache and token loop.
- Constraints: **static shapes**, fixed-point-oriented numerics (INT8 or INT16 activations, INT4/INT8 weights, i.e. "W4A16" in their terminology), limits on graph and context-binary size (large models are split into several parts), and **weight sharing** between graphs (prefill and decode variants) to avoid duplicating weights.

**Samsung (Exynos NPU SDK and the ONE toolchain):**
- **Exynos AI Studio / ENN SDK** for the Exynos NPU: convert (TFLite- or ONNX-based flows), compile for the NPU and run through the ENN runtime on Android (verify current names and versions).
- **ONE (On-device Neural Engine)** is Samsung Research's open-source compiler and runtime toolchain (the Circle IR, `one-cc`, quantization tools, and the `onert` runtime). It targets on-device and embedded products and appears aimed at devices such as TVs and appliances (verify scope).
- **Why this matters for a Samsung role:** Galaxy devices ship with **both Snapdragon and Exynos** variants depending on region and model, and TVs and appliances use other SoCs. A production team needs a portable approach (section 5).

**Others:** MediaTek NeuroPilot, Apple Core ML (ANE), and Google's AICore / Gemini Nano APIs on supported phones.

### Summary table

| Stack | Entry point | Hardware | LLM/VLM loop built in | Dynamic shapes | Strength | Weakness |
|---|---|---|---|---|---|---|
| ExecuTorch | `torch.export` → `.pte` | CPU, GPU, several NPUs via delegates | Examples + runner | Limited, AOT planned | One PyTorch path, small runtime | Newer, export strictness |
| ONNX Runtime | ONNX | CPU, GPU, QNN EP, others | ORT GenAI | Yes on CPU, static on QNN | Portable, mature | Partitioning, QDQ and static shapes for NPU |
| LiteRT/TFLite | TFLite | CPU, GPU, vendor NPUs | MediaPipe / LiteRT-LM | Limited | Android reach | Conversion pain |
| llama.cpp | GGUF | CPU, GPU (Metal/Vulkan/OpenCL/CUDA) | Yes | **Yes** | Fast bring-up, broad model support | Mostly no NPU, higher power |
| MLC-LLM | TVM compile | GPU (Vulkan/OpenCL/Metal/...) | Yes | Yes (compiled symbolic) | Good mobile GPU speed | Complexity, no NPU |
| TensorRT-LLM | TRT engines | NVIDIA only | Yes | Engine profiles | Top NVIDIA performance | Lock-in |
| Qualcomm QNN / Genie | Convert + compile | Hexagon NPU, GPU | Genie (LLM) | **Static** | Best Snapdragon NPU efficiency | Vendor-specific, static |
| Samsung ENN / ONE | Convert + compile | Exynos NPU, embedded targets | Varies (verify) | Mostly static | Samsung silicon and devices | Vendor-specific, smaller community |

## 2. Choosing a stack (decision framework)
- **Phone, best latency and energy:** vendor NPU path (QNN on Snapdragon, ENN on Exynos), possibly through ExecuTorch delegates or ONNX/TFLite as the interchange.
- **Fast prototyping and wide model coverage:** llama.cpp (GGUF), or PyTorch eager first.
- **Phone GPU fallback:** llama.cpp (OpenCL/Vulkan) or MLC-LLM.
- **Jetson robots and the cloud tier:** TensorRT-LLM (and vLLM for general serving).
- **TV and appliance:** check the SoC. Often CPU or a small NPU, with the ONE/onert toolchain as an option.
- **Other criteria:** team skills, time to market, license, update path (OTA of models and adapters), and how many vendors you must support.

**A realistic staged strategy:**
1. Prototype and validate accuracy in PyTorch eager and llama.cpp.
2. Build the production NPU path per vendor (or via a shared export).
3. Keep a CPU/GPU fallback for devices without a usable NPU.
4. Keep a cloud tier (TensorRT-LLM / vLLM) for hybrid routing (Q17).

## 3. How to export a VLM

### 3.1 Split into several graphs
A VLM is not one graph. A typical deployment uses:

| Graph | Inputs | Outputs | Notes |
|---|---|---|---|
| **Vision encoder (+ projector)** | `pixel_values [tiles, 3, H, W]` (fixed bucket) | visual embeddings `[tiles × tokens, D_llm]` | The projector can be fused into this graph |
| **Token embedding** | token IDs | embeddings | Often a CPU gather on a memory-mapped table, saving NPU memory |
| **LLM prefill** (one or several chunk sizes) | `inputs_embeds [1, chunk, D]`, position data (or cos/sin), attention mask, KV-cache in | KV-cache out, logits for the **last** position only | Skipping the LM head on other positions saves compute |
| **LLM decode** | one token's embedding, position data, KV-cache | logits `[1, V]`, updated KV | Sequence length 1 |

- **Merging visual and text embeddings happens on the host** (CPU), by writing the visual embeddings into `inputs_embeds` at the image-placeholder positions. This avoids dynamic scatter inside the graph.
- The generation loop (sampling, stopping, streaming) lives in host code.
- **Share weights across the prefill and decode graphs** (ExecuTorch multi-method programs and QNN weight sharing, verify specifics), otherwise you store the LLM weights several times.

### 3.2 KV-cache as I/O
Options:
1. **Functional I/O:** pass the cache in and out each step. Simple, but copying a large cache every step is expensive.
2. **Mutable state or in-place update:** the graph updates a buffer in place (ExecuTorch mutable buffers, or zero-copy shared buffers in vendor runtimes). Needs delegate support.
3. **Static fixed-capacity cache:** a buffer `[layers, 2, kv_heads, max_ctx, head_dim]` plus a write index and a mask for unused positions. No dynamic shapes, but attention then costs `max_ctx` every step, so compile **several context buckets** (e.g. 512, 1,024, 2,048, 4,096) and switch between them.

Details that matter:
- Writing into the cache at an offset uses `dynamic_update_slice`/`index_copy`/`scatter`, which may be unsupported on an NPU. Workarounds are a concat-and-slice scheme, a one-hot formulation, or fixed chunk positions with masking.
- NPU-friendly layouts often store K transposed (`[head_dim, seq]`).
- Quantized KV needs static scales per layer and head.
- For long-running sessions, use a ring buffer or sliding window.

### 3.3 Dynamic shapes: design them out
- **Static by design:** fixed tile size and tile-count buckets for the encoder, fixed prefill chunk sizes, decode at sequence length 1, and a maximum context. The host picks the right graph at runtime.
- Use truly dynamic shapes only where the backend supports them (CPU or GPU paths, ORT CPU). In `torch.export`, bound them with `Dim(min, max)`.
- **Variable number of visual tokens:** pad to a bucket and mask (padded tokens still cost compute), or export one graph per bucket.
- **Native-resolution encoders** (Qwen2-VL style): export several resolution buckets, and replace variable-length attention (`cu_seqlens` / flash-attn varlen) with a block-diagonal or windowed mask per bucket.
- Precompute position tables (2D RoPE, interpolated position embeddings) per bucket (Q3).

## 4. What breaks during export, and the workarounds

| Problem | Symptom | Workaround |
|---|---|---|
| **Unsupported ops** (gather/scatter, `topk`, `cumsum`, `einsum`, bicubic `interpolate`, `unfold`, `pixel_shuffle`, `grid_sample`, `erf`-based GELU, fused SDPA) | Graph partitions, CPU fallbacks, compile errors | Rewrite using supported primitives (matmul, add, softmax, reshape, transpose, concat, slice). Precompute things offline (interpolated position embeddings). Decompose SDPA into matmul + softmax. Move unsupported ops to the host. Add custom ops or delegate kernels. Check the op-coverage report |
| **Dynamic control flow and data-dependent shapes** (`if tensor`, `.item()`, `nonzero`, `masked_select`, loops over values) | `torch.export` fails, or `jit.trace` silently bakes in one branch | Refactor to static code, unroll loops with constant bounds, use `torch.cond` where supported, and move decisions (tile selection, stopping) to the host |
| **Rotary embeddings** | Garbage output, NaNs, or slow graphs | See below |
| **Attention masks** | NaNs, wrong attention | Precompute or pass masks as inputs. Use a large finite negative instead of `-inf` (FP16/INT can't hold -1e9). Avoid fully masked rows in softmax (NaN) |
| **Embedding merge** (`masked_scatter`, `index_put` at image-token positions) | Dynamic-shape or unsupported op errors | Do it on the host |
| **Large models** | Protobuf or FlatBuffers 2 GB limits, export OOM, duplicated tied weights | External-data files, splitting the model into parts (QNN context binaries are also split), de-duplicating tied embedding/LM-head weights |
| **Numerical issues** | FP16 overflow, RMSNorm/LayerNorm instability | Scale q before the matmul, compute norms' sum of squares in FP32, double-check eps, and keep the quantization recipe in mind (Q11) |
| **Custom model code** (`trust_remote_code`, flash-attn varlen, `einops`) | Export fails | Replace it with an export-friendly reference implementation |
| **Preprocessing parity** | Accuracy drops on device only | Match the reference image processor exactly: interpolation and antialiasing, mean/std, RGB vs BGR, EXIF orientation, tiling rules, chat template |
| **Memory lifecycle** | OOM from several resident graphs | Memory planning, time-multiplexing the encoder and LLM (load, run, unload), and sharing weights |

### Rotary embeddings in more detail
1. **Precompute cos/sin tables in FP32** (host or constants), and feed the slice for the current positions as graph inputs. This avoids `arange`/`pow`/gather inside the graph and avoids FP16 precision loss at large positions.
2. **Avoid complex-number implementations** (`view_as_complex`, `polar`). Use the real-valued rotate-half form.
3. **Match the weight layout:** the HF and original Llama layouts differ (the q/k projection weights are permuted in conversions). Mixing a layout and a rotate-half convention gives silently wrong output.
4. **Scaling variants** (YaRN, NTK, llama3 scaling): compute the inverse frequencies offline.
5. **M-RoPE** (Qwen2-VL): build the three-section position IDs on the host, produce the final cos/sin, and track `rope_delta` for decode (Q3).
6. **2D RoPE in the ViT:** precompute row and column tables per resolution bucket.
7. **Don't quantize the tables.**

### Verification process
- **Layer-wise comparison** against PyTorch eager (e.g. cosine similarity above 0.99 per layer in FP16).
- **End-to-end greedy decode match** for the first N tokens on a test set.
- **Op-coverage and partition report** after every change, and a performance baseline.
- A common practice is an **export-friendly rewritten model definition** kept in sync with the reference through tests (as in Meta's Llama export code or `ai_edge_torch.generative`). Some NPU toolchains also prefer rewriting linear layers as 1×1 convolutions with specific layouts (verify for your target).

## 5. Multi-vendor strategy (relevant for Samsung)
- Galaxy devices use both Snapdragon and Exynos, TVs and appliances use other SoCs, and robots may use yet another platform.
- **Single source of truth:** one export-friendly model definition and one evaluation harness.
- **Per-target lowering recipes:** each vendor gets its own conversion, quantization (Q11) and shape-bucket configuration, and a capability matrix (ops, precisions, max graph size).
- **A fallback ladder:** NPU → GPU → CPU → cloud.
- **Shared CI:** golden vectors, accuracy, hallucination (Q9) and latency gates run per target on real devices.
- **OTA updates:** versioned models, adapters and compiled binaries per device class.

## 6. Strong interview answer
> "I choose the stack by hardware reach, how the model gets in, and whether it handles the LLM loop. For phone NPUs the best latency and energy come from the vendor toolchains, QNN with Genie on Snapdragon and the Exynos NPU SDK on Exynos, which need static shapes, fixed-point-friendly quantization and offline compilation. ExecuTorch and ONNX Runtime give a shared export path with vendor delegates or EPs. LiteRT is strong on Android GPU and Google's LLM APIs. llama.cpp is the fastest bring-up with native dynamic shapes, but it mostly runs on CPU or GPU. MLC-LLM is good on mobile GPUs, and TensorRT-LLM covers NVIDIA, i.e. Jetson robots and the cloud fallback. I'd prototype in eager PyTorch or llama.cpp, build the NPU path per vendor, and keep a GPU or CPU fallback. For export, I'd split the VLM into an encoder plus projector graph, a token-embedding lookup on the host, prefill graphs at fixed chunk sizes, and a decode graph, sharing weights. The embedding merge, sampling and the generation loop run on the host. KV-cache is a static, fixed-capacity buffer passed as I/O or updated in place, with a few context buckets. I'd design out dynamic shapes with buckets and masks. The things that break are unsupported ops, which I rewrite to primitives or move to the host, data-dependent control flow, which I make static or move to the host, and rotary embeddings, where I precompute FP32 cos/sin tables, avoid complex numbers, match the weight layout, and handle M-RoPE and `rope_delta` on the host. I'd validate with layer-wise comparisons and end-to-end greedy-token matches, check preprocessing parity, and gate per target on real devices, since Samsung ships both Snapdragon and Exynos."

## 7. Follow-up questions to prepare for
1. Why do NPU toolchains require static shapes, and how do buckets and masks work around that?
2. What are the trade-offs between functional KV-cache I/O, in-place mutable buffers, and a static fixed-capacity cache?
3. Why compute the LM head only for the last position in prefill?
4. How would you handle a vision encoder with variable-length attention (`cu_seqlens`) on an NPU?
5. What silent bugs can come from mixing RoPE weight layouts and rotate-half conventions?
6. When would you choose ExecuTorch over ONNX Runtime, or the reverse, for a multi-vendor product?
7. How does weight sharing across prefill and decode graphs work, and what does it save?
8. How would you structure CI so that Snapdragon, Exynos and CPU fallback paths all stay correct after each model change?

---

# Q14: Memory budgeting on a phone

**Question:** The device has 12 GB RAM shared with the OS and other apps, and your VLM gets about 2–3 GB. Compute the memory for weights (INT4), KV-cache (layers × heads × dim × seq × bytes), vision activations and the image buffer. How would you use mmap/lazy loading, KV-cache eviction, sliding-window or sink attention, and GQA/MQA to fit? What are the OOM and thermal risks, and how do you handle them gracefully?

Model dimensions are those of public models (Qwen2.5-1.5B and 3B, Llama-3.2-1B, Llama-3.1-8B) as I remember them, and OS behavior is Android-centric. Treat all numbers as illustrative and measure on your own device.

## 0. "12 GB" is not 12 GB for you
- The OS, system services, GPU/ISP/modem carve-outs and other apps use a big share. Your app might realistically see **about 3 to 5 GB** before the system starts killing things, and a VLM feature gets a fraction of that.
- **Android's low-memory killer (lmkd)** kills processes under memory pressure, based on PSI (pressure stall information) and each process's priority. A **kill is silent**: no exception, the process just disappears. Foreground apps are safer but not immune.
- **Memory types behave differently:**
  - **Anonymous memory** (malloc'd weights, KV-cache, activations): can't simply be dropped. Android compresses it to zram, but INT4 weights are essentially incompressible, so zram doesn't help.
  - **File-backed mmap pages:** clean pages are reclaimable and re-readable from flash, so they are cheaper for the OS, but re-reading is slow (see section 3).
  - **dma-buf / ION / NPU buffers:** allocated from shared system heaps, accounted separately, and often limited by the NPU driver (contiguous or pinned memory, virtual-address limits).
- **Budget per RAM class** (a policy, not a fact): an 8 GB phone might get about 1.5 to 2 GB, a 12 GB phone about 2 to 3 GB, a 16 GB phone about 3 to 4 GB. Test on the **lowest RAM tier** you ship.

## 1. The memory components, with formulas and numbers

### 1.1 Weights
`bytes = params × bits / 8`, plus quantization overhead. With group size 128 and FP16 scales (and zero-points) add about 0.125 to 0.25 bits per weight, so use about **4.25 to 4.5 bits** for INT4 g128.

**Example: ~2B VLM** (Qwen2.5-1.5B-class LLM, SigLIP-B-class encoder, pixel-shuffle projector):

| Component | Size |
|---|---|
| LLM non-embedding, 1.31B × 4.5 bits | 0.74 GB |
| Embedding / LM head (tied), 233M × INT8 | 0.23 GB |
| Vision encoder, ~93M × INT8 | 0.09 GB |
| Projector, ~13M × FP16 | 0.03 GB |
| **Total weights** | **≈ 1.1 GB** (vs about 3.8 GB in FP16) |

**Weight-memory traps (double residency):**
- Keeping **both** a file-backed mmap copy and a repacked copy in NPU memory (2x). Drop the file pages after upload (`madvise(MADV_DONTNEED)` / unmap), or ship weights in the device-ready layout.
- **Separate prefill and decode graphs** that don't share weights (2 to 3x). Use weight sharing (Q13).
- **Tied weights duplicated** by an exporter (+0.23 GB).
- Temporary FP16 or FP32 staging buffers at load time. **Peak memory occurs at load**, not only at steady state.

### 1.2 KV-cache
`bytes/token = 2 (K,V) × layers × kv_heads × head_dim × bytes_per_element`.

| Model | Layers × KV heads × head_dim | KV per token (FP16) | 4k context FP16 | 4k context INT8 |
|---|---|---|---|---|
| Qwen2.5-1.5B (GQA) | 28 × 2 × 128 | 28 KB | ~117 MB | ~59 MB |
| Same model if it used MHA (12 KV heads) | 28 × 12 × 128 | 168 KB | ~705 MB | ~350 MB |
| Llama-3.2-1B (GQA) | 16 × 8 × 64 | 32 KB | ~134 MB | ~67 MB |
| Llama-3.1-8B (GQA) | 32 × 8 × 128 | 128 KB | ~537 MB | ~268 MB |

**Key points:**
- **GQA/MQA cuts KV by the ratio of query heads to KV heads** (6x in the example above). Pick a GQA backbone for edge.
- On-device runtimes usually allocate a **static buffer at `max_ctx`**, not the used length. If the default is 32k, Qwen2.5-1.5B pays about 0.94 GB. **Set `max_ctx` to what the feature actually needs** (e.g. 2k to 4k).
- Several context buckets should **share one buffer** sized for the largest bucket (Q13), not one buffer each.
- **Visual tokens:** 5 tiles × 64 tokens + text is only a few hundred tokens, so the cost is small. **Video is where it grows:** 30 s at 8 fps × 64 tokens = 15,360 tokens ≈ 440 MB in FP16 for this model.

### 1.3 Vision activations
- With a static memory planner (ExecuTorch AOT planning, TFLite arena, ORT memory pattern, NPU graph memory), **peak = the largest set of simultaneously live tensors**, not the sum.
- **Example, SigLIP-B/16 at 512 px** (N = 1,024 tokens, D = 768, 12 heads): the hidden state is about 1.5 MB, the MLP intermediate (3,072) about 6.3 MB, and **unfused attention scores 12 × 1,024² × 2 B ≈ 25 MB** per layer (reused across layers). Fused attention removes the 25 MB term.
- Encoding tiles **one at a time** (batch 1) keeps the peak flat. Batching tiles multiplies it.
- NPU graph "arenas" are allocated at graph initialization, and can be tens to hundreds of MB. Check them in the compiler's memory report.
- The encoder output (e.g. 5 × 64 × 1,536 × 2 B ≈ 1 MB) is negligible.

### 1.4 LLM activations and logits
- Prefill with a 256-token chunk: hidden states about 0.8 MB, MLP intermediate (8,960) about 4.6 MB, and unfused attention scores 12 heads × 256 × 4,096 × 2 B ≈ 25 MB.
- **Logits are the hidden trap:** `256 × 151,936 × 4 B ≈ 156 MB` if you compute the LM head for every position. Compute it **only for the last position** (about 0.6 MB), as in Q13.

### 1.5 Image buffers
- 12 MP RGBA bitmap: 4,000 × 3,000 × 4 B = **48 MB**. YUV420 from the camera is 18 MB.
- Five 512×512 tiles as normalized FP32: 5 × 512 × 512 × 3 × 4 B ≈ **15.7 MB** (FP16 about 7.9 MB, uint8 about 3.9 MB).
- The camera pipeline holds several frames in flight (a ring of buffers).
- **Reduce:** decode-time downscaling (`inSampleSize` / scaled JPEG decode) so you never hold the full bitmap, reuse buffers, feed uint8 into the graph, zero-copy hardware buffers, and release the original promptly.

### 1.6 Runtime and miscellaneous
- Runtime and framework state (ExecuTorch is small, ONNX Runtime tens of MB, llama.cpp compute buffers can be hundreds of MB for large batch sizes), tokenizer vocabulary (about 10 to 20 MB), NPU driver and context memory, thread stacks, sampling buffers, UI and app overhead.
- Add **10 to 20% margin** for allocator fragmentation and peaks.

### Worked budget (illustrative)

| Item | Size |
|---|---|
| Weights | 1.1 GB |
| KV-cache (4k context, INT8) | 0.06 GB (FP16: 0.12 GB) |
| Vision activations arena | 0.10 to 0.20 GB |
| LLM activations arena | 0.05 to 0.10 GB (the encoder and LLM arenas can be **shared**, since they don't run at the same time) |
| Image buffers | 0.05 to 0.10 GB |
| Runtime, tokenizer, misc | 0.15 to 0.25 GB |
| NPU context / driver | 0.10 to 0.30 GB |
| 15% margin | ~0.25 GB |
| **Total** | **≈ 1.9 to 2.4 GB**, fits a 2 to 3 GB budget |

**Will a bigger model fit?** A 3B-class LLM (Qwen2.5-3B-like): non-embedding 2.77B × 4.5 bits ≈ 1.56 GB, embedding/LM head 311M × INT8 ≈ 0.31 GB, plus encoder and projector ≈ 0.12 GB → about 2.0 GB of weights, and **about 2.8 GB total**, which is tight against a 3 GB budget and risky.

```python
def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per=2):
    return 2 * layers * kv_heads * head_dim * bytes_per

def weights_gb(params_b, bits=4.5):
    return params_b * 1e9 * bits / 8 / 1e9

llm   = weights_gb(1.31)                    # ~0.74 GB
embed = 233e6 * 1 / 1e9                     # ~0.23 GB (INT8)
vit, proj = 0.09, 0.03
kv    = 4096 * kv_bytes_per_token(28, 2, 128, 1) / 1e9   # INT8 KV, ~0.06 GB
fixed = 0.55                                # activations + images + runtime + NPU ctx (assumed)
total = (llm + embed + vit + proj + kv + fixed) * 1.15
print(f"{total:.2f} GB")                    # ~2.1 GB
```

## 2. Techniques to fit, by component

**Weights**
- INT4 g128 with sensitive layers at INT8 (Q11), a de-duplicated tied embedding, weight sharing across graphs.
- **Vocabulary pruning** shrinks both the embedding and the LM head (Q1 deep dive, section 5).
- If the model is bigger than the budget: **flash offloading** ideas (windowed loading, as in Apple's "LLM in a flash", or PowerInfer-2-style neuron-level streaming) trade latency for memory. They are a last resort, not for a ≤2B model.

**KV-cache**
- GQA/MQA backbone, INT8 or INT4 KV (Q11), and `max_ctx` set to the product need.
- **Sliding-window attention:** only the last W tokens are cached on local layers. Gemma 2/3 interleave local and global layers (Gemma 3 uses about 5 local layers per global one with a 1,024-token window), which cuts KV a lot.
- **Attention sinks (StreamingLLM):** keep the first few "sink" tokens plus a recent window and evict the middle. It allows indefinite streaming but loses the middle content, which is acceptable for continuous video chat but not for "what did I see 2 minutes ago".
  - Detail: with eviction, positions need care. Store keys **before RoPE** and apply rotation using positions within the cache, or keep original positions consistently.
- **Heavy-hitter eviction (H2O, SnapKV, Scissorhands):** keep tokens that received the most attention, plus recent ones. Prompt-KV compression for long prompts.
- **For visual tokens:** after the model has answered, drop or compress the image's KV. Because the visual **embeddings** are tiny (320 tokens × 1,536 × 2 B ≈ 1 MB, versus about 9 MB of KV), you can keep embeddings and **re-prefill** if the user asks a follow-up.
- **Video:** a fixed token budget per frame and a rolling window of K frames, so memory is bounded by design (Q22).
- **Paged KV (vLLM-style):** avoids fragmentation and helps multi-session serving. On a single-user phone, a static buffer is simpler.

**Activations**
- A static memory planner, **chunked prefill** (smaller chunks, smaller activations), tile-by-tile encoding, **fused attention** (no N² score matrix), in-place ops, lower-precision activations, **logits for the last position only**, and **one shared arena** for graphs that never run concurrently (the arena size is the max, not the sum).

**Process level**
- One runtime instance in one process, no duplicate model copies, native buffers instead of Java bitmap copies, large allocations through anonymous `mmap` so they return to the OS when freed, and a scalable allocator (jemalloc/scudo) to limit fragmentation.

## 3. mmap and lazy loading, with the latency catch

**What mmap gives you:**
- Weights are mapped read-only from the file, and pages load on demand. Clean pages are **reclaimable**, so the OS can free them instead of killing you.
- No staging copy at load time, and fast startup.

**The catch for LLM decode:** decode reads **all** weights every token. If the working set doesn't stay in RAM, every token page-faults from flash. At about 1 GB per token and flash reads of a few GB/s (UFS 4-class), that is roughly 3 tokens/s or worse, a latency disaster. So:
- **Hot weights** (LLM layers, LM head): keep resident. Pre-fault them at load (`MAP_POPULATE`, `madvise(MADV_WILLNEED)`, or touch pages), or copy into anonymous or device memory. `mlock` is restricted for apps (RLIMIT_MEMLOCK).
- **Cold or rarely used data** (embedding rows for input lookup when untied, vision tower while idle, LoRA adapters, alternate-language resources): mmap and let them be paged.
- Pre-fault also removes **first-token latency spikes** from cold page faults (Q12).

**Lazy loading of components:**
- Load the vision tower only when a camera feature is invoked, and unload it afterwards (**time-multiplexing**). The benefit is small for an INT8 90M encoder (about 0.09 GB), but meaningful for a 400M encoder (about 0.4 GB). Weigh it against the reload latency.

## 4. OOM risks and graceful degradation

### 4.1 Risks
- **Silent kill by lmkd** under memory pressure. In-process allocation failures (`ENOMEM`, `std::bad_alloc`) are catchable, but a kernel or lmkd `SIGKILL` is not.
- **Peak vs steady state:** load-time staging, the largest prefill chunk, and camera frames in flight can coincide. Budget for the **peak**.
- **Fragmentation and slow leaks** during long sessions.
- **Background state:** a backgrounded app is killed first. Reloading means paying the cold start again (Q12).
- **Co-resident work:** the camera, ISP and other apps compete for memory and bandwidth.

### 4.2 Signals to watch
- Android: `ComponentCallbacks2.onTrimMemory` levels, `ActivityManager.getMemoryInfo()` (`availMem`, `lowMemory`), native heap size, PSS, and PSI where accessible (`/proc/pressure/memory`, restricted on recent versions). iOS: `os_proc_available_memory()` and memory warnings.

### 4.3 Graceful-degradation ladder
1. **Pre-flight check** when the feature starts: read available memory and choose a configuration tier.
   - **Tier A (full):** up to 6 tiles, 4k context, INT8 KV.
   - **Tier B (reduced):** 2 tiles, 2k context.
   - **Tier C (minimal):** one tile, 1k context.
   - **Fallback:** cloud, or a clear "feature unavailable" message.
2. **On trim-memory callbacks:** free caches (encoder-output cache, prefix cache, bitmap pools), unload the vision tower, evict or shrink the KV-cache, and cancel low-priority work.
3. **On backgrounding:** unload the model (or keep it only if policy allows).
4. **On allocation failure:** catch it, retry at a lower tier, and never crash.
5. **Persist conversation state** (text and compact summaries) so a kill leads to a fast restore.
6. **Admission control:** one inference at a time, with a queue.
7. **Consider a separate service process:** a kill or crash doesn't take down the UI app. Image transfer then needs shared memory or hardware buffers, since IPC copies are expensive.

### 4.4 Testing
- Test on the **lowest RAM tier** and under induced pressure (a memory-hog app).
- **Soak tests:** thousands of inferences, tracking the PSS and native-heap **slope** to catch leaks and fragmentation.
- Tools: `dumpsys meminfo <pid>` (native heap, graphics), `smaps`/`smaps_rollup`, Perfetto memory counters, `heapprofd` for native allocations, and dma-buf dumps for NPU and camera buffers.

## 5. Thermal risks and memory–thermal coupling
- Sustained NPU and memory traffic heats the SoC, and the OS throttles clocks, which can make latency **2x or worse** over minutes (full treatment in Q16).
- DRAM traffic is a major energy cost, so **less memory traffic means less heat**. INT4 weights, KV quantization, GQA and fewer visual tokens all help.
- **Respond to thermal state:** Android's `PowerManager.getCurrentThermalStatus()` and `OnThermalStatusChangedListener`, and `getThermalHeadroom()` forecasts. Reactions include fewer tiles or lower resolution, shorter outputs, a lower frame rate for continuous video, a duty cycle with pauses, deferring background work, and routing to the cloud.
- Also handle battery saver and low-power modes.

## 6. Strong interview answer
> "I'd budget by component, for the peak, not the steady state. Weights are params × bits/8 plus group-scale overhead: for a ~2B VLM that's about 0.74 GB for the INT4 LLM, 0.23 GB for an INT8 tied embedding and LM head, and about 0.12 GB for encoder and projector, so about 1.1 GB. KV is 2 × layers × KV heads × head_dim × bytes per token, about 28 KB per token for a Qwen2.5-1.5B-style GQA model, so about 117 MB at 4k in FP16 and half that in INT8. It matters mainly for video and for non-GQA models, and runtimes allocate at max_ctx, so I'd set max_ctx to the feature's need. Vision activations come from a static planner. I'd encode tiles one at a time, fuse attention to avoid N² score matrices, share one arena between the encoder and the LLM, and compute logits for the last position only, since full-position logits are about 156 MB. Image buffers are controlled with decode-time downscaling and buffer reuse. That adds up to about 2 to 2.4 GB with a 15% margin. To fit, I'd choose a GQA backbone, INT4 and INT8 KV, sliding-window or sink attention plus heavy-hitter eviction for long sessions, vocabulary pruning, weight sharing across graphs, and avoid double residency of mmap and device copies. mmap saves memory and survives pressure, but decode reads all weights per token, so hot weights must be pre-faulted and resident, and only cold data is paged. For OOM, I'd handle the lmkd's silent kills with a pre-flight tier selection, trim-memory handlers that drop caches and unload the encoder, catchable allocation failures with fallback to a lower tier, persisted state, and soak tests on the lowest RAM tier. For thermals, I'd listen to thermal status and headroom and reduce tiles, output length and frame rate, or route to the cloud."

## 7. Follow-up questions to prepare for
1. Why do INT4 weights not benefit from zram, and what does that imply for memory pressure?
2. Why can mmap make memory safer but latency worse for LLM decode?
3. How does GQA reduce KV-cache size, and what is the quality trade-off?
4. How do attention sinks work, and what breaks if you evict the first tokens?
5. How do you handle RoPE positions when you evict tokens from the middle of the cache?
6. Why is "double residency" a common bug, and how would you detect it?
7. How would you decide the tiers for the graceful-degradation ladder?
8. How would you detect a slow memory leak or fragmentation in a long-running assistant?

---

# Q15: Speculative decoding, batching and scheduling on-device

**Question:** Does speculative decoding help on a memory-bound mobile decoder? When would it hurt (draft cost, NPU kernel support)? How do you do continuous batching, prefix caching for repeated system prompts or repeated images, and chunked prefill when the device also runs camera and ISP pipelines concurrently?

Speedup numbers come from the simple cost model in section 1, with assumed acceptance rates and cost ratios. They are for reasoning, not benchmarks. Paper details (EAGLE, Medusa, vLLM, Sarathi and others) are from memory, so check them before quoting.

## 0. Framing: what "batching" means on a phone
Server inference is about throughput across many users. On a phone you usually have **one user and batch size 1**, so latency matters most. But batching ideas still apply in three places:
1. **Several concurrent sessions or clients** (an assistant, live captions and another app sharing one on-device model service).
2. **Several frames or tiles** processed together.
3. **Speculative verification**, which is a small batch of candidate tokens checked in one pass.

The unifying fact from Q12: **decode is memory-bandwidth-bound**, so extra tokens or streams in the same pass are nearly free until you reach the compute ridge (a few hundred tokens in flight). All three techniques below exploit that slack.

## 1. Speculative decoding

### 1.1 How it works
1. A cheap **draft** proposes `k` tokens autoregressively.
2. The **target** model verifies all `k` in **one forward pass** (`k+1` positions).
3. Accept the longest correct prefix, and sample one extra token from the target. With rejection sampling, the output distribution is **identical to the target's** (lossless). For greedy decoding, accept while the target's argmax equals the draft token.

### 1.2 Why it can help on a memory-bound decoder
Verifying `k+1` tokens reads the weights **once**, the same as decoding one token. With INT4 weights the arithmetic intensity at `k+1` ≈ 5 to 9 tokens is still far below the ridge point of roughly 200 to 400 tokens (Q12). So the extra verification compute is almost free, and the speedup is limited by acceptance, not by hardware.

### 1.3 Cost model
- Per-token acceptance rate α, draft length k. Expected tokens per iteration: `E = (1 − α^(k+1)) / (1 − α)`.
- Cost per iteration (in units of one target decode step): `k·c + v`, where `c = T_draft / T_target` and `v ≈ 1.05` (verify is slightly more than one step).
- Speedup ≈ `E / (k·c + v)`.

| α | k | c (draft cost ratio) | E | Cost | Speedup |
|---|---|---|---|---|---|
| 0.7 | 4 | 0.1 | 2.77 | 1.45 | **~1.9x** |
| 0.7 | 4 | 0.3 | 2.77 | 2.25 | ~1.2x |
| 0.5 | 4 | 0.1 | 1.94 | 1.45 | ~1.3x |
| 0.5 | 4 | 0.3 | 1.94 | 2.25 | **~0.86x (slower)** |

**The on-device catch:** the target is already small (1 to 2B), so a draft that is **10x cheaper** would have to be about 100M parameters. A 0.5B draft for a 1.5B target has `c ≈ 0.33`, which erases most of the benefit. Cheap drafting is the whole game on-device.

### 1.4 Variants and how they fit the edge

| Variant | Idea | Edge fit |
|---|---|---|
| **Draft model** (small LM, same tokenizer) | Separate small model | Needs a tiny draft (about 50 to 150M). Extra weights and its own KV |
| **Medusa** | Extra heads on the target predict the next k tokens, verified with tree attention | Needs fine-tuning. Each head needs a vocabulary projection (hidden × vocab). With a 152k vocabulary, that is about 233M parameters **per head**, which is expensive (Q1 deep dive) |
| **EAGLE / EAGLE-2** | A light autoregressive head on the target's **hidden features**, reusing the target's LM head. EAGLE-2 builds a dynamic draft tree | Small overhead (about one decoder layer), high acceptance. A natural fit for edge, and for VLMs, because the target's hidden states already contain the visual information |
| **Prompt-lookup / n-gram** | Propose tokens that copy from the prompt or earlier text | **Free** (no model). Excellent for copy-heavy outputs: OCR text, JSON extraction, quoting UI text |
| **Lookahead decoding** | Jacobi-style parallel guesses and an n-gram pool | No draft model, modest gains |
| **Layer-skip / early-exit self-speculation** (e.g. LayerSkip) | Draft with the first layers of the target, verify with all | No extra weights, but needs training for good acceptance |
| **MTP heads** | Multi-token-prediction heads used for speculation | Needs a model trained that way |

### 1.5 Multimodal specifics
- The draft must know the image to predict visually grounded tokens. Options:
  - **Text-only draft:** simpler, lower acceptance on image-dependent tokens (reported to still give useful speedups on LLaVA-style models, verify).
  - **Small VLM draft:** better acceptance, but has its own visual path.
  - **EAGLE-style head on the target's features:** reuses the target's visual information and needs no separate image prefill. This is the cleanest option.
- On-device VLM answers are often **short** (5 to 30 tokens), so fixed overheads (draft warm-up, extra graph launches) matter. Acceptance is high for **structured or copy-heavy** outputs (OCR, JSON, UI labels) and low for creative text.

### 1.6 When it hurts (the checklist)
1. **Low acceptance:** high-temperature sampling, creative tasks, or a draft that doesn't match the target after fine-tuning. If you LoRA-tune the target on your domain (Q7), you must also tune or distill the draft on the target's outputs (Q10), otherwise α drops.
2. **A draft that is too expensive:** `c` above roughly 0.2 to 0.3 (the table above).
3. **No small-M verify graph on the NPU.** If the only compiled prefill graph is 128 tokens, verification becomes a compute-bound 128-token pass, which can be slower than the decode steps it replaces. You need verify graphs with **M = 4 to 16** (padding up to 8 or 16 is nearly free because decode is memory-bound).
4. **Missing kernel support:** tree attention masks, KV compaction for accepted branches, and per-iteration host logic. Static-graph NPUs may not support them.
5. **Host-device sync overhead per iteration:** drafting k tokens in sequence, a verify launch, then host-side accept/reject using `(k+1) × V` logits (about 1.8 MB for k = 5, V = 152k, FP16). Each sync costs milliseconds.
6. **Memory budget:** extra draft weights and KV (Q14). A 150M INT4 draft is about 0.08 GB, an EAGLE head about 2 to 3% of the model.
7. **Short outputs,** where setup costs dominate.
8. **Compute is not free anymore:** if another workload shares the NPU, or you use large trees (M in the tens to hundreds).
9. **Numerical non-determinism:** the verification pass uses a different kernel shape (M = k+1 vs 1), so tiny numerical differences in FP16/INT4 can flip near-tied tokens. Outputs may not be bitwise identical to non-speculative decoding.

### 1.7 Engineering details on a static-graph NPU
- **KV rollback:** with a static buffer and a write index, rejected tokens are simply not committed (the index doesn't advance, and the mask hides them). **Tree** verification needs the accepted branch's KV copied to contiguous positions.
- **Where to run the draft:** on the same NPU (sequential launches), or on the CPU/GPU. A CPU draft is slower (larger `c`) and competes for shared DRAM bandwidth with the NPU.
- **Adaptive k:** adjust the draft length from a running acceptance estimate, or draft until the draft's confidence drops below a threshold (EAGLE-2-style dynamic trees).
- **Per-task enable/disable:** turn it on for OCR and JSON outputs, off for creative chat.
- **Energy:** wasted draft compute costs energy, but fewer weight reads per accepted token usually lowers energy per token. Measure joules per token, not just latency.
- **Evaluate** acceptance length (tokens per verify step), TPOT, E2E, energy per accepted token, memory, and a lossless-equivalence check against non-speculative output (within numeric tolerance), on **product workloads**, not generic chat.

## 2. Continuous batching and scheduling

### 2.1 Server ideas (what they are)
- **Iteration-level scheduling (Orca):** requests join and leave the batch at every decode step, instead of waiting for a whole batch to finish.
- **Paged KV (vLLM PagedAttention):** KV stored in blocks, so no big contiguous buffers, little fragmentation, and easy sharing.
- **Chunked prefill (Sarathi-style, "stall-free"):** split long prefills into chunks and interleave them with decode steps, so decodes aren't blocked.
- **Prefix caching:** hash blocks (vLLM) or a radix tree (SGLang RadixAttention) to reuse shared prefixes.

### 2.2 What carries over on-device
- Decode is memory-bound, so **B concurrent streams cost about the same per step as one** (until the ridge). Throughput scales almost for free if there is memory for B KV-caches. At about 28 KB per token (Qwen2.5-1.5B-class), four 2k-token sessions are about 230 MB, which is affordable.
- NPU graphs are **static**, so use **slot-based batching**:
  - Compile the decode graph for a fixed number of slots S (e.g. 1, 2, 4), each with its own KV region.
  - Each step runs all S slots, with per-slot positions and an attention mask `[S, 1, ctx]`. Inactive slots are masked, and because decode is memory-bound, running 4 slots with 1 active costs about the same as running 1.
  - Admit a request by assigning a free slot and running its (chunked) prefill into that slot. Free the slot when it finishes.
- **When it matters:** a system-wide model service shared by several apps, parallel sampling (n-best), and multi-agent flows. For a single-user camera assistant, scheduling is mostly about priorities, not batching.

### 2.3 Scheduling policy
- **Priorities:** interactive foreground request > background summarization or indexing.
- **Preemption:** pause a low-priority decode to run a high-priority prefill, and either keep the paused KV (if memory allows) or recompute it later.
- **Deadline-aware** policies (e.g. earliest-deadline-first) with explicit TTFT and TPOT targets.
- **Per-iteration token budget:** cap the tokens processed per iteration so decode streams stay within the TPOT target.
- **Admission control by memory** (slots, KV budget), tied to Q14's tiers.

### 2.4 Chunked prefill, with camera and ISP running at the same time
- **Why chunk:** a long prefill (say 300 to 2,000 tokens) would stall every decoding stream. A chunk of C tokens bounds the stall to about `T_prefill(C)`.
- **Chunk size trade-off:** small C means shorter stalls but less efficient (each chunk re-reads the weights and attends over the growing KV, and below the ridge of about 200 to 400 tokens the NPU is memory-bound). Large C means longer stalls. Choose C near the ridge (e.g. 128 to 256) and match a compiled graph size.
- **Static-graph details:** the last chunk is padded and masked, the KV write offset advances per chunk, and prefix reuse starts prefill at a nonzero offset.

**Co-running with the camera and ISP:**
- The ISP and camera pipeline have **hard real-time deadlines** (a 30 fps preview needs a frame every 33 ms). They share **DRAM bandwidth, the power budget and the thermal budget** with the NPU.
- If the NPU hogs bandwidth or a long graph blocks, the camera can **drop frames or jank**.
- Design principles:
  1. **Treat camera/ISP as hard real-time and the VLM as best effort.**
  2. **Make work preemptible by splitting graphs.** Many NPUs can't preempt a running graph, so a 150 ms encoder graph blocks everything. Split into sub-graphs (per tile, per group of layers) to create yield points of about 10 to 20 ms.
  3. **Use scheduling hints:** NPU graph priority settings where the SDK has them, thread priorities and cgroups, and Android's performance-hint APIs. DRAM QoS for the ISP is mostly system-level and not controllable by an app (verify).
  4. **Schedule between frames,** or during idle gaps, and reduce VLM activity during capture bursts.
  5. **Sample frames sparsely** for live use (e.g. 1 to 2 fps or event-triggered), and use a "latest frame wins" queue so stale frames are dropped instead of queued.
  6. **Mind shared power and thermals:** NPU activity can lower CPU/ISP clocks under a shared budget (Q16).
  7. **Measure the interference:** in Perfetto, watch camera frame drops, SurfaceFlinger missed deadlines and HAL request latency while the VLM runs, and set an acceptable jank budget.

## 3. Prefix caching for system prompts and repeated images

### 3.1 What and why
Causal attention means the KV of a token depends only on the tokens **before** it. If two requests share an identical prefix, the prefix's KV can be reused, so you skip its prefill.

**System prompt:** a 500-token system prompt is about 14 MB of KV (at 28 KB per token) and about 100 to 200 ms of prefill. Precompute its KV **once** (even offline at install or first launch) and load it in milliseconds on every request.

### 3.2 Layered caches
1. **Encoder-output cache** (key = hash of image pixels + preprocessing config): skips the vision encoder (about 100 ms) when the same image appears again.
2. **LLM KV prefix cache** (key = hash of the prefix tokens + image hash + model/adapter/KV-format version): for `[system prompt + image]`, so follow-up questions only prefill the new text tokens.

### 3.3 Rules and pitfalls
- **Order matters.** Put stable content first: system prompt, then image, then the variable user question. Anything that changes early (timestamps, user names in the system prompt) invalidates everything after it.
- **The image KV depends on the whole prefix and on positions.** With a fixed-length system prompt, M-RoPE and 1D positions line up, so the cached image KV is reusable (Q3).
- **Cache key must include everything that affects the KV:** model hash, tokenizer, quantization and KV format, RoPE configuration, **LoRA adapter ID** (different weights mean different KV), and chat template version.
- **Static-graph handling:** loading cached KV is a memcpy into the KV buffer plus setting the write index, then chunked prefill of the suffix at an offset. Align the prefix length to the chunk size (or design the system prompt length as a multiple of it) to avoid wasting a partial chunk.
- **Capacity:** one cached 320-token image KV is about 9 MB. Use an LRU cache with a memory cap (e.g. 64 to 128 MB), and drop it under memory pressure (Q14).
- **Persist the system-prompt KV on disk** (memory-mapped, quantized), keyed by version, so it survives restarts.
- **Privacy and security:** caches contain user-derived data, so keep them in app-private storage, encrypt as required, and clear on logout. In a shared multi-app service, isolate caches per client, since prefix-cache hit timing can leak whether another client used a prompt.
- **Multi-turn:** reuse the whole conversation prefix, and apply an eviction or summarization policy when the context grows (Q14).

## 4. Putting it together: a scheduler loop (sketch)

```
each iteration:
    budget = TOKENS_PER_ITER                     # bounds the stall seen by decoding streams
    decode all active slots (1 token each, or k+1 with speculation if enabled for the task)
    if pending prefill and camera_not_in_burst:
        run one prefill chunk (size C) for the highest-priority request   # preemptible unit
    adapt: read thermal / memory / camera-jank signals
           → lower k, C, tiles, or slots; or defer background work
```

## 5. Strong interview answer
> "Decode is memory-bandwidth-bound, so verifying k+1 tokens in one pass costs roughly the same as decoding one, because the weights are read once and we stay far below the compute ridge. That is why speculative decoding can help on a mobile NPU. The speedup is E over cost, where E = (1 − α^(k+1))/(1 − α) and the cost is k·c + v, with c the draft-to-target cost ratio. With α = 0.7, k = 4 and c = 0.1 that is about 1.9x, but with α = 0.5 and c = 0.3 it is slower than not speculating. The catch on-device is that the target is already 1 to 2B, so a 10x cheaper draft is about 100M parameters. That makes EAGLE-style heads on the target's hidden states, which reuse its LM head and visual information, or free prompt-lookup for copy-heavy OCR and JSON outputs, better fits than a separate draft model. Medusa heads are expensive with a 152k vocabulary. It hurts with low acceptance, for example after domain fine-tuning without re-tuning the draft, with a draft that is too costly, with no small-M verify graph or tree-mask support on the NPU, with per-iteration host sync overhead, and for short outputs. For batching, I'd use slot-based static batching on the NPU: a decode graph for S slots with per-slot positions and masks, since an extra stream is nearly free while memory-bound. I'd use chunked prefill with C near the ridge, interleaved with decode, and priorities and preemption. With the camera and ISP running, I'd treat them as hard real-time, split long graphs into sub-graphs so the NPU has yield points, schedule between frames, use latest-frame-wins queues, and measure frame drops in Perfetto. Prefix caching has two layers: an encoder-output cache keyed by image hash, and a KV prefix cache, with system prompt first, then the image, then the question, keyed by model, adapter, KV format and RoPE configuration. The system-prompt KV can be precomputed offline and mmapped, which saves 100 to 200 ms of TTFT per request, and loading it is a memcpy plus an offset into chunked prefill. I'd keep caches app-private for privacy."

## 6. Follow-up questions to prepare for
1. Derive the expected accepted tokens per iteration and the speedup. When is the speedup below 1?
2. Why does verifying k+1 tokens cost about the same as 1 on a memory-bound decoder, and when does that stop being true?
3. Why are Medusa heads costly with a large vocabulary, and how does EAGLE avoid that?
4. What does a verify graph need on a static-graph NPU, and how do you roll back rejected tokens?
5. Why must the system prompt come before the image for prefix caching to work?
6. What must the prefix-cache key include, and what happens if you forget the LoRA adapter ID?
7. How would you give a camera pipeline priority over a long VLM encoder graph on an NPU that can't preempt?
8. How would you size the chunk for chunked prefill on a given NPU?

---

# Q16: Thermal, power and QoS

**Question:** How do you design for sustained performance under thermal throttling and a battery budget (J/token, mJ/inference)? What are the strategies: adaptive resolution, early exit, model cascading (tiny classifier → small VLM → cloud), DVFS-aware scheduling, and heterogeneous CPU/GPU/NPU partitioning? How do you benchmark reproducibly?

Power and energy figures are **assumption-based illustrations** (for example 4 pJ/bit for DRAM and 5 TOPS/W for an NPU). Real values vary by device, process node and what you count, so always measure. Android API details are from memory, so verify the API levels.

## 0. Why this matters
- A phone is passively cooled. The comfortable **skin temperature** limit (roughly the low 40s °C) sets the sustained power budget, which is only a few watts, far below the SoC's burst capability.
- A cold device can burst for tens of seconds to a few minutes, then **throttle**: the thermal governor caps CPU, GPU, NPU and DRAM frequencies. A benchmark run on a cold phone therefore **overstates** real sustained performance.
- Battery is finite: a 5,000 mAh battery at 3.85 V is about 19 Wh, or roughly **69 kJ**.
- The NPU shares the SoC power and thermal budget with the CPU, GPU, ISP and modem, so a camera session and a VLM compete.

## 1. Where the energy goes (illustrative model)

Basic power model: `P = P_static (leakage, rises with temperature) + P_dynamic ≈ C·V²·f`. Lowering voltage and frequency cuts energy per operation, but runs longer, so static energy accumulates ("race to idle" vs "slow and steady").

**Assumptions:** DRAM access about 4 pJ/bit, NPU about 5 TOPS/W effective at INT8, the ~2B VLM from Q14 (about 1.0 GB of weights read per decode token).

| Stage | Energy estimate | Reasoning |
|---|---|---|
| Decode, per token | **~35 to 40 mJ** | Weight reads: 1 GB × 8 bits × 4 pJ ≈ 32 mJ. Compute: 2.6 GFLOP ÷ 5 TOPS/W ≈ 0.5 mJ. Plus KV, host and sampling |
| Vision encoder (5 tiles) | ~0.25 J | About 1 TFLOP ÷ 5 TOPS/W ≈ 0.2 J, plus traffic |
| Prefill (~380 tokens) | ~0.2 J | About 1 TFLOP |
| 30-token answer | ~1.1 J | 30 × ~38 mJ |
| **One query end to end** | **~1.6 J** | |

**Consequences:**
- **Decode energy is dominated by DRAM data movement, not arithmetic** (32 mJ vs 0.5 mJ). So the levers are **fewer bytes per token**: INT4 weights, KV quantization, GQA, a smaller vocabulary and LM head (Q1 deep dive), and **speculative decoding**, which reads the weights once per several accepted tokens and can roughly halve decode energy per token (Q15).
- **Battery math:** 1% of the battery is about 690 J, so about **430 queries per 1%** at 1.6 J each. Always-on video at 1 frame per second, with about 0.4 J per frame, costs about 0.4 W, i.e. roughly **2% battery per hour**.
- **Average power during decode:** 38 mJ × 25 tokens/s ≈ 1 W for NPU + DRAM, and roughly 2 to 3 W for the whole system. That is near a typical sustained envelope, so **continuous generation can throttle within minutes**, while bursty use does not.
- For long answers, decode dominates energy, so **output length** is an energy lever as well as a latency lever.

## 2. How throttling works
- **Thermal zones** (CPU, GPU, NPU, skin, battery) feed a thermal governor (step-wise or power-allocator style) and a vendor thermal engine. Actions include capping CPU, GPU, NPU and DDR frequencies, reducing active cores, and, eventually, shutdown.
- **Skin temperature** is prioritized for user comfort, so it often sets the sustained limit before the silicon junction temperature does. Heat spreads slowly, so the first 30 to 120 seconds are "free".
- **Environment matters:** ambient temperature, a case, charging (heat from charging, especially wireless), sunlight, other apps, camera and ISP use, and the 5G modem.
- **Throttling is not always smooth:** governor hysteresis can cause oscillation, which shows up as latency jitter and bad p99.
- **Android signals:**
  - `PowerManager.getCurrentThermalStatus()`: NONE, LIGHT, MODERATE, SEVERE, CRITICAL, EMERGENCY, SHUTDOWN, plus a listener for changes.
  - `getThermalHeadroom(forecastSeconds)` (API 30+, forecasts up to about 60 s): a ratio where about 1.0 corresponds to severe throttling. Use it to **act before the cliff**.
  - Per-zone temperatures in `/sys/class/thermal/*/temp` (for measurement).

## 3. Strategies

### 3.1 Reduce work per query
Energy scales with work, so the cheapest watt is the one you don't spend: INT4/INT8 (fewer bytes), fewer visual tokens (Q2), shorter outputs, encoder and prefix caches (Q15), speculative decoding where it pays off, and no re-encoding of unchanged frames.

### 3.2 Adaptive quality ("anytime" operation)
Build a **ladder of operating points** and move along it based on thermal and battery state:

| Level | Tiles / resolution | Visual-token budget | Max output tokens | Speculation k | Model tier | Frame rate (video) |
|---|---|---|---|---|---|---|
| L0 (cool, plugged) | Full | Full | Full | On | 1.5B | Full |
| L1 | Fewer tiles | Reduced | Capped | On | 1.5B | Reduced |
| L2 | 1 tile | Low | Short | Off | 1.5B | Low |
| L3 | 1 tile | Minimal | Very short | Off | 0.5B | Event-triggered |
| L4 | Offload or decline | — | — | — | Cloud / "try later" | — |

Implementation notes:
- **Pre-compiled static graphs per level**, kept resident (memory cost, Q14), so switching is fast on a static-graph NPU (Q13).
- Matryoshka-style nested token budgets (Q2) let one model run at several token counts.
- **Use accuracy-per-level data** (the Pareto curve from Q2) and set **quality floors** for safety-critical tasks, so hazard detection is never degraded below a set level.
- **Controller design:** a state machine with **hysteresis** (separate up and down thresholds, minimum dwell time) to avoid oscillation, driven by thermal status, headroom forecast, battery level and power-saver state. Prefer **gradual degradation before the hardware cliff**.

```python
def choose_level(status, headroom, battery, current):
    want = current
    if status >= SEVERE or headroom > 0.95:   want = min(current + 1, MAX_LEVEL)
    elif status <= LIGHT and headroom < 0.6 and battery > 0.2:
                                               want = max(current - 1, 0)   # recover slowly
    if battery < 0.15 or power_saver:          want = max(want, 2)
    return want if dwell_time(current) > MIN_DWELL or want > current else current
```

### 3.3 Early exit and dynamic compute
- Confidence-based **early exit** in the vision encoder or LLM layers (layer-skip style), adaptive token pruning or merging, and skipping the LLM for easy classification-like questions. For vision, adapt tiles to text density.
- **Static-graph caveats:** early exit needs graph segments with host-side checks or exit-specific graphs. For LLM decode, skipped layers still need their KV filled in (state copying), which reduces the benefit.
- **Risks:** exit confidence must be calibrated, and exit heads need training. Quality can drop silently, so evaluate per slice (Q9, Q24).

### 3.4 Model cascades (tiny → small VLM → cloud)
- **Stage 0:** an always-on tiny detector or classifier on a low-power unit (DSP, sensor hub, small NPU island) at milliwatt scale, tuned for **high recall**, which wakes the next stage. Add event triggers such as frame differencing and scene-change gating.
- **Stage 1:** the small on-device VLM (≤2B).
- **Stage 2:** a larger model or the cloud, only for low-confidence or hard cases (Q17).
- **Cost model:** `E = c0 + p1·c1 + p1·p2·c2`, where `p_i` is the fraction of queries reaching stage i. Choose thresholds to minimize energy subject to an accuracy floor, and plot the accuracy-vs-energy curve.
- **Calibration** of confidences is essential. The costs of a missed event (false negative at stage 0) and of a bad deferral should be explicit.
- **Cloud is not free in energy:** transmitting an image over a cellular radio costs energy, and the radio stays in a high-power state for seconds afterwards ("tail energy"). With a weak signal, offloading can cost as much as, or more than, local inference. Measure the break-even, and respect privacy constraints (Q17).

### 3.5 DVFS-aware scheduling
- **Phase-aware operating points:**
  - The encoder and prefill are **compute-bound**, so they benefit from a high NPU clock (a burst profile).
  - Decode is **memory-bound**, so a higher NPU clock adds power without speed. Prefer a lower or balanced NPU clock with enough DRAM bandwidth, which saves energy at nearly the same speed.
  - Mechanisms: vendor performance profiles (QNN/HTP burst, balanced, power-saver style settings, and Exynos NPU performance modes, verify), NNAPI-style preferences (sustained speed vs low power, now deprecated), and Android's performance-hint APIs for CPU threads.
- **Avoid governor lag:** clocks take tens of milliseconds to ramp, so the first inference is slower. Send a **pre-boost hint** when the user opens the camera, before the request arrives.
- **Race-to-idle for background work:** batch background jobs into bursts so the SoC can enter deep idle, and use constraints like "charging, idle and battery not low" (e.g. WorkManager).
- **Don't busy-wait:** while the NPU runs, block on a fence or interrupt instead of spinning a CPU core. Pin orchestration threads to a mid or little core to avoid waking the prime core.
- **Smooth over hard cliffs:** use the headroom forecast to reduce load gradually.

### 3.6 Heterogeneous partitioning
- Use the **most energy-efficient unit for each stage**: NPU for INT8 matmul-heavy work (often several times more energy-efficient than CPU or GPU for the same workload, verify on your target), GPU for flexible FP16 ops and fallback, CPU for control, sampling and small ops, and a low-power island for the stage-0 cascade.
- **Avoid ping-pong:** each unit switch costs latency, copies, cache maintenance and synchronization, so keep partition boundaries few and use zero-copy buffers.
- **Thermal spreading:** splitting work across units spreads heat across different thermal zones and can raise sustained throughput, but it increases total power, so it can hurt skin temperature and battery. Measure rather than assume.
- A CPU decode path is a fallback, and costs much more energy per token.

### 3.7 Battery and system policies
- Respect **battery saver**, low battery, and battery temperature. Charging adds heat, so plugged-in is not automatically a free pass for heavy compute.
- Unload models when idle, avoid wake locks and unnecessary wakeups, and schedule heavy background indexing for when the phone is charging, idle and cool.
- Reduce the always-on frame rate when no one is looking.

## 4. QoS: define the contract
- **Service classes:** interactive (TTFT and TPOT targets) vs background (best effort).
- **Under throttling, degrade quality in a controlled order, not latency randomly,** for example fewer tiles, then shorter outputs, then a smaller model, then offload.
- Each feature gets: an accuracy floor, a latency target, an energy budget, and a fallback.
- Android has no OS-level NPU QoS arbitration that you can fully rely on (verify), so cooperation inside the app and with the vendor SDK's priority settings matters.
- Report **p95 steady-state latency** after throttling, not just cold performance.

## 5. How to benchmark reproducibly

### 5.1 Sources of variance
Ambient temperature, **starting temperature** (cold vs heat-soaked), battery state of charge (voltage and peak power limits, battery aging and internal resistance), charging state, USB cable (it charges the phone and changes thermals), screen brightness and refresh rate, background apps and sync, radios, OS and firmware version, governor settings, DRAM state and page cache, device-to-device silicon variation, case and orientation, airflow.

### 5.2 Protocol
1. **Controlled setup:** a thermally controlled room (e.g. 25 ± 1 °C), a standard fixture (same stand, no case, fixed orientation, still air or a fixed fan), screen off or fixed, airplane mode or a fixed network, background apps killed, a fixed OS build and power-mode settings.
2. **Power measurement:** an external monitor with battery bypass (Monsoon, Otii, Joulescope-style) for accuracy. The on-device fuel gauge or `BatteryManager` is coarse (about 1 s, low accuracy). Pixel-class devices expose on-device rail monitors (verify per device). Subtract the **idle baseline** measured under identical conditions.
3. **Pre-conditioning:** cool to a known start temperature before each run (e.g. skin temperature below a set value), and also run a **heat-soaked** variant (a fixed load for about 10 minutes first). Report both.
4. **Workload:** fixed prompts, images and decoding settings, a **fixed output length for benchmarking** (ignore EOS to remove variance), plus a separate realistic mix. Run for a fixed duration (15 to 30 minutes), with both a continuous duty cycle and a bursty one (e.g. one query every 10 s).
5. **Logging at 1 Hz or better:** per-inference TTFT and TPOT, tokens/s, power, energy per inference, all thermal zones plus a skin-temperature measurement (IR sensor or thermal camera), CPU/GPU/NPU/DDR frequencies, throttle state and battery.
6. **Statistics:** at least 5 runs per configuration, **interleaved A/B/A/B** order to cancel drift, rest between runs to return to baseline, at least 3 device units to measure unit variance, and report medians with spread (p5 to p95) or confidence intervals.
7. **Report the right metrics:**
   - Peak (first minute) vs **sustained** (e.g. the last 5 minutes) performance, and **time to throttle onset**.
   - **Energy per inference and per token**, and battery % per hour for the use case.
   - Maximum skin temperature.
   - **Quality under each throttling state**, since adaptive policies change accuracy. Always report the accuracy–latency–energy triple.
8. **Reference:** MLPerf Mobile defines controlled conditions and cooldown intervals, a useful model for protocol design (verify the exact rules).

### 5.3 Test the controller deterministically
- You can't wait for a phone to heat up on every CI run. On Android, `adb shell cmd thermalservice override-status <level>` (and `reset`) lets you emulate thermal states to test that your policy reacts correctly (verify the command on your OS version).
- Add **nightly power and thermal regression tests** on real devices, with relative gates (e.g. fail if J/token regresses by more than a set percentage).

### 5.4 Pitfalls
- Measuring on a cold device, or over USB power.
- Ignoring the idle baseline.
- Lab results that differ from the field (pocket, sunlight, a hot car, a case).
- A controller that oscillates, or lowers quality without measuring it.
- Differences between device variants and regions (Snapdragon vs Exynos thermal behavior differ).
- OS updates that change governors, and battery aging.

## 6. Strong interview answer
> "I'd start from an energy model. Decode energy is dominated by DRAM traffic, not arithmetic: with about 1 GB of weights per token at roughly 4 pJ/bit that's about 32 mJ against about 0.5 mJ of compute, so INT4, KV quantization, GQA, a smaller LM head and speculative decoding directly cut J/token. At about 1.6 J per query, 1% of a phone battery is about 400 queries, and continuous generation near 2 to 3 W will hit a skin-temperature-limited sustained envelope within minutes. So I'd design for sustained, not peak, performance. I'd build a ladder of operating points, with tiles or resolution, visual-token budget, output length, speculation and model tier, pre-compiled as static graphs, driven by a controller that uses thermal status and the headroom forecast, with hysteresis and quality floors for safety-critical tasks, so it degrades gradually before the cliff. I'd use a cascade, with an always-on tiny detector tuned for recall waking the small VLM, and low-confidence cases going to the cloud, after measuring the radio and tail energy, because offload can cost as much as local inference. For DVFS, compute-bound encoder and prefill phases get a burst profile and memory-bound decode gets a balanced one, with pre-boost hints and no busy-waiting. For partitioning, I'd put matmul on the NPU, keep the boundaries few and zero-copy, and measure thermal spreading rather than assume it. To benchmark, I'd control ambient temperature, fixture, screen, radios and battery state, pre-condition to known start temperatures including a heat-soaked case, use an external power monitor with an idle baseline, run 15 to 30 minutes with fixed output lengths, interleave A/B runs across several units, and report peak vs sustained throughput, throttle-onset time, J/token, skin temperature and accuracy at each operating point. I'd emulate thermal states for CI, and add nightly relative power and thermal gates."

## 7. Follow-up questions to prepare for
1. Why is decode energy dominated by memory traffic rather than compute, and how does that change which optimizations pay off?
2. Why can lowering the NPU clock during decode save energy at almost no latency cost?
3. How does the thermal headroom forecast help avoid a hard throttling cliff, and how do you prevent oscillation in the controller?
4. When is offloading to the cloud more expensive in energy than running locally?
5. How would you choose the thresholds in a three-stage cascade to meet an accuracy floor at minimum energy?
6. Why does a cold-start benchmark misrepresent real-world performance, and how do you design a fair comparison?
7. What are the trade-offs of spreading work across the NPU, GPU and CPU for thermals?
8. How would you test the thermal controller in CI without heating a physical device?

---

# Q17: Hybrid on-device and cloud routing

**Question:** Design the routing logic that decides when a query is handled by the on-device VLM vs a larger cloud model. Cover confidence calibration, uncertainty estimation, privacy constraints, latency SLOs, offline fallback and consistency of behaviour between the two.

Numbers in the worked examples are assumptions for illustration. Paper and product references (RouteLLM, FrugalGPT, semantic entropy, conformal methods, Apple's Private Cloud Compute) are from memory, so check them before quoting.

## 0. Framing
Routing is a **decision under constraints**. You choose the route `r ∈ {on-device, cloud, hybrid, ask/decline}` to maximize quality subject to:
- **Privacy** (a hard constraint, never traded away),
- **Latency SLOs**,
- **Cloud cost and energy/battery** (Q16),
- **Availability** (offline, outages),
- **Behavioural consistency** between the two models.

A useful way to write it: `U(route) = Q − λ_L·max(0, T − SLO) − λ_C·cost − λ_E·energy`, subject to `privacy_ok(route) = true`. Privacy is a gate, not a term in the sum.

A good design is **layered**:
1. Hard policy gates (privacy, offline, user settings), fail closed.
2. A cheap pre-router (task type, out-of-distribution score, knowledge need).
3. On-device inference plus calibrated confidence.
4. A context-aware deferral threshold.
5. A redacted cloud call, with timeout and fallback.
6. Response harmonization.

## 1. Routing signals

### 1.1 Query and task features (cheap, before running any model)
- **Task type:** OCR, scene Q&A, counting, grounding, long-form reasoning, math, "needs current knowledge or the web".
- **Query features:** language, question length, expected output length, multi-image or video, tool or API need.
- **Image features:** text density, resolution, blur or low light, similarity to the training distribution.

### 1.2 Model-based confidence (after the small model has run)
- **Token-level statistics:** mean and minimum log-probability, entropy, top-2 margin, length-normalized sequence probability. For classification-style or multiple-choice answers, use the calibrated probability of the chosen option.
- **Self-consistency and semantic entropy:** sample n answers and measure semantic agreement. Strong, but n× decode cost, so use it only for borderline cases. A cheaper variant compares the greedy answer with one or two sampled answers, or with answers from a second view of the image (different crop or resolution).
- **Learned correctness probe:** a small linear or MLP head on hidden states the model already computed, trained to predict "is this answer correct". Almost free at inference.
- **Verbalized confidence** ("I'm 80% sure"): usually poorly calibrated in small models, so use with caution.
- **Visual-grounding checks (Q9):** token probabilities fail when the model is confidently wrong because of language priors. Add:
  - an answer-depends-on-the-image test (compare the answer or logits with a corrupted or blank image),
  - an OCR-engine cross-check for text answers,
  - a detector cross-check for object claims.
- **Out-of-distribution score:** distance of the image embedding from the training distribution (kNN or Mahalanobis). Unusual conditions (night, a rare device, strange screens) should push the query up.

### 1.3 Context signals
Connectivity quality (RTT, bandwidth, signal, metered or roaming), battery, thermal state (Q16), memory pressure (Q14), cloud health and load, user settings ("never send images", offline mode), enterprise policy, region, cost quota.

### 1.4 Privacy signals
A **sensitivity classifier** on the input (faces, documents, financial or health content, screens with personal data, children, home interiors) plus user consent state. See section 4.

## 2. Routing architectures

| Pattern | How it works | Pros | Cons |
|---|---|---|---|
| **A. Rules** | Static policy by task, privacy and connectivity | Predictable, auditable, the right base layer | Coarse, no adaptation to difficulty |
| **B. Pre-routing classifier** | A learned model predicts, before any inference, which model can handle the query (RouteLLM-style routers trained on preference or outcome data) | No wasted on-device compute for queries that will go to the cloud | Less accurate than post-hoc confidence |
| **C. Cascade (deferral)** | Run the small model, compute confidence, defer to cloud if low (FrugalGPT-style, "learning to defer") | Most queries stay local, high accuracy per cost | A deferred query pays `T_local + T_cloud` |
| **D. Hybrid collaboration** | Split the work (see below) | Can improve quality, privacy, bandwidth | More complex, more failure modes |
| **E. Learned policy (contextual bandit / RL)** | Optimize a utility reward with exploration, evaluated off-policy from logs | Adapts to real traffic | Needs careful safety constraints and exploration design |

**Hybrid collaboration patterns:**
- **Perception on-device, reasoning in the cloud:** the device turns the image into structured text (objects, OCR text, scene graph, PII scrubbed), and a cloud LLM reasons over a **text-only payload of a few KB**. This is privacy-friendly and cheap in bandwidth, and fits a home robot well.
- **Split computing:** run the vision encoder on-device and send visual tokens to the cloud LLM. Bandwidth is rarely better: 320 tokens × 1,536 × 2 B ≈ 1 MB of features versus about 200 to 300 KB for a compressed JPEG. Embeddings can also be **inverted** to reconstruct images, so they are not automatically private (verify against the literature for your encoder).
- **On-device draft, cloud verification:** a network round trip per verification step makes this costly unless you verify long chunks per round trip.
- **Cloud for knowledge, device for personal context:** the cloud handles retrieval and the web, the device keeps personal data local.
- **Cloud critic:** the cloud checks or corrects an on-device answer in the background (privacy permitting).

**Hedged requests:** when the pre-router is unsure, launch the cloud call in parallel with local inference, and use whichever finishes first or better. This cuts tail latency, but costs radio energy, cloud cost and privacy exposure, so allow it only where policy permits.

## 3. Confidence calibration and uncertainty

**Why raw confidence is unreliable:** instruction-tuned and RLHF'd models are overconfident, quantization shifts logits, small models are more fragile, and domain shift breaks calibration.

**What you need for routing:** mainly **discrimination** (does confidence rank correct above incorrect answers?) plus a good threshold. Perfect probability calibration is secondary.

**Metrics:**
- **AUROC of confidence for correctness** (ranking quality).
- **Risk–coverage curve** and its area (AURC): plot error rate among accepted answers vs the fraction accepted.
- **Selective accuracy** at a fixed coverage.
- **ECE, Brier score and NLL** for calibration, with reliability diagrams.

**Methods:**
- **Temperature scaling** per task type on a held-out set.
- **A confidence model:** logistic regression or gradient-boosted trees on features (min log-prob, entropy, answer length, task type, image quality, OOD score, probe output), with isotonic or Platt calibration on top.
- **Per-slice calibration** (OCR vs scene Q&A, low light), since calibration varies a lot by slice.
- **Re-calibrate after every model update and quantization change** (INT4 shifts logits).
- **Conformal risk control:** choose a threshold that guarantees an error rate of at most ε among accepted answers, assuming exchangeability. Distribution shift breaks the guarantee, so monitor it.

**Labels for training the confidence model:**
- Gold labels are best but expensive.
- **Agreement with the cloud model** is a proxy for "matches the stronger model", not for truth. Combine it with LLM-judge scores and human audits.
- Closed-form tasks (classification, OCR strings) can use exact match.

**Free-form outputs are hard.** Use the confidence of the first tokens or the answer span, claim-level checks, or semantic entropy.

**Dynamic thresholds:** `τ(context)`. For example, a lower deferral rate when offline or on low battery, a higher one on Wi-Fi while charging. A control loop can adjust τ online to hit a **target deferral rate** (a cloud cost budget).

**Production monitoring:** audit calibration on sampled traffic, and track drift in the deferral rate. A small, consented **shadow sample** can measure agreement with the cloud.

## 4. Privacy constraints

### 4.1 Data classes and routes

| Class | Examples | Allowed routes |
|---|---|---|
| P0: non-sensitive | Generic scenes, public objects | Any |
| P1: personal, low sensitivity | Everyday photos without faces or documents | Cloud with consent and redaction |
| P2: sensitive | Faces, documents, financial or health content, children, home interiors, screens with personal data | **On-device only**, or cloud only with explicit opt-in and redaction |
| P3: regulated or region-bound | Data-residency rules (GDPR, India's DPDP, Korea's PIPA, verify) | Region-specific endpoints, or on-device |

### 4.2 Design rules
- **The privacy gate runs before any confidence logic and fails closed.** If the sensitivity classifier is unsure, treat the input as sensitive. Optimize it for **recall**, since a false negative is the worst failure (a leak).
- **Data minimization:** send only what the cloud needs. Crop to the region of interest, downscale, redact faces and plates, mask text PII (the Q8 pipeline, run on-device), or send a **structured text description** instead of pixels.
- **Per-category user consent**, a visible indicator when data leaves the device, and enterprise or device-management policy support (verify the specifics for Samsung's management stack).
- **Cloud side:** encrypted transport, **no retention or short TTL**, no training on user data without consent, attested confidential-computing enclaves (as in Apple's Private Cloud Compute design), and isolation of shared caches (prefix-cache timing channels, Q15).
- **Prompt injection (Q25):** text in a scene can try to steer a cloud model that has tool access. Sanitize inputs and restrict tool capabilities.
- **Auditability:** log routing decisions with reason codes, not content.
- **Router training data:** use only consented or privacy-preserving data (opt-in sampling, on-device labeling, federated analytics).

## 5. Latency SLOs and routing economics

**Latency model:**
- `T_local = T_enc + T_prefill + n · TPOT_local` (Q12).
- `T_cloud = T_upload + RTT/handshake + T_queue + T_prefill_cloud + n · TPOT_cloud + T_download`.
- A **deferred** query costs `T_local_partial + T_cloud`. Expected latency of a cascade: `E[T] = T_local + p_defer · T_cloud`.

**Typical shape:**
- **Upload is the dominant cost for images.** A 12 MP JPEG can be about 3 MB, so downscale to about 1,024 px (roughly 200 to 300 KB) before sending. At 10 Mbps uplink that is about 0.25 s, at 2 Mbps about 1.2 s.
- Keep connections warm (HTTP/2 or QUIC, connection reuse) to avoid repeated TLS handshakes.
- For short answers, local is often faster. For long or hard answers, cloud decode speed can win. For example, 200 tokens at 25 tokens/s locally takes about 8 s, versus perhaps 2 to 3 s end to end in the cloud (assumed).

**Reducing the deferral penalty:**
- **Decide early.** After prefill, but before decode, check a cheap signal (the probe on hidden states) and defer immediately if it is bad.
- **Hedged requests** for latency-critical classes where policy allows.
- **Progressive refinement:** show the local answer first and update it if the cloud answer is materially better. Beware flip-flopping (see section 7).

**Cost and the budget:** cloud cost per query (image tokens and GPU time), a target deferral rate (for example 10 to 20%), and radio and tail energy (Q16).

**Worked example (assumed numbers):** the local model is 80% accurate overall. With good confidence ranking, deferring the 20% least-confident queries leaves 88% accuracy on the 80% answered locally. The cloud is 93% accurate on the deferred 20%.
- Overall accuracy = `0.8 × 0.88 + 0.2 × 0.93 = 0.89`, against 0.93 for all-cloud.
- That is about **96% of all-cloud quality at 20% of the cloud cost**, and most queries get a fast local answer.

## 6. Offline fallback and resilience
- **Detect connectivity quality, not just presence:** passive measurements of recent RTT and throughput, quick probes, and a **circuit breaker** that stops calling the cloud after repeated failures and retries after a cooldown.
- **Timeouts from the latency budget:** `cloud_timeout = SLO − elapsed`, with a local fallback if the cloud is late.
- **Graceful offline behaviour:**
  - The on-device model is a **complete product at its tier**, with clear UX about what works offline.
  - For a knowledge-heavy query while offline, answer from what is visible with explicit uncertainty, or queue the request for when connectivity returns (async, with a notification).
  - Cache previous cloud answers (keyed by image hash and question, privacy permitting).
- **Retries and idempotency:** exponential backoff, idempotency tokens, resumable uploads.
- **Overload handling:** on rate limits (429) or outages, fall back to local. Support feature flags, a **kill switch** for remote routing, and load shedding.

### Mode matrix

| Mode | Behaviour |
|---|---|
| Online, good link | Normal cascade, normal threshold |
| Online, poor or metered link | Higher confidence needed to defer, smaller payloads, longer timeouts avoided |
| Offline | On-device only, explicit uncertainty, queue knowledge-heavy requests |
| Privacy mode | On-device only, no hedging, no logging |
| Low battery or thermal stress | Prefer the cheaper route by energy (can favour cloud on strong signal, or a smaller local tier), degrade gracefully (Q16) |

## 7. Consistency of behaviour between the two models

**What goes wrong:** different style and level of detail, different output formats or JSON structure, different refusal and safety behaviour, different units and language, different perception (one model reads small text and the other doesn't), contradictory facts across turns after a route switch, different tool-call schemas, and different handling of "I don't know".

**Mitigations:**
- **One behaviour contract:** the same system prompt and persona, output schema, safety policy, tool interface and language policy. Run a **golden behaviour test suite on both models** (schema validity, style, refusal parity).
- **Align the models by training:** distill the cloud model's style, formats and safety behaviour into the on-device model (Q10), and share the same input and output guardrails on both paths.
- **Canonical conversation state in the app:** keep the conversation as text plus structured memory, and pass a compact summary and recent turns on a route change (not KV-cache). Send only the necessary context, for privacy.
- **Sticky routing with hysteresis:** once a conversation escalates to the cloud, stay there for the session or until the topic changes, so answers don't flip-flop.
- **Answer reconciliation:** if the cloud answer differs materially from a local answer that was already shown, present it explicitly as an update, and only when the cloud is more confident. Avoid silent contradictions.
- **Version management:** the on-device model updates slowly and the cloud model updates quickly, so keep a compatibility matrix, canary and A/B tests, and regression tests across model pairs.
- **Measure it:** agreement rate on a shared evaluation set, format-valid rate, refusal-rate parity, style similarity, and the **flip-flop rate** within sessions. Treat user-visible inconsistencies as bugs.

## 8. Evaluating and rolling out the router

**Offline evaluation:**
- Build a dataset of (query, image, local answer and features, cloud answer, gold or judge label). Train the router and evaluate the risk–coverage curve, the cost–accuracy Pareto front, and latency under **real network traces** and thermal states.
- Slice analysis, with the privacy-sensitive slices required to show **zero** leaks.
- **Off-policy evaluation** (inverse propensity or doubly robust estimators) to evaluate new policies from logs without deploying them.

**Online rollout:** shadow mode (compute decisions without acting), then A/B with guardrail metrics (task success, p95 latency, cloud cost per user, deferral rate, privacy violations = 0, user feedback, timeouts), staged rollout and a kill switch.

**Monitoring:** deferral-rate drift, calibration drift (ECE on audited samples), cloud timeouts and errors, privacy-classifier recall audits, and version-skew dashboards.

**Failure modes:**
- **Confidently wrong local answers** (silent hallucinations).
- Over-sending (cost) or under-sending (quality).
- **Privacy classifier false negatives** (the worst case).
- Threshold drift after a model update.
- **Selection bias in router training:** only deferred queries have cloud answers. Keep a small, consented random exploration sample to get unbiased estimates.
- Adversarial inputs designed to force or avoid cloud routing.

## 9. Sketch of the routing logic

```python
def route(req, ctx):
    # 1. hard policy gates (fail closed)
    if ctx.privacy_mode or sensitivity(req) >= SENSITIVE or not user_consent(req):
        return LOCAL
    if not ctx.online or breaker_open():
        return LOCAL
    # 2. cheap pre-router
    pre = pre_router(req.features)               # task type, OOD score, needs_knowledge
    if pre.needs_knowledge or pre.ood_score > T_OOD:
        return CLOUD_REDACTED
    # 3. run local, then decide post-hoc
    ans, feats = local_vlm(req)                  # logprob stats, entropy, hidden-state probe
    conf = confidence_model(feats, req.task)     # calibrated P(correct)
    tau = threshold(ctx)                         # connectivity, battery, cost budget
    if conf >= tau:
        return ans
    # 4. defer, with a timeout derived from the SLO and a local fallback
    return cloud(redact(req), timeout=ctx.slo - elapsed()) or ans
```

## 10. Strong interview answer
> "I treat routing as a constrained decision. Privacy is a hard gate that fails closed, and the rest is utility: quality minus latency, cost and energy penalties. The design is layered. First, policy gates: a sensitivity classifier tuned for recall, user consent, region rules, and offline or privacy mode, which force on-device processing. Second, a cheap pre-router on task type, out-of-distribution score and knowledge need, so queries that clearly need the cloud don't waste local compute. Third, run the on-device VLM and compute a calibrated confidence from token statistics, a probe on hidden states, and grounding checks like whether the answer depends on the image, because token probabilities alone miss confident hallucinations. I'd measure ranking quality with AUROC and the risk–coverage curve, calibrate per task slice, re-calibrate after each model or quantization update, and use conformal-style thresholds. The deferral threshold is dynamic by context, like connectivity, battery and a target deferral-rate budget. On latency, I'd model both paths, compress images to about 300 KB before upload, reuse warm connections, decide early after prefill, and use hedged requests only where policy allows. With assumed numbers, deferring the least-confident 20% could give about 96% of all-cloud accuracy at 20% of the cost. For privacy, I'd minimize what leaves the device: crop, downscale, redact PII, or send a text scene description instead of pixels, since features can be inverted. For offline, I'd use quality-aware connectivity detection, a circuit breaker, timeouts from the SLO, queueing and complete offline behaviour at the on-device tier. For consistency, I'd use one behaviour contract and tests on both models, distill the cloud model's style and safety into the on-device model, keep canonical conversation state in the app, use sticky routing to avoid flip-flopping, and measure agreement and flip-flop rates. I'd roll the router out with shadow mode, off-policy evaluation, A/B guardrails, and a small consented exploration sample to avoid selection bias."

## 11. Follow-up questions to prepare for
1. Why does discrimination (AUROC) matter more than calibration for routing, and when does calibration still matter?
2. Why do token probabilities fail for confident hallucinations, and what extra signals fix that?
3. How would you choose the deferral threshold to hit a cloud cost budget while keeping an accuracy floor?
4. Why is sending visual features to the cloud not automatically private or bandwidth-efficient?
5. How would you handle a user conversation that starts local and escalates to the cloud mid-way?
6. What are the risks of hedged requests, and when would you allow them?
7. How do you avoid selection bias when training the router from logs where only deferred queries have cloud answers?
8. How would you build and evaluate the privacy sensitivity gate, and what recall would you require?

---

# Q18: Efficient vision backbones

**Question:** Compare MobileNetV3/V4, EfficientNet, MobileViT, FastViT, EfficientFormer and RepViT, and distilled ViTs (DeiT-Tiny, TinySAM, EdgeSAM). What operators are NPU-friendly (e.g. depthwise convs, GELU or LayerNorm costs, attention quantization)? How do structural re-parameterization and NAS help on a specific target device?

Accuracy and parameter numbers are as reported in the papers at 224 px ImageNet, recalled from memory. Verify before quoting. Latency claims depend on the device and compiler, so treat them as directional.

## 0. The core lesson: FLOPs are not latency
On a mobile accelerator, latency is set by **data movement, operator support, parallelism and kernel overhead**, not just MAC count. ShuffleNetV2's guidelines state this well:
- Equal channel widths minimize memory access cost.
- Excessive group convolution raises memory cost.
- Fragmented, many-branch structures reduce parallelism.
- Element-wise ops (activations, adds, concatenations) are **not negligible**.

So a model with 3x fewer FLOPs can be slower on your NPU. The only reliable metric is **measured INT8 latency on the target, compiled with the real toolchain**.

## 1. The families

| Family | Type | Params / ImageNet top-1 (reported) | Key ideas | Edge notes |
|---|---|---|---|---|
| **MobileNetV3** | CNN | L: ~5.4M, ~75.2% | Inverted residual + depthwise convs, squeeze-excite (SE), h-swish, NAS (MnasNet/NetAdapt) | Depthwise and SE are awkward on matrix-engine NPUs. Per-tensor PTQ is fragile |
| **MobileNetV4** | CNN / hybrid | Several sizes | **Universal Inverted Bottleneck (UIB)** that unifies several block variants, **Mobile MQA** attention, NAS, designed with a roofline view to be Pareto-efficient across CPUs, GPUs and accelerators | Explicitly device-agnostic, and the multi-query attention cuts memory traffic (reported large speedup on accelerators, verify) |
| **EfficientNet** | CNN | B0: ~5.3M, ~77.1% | Compound scaling of depth, width and resolution, MBConv + SE + Swish | **EfficientNet-Lite** removes SE and Swish for edge accelerators and quantization. **EfficientNetV2** adds **Fused-MBConv** in early stages |
| **MobileViT** | CNN + ViT hybrid | S: ~5.6M, ~78.4% | Local conv, then unfold into patches, a transformer across patches, fold back | `unfold/fold` reshapes and attention make it **slower on-device than its FLOPs suggest**. MobileViTv2 uses linear-complexity separable attention |
| **EfficientFormer** | Hybrid | L1: ~12.3M, ~79.2% | **Latency-driven design** measured on a phone. Found that large-kernel patchify, LayerNorm and GELU were slow. Uses 4D conv/pool blocks with BN early and attention only in late stages | Reported about 1.6 ms on an iPhone 12 (Core ML) for L1 |
| **RepViT** | Pure CNN | M1.0: ~6.8M, ~80.0% | Brings ViT-style macro design (separate token and channel mixing, early convs, downsampling) into a MobileNet-like CNN, with **re-parameterized** depthwise convs | Reported about 1 ms on an iPhone 12. No attention to quantize |
| **FastViT** | Hybrid | T8: ~3.6M, ~76%; SA12: ~11M, ~80% | **RepMixer** (re-parameterizable token mixer without skip connections at inference), large-kernel depthwise convs, train-time overparameterization, attention only in the last stage | Reported 2 to 5x speedups over several baselines at similar accuracy |
| **MobileOne** | CNN | Several sizes | RepVGG-style structural re-parameterization with many train-time branches | Simple single-path inference graph |
| **DeiT-Tiny** | Pure ViT | ~5.7M, ~72.2% (higher with distillation) | Standard ViT trained data-efficiently, with a distillation token | Global attention, LayerNorm and GELU in every block, which are costly on NPUs |
| **TinyViT** | Hybrid | 5M: ~79%, 11M: ~81.5% (with large-data distillation) | Fast pretraining distillation with cached teacher logits | Good for distilled small encoders |
| **MobileSAM / TinySAM / EdgeSAM** | Distilled segmentation encoders | — | MobileSAM: distills SAM's ViT-H image encoder (~632M) into a TinyViT-5M. TinySAM: online hard-prompt distillation + W8A8 PTQ. **EdgeSAM**: prompt-in-the-loop distillation into a RepViT-based CNN, reported above 30 FPS on an iPhone 14 | Shows how to move heavy ViT capabilities to the edge by distilling into a CNN-friendly backbone |
| **EfficientViT** (MIT) | Hybrid | Several sizes | ReLU-based **linear attention** with multi-scale depthwise convs | Strong for high-resolution dense tasks. Note: Microsoft's "EfficientViT" with cascaded group attention is a different model |

**Also worth knowing:** FasterNet (partial convolution to cut memory access), ConvNeXt (7×7 depthwise + LayerNorm + GELU, strong but heavy for NPUs), GhostNet, ShuffleNet (shuffle and concat cost memory traffic), and **MobileCLIP** (Apple), whose image encoders are FastViT-style hybrids trained with multimodal reinforced training for language-aligned features.

## 2. Operator-by-operator view of NPU friendliness

| Operator | NPU friendliness | Why and what to do |
|---|---|---|
| **1×1 conv / matmul** | Excellent | Dense, high arithmetic intensity, maps straight to the MAC array |
| **Standard 3×3 conv** | Excellent | Good reuse. **Fused-MBConv** uses it in early stages for that reason |
| **Depthwise conv (3×3, 5×5)** | Mixed | Each activation is reused only K² times, so it is **memory-bound** and underuses large MAC arrays (some NPUs run it on slower vector units). Fine in later stages with many channels at low resolution. Per-tensor INT8 is fragile because per-channel weight ranges vary a lot (use per-channel weights, equalization, QAT) |
| **Large-kernel depthwise (7×7)** | Check | May be slow or unsupported on some NPUs. Verify |
| **ReLU / ReLU6** | Excellent | Cheapest. ReLU6's bounded range helps quantization |
| **h-swish / h-sigmoid** | Good | Piecewise linear, often supported. Fused support varies |
| **SiLU / Swish** | Fair | Needs sigmoid and a multiply, or a LUT. More quantization-sensitive |
| **GELU** | Fair to poor | Erf or tanh approximations, often a LUT or FP16/vector fallback |
| **Softmax** | Fair | Needs exp, a sum and a divide, usually FP16 or INT16 and memory-bound |
| **BatchNorm** | Excellent | Folds into the preceding conv at inference, so it costs nothing |
| **LayerNorm / RMSNorm** | Poor | A reduction per token, not foldable, memory-bound, needs higher precision, and often needs layout transposes. Prefer BN in conv stages and keep LN to a few attention blocks |
| **Squeeze-excite (SE)** | Poor | Global pooling + FCs + broadcast multiply: a serialization point and memory-bound. EfficientNet-Lite drops it |
| **Self-attention (MHSA)** | Poor to fair | QKᵀ is N², softmax, AV, plus head reshapes and transposes (layout changes cost DMA). Use only at the lowest-resolution stage |
| **Window partition / shift, unfold / fold, roll, gather, einsum** | Poor | Often unsupported or CPU fallbacks. Avoid in deployment backbones |
| **Concat / channel shuffle / split** | Poor to fair | Pure memory copies |
| **Skip connections** | Fair | Need the activation kept in memory. RepMixer-style designs remove them at inference |
| **Resize (bilinear)** | Fair | Usually supported. Bicubic often is not (precompute offline, Q3) |

**Layout and shape rules:** many NPUs prefer **NHWC** with channel counts a multiple of 8, 16 or 32, a limited tensor rank (often ≤4), and supported kernel and stride combinations. Pick channel widths and resolutions accordingly. A width multiplier that produces odd channel counts wastes MAC lanes.

**Quantization notes (Q11):**
- Depthwise-heavy CNNs need per-channel weight quantization and often QAT or equalization.
- LayerNorm and softmax outliers hurt attention-based backbones.
- **Attention quantization:** keep softmax and LayerNorm in FP16 or INT16, use W8A8 for the projections, and watch for long-tailed post-softmax distributions.
- Re-parameterized kernels can be hard to quantize (section 3).

## 3. Structural re-parameterization

**Idea:** train with a rich multi-branch structure, then **merge the branches into a single operator** for inference. The inference model is a plain single-path network.

**RepVGG-style recipe (3×3 + 1×1 + identity, each with BN):**
1. Fold each BN into its conv: `W' = W·γ/√(σ² + ε)`, `b' = β − μ·γ/√(σ² + ε)`.
2. Pad the 1×1 kernel to 3×3, and express the identity as a 3×3 kernel (a 1 at the center per channel).
3. Sum the kernels and the biases. The result is one 3×3 conv with the same function.

**Why it helps on-device:**
- Fewer memory accesses, no skip-connection activations to keep, fewer kernel launches, and a graph that compilers fuse well.
- You keep the **accuracy benefit of train-time capacity** with inference-time simplicity (RepVGG, MobileOne, RepViT, FastViT's RepMixer, and RepLKNet-style large kernels made from parallel small kernels).

**Caveats:**
- **Quantization:** the fused kernel can have a very different weight and activation distribution, and PTQ on a re-parameterized model can degrade a lot (the known RepVGG quantization problem, addressed by approaches like RepOpt or QAT). Plan QAT.
- **Training cost:** multi-branch training uses more compute and memory.
- **Exactness:** do the fusion in FP32 and in eval mode, and check numeric equivalence (for example max error around 1e-5) before export. Fusing in FP16 can lose precision.
- Fusion must happen **before** export, and it only works across linear operations.

## 4. NAS and hardware-aware design for a specific target

**What hardware-aware search does:**
- **MnasNet:** the search reward includes **measured latency on a phone**, roughly `ACC × (LAT/T)^w`.
- **FBNet:** differentiable NAS with a **latency lookup table**.
- **ProxylessNAS:** showed that the best architecture differs for GPU, CPU and mobile.
- **Once-for-All:** train one supernet, then specialize subnets for many devices with a latency predictor.
- **MobileNetV3:** NAS plus NetAdapt layer-wise tuning.
- **MobileNetV4:** designs with a roofline view of hardware (the ridge point) to find architectures that are efficient across many devices.
- Latency predictors: nn-Meter, HELP (meta-learned predictors).

**A practical workflow for your target NPU:**
1. **Characterize the device** with the real compile stack (QNN, ENN, ExecuTorch delegate, and so on). Micro-benchmark every op type across shapes (convs, depthwise, 1×1, matmul, LayerNorm, softmax, GELU, resize) in **INT8**, to build a latency table and a roofline (peak compute vs bandwidth).
2. **Define the search space** using only supported ops, hardware-aligned channels, a limited attention placement, fixed-shape buckets and a bounded number of tokens for the LLM (Q2).
3. **Multi-objective search:** maximize accuracy (supernet or zero-cost proxies) subject to latency ≤ T, memory and energy on the target. Remember that per-op latency tables are not perfectly additive (fusion and layout effects), so validate with real compiles, or use a predictor trained on device measurements.
4. **Train the winner** with distillation from a large teacher, then QAT.
5. **Re-search when the hardware or compiler changes.** Results are device- and compiler-version-specific. An architecture tuned for one NPU may be suboptimal on another, which matters for a multi-vendor product (Q13).

**Beyond architecture:** the resolution, width multiplier and depth are also knobs, and they tie directly to the token budget (Q2).

## 5. Relevance to small VLMs
- **A flat ViT patchify (stride 14 or 16) makes many tokens.** Hierarchical **conv or hybrid encoders downsample progressively** (stride 32 or 64), so they can run at high resolution and output **far fewer tokens** to the LLM. Apple's FastVLM, for example, uses a hybrid FastViT-style encoder (FastViTHD) with a large output stride, reporting large time-to-first-token improvements at high resolution (verify the numbers). This attacks the visual-token explosion (Q2) at the encoder, not only in the projector.
- **Language-aligned pretraining matters:** the encoder must produce features the LLM can use. MobileCLIP-style multimodal reinforced training, or distillation from a big SigLIP-like encoder (Q10), is how small hybrid encoders get there. ImageNet top-1 alone does not predict VLM quality (Q4).
- **Dense tasks share the backbone:** RepViT-SAM and EfficientViT-SAM show that one efficient backbone can serve segmentation and detection (Q19).
- **Evaluate with a plug-in test (Q4):** plug each candidate into a small VLM alignment run and compare on TextVQA, DocVQA and OCR tasks. OCR needs fine features, so a very large stride can hurt text detail unless the input resolution is raised (verify per model).

## 6. NPU-friendly backbone checklist
1. Prefer **1×1 and 3×3 standard convs and matmuls**. Use **Fused-MBConv** in early high-resolution stages and depthwise (MBConv) in late low-resolution, high-channel stages.
2. Use **BN in conv stages** (foldable), and LayerNorm in only a few attention blocks.
3. Prefer **ReLU, ReLU6 or h-swish** over GELU and SiLU when accuracy allows. Avoid or replace SE blocks.
4. Put **attention only at the lowest-resolution stage**, and consider MQA or ReLU linear attention. Avoid window shifts, unfold/fold, roll, gather and einsum.
5. **Re-parameterize** to a single-path inference graph, and plan QAT.
6. Use **hardware-aligned channel counts** (multiples of 8/16/32) and fixed shape buckets.
7. Use **bounded activations and per-channel weights** for quantization.
8. **Downsample progressively** so the LLM gets few tokens.
9. Check the **compiler's partition report early** and aim for zero CPU fallbacks.
10. **Measure INT8 latency on the device**, not FLOPs.

## 7. Strong interview answer
> "On an NPU, FLOPs don't predict latency, because memory movement, operator support and parallelism do, so I benchmark INT8 latency on the target with the real compiler. Among the families, MobileNetV3 and EfficientNet rely on depthwise convs and SE, which are memory-bound and under-use matrix engines, and EfficientNet-Lite and Fused-MBConv fix that. MobileViT looks cheap in FLOPs but unfold/fold and attention make it slow. EfficientFormer was designed by measuring latency on a phone and found large-kernel patchify, LayerNorm and GELU to be slow, so it uses BN and conv or pooling early and attention only at the end. FastViT and RepViT use structural re-parameterization to get a plain single-path inference graph, and MobileNetV4 designs for many devices with UIB and mobile multi-query attention. DeiT-Tiny is a pure ViT with LayerNorm, GELU and global attention everywhere, which is costly on NPUs. TinyViT, MobileSAM, TinySAM and EdgeSAM show how to distill heavy ViTs into edge-friendly backbones. NPU-friendly ops are 1×1 and 3×3 convs and matmuls, ReLU or h-swish, and foldable BN. Costly ones are LayerNorm, GELU, softmax, SE, window attention and reshape-heavy ops. For attention quantization, I'd keep softmax and LayerNorm in higher precision. Re-parameterization merges multi-branch training structures into one conv for faster inference with accuracy kept, but it makes PTQ harder, so I'd plan QAT. Hardware-aware NAS, like MnasNet, FBNet and Once-for-All, puts measured device latency into the objective, but results are device- and compiler-specific, so I'd build an op-level latency table for the target, search within supported ops and aligned channel counts, validate with real compiles, then distill and QAT. For small VLMs, hierarchical hybrid encoders with a large output stride, like the FastViTHD approach, also cut visual tokens at the source, and I'd pick the encoder with a plug-in test on OCR and VQA, not ImageNet accuracy."

## 8. Follow-up questions to prepare for
1. Why are depthwise convolutions slow on matrix-engine NPUs despite having few FLOPs?
2. Why did EfficientNetV2 use Fused-MBConv in early stages and MBConv later?
3. Derive how a RepVGG block's three branches merge into one 3×3 conv. Why is the result hard to quantize?
4. Why can LayerNorm be expensive on an NPU, and how do BN-based hybrids avoid it?
5. How does ReLU linear attention (EfficientViT) achieve O(N) attention, and what does it give up?
6. What is the trade-off between per-op latency lookup tables and a latency predictor for NAS?
7. How does a conv encoder with stride 64 reduce visual tokens, and what risks does it carry for OCR?
8. How would you build a MobileSAM-style distillation of a large vision encoder for your VLM?

---

# Q19: Dense perception on-device

**Question:** How would you deploy open-vocabulary detection or segmentation (Grounding-DINO, OWL-ViT, YOLO-World, SAM-lite, CLIPSeg) on-device? Cover real-time constraints, tracking-by-detection and keeping temporal consistency across frames. How do you combine these with a VLM for grounded reasoning (box or point outputs, visual prompts)?

Model facts (sizes, speeds, reported numbers) are from memory of the papers and project pages. Latency figures in budgets are illustrative assumptions. Verify specifics before quoting them.

## 0. The core design principle
Don't deploy the heavy, universal model. Deploy a **composed pipeline of right-sized models** that runs at different rates:
- A cheap detector and tracker run continuously.
- A SAM-lite segmenter runs on demand.
- The VLM runs only on user queries or events.

The key tricks are **caching or baking in text embeddings** (so the text encoder is not in the loop), **"ground once, then track"**, and **open-vocabulary at training time, closed-set at deploy time**.

## 1. The model families and how deployable they are

### 1.1 Open-vocabulary detection

| Model | Architecture | Text handling | On-device verdict |
|---|---|---|---|
| **OWL-ViT / OWLv2** | CLIP-style ViT, each output patch token predicts a box and a class embedding. Classification = dot product with text embeddings | Text and image are **decoupled**, so text embeddings can be precomputed and cached. Also supports image-conditioned (query-by-example) detection | Practical if the ViT is affordable. High resolution (e.g. 960 px, thousands of tokens) is heavy. ViT ops (LayerNorm, GELU, attention) are NPU-costly (Q18) |
| **Grounding DINO** | DETR-style detector (Swin backbone), a BERT text encoder, **cross-modality feature enhancer**, language-guided query selection, cross-modality decoder | Image features depend on the text prompt (fusion inside the encoder), so nothing about the image can be cached across prompts | **Hard to deploy:** heavy, **multi-scale deformable attention** (a gather-based custom op, poorly supported on NPUs). Use in the cloud or offline as an annotator. Lighter "edge" variants have been released (reported real-time-class on Jetson-level hardware, verify) |
| **YOLO-World** | YOLOv8-style CNN backbone + **RepVL-PAN** (vision-language path aggregation) with a region-text contrastive loss | **"Prompt-then-detect":** compute text embeddings offline and **re-parameterize them into the head weights**, so inference is a plain YOLO with no text encoder | **Best fit.** Standard NPU ops (convs, upsample, concat), small sizes (S/M/L), reported real-time speeds on a server GPU |
| **Florence-2** (0.23B/0.77B) | Seq2seq VLM with a DaViT encoder. Boxes and masks as location tokens | Prompted by text | Small enough for on-device, flexible (detection, grounding, captioning, segmentation), but slow for many objects (see section 2D) |
| **VLM-native grounding** (PaliGemma, Qwen2.5-VL, Molmo) | The VLM outputs boxes or points as text or special tokens | Prompted by text | Flexible, good on complex referring expressions, but autoregressive and slow |

### 1.2 Segmentation

| Model | Idea | Notes |
|---|---|---|
| **SAM** | Heavy ViT-H image encoder (~632M), light prompt encoder and mask decoder (a few M parameters). The encoder runs **once per image**, and the decoder runs **per prompt** (box or point) in milliseconds | The split design (amortized encoder, interactive decoder) is ideal for on-device |
| **MobileSAM, EdgeSAM, TinySAM, EfficientSAM, EfficientViT-SAM** | Distilled SAM encoders (TinyViT, RepViT, EfficientViT) | EdgeSAM is reported above 30 FPS on an iPhone 14 (Q18). Typically INT8-friendly |
| **SAM 2 and edge variants** | Streaming video segmentation with a memory bank. EdgeTAM is an on-device variant (reported around 16 FPS on a recent iPhone, verify) | Handles temporal consistency natively |
| **CLIPSeg** | Frozen CLIP ViT-B/16 + a small decoder conditioned on a text or image prompt, producing a coarse mask | Cheap and zero-shot, but low-resolution masks with blurry boundaries. Good for rough regions of interest |
| **Other open-vocab segmentation** | MaskCLIP (training-free dense CLIP features), SAN, CAT-Seg, SEEM, FC-CLIP | Mostly research-grade for edge |

## 2. Deployment decisions

### 2A. Detection: choose by how dynamic the vocabulary must be

| Need | Choice |
|---|---|
| Fixed set of 100 to 1,000 domain classes (home objects, appliances), real-time | **YOLO-World-style, vocabulary baked in** (OTA updates ship only new head weights, tiny) |
| Vocabulary changes at runtime but uses cached embeddings | Detector head takes text embeddings as input (non-re-parameterized mode), so a new prompt costs one text-encoder pass (CLIP text encoder ~63M, can run on-demand on CPU or NPU) plus a matmul |
| Truly free-form queries and complex relations | VLM-native grounding, on demand, or the cloud |
| High-quality open-vocab labels for your domain | **Grounding DINO + SAM as an offline annotator** (Q8), then **distill into a small closed-set detector** |

### 2B. The vocabulary-size problem
Open-vocab heads compute a score for every anchor and every class. With 8,400 anchors × 1,000 classes that is 8.4M scores per frame, which makes the head and the NMS expensive.
- **Restrict the vocabulary by context** (scene-conditioned: kitchen classes in a kitchen), or use a hierarchical vocabulary.
- Pre-filter to a top-K of candidate boxes before class scoring.
- Use mixed precision for the head (the cosine-similarity logits are quantization-sensitive, Q11).

### 2C. Segmentation: detector → box prompt → SAM-lite
- Run the SAM-lite **encoder** at a low rate (keyframes, or on demand) and cache the image embedding (e.g. 256×64×64 FP16 ≈ 2 MB).
- Run the **decoder** per prompt, batching all boxes from the detector together. A few ms.
- Resize and threshold the mask logits on the GPU or in the graph. Mask post-processing (upsampling, argmax, RLE or polygon conversion) is a hidden cost.
- Export notes: SAM-lite encoders use a fixed input size. The decoder has a variable number of prompts, so export fixed buckets (e.g. 1, 4, 8 prompts, padded) (Q13).

### 2D. Why VLM-native grounding is not for continuous use
Boxes are text tokens, about 4 to 8 tokens per box, so a scene with 10 objects can need about 200 output tokens. At 25 tokens/s that is about **8 seconds** (Q12). Use it for **on-demand** queries, then hand off to a tracker.

### 2E. Export and compiler gotchas
- Deformable attention (Grounding DINO) needs a `grid_sample`-style rewrite, which NPUs handle poorly.
- NMS is dynamic and data-dependent. Keep it **outside the NPU graph** (on the CPU), or use an NMS-free head or fixed top-K selection in-graph.
- Verify the op coverage report for every model early (zero CPU fallbacks in the hot path).

## 3. Real-time constraints
- **Frame budget:** 30 FPS gives about 33 ms per frame for the continuous path (preprocess + detect + postprocess + track + render).
- **Hierarchical rates** (the central idea):
  - Tracker at the camera rate (30 Hz).
  - Detector at 5 to 10 Hz, or event-triggered.
  - SAM-lite encoder at 2 to 5 Hz, or on demand.
  - VLM on a user query or every N seconds with an object list.
- **Latency vs throughput:** pipelining (ISP/GPU preprocess, NPU inference, CPU postprocess) raises throughput but latency is still the sum. AR overlays and robot control need end-to-end latency, not only FPS.
- **Resolution vs small objects:** use ROI crops predicted by the tracker, or tile at a lower rate, or adapt resolution to the scene.
- **Frame handling:** zero-copy YUV into the model, sensor timestamps, and "latest frame wins" queues so stale frames are dropped (Q15).
- **Memory (Q14):** detector tens of MB, SAM-lite encoder and decoder tens of MB, text encoder 30 to 60 MB INT8 (load on demand), plus the VLM (~1.1 GB).
- **Energy and thermal (Q16):** continuous detection costs power. Use event triggers, adaptive frame rate and ROI processing.

**Illustrative continuous-path budget:**

| Step | Time |
|---|---|
| Preprocess | ~3 ms |
| Detector (small YOLO-World, INT8, 640 px, on NPU) | ~15 ms |
| Postprocess + NMS | ~3 ms |
| Tracker | ~2 ms |
| Render / overlay | ~5 ms |
| **Total** | **~28 ms (about 30 FPS)** |

SAM-lite encoder (tens of ms) and the VLM stay **off** the per-frame path.

## 4. Tracking-by-detection and temporal consistency

### 4.1 Tracking algorithms
- **SORT:** a Kalman filter (constant-velocity box state) plus Hungarian assignment on IoU. Very cheap.
- **DeepSORT:** adds an appearance embedding (ReID network) and cascade matching.
- **ByteTrack:** a second association pass that also uses **low-confidence** detections, recovering occluded objects. No appearance model, very cheap, and popular for edge.
- **OC-SORT:** observation-centric, better with non-linear motion and occlusion.
- **BoT-SORT / StrongSORT:** camera-motion compensation and ReID.
- Kalman filters and the Hungarian algorithm are CPU operations on tens of objects, which is negligible.

### 4.2 Open-vocabulary specific problems
- **Label flicker:** text similarity scores hover near thresholds, so a "cup" becomes a "mug" or a "glass" from frame to frame.
- **Fixes:**
  - Aggregate class scores **per track** (running average or majority vote with hysteresis).
  - Average the **region embedding** over frames and compare that to text.
  - Use a high threshold to start a track and a lower one to keep it.
  - Decide the label after N frames.

### 4.3 Temporal consistency for masks
- **SAM 2-style memory** for streaming propagation, or per-frame SAM with flow-based warping.
- **Keyframes + propagation:** run the heavy model at keyframes. Propagate between them with a light method: optical flow (a small learned model or classical Lucas–Kanade), template trackers, or video-encoder motion vectors where the hardware exposes them (verify).
- **Smooth the outputs:** exponential moving average on mask logits, temporal median, and mask-IoU association.
- **Ego-motion compensation:** use the IMU to warp prior boxes during camera motion (Q20, Q21).
- **Re-detection and re-ID:** periodically re-prompt SAM with detector boxes, and re-identify objects after occlusion using appearance embeddings (DINO or CLIP crop embeddings).
- **Adaptive keyframes:** trigger the detector when tracker confidence drops, a new object enters (frame differencing), or the IMU shows large motion. Otherwise skip.

### 4.4 Metrics and failure modes
- **Metrics:** MOTA, IDF1, **HOTA** (more balanced), ID switches, fragmentation, plus a **flicker rate** (how often object presence toggles) and temporal mask-IoU stability.
- **Failure modes:** motion blur, rolling shutter, low light, occlusion, two identical objects (two chairs), objects leaving and re-entering, fast robot turns causing track loss.
- **Identity persistence matters downstream:** "the red cup" in the VLM context should refer to a stable track ID.

## 5. Combining with a VLM for grounded reasoning

### 5.1 Patterns

1. **Perception as structured context for the VLM.**
   - The on-device detector produces a compact scene description, for example `[{id, label, box, mask_area, depth, state}]`, which is added to the prompt alongside the image.
   - The VLM then reasons over objects ("which object is closest to the door?").
   - Because the object list is grounded, this reduces hallucination (Q9).
   - Token cost: about 20 objects × 25 tokens ≈ 500 tokens, similar to the image tokens, so balance the two (fewer visual tokens plus an object list).
   - **Set-of-Mark prompting:** overlay numbered masks on the image (from SAM) so the VLM can refer to "object 5". It helps small VLMs, but needs readable marks at the encoder's resolution.

2. **VLM as grounder, then segmenter and tracker ("ground once, track continuously").**
   - The VLM outputs boxes or points for a complex referring expression ("the mug left of the laptop that is not blue").
   - Those become SAM-lite prompts, and the resulting mask goes to the tracker.
   - The VLM re-grounds only on demand.
   - Output formats: location tokens (Florence-2, PaliGemma), JSON absolute coordinates (Qwen2.5-VL), points (Molmo).

3. **Region-conditioned VLM (visual prompts).**
   - **Crop-and-zoom:** use a detector box to crop the region at high resolution and feed the crop plus a global thumbnail. This spends tokens only where needed (Q2) and helps small objects and OCR.
   - **Drawn visual prompts:** boxes, arrows or circles overlaid on the image (ViP-LLaVA).
   - **Region features as tokens:** RoI-aligned features fed into the LLM (Ferret, GPT4RoI, Osprey style).
   - **Tap-to-ask UX:** the user taps a point, SAM-lite gives a mask, and the VLM answers about that region.

4. **Agentic loop with perception tools.**
   - The VLM requests tools ("detect cups", "segment object 3", "zoom into region X"), the on-device orchestrator executes them, and the VLM iterates.
   - Each call adds detector time plus a re-prefill for new crop tokens, so cap the number of steps.

5. **Verification.**
   - Check VLM claims against the detector ("there is a cat" → open-vocab score), to reduce object hallucination (Q9) and to feed uncertainty into routing and abstention (Q17).

6. **Add geometry from other sensors.**
   - Attach depth or LiDAR to each mask (for example the median depth inside the mask), project to 3D using the intrinsics, and give the VLM or planner metric positions ("how far is the cup?") (Q20, Q21).

### 5.2 Coordinate-frame bugs (a major source of errors)
The VLM sees a **resized, letterboxed, tiled or cropped** input, and its coordinates are relative to that input, in a normalized (for example 0 to 1000) or absolute-pixel format.
- Keep **explicit transform matrices** to map between the original frame, the model input, tiles and crops. For tiles, outputs are per-tile with offsets.
- Dynamic-resolution models (Qwen2-VL style) output absolute coordinates in the resized image.
- Unit-test with synthetic images that have known boxes, and check round trips for every preprocessing path.

### 5.3 Training and data
- Add grounding data (boxes in text) when fine-tuning the VLM (Q9), and choose one coordinate convention across the VLM and the detector.
- Auto-label domain data with Grounding DINO + SAM (Q8), then distill into the small detector and into the VLM's grounding skill.

## 6. Pipeline sketch

```python
def run_frame(frame, state):
    t = tracker.update(frame)                          # every frame, ~2 ms (flow / Kalman / ByteTrack)
    if state.need_detect(t):                           # every ~3rd frame, or tracker confidence drop, new motion, IMU jolt
        dets = detector(frame, vocab=state.vocab)      # baked or cached text embeddings
        tracks = associate(t, dets)                    # ByteTrack-style, class scores smoothed per track
    if state.need_masks(tracks):                       # on demand, keyframes only
        emb = sam_encoder(frame)                       # cached 2 MB embedding
        masks = sam_decoder(emb, boxes=[tr.box for tr in tracks])
    if state.user_query:                               # on-demand, off the per-frame path
        ctx = scene_json(tracks, masks, depth)         # grounded, compact
        answer, new_boxes = vlm(frame_or_crops, ctx, state.user_query)
        tracker.seed(map_to_frame(new_boxes))          # ground once, then track
    return tracks, masks
```

## 7. Evaluation
- **Open-vocab detection:** LVIS AP (especially rare classes) and COCO zero-shot, plus **domain AP per class**, small-object AP, and **false positives per image** (for safety-relevant classes like "person").
- **Runtime:** FPS, p95 frame time, energy per frame, thermal behaviour (Q16).
- **Tracking:** HOTA, IDF1, ID switches, flicker rate on **domain video** (home scenes, egocentric motion).
- **Segmentation:** mIoU and boundary F-score.
- **End to end:** grounded VQA accuracy **with vs without** the detector context, hallucination rate (POPE, Q9), and coordinate-mapping unit tests.
- **Slices:** low light, clutter, small objects, fast motion.

## 8. Strong interview answer
> "I'd deploy a composed pipeline, not one big model, with different components at different rates. For open-vocabulary detection, Grounding DINO is too heavy for edge: its image features depend on the text prompt and it uses deformable attention, a gather-based op NPUs handle badly. I'd use it offline as an annotator, together with SAM, and distill into a small closed-set detector. YOLO-World's prompt-then-detect design is the best on-device fit: text embeddings are computed offline and re-parameterized into the head, so inference is a plain CNN, and a vocabulary update ships as a tiny head-weights OTA. OWL-ViT also decouples text, so embeddings can be cached, but its ViT is costly at high resolution. I'd restrict vocabulary by scene context to avoid huge class-score tensors, keep NMS outside the NPU graph, and use mixed precision for the head. For masks, I'd run a SAM-lite encoder like EdgeSAM or MobileSAM at a low rate and cache the embedding, with the light decoder per box prompt batched. For real time, I'd split rates: tracker at 30 Hz, detector at 5 to 10 Hz or event-triggered, SAM encoder on demand, VLM on query. Tracking is ByteTrack or OC-SORT, with per-track class-score smoothing and hysteresis to stop open-vocabulary label flicker, SAM 2-style memory or flow-based propagation for masks, IMU ego-motion compensation, and periodic re-prompting. I'd measure with HOTA, IDF1 and a flicker rate. To combine with the VLM, I'd use four patterns: structured object lists as grounded context, Set-of-Mark prompts, 'ground once, then track', where the VLM emits boxes or points on demand that go to SAM and the tracker, and region crops or visual prompts for small objects. I'd also verify VLM claims with the detector. The big engineering risk is coordinate-frame mapping across resizing, letterboxing, tiling and crops, so I'd keep explicit transforms and test them with synthetic known-box images. VLM-native grounding is too slow for continuous use, since boxes cost many tokens and decode is bandwidth-bound."

## 9. Follow-up questions to prepare for
1. Why can YOLO-World re-parameterize the text embeddings into the head, and what does that cost in flexibility?
2. Why is Grounding DINO hard to deploy on an NPU, and what is the better use of it?
3. Why does SAM's encoder/decoder split fit on-device use, and how would you batch several box prompts?
4. How would you stop an open-vocabulary detector's labels from flickering across frames?
5. What triggers would you use to decide when to run the detector instead of the tracker?
6. Why is VLM-native grounding too slow for continuous detection, and how does "ground once, then track" fix that?
7. What are the common coordinate-mapping bugs between a tiled VLM input and the original frame?
8. How would you use depth or LiDAR to give a VLM metric positions of masked objects?

---

# Q20: Multimodal sensor fusion (RGB, depth, LiDAR, IMU)

**Question:** You have RGB, depth, LiDAR and IMU. Compare early, mid and late fusion. How would you feed depth or point clouds into a VLM (depth as a pseudo-image, 3D tokens like 3D-LLM / PointLLM / LLaVA-3D, BEV features)? Cover time synchronization, calibration (intrinsics, extrinsics, camera-LiDAR, camera-IMU), missing or degraded sensors and sensor dropout during training.

Paper and benchmark details are from memory (3D-LLM, PointLLM, LLaVA-3D, SpatialVLM, SpatialRGPT, ImageBind, BEVFusion, MetaBEV, Kalibr, VSI-Bench and others). Error numbers in the worked tables are simple geometry. Verify specifics before quoting them.

## 0. What each sensor brings

| Sensor | Gives | Weaknesses |
|---|---|---|
| **RGB camera** | Semantics, texture, color, text, high resolution | No metric scale, depends on lighting, motion blur |
| **Depth (ToF, stereo, structured light, or monocular-estimated)** | Metric (or relative) geometry per pixel | Low resolution (ToF), holes on glass, black or shiny surfaces, sunlight interference for IR sensors, range limits. Monocular depth has scale ambiguity |
| **LiDAR** | Accurate sparse 3D points, long range, lighting-independent | Sparse, no color or texture, noisy on glass and dark surfaces, motion distortion in spinning units |
| **IMU (accelerometer + gyroscope, 100 to 1,000 Hz)** | Orientation, gravity direction, short-term motion | Drift and bias, no appearance, needs integration or fusion |

They are complementary: RGB says **what**, depth and LiDAR say **where and how far**, IMU says **how the sensor moved**.

## 1. Early, mid and late fusion

| Level | What is fused | Examples | Pros | Cons |
|---|---|---|---|---|
| **Early (input or data level)** | Raw or low-level data combined before feature extraction | RGB-D as a 4-channel input, projecting LiDAR points into the image as a sparse depth channel, **PointPainting** (append image segmentation scores to LiDAR points) | One backbone, cheap, allows low-level cross-modal cues | Needs precise calibration and sync, breaks pretrained RGB encoders (the first conv expects 3 channels), modalities have different statistics, a failed sensor corrupts the input |
| **Mid (feature level)** | Per-modality encoders, then fused features or tokens (concat, sum, cross-attention, shared BEV space) | **BEVFusion** (camera and LiDAR features fused in BEV), TransFusion, FUTR3D | Usually the best accuracy, reuses pretrained per-modality encoders, handles different resolutions, supports modality gating and dropout | More compute (several encoders), needs a shared representation, more complex training |
| **Late (decision level)** | Per-modality models output detections or predictions, merged afterwards (NMS merge, Kalman or Bayesian fusion, voting, learned late fusion) | Detector per sensor + tracker | Modular, robust to a failed sensor, easy to synchronize asynchronous sensors, each model validated separately | Loses cross-modal cues, correlated errors, confidences must be calibrated across modalities |

**Mapping to a VLM (the three practical routes):**
1. **Pseudo-image route (early):** convert depth to an image and reuse the VLM's existing vision encoder.
2. **Token route (mid):** add a modality-specific encoder + projector and append the resulting tokens to the LLM input.
3. **Symbolic route (late):** convert sensor data into structured text (objects with metric positions, distances, motion state) and put it in the prompt.

## 2. Feeding non-RGB data into a VLM

### 2.1 Depth as a pseudo-image
- Render depth as a 3-channel image: a colormap (e.g. Turbo), a normalized grayscale replicated across channels, or **HHA encoding** (horizontal disparity, height above ground, angle with gravity).
- **Pros:** no new architecture, cheap, works with the existing encoder after fine-tuning.
- **Cons:**
  - 8-bit quantization and per-image normalization lose **metric precision**. Use a fixed range (e.g. 0 to 10 m mapped to 0 to 255) or a log scale.
  - Pretrained encoders have rarely seen colormapped depth, so it needs alignment data.
  - Depth-to-3D needs the **intrinsics**, so provide the focal length or normalize with it.
- **Related work:** SpatialVLM (builds metric spatial QA data using estimated depth, then trains a VLM that needs no depth at inference), SpatialRGPT (a depth plugin giving region-level relative depth), SpatialBot (RGB-D inputs). Verify details.
- **A separate depth encoder + projector** (mid-level) is the alternative: a small CNN or ViT whose tokens are appended or fused. It costs extra tokens (Q2), so budget them (e.g. 64 depth tokens next to ~320 RGB tokens), or fuse them into the RGB tokens with a small resampler.

### 2.2 Metric depth sources on-device
- **Hardware depth** (ToF, LiDAR): metric but low-resolution and sparse. Upsample or complete it with RGB guidance.
- **Monocular depth network** (Depth Anything V2-Small class, about 25M parameters): dense, but usually **relative** (scale and shift unknown). Metric variants (Metric3D, UniDepth, ZoeDepth, Depth Pro) exist but are heavier or intrinsics-dependent.
- **A strong on-device combination:** a dense relative-depth network plus **sparse metric points** (ToF or LiDAR). Fit a per-frame scale and shift (robust least squares or RANSAC), which gives **dense metric depth**.

### 2.3 Point clouds as 3D tokens
- **Point-based encoders** (PointNet++, Point-BERT, Point-MAE, Point Transformer v3, CLIP-aligned 3D encoders like ULIP-2, OpenShape and Uni3D) produce a set of tokens (say 512), which an MLP projector maps into the LLM.
  - **PointLLM** (object-level) follows this recipe with point-text data from Objaverse.
- **Lifting 2D features into 3D** (3D-LLM and successors): extract features from multi-view images with a strong 2D model and aggregate them onto 3D points. This leverages strong 2D pretraining and avoids scarce 3D-text data.
- **LLaVA-3D** (as I recall): adds **3D position embeddings** (from depth and camera poses) to the 2D patch tokens of multi-view images, so the model becomes 3D-aware **without extra tokens or a separate point encoder**. Video-3D-LLM and Chat-Scene (object-centric tokens with object IDs) are related designs.
- **Design choices:** object-level vs scene-level; point encoder vs 2D lifting; token budget (farthest-point sampling and voxel downsampling, grouping k-nearest points into patch tokens); 3D positional encoding (Fourier or MLP on xyz); a **gravity-aligned frame** (from the IMU) so "up" is consistent; **keep metric scale** (don't normalize clouds to a unit sphere if you need distances).
- **On-device issues:**
  - Mobile ToF and LiDAR clouds are small (hundreds to about 20k points per frame).
  - PointNet++-style ops (farthest-point sampling, ball query, kNN gather) are poorly supported on NPUs (Q13, Q18). Prefer **voxel or pillar encoders that produce a 2D pseudo-image** (PointPillars) or **range-image CNNs** for spinning LiDAR.
  - Sparse 3D convolutions are also poorly supported on NPUs.

### 2.4 BEV (bird's-eye-view) features
- Lift RGB to BEV (Lift-Splat-Shoot, BEVFormer, or inverse perspective mapping on a ground plane), turn LiDAR into BEV (pillars or voxels), and fuse in BEV. The BEV map `[C, H, W]` is patchified into tokens (pooled to about 64) and projected into the LLM.
- **Pros:** metric, ego-centric, natural for navigation and planning, and a shared space for all sensors.
- **Cons:**
  - The ground-plane assumption is weak indoors, where objects sit at different heights. Use multi-layer BEV, an occupancy or voxel grid.
  - View transforms (deformable attention, scatter/cumsum splatting) are NPU-unfriendly. With a fixed calibration you can precompute the mapping as a lookup table.
- **Fit:** a home robot needs it for navigation, while a phone camera usually doesn't.

### 2.5 Symbolic (late) route
- Provide structured text: object list with 3D positions and sizes, room layout, distances, and motion state ("robot moving 0.3 m/s", "device rotating"). Few tokens, interpretable, and robust.
- **Caveat:** LLMs are weaker at numeric reasoning, so use quantized coordinate bins or pre-compute the answers (e.g. nearest-object distance) in code, and let the VLM explain.

### 2.6 IMU
- Raw IMU rarely goes into the VLM as tokens. Options: a small 1D-CNN or transformer encoder aligned to the other modalities (IMU2CLIP, ImageBind includes IMU), or, more commonly, use the IMU to:
  1. provide **gravity direction and orientation**,
  2. **stabilize and compensate motion** (deskew LiDAR, warp tracks, Q19),
  3. enable **visual-inertial odometry** for camera poses used in multi-view fusion (Q21),
  4. classify **activity context** ("walking", "stationary") as text or metadata,
  5. **gate inference** (reuse results when stationary, skip frames during violent motion).

### 2.7 Which route for which task

| Task | Route |
|---|---|
| Metric distance or size questions | Dense metric depth + symbolic values (or 3D-aware tokens) |
| Navigation, "is the path clear" | BEV or occupancy + symbolic summary |
| Object-level 3D QA and grounding | 3D object tokens or lifted features |
| General scene description | RGB only (don't pay for other sensors) |
| Low-light, textureless scenes | Depth or LiDAR assisted, with RGB fallback |

## 3. Time synchronization

**Why it matters:** sensors differ in rate (camera 30 Hz, LiDAR 10 to 20 Hz, ToF 30 to 60 Hz, IMU 200 to 1,000 Hz), latency and clock domain. Cameras may be rolling shutter (each row has a different timestamp). A spinning LiDAR collects points over a whole sweep, so a moving platform **distorts** the cloud.

**Cost of small timing errors:**

| Error | Effect |
|---|---|
| 10 ms at 1 m/s translation | 1 cm |
| 10 ms at 100°/s rotation | 1° → about 8.7 cm lateral at 5 m, about 17 px at fx = 1,000 px |

For camera-IMU visual-inertial systems, even a few milliseconds of **time offset** hurts accuracy, so it is usually estimated online.

**Methods:**
- **Hardware sync:** a common trigger or shared clock (PTP / IEEE 1588, PPS, camera strobe signals) so all sensors timestamp against the same clock. On Android, check the camera's timestamp source (`REALTIME` means the camera timestamps are comparable with sensor-event timestamps, verify on your device).
- **Timestamp at capture, not at arrival.** A common bug is using message arrival time (e.g. ROS `header.stamp` set when received).
- **Software alignment:** ring buffers per sensor, **interpolate** IMU to the camera timestamp, **SLERP + linear** pose interpolation, tolerance-based nearest-neighbor matching (e.g. ≤ 5 ms, otherwise drop), handling of jitter, drops and out-of-order data, and a linear clock-drift model.
- **LiDAR deskewing:** use IMU or odometry to transform each point to a common timestamp (e.g. the camera exposure time).
- **Rolling shutter:** model per-row timing and use per-row poses from the IMU for fast motion.
- **Estimate offsets by calibration:** e.g. cross-correlate gyro angular velocity with optical-flow-derived rotation, or use Kalibr and VINS/OpenVINS-style online temporal calibration.
- **Bundle assembly for the VLM:** a snapshot at time t: `{rgb(t), depth(t±δ), deskewed lidar(t), IMU window [t−Δ, t]}`. Use sliding windows for video.
- **Test it:** a blinking LED or tap test (a visible event captured by the camera and a spike in the IMU), checkerboard-motion tests, and production monitoring of timestamp deltas with alerts.

## 4. Calibration

### 4.1 Intrinsics
- **Camera:** focal lengths and principal point `(fx, fy, cx, cy)` plus distortion (radial-tangential, or fisheye / equidistant models). Calibrated with checkerboard or AprilGrid targets (Zhang's method, OpenCV, Kalibr).
- **Per-frame variation:** on phones with OIS and autofocus, intrinsics and distortion can change per frame (focus breathing, lens shift). Android exposes per-frame calibration metadata (verify the keys on your device).
- **LiDAR:** beam angles and per-beam offsets (factory calibration). **Depth sensors:** their own intrinsics, and the aligned-to-color depth mapping.
- **IMU intrinsics:** scale and axis misalignment, bias, noise densities (Allan variance).

### 4.2 Extrinsics (rigid transforms between sensors)
- **Camera ↔ LiDAR:** target-based (checkerboard plane fitted in LiDAR points), or targetless (edge or mutual-information alignment, motion-based hand-eye calibration `AX = XB`, deep methods such as CalibNet / LCCNet).
- **Camera ↔ IMU:** spatio-temporal calibration (Kalibr) that estimates the transform **and** the time offset, using excitation motion in front of a target.
- **Camera ↔ depth:** register depth to the color frame using the extrinsics, with occlusion handling (z-buffering) when reprojecting.
- **Conventions are a classic bug:** optical frame (z forward) vs robot base frame (x forward), handedness, **quaternion order (xyzw vs wxyz)**, units. Store everything as a transform tree and unit-test it.

### 4.3 Sensitivity to calibration error (geometry)

| Rotation error | Lateral error at 5 m | At 10 m | Pixel error at fx = 1,000 px |
|---|---|---|---|
| 0.1° | 0.9 cm | 1.7 cm | 1.7 px |
| 0.5° | 4.4 cm | 8.7 cm | 8.7 px |
| 1.0° | 8.7 cm | 17.5 cm | 17.5 px |

A few pixels of error cause **semantic bleeding** when painting LiDAR points with image labels, and wrong depth at object edges.

### 4.4 Drift and robustness
- Thermal expansion, shocks, aging and device flex all change calibration. Monitor **reprojection and photometric consistency** (LiDAR edges vs image edges), detect drift, and re-calibrate online with targetless methods.
- **Augment training with random extrinsic perturbations** (rotation noise of about 0.1 to 1°, translation noise of centimeters) so the model tolerates miscalibration.
- **Multi-device products:** per-unit factory calibration must ship with the data and the app. Make models **camera-aware**, for example by conditioning on intrinsics or normalizing depth to a canonical focal length (Metric3D-style), so they generalize across devices.
- **Scale from sparse metric points:** align a dense relative depth network to sparse ToF or LiDAR points with a robust per-frame scale and shift fit (section 2.2).

## 5. Missing or degraded sensors, and sensor dropout in training

**Failure modes:**
- **Complete dropout:** sensor off, driver error, packet loss.
- **Partial degradation:** ToF holes on glass, black or shiny surfaces, sunlight interference, LiDAR in rain, camera low light, overexposure, motion blur or a dirty lens, IMU bias drift or saturation under shock.
- **Stale data:** timestamp lag.
- **Calibration drift:** misaligned sensors.

### 5.1 Training for robustness
- **Modality dropout:** randomly drop **entire modalities** (replace with zeros or a learned "missing" token plus a mask), for example 10 to 30% per non-RGB modality, always keeping at least one. Train on mixtures of availability patterns (RGB-only, RGB+D, RGB+LiDAR, all) so the model works with **any subset**.
- **Partial corruption augmentation:** random holes in depth, dropping 20 to 80% of LiDAR points, sensor-specific noise models (ToF multipath, quantization), time-offset jitter (±10 to 30 ms), extrinsic perturbation, exposure and blur augmentation for RGB, bias and noise for IMU.
- **Modality competition:** multimodal models tend to over-rely on the easiest modality (modality collapse or "greedy learning"). Dropout helps, and you can add per-modality auxiliary heads or gradient-balancing methods.
- **Efficient data strategy:** synchronized multi-sensor data is expensive.
  - Use simulation (Gazebo, Isaac, Habitat, Q23) with realistic sensor noise.
  - Use pretrained single-modality encoders (depth: Depth Anything, LiDAR: BEV or point pretraining).
  - **Anchor alignment on RGB** (ImageBind-style): you only need (RGB, X) pairs to bind each modality into a shared space with text.

### 5.2 Architecture and runtime
- **Mask-aware encoders** that take explicit availability masks, **gated fusion** or attention with learned per-modality reliability (sensor confidence), separate unimodal pathways so removing one doesn't break the others, and a late-fusion fallback head. Some work (e.g. MetaBEV) targets arbitrary sensor failure.
- **Health monitoring per sensor:** frame rate, timestamp monotonicity, valid-pixel ratio of depth, point count, IMU variance and saturation, image brightness and blur metrics (variance of the Laplacian), and cross-sensor consistency (depth vs monocular depth disagreement, LiDAR-image edge alignment).
- **Graceful degradation:** fall back to RGB-only with monocular depth, lower confidence, add explicit uncertainty to the answer ("I can't measure distances reliably in this lighting"), and trigger re-calibration. For safety-critical functions (obstacle avoidance on a robot), use redundancy and fail-safe behaviour (e.g. stop on LiDAR failure).
- **Robustness evaluation:** systematically corrupt each modality (dropout-rate sweeps, noise, time offset, calibration error, weather) and plot accuracy vs severity. Require graceful degradation (for example, a stated fraction of full performance with one sensor missing), and make sure multimodal training **doesn't regress RGB-only performance**.

## 6. On-device architecture considerations
- **Per-modality encoders mapped to compute units:** RGB hybrid or ViT on the NPU (Q18), depth CNN (INT8), LiDAR pillars → 2D CNN, IMU 1D CNN or GRU on the DSP or CPU. Run them in parallel on different units where possible.
- **Token budget (Q2):** fuse before the LLM into a few tokens (a Perceiver-style resampler to 32 to 64 tokens), or add 3D position embeddings to the RGB tokens (no extra tokens).
- **Memory (Q14):** extra encoders add weights and activations, roughly tens of MB each at INT8.
- **Rate limits:** fusion runs at the slowest sensor's rate (e.g. LiDAR at 10 Hz). Reuse static-background features and temporal context.
- **Energy (Q16):** active sensors (ToF, LiDAR) draw power. Duty-cycle them, use passive RGB + monocular depth when enough, and trigger active sensing for metric queries ("sensor scheduling").
- **Privacy (Q17):** depth and LiDAR reveal home geometry and should be treated as sensitive, but they are **less identifying than RGB**, so they are a good basis for privacy-preserving presence, navigation and activity perception without storing images.

## 7. Evaluation
- **3D and spatial QA:** ScanQA, SQA3D, ScanRefer, Nr3D/Sr3D, OpenEQA, VSI-Bench (video spatial intelligence: distances, sizes, counting, route planning), BLINK and CV-Bench (relative depth), SpatialVLM-style spatial QA.
- **Metric quality:** absolute relative error and δ < 1.25 accuracy for depth, distance error in centimeters for the product's range, ATE and RPE for pose.
- **Robustness:** corruption sweeps and sensor-dropout matrices (section 5).
- **Product tests:** for example "distance to obstacle within ±10 cm at 0.3 to 3 m", measured against a ground-truth rig.

## 8. Strong interview answer
> "Early fusion combines raw data before feature extraction, for example RGB-D channels or projecting LiDAR into the image or painting points with semantics. It is cheap, but needs precise calibration and sync, and it breaks pretrained RGB encoders. Mid fusion keeps separate encoders and fuses features, for example in BEV as in BEVFusion, and usually gives the best accuracy and supports modality gating. Late fusion merges per-sensor decisions, which is robust and modular but loses cross-modal cues. For a VLM, I'd map those to three routes. First, depth as a pseudo-image into the existing encoder, with a fixed metric range and the intrinsics provided, which is cheap but loses precision. Second, modality tokens: a small depth or pillar encoder with a projector, or lifting 2D features into 3D as in 3D-LLM, or adding 3D position embeddings to 2D patches as in LLaVA-3D, which adds no tokens. On-device I'd use pillar or range-image CNNs rather than point ops like FPS and kNN, which NPUs handle badly. Third, symbolic text, with metric values precomputed in code. For BEV, I'd note the ground-plane limitation indoors. A pragmatic on-device depth path is a dense relative-depth network aligned to sparse ToF or LiDAR with a robust scale and shift fit. IMU is mostly for gravity, motion compensation, VIO poses and gating. For synchronization, I'd use hardware timestamps on a common clock, timestamp at capture, interpolate IMU, deskew LiDAR, and estimate camera-IMU offsets, because 10 ms at 100°/s is about 9 cm at 5 m. For calibration, I'd manage intrinsics and extrinsics with a tested transform tree, watch conventions like quaternion order, calibrate camera-IMU with Kalibr, camera-LiDAR with target-based or targetless methods, monitor drift online, and train with extrinsic perturbation. For robustness, I'd train with modality dropout and partial corruption on mixtures of availability patterns, use mask-aware gated fusion, add health monitors and graceful fallbacks to RGB-only with explicit uncertainty, and evaluate with corruption sweeps while making sure RGB-only performance doesn't regress. I'd also budget tokens, duty-cycle active sensors for energy, and use non-RGB sensing for privacy where possible."

## 9. Follow-up questions to prepare for
1. Why does early fusion of a depth channel hurt a pretrained RGB encoder, and how would you mitigate it?
2. How does LLaVA-3D add 3D awareness without adding tokens, and what does it need at inference?
3. Why are PointNet++-style ops hard on NPUs, and what alternatives would you use?
4. How would you recover metric scale from a monocular depth network and a sparse ToF sensor?
5. How big is the error from a 10 ms time offset or a 0.5° extrinsic error, and how would you test for it?
6. How does Kalibr estimate the camera-IMU transform and the time offset together?
7. How does modality dropout prevent modality collapse, and what are its risks?
8. How would you design runtime sensor health monitoring and a graceful-degradation policy for a home robot?

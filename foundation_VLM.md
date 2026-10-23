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

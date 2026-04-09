# LLM Inference Knowledge Base

## Q1: Why do we divide by √d in attention calculation?

### Beginner Level: Simple Intuition

When you compute attention scores, you take the dot product of query and key vectors. As these vectors get longer (higher dimension $d$), their dot products get **really large** — they grow like $\sqrt{d}$ on average.

**The problem:** Large dot products fed into softmax become very "peaked" (one value gets almost 100% of the attention, others get nearly 0%). This makes the gradients tiny during backprop, so the model learns slowly.

**The fix:** Divide by $\sqrt{d}$ to keep the dot products in a reasonable range, so softmax outputs are more spread out and gradients flow better.

---

### Intermediate Level: The Math Behind It

Recall that: random vectors of dimension $d$ have dot products that grow as $\sqrt{d}$ in expectation.

In attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

**Why divide specifically by $\sqrt{d}$?**

- Each element in $QK^T$ involves a sum of $d$ random products
- The **variance** of this sum is proportional to $d$
- The **standard deviation** (scale) is proportional to $\sqrt{d}$
- Dividing by $\sqrt{d}$ **normalizes the variance** to stay around 1

Without this scaling, the softmax receives values that are $O(\sqrt{d})$ larger, making the distribution extremely concentrated.

---

### Advanced Level: Information-Theoretic & Stability Perspective

#### The gradient flow problem:
When pre-softmax logits have variance $\sigma^2 = d$:
$$\nabla_{\text{logits}} \mathcal{L} = \text{softmax}(\text{logits}) - \text{target}$$

With large logits, softmax becomes one-hot-like. The gradient has very small variance in early positions. This creates **vanishing gradients** during backprop through attention.

#### The solution as variance stabilization:
Dividing by $\sqrt{d}$ ensures:
$$\mathbb{Var}(QK^T / \sqrt{d}) \approx 1$$

This keeps the softmax output in the **high-sensitivity regime** where gradients remain informative.

#### Connection to initialization theory:
This is related to **Xavier/Glorot initialization** — the principle that gradients should have stable variance across layers. The $\sqrt{d}$ scaling ensures attention doesn't "collapse" the gradient signal.

#### Alternative perspective:
Think of it as **entropy regularization**:
- Unscaled: softmax ≈ one-hot (entropy ≈ 0, no information flow)
- Scaled by $\sqrt{d}$: softmax is more uniform (higher entropy, better gradient flow)

---

### Summary Across Levels

| Level | Answer |
|-------|--------|
| **Beginner** | Makes softmax outputs less extreme so the model can learn better |
| **Intermediate** | Normalizes dot product variance from $d$ to 1 |
| **Advanced** | Stabilizes gradient flow and maintains information flow through the softmax distribution |

**Key Insight:** It's not about the dot product size directly—it's about keeping the softmax in its optimal learning regime.

---

## Q2: What is Softmax? Why Exponential?

### Beginner Level: Simple Intuition

**What is softmax?**

Softmax is a function that takes any list of numbers and converts them into **probabilities** (numbers between 0 and 1 that add up to 1).

**Why exponential?**

Think of it this way: we want larger numbers to get more probability, while smaller numbers get less. The exponential function does this perfectly—it **amplifies differences** between numbers.

**Simple example:**
- If numbers are [1, 2, 3], softmax doesn't just say "3 is biggest, give it equal share"
- Instead, it says "3 is much bigger than others, give it WAY more probability"
- The exponential makes this "preference" very strong

#### Example with NLP tokens:

Imagine you're predicting the next word after "The cat sat on the ___"

Your model outputs raw scores (called **logits**) for different tokens:

| Token | Raw Score |
|-------|-----------|
| mat | 2.0 |
| floor | 3.5 |
| chair | 1.5 |
| dog | 0.8 |

**Without softmax:** These are just numbers—not probabilities.

**With softmax:**
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

| Token | Probability |
|-------|-----------|
| mat | 0.10 |
| floor | 0.74 |
| chair | 0.12 |
| dog | 0.04 |

Notice: "floor" had the highest score (3.5), and now gets the highest probability (74%). The exponential amplified this advantage!

---

### Intermediate Level: The Math and Why Exponential

#### The softmax formula:
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$

Where $z$ is the raw score (logit).

#### Why exponential specifically?

1. **Always positive:** $e^x > 0$ for any $x$ (negative or positive), so division always works
2. **Monotonic:** If $x_1 > x_2$, then $e^{x_1} > e^{x_2}$ (order is preserved)
3. **Amplifies differences:** Exponential grows fast, so larger scores get exponentially bigger probability
   - Score difference of 1: probability ratio is $e \approx 2.7$
   - Score difference of 2: probability ratio is $e^2 \approx 7.4$
   - Score difference of 10: probability ratio is $e^{10} \approx 22,000$!

4. **Mathematical convenience:** Easy to differentiate, pairs well with cross-entropy loss

#### Why not linear or other functions?

With a linear function like $\frac{z_i}{\sum_j z_j}$:
- Negative scores become problematic (could give negative probabilities)
- Differences aren't amplified—score of 10 vs 11 gives similar probabilities
- Model can't express strong preferences

---

### Advanced Level: Information Theory & Optimization

#### Exponential as temperature scaling:

The softmax can be generalized with a temperature parameter:
$$\text{softmax}_T(z_i) = \frac{e^{z_i/T}}{\sum_{j} e^{z_j/T}}$$

- **Low temperature** (T → 0): Exponential amplifies even more → one-hot-like (sharp decision)
- **High temperature** (T → ∞): More uniform distribution (softer probabilities)

The exponential is the natural temperature scaling that arises from **maximum entropy distributions** under moment constraints.

#### Connection to cross-entropy loss:

When you use softmax + cross-entropy loss:
$$\mathcal{L} = -\log(\text{softmax}(z_{\text{true}}))$$

The exponential in softmax makes this loss **convex** (single global minimum), which is ideal for optimization.

#### Why exponential from a probability theory perspective:

The exponential family is the **most natural** way to express probabilities under constraints. Softmax with exponential is the unique solution to:

*"Find probabilities that maximize entropy while fitting observed feature expectations"*

This is the Boltzmann distribution from statistical mechanics!

---

### Summary Across Levels

| Aspect | Answer |
|--------|--------|
| **What** | Converts raw scores to probabilities (0 to 1, sum to 1) |
| **Why exponential** | Amplifies differences, ensures positivity, mathematically convenient |
| **Intuition** | Larger scores get exponentially larger probabilities |
| **In NLP** | Converts model logits to token probabilities for next-token prediction |
| **Advanced** | Maximum entropy distribution + Boltzmann statistics |

---

## Q3: What is top-p (nucleus) sampling, and how does it combine with temperature and attention scaling?

### Beginner Level: Simple Intuition

When an LLM predicts the next token, it first gives scores to all tokens.

- **Temperature ($T$)** changes how sharp or random those scores feel before sampling.
   - Low $T$ (for example, 0.7): more focused, safer outputs.
   - High $T$ (for example, 1.3): more diverse, creative outputs.
- **Top-p (nucleus) sampling** keeps only the smallest set of top tokens whose total probability reaches $p$ (for example, 0.9), then samples from only that set.

So in practice:
1. Apply temperature to logits.
2. Convert to probabilities with softmax.
3. Keep nucleus tokens up to cumulative probability $p$.
4. Sample one token from this nucleus.

#### NLP token example:

Prompt: "The cat sat on the ___"

Assume after temperature + softmax we get:

| Token | Probability |
|-------|-------------|
| mat | 0.48 |
| floor | 0.25 |
| sofa | 0.10 |
| chair | 0.08 |
| roof | 0.03 |
| banana | 0.02 |
| others | 0.04 |

If $p = 0.90$, cumulative sum is:
- mat: 0.48
- floor: 0.73
- sofa: 0.83
- chair: 0.91 (stop)

Nucleus is {mat, floor, sofa, chair}. Tokens like roof and banana are dropped for this step.

---

### Intermediate Level: How temperature and top-p work together

Start with logits $z_i$ and apply temperature:
$$P(i) = \text{softmax}\left(\frac{z_i}{T}\right)$$

Then for top-p:
1. Sort tokens by $P(i)$ descending.
2. Find smallest prefix $S$ such that:
$$\sum_{i \in S} P(i) \ge p$$
3. Renormalize within $S$ and sample.

Key interaction:
- Lower $T$ makes distribution peakier, so nucleus size is usually smaller.
- Higher $T$ flattens distribution, so nucleus size is usually larger.

So temperature controls distribution shape, and top-p cuts the tail adaptively.

---

### Advanced Level: Relation to attention scaling

There are two different "temperature-like" mechanisms in transformers:

1. **Attention scaling during score computation**
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- Purpose: keep attention logits numerically stable as dimension grows.
- Stage: inside each transformer layer.
- Effect: stabilizes training/inference signal quality.

2. **Sampling temperature during decoding**
$$P_{\text{next}}(i) = \text{softmax}\left(\frac{z_i}{T}\right)$$
- Purpose: control randomness of final token choice.
- Stage: after model outputs next-token logits.
- Effect: changes creativity vs determinism.

Then top-p is applied on $P_{\text{next}}$ to truncate low-probability tail.

Practical interpretation:
- $1/\sqrt{d_k}$ is a model-internal stabilization constant.
- $T$ and $p$ are user-facing decoding knobs.

Common settings:
- Factual QA: low-to-mid $T$ (0.2 to 0.8), top-p around 0.8 to 0.95
- Creative writing: mid-to-high $T$ (0.8 to 1.3), top-p around 0.9 to 0.98

---

### One-line summary

Use temperature to shape randomness, top-p to keep only the likely token mass, and remember this is different from attention's $\frac{1}{\sqrt{d_k}}$ scaling, which is for internal score stability.

---

## Q4: What is the difference between top-k and top-p sampling, and when should we use each?

### Beginner Level: Simple Intuition

Both methods pick the next token from a limited candidate set.

- **Top-k**: keep exactly $k$ most likely tokens, then sample from them.
- **Top-p (nucleus)**: keep as many top tokens as needed until total probability reaches $p$.

Simple view:
- Top-k uses a fixed count.
- Top-p uses a fixed probability mass.

#### NLP token example:

Suppose next-token probabilities are:

| Token | Probability |
|-------|-------------|
| mat | 0.40 |
| floor | 0.22 |
| sofa | 0.14 |
| chair | 0.08 |
| bed | 0.06 |
| carpet | 0.04 |
| banana | 0.03 |
| others | 0.03 |

If **top-k = 3**, candidate set is {mat, floor, sofa}.

If **top-p = 0.85**, cumulative is:
- mat: 0.40
- floor: 0.62
- sofa: 0.76
- chair: 0.84
- bed: 0.90 (stop)

Candidate set is {mat, floor, sofa, chair, bed}.

---

### Intermediate Level: Behavior Differences

1. **Adaptivity**
- Top-k is not adaptive: always $k$ tokens.
- Top-p is adaptive: number of tokens changes by confidence.

2. **When model is very confident**
- Top-k may still include unnecessary lower-quality tokens.
- Top-p may keep very few tokens, often improving quality.

3. **When model is uncertain**
- Top-k may be too narrow and repetitive if $k$ is small.
- Top-p expands naturally and can preserve diversity.

4. **Failure mode**
- Top-k with large $k$ can allow noisy tail tokens.
- Top-p with very high $p$ (for example, 0.98 to 1.0) can also admit too much tail noise.

---

### Advanced Level: Production Decoding Guidance

In production, these are often combined with temperature:
$$P(i) = \text{softmax}(z_i / T)$$
then apply top-k and/or top-p filtering before sampling.

Common practical choices:
- **Reliable factual responses**: lower $T$, moderate top-p (for example, 0.85 to 0.95), optional small top-k cap.
- **Creative generation**: mid/high $T$, higher top-p (for example, 0.9 to 0.98), sometimes no top-k cap.
- **Latency-sensitive systems**: top-k can be simpler and faster to tune because candidate count is fixed.

Rule-of-thumb:
- Prefer **top-p** as default for natural generation because it adapts to uncertainty.
- Add **top-k cap** (for example, 40 to 100) as a safety guard against very large candidate sets.

---

### Quick decision table

| Goal | Suggested setting |
|------|-------------------|
| Stable and concise answers | Lower $T$ + top-p 0.85 to 0.92 |
| Balanced chat quality | $T$ around 0.7 to 1.0 + top-p 0.9 to 0.95 |
| High creativity | Higher $T$ + top-p 0.95 to 0.98 |
| Hard cap on randomness/compute | Use top-k (or top-k + top-p) |

---

## Q5: Why do we call model.eval() during inference?

### Beginner Level: Simple Intuition

Think of a model like a student in two different modes:

- **Training mode**: practicing with noise so it learns better.
- **Inference mode**: taking the final exam, no random tricks.

In training mode, layers like dropout randomly turn off parts of the network. This helps learning, but it also means outputs can change each run.

During inference, we usually want stable and repeatable outputs. Calling `model.eval()` turns off that training-time randomness.

If you forget `model.eval()`, you may see puzzling non-determinism: same input, different output.

---

### Intermediate Level: What changes internally

`model.eval()` switches certain layers to evaluation behavior:

1. **Dropout**
- Train mode: randomly zeroes activations.
- Eval mode: passes activations through (no random masking).

2. **BatchNorm**
- Train mode: uses current mini-batch statistics.
- Eval mode: uses running statistics collected during training.

So `eval()` is not just a style preference. It changes math in key layers and directly affects output consistency.

Also common with inference:
- `torch.no_grad()` or `torch.inference_mode()` to avoid gradient tracking, reduce memory, and improve speed.

---

### Senior Level: Production mindset

In real systems, inference correctness is not only about model weights; it is also about **runtime mode discipline**.

Why this matters operationally:

1. **Reproducibility risk**
- Missing `eval()` introduces stochastic behavior from dropout and train-mode normalization.
- This can look like flaky bugs, data drift, or unstable quality.

2. **Observability confusion**
- A/B tests and regression checks become noisy if mode is wrong.
- Debugging is harder because outputs differ across identical requests.

3. **Performance cost**
- Forgetting no-grad/inference-mode wastes memory and compute.
- Under load, this can reduce throughput and increase latency.

Recommended serving pattern (PyTorch):

```python
model.eval()
with torch.inference_mode():
   logits = model(input_ids)
```

For strict reproducibility, combine this with fixed seeds and deterministic backend settings when needed.

---

### Quick checklist

- Call `model.eval()` before serving.
- Use `torch.inference_mode()` (or `torch.no_grad()`) during forward pass.
- Validate deterministic behavior with repeated same-input tests.
- Log model mode and decoding parameters in inference telemetry.

---

## Q6: How should we understand dropout from simple to advanced? How does it impact training and inference? What happens in energy or entropy sense during inference?

### Beginner Level: Simple Intuition

Dropout is like training with a team where, in every practice round, some players are randomly asked to sit out.

- The model cannot depend too much on any one neuron.
- It learns more robust patterns.
- This reduces overfitting.

During training:
- Random neurons are temporarily turned off.
- Learning is noisier but usually generalizes better.

During inference:
- Dropout is turned off using eval mode.
- All neurons are available.
- Output becomes stable and deterministic for the same input.

---

### Intermediate Level: What changes in training and inference time

Dropout operation:
- In train mode, each activation is multiplied by a random mask sampled from Bernoulli distribution.
- Surviving activations are scaled so expected activation magnitude stays consistent.

Impact on training:
- Acts as regularization.
- Slows convergence a bit because of noise.
- Often improves validation performance and robustness.

Impact on inference:
- Standard inference disables dropout with eval mode.
- No random masking, so output variance drops.
- Throughput is typically better than train mode because there is no dropout sampling overhead and no gradient computation.

If dropout is accidentally left on at inference:
- Same prompt can produce different logits even before sampling randomness.
- This appears as unexplained non-determinism.

---

### Senior Level: Energy and entropy perspective

Consider logits as negative energies (up to a constant), where lower energy means more preferred tokens.

Train-time dropout effect:
- Random masks perturb internal features, which perturbs logits.
- In energy terms, each mask gives a slightly different energy landscape.
- The model learns parameters that remain good across many such perturbations.

Inference with dropout off:
- You evaluate one deterministic network (the mean behavior induced by training-time scaling).
- Energy landscape is fixed for a given input.
- Predictive entropy reflects data ambiguity, not extra dropout noise.

Inference with dropout on intentionally (MC dropout):
- You sample multiple stochastic forward passes.
- Each pass gives different energies and probabilities.
- Averaging predictions often increases predictive entropy when model uncertainty is high.
- This can estimate epistemic uncertainty.

Practical implication:
- Standard serving: dropout off for stability and reproducibility.
- Uncertainty-aware serving: optional MC dropout with multiple passes, trading latency for uncertainty estimates.

---

### Quick summary

- Dropout adds structured noise during training to improve generalization.
- Dropout is usually disabled during inference to avoid random output drift.
- In entropy and energy terms, train-time dropout smooths learning across many perturbed energy landscapes; inference without dropout uses one stable landscape, while MC dropout averages multiple stochastic landscapes to quantify uncertainty.

---

## Q7: Numeric example - entropy with dropout-off inference vs MC-dropout inference

### Setup

Prompt: "The cat sat on the ___"

Candidate tokens: `{mat, floor, sofa}`

Entropy formula:
$$H(p) = -\sum_i p_i \log p_i$$

---

### Case A: Standard inference (dropout off)

Single forward-pass probabilities:
- $p(\text{mat}) = 0.80$
- $p(\text{floor}) = 0.15$
- $p(\text{sofa}) = 0.05$

Entropy:
$$H_A = -(0.80\log 0.80 + 0.15\log 0.15 + 0.05\log 0.05) \approx 0.61$$

Interpretation: low entropy, model is fairly confident.

---

### Case B: MC-dropout inference (dropout on, multiple passes)

Assume 3 stochastic passes:

1. Pass 1: $(0.75, 0.20, 0.05)$
2. Pass 2: $(0.55, 0.35, 0.10)$
3. Pass 3: $(0.40, 0.45, 0.15)$

Mean predictive distribution:
$$\bar{p} = \left(\frac{0.75+0.55+0.40}{3},\frac{0.20+0.35+0.45}{3},\frac{0.05+0.10+0.15}{3}\right) = (0.57, 0.33, 0.10)$$

Entropy of mean prediction:
$$H_B = -(0.57\log 0.57 + 0.33\log 0.33 + 0.10\log 0.10) \approx 0.92$$

Interpretation: higher entropy than Case A, showing more predictive uncertainty.

---

### Energy viewpoint (intuition)

If logits behave like negative energies:
- Dropout off: one fixed energy landscape gives one sharp distribution.
- MC-dropout: each mask creates a slightly different landscape; averaging them broadens the final distribution, often raising entropy.

---

### Practical takeaway

- For stable production outputs: use dropout off (`model.eval()`).
- For uncertainty estimation: use MC-dropout with multiple passes and aggregate probabilities.
- Increased uncertainty signal is useful, but latency and compute cost increase roughly with number of passes.

---

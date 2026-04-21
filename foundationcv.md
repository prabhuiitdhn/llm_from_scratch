# Computer Vision Interview Preparation
## 50+ Q&A for Senior Researcher Role
### Foundation to Intermediate Questions with Beginner-to-Expert Explanations

---

## Table of Contents
1. [Image Processing Fundamentals](#image-processing-fundamentals)
2. [Deep Learning Architectures](#deep-learning-architectures)
3. [Object Detection & Localization](#object-detection--localization)
4. [Semantic & Instance Segmentation](#semantic--instance-segmentation)
5. [Optical Flow & Motion](#optical-flow--motion)
6. [3D Vision & Geometry](#3d-vision--geometry)
7. [Advanced Topics](#advanced-topics)

---

## Image Processing Fundamentals

### Q1. What is convolution and why is it fundamental to computer vision?

**A:** Convolution is a mathematical operation that slides a small matrix (kernel) over an image and computes a weighted sum at each position, producing a new image. It is the foundation of filtering, edge detection, feature extraction, and neural networks.

**Beginner level:**
- Convolution is like sliding a stamp over paper—at each position you compute what "matches" the stamp pattern.
- A kernel is a small grid of numbers (for example, 3×3 or 5×5).
- The output at each pixel is the sum of element-wise products between the kernel and the image patch.

**Mathematical formulation:**
$$
(I * K)(x, y) = \sum_{i=-n}^{n} \sum_{j=-n}^{n} K(i, j) \cdot I(x+i, y+j)
$$

where $I$ is the image, $K$ is the kernel, and the result is a filtered image.

**Why it's fundamental:**
1. **Locality:** Convolution respects spatial locality—output depends only on nearby pixels.
2. **Parameter sharing:** The same kernel weights are applied everywhere, reducing parameters.
3. **Equivariance:** Shifting the input shifts the output (translation equivariance).
4. **Interpretability:** Different kernels detect different patterns (edges, corners, textures).

**Common kernels:**
- **Sobel:** detects edges (horizontal or vertical).
- **Gaussian:** smoothing and blur.
- **Laplacian:** second derivative, edge enhancement.

**Expert perspective:**
Convolution is a linear operation in spatial domain, equivalent to multiplication in frequency domain (Fourier domain). This duality is powerful:
- In spatial domain: local pattern matching via sliding kernel.
- In frequency domain: selective frequency attenuation via filtering.

Convolutional Neural Networks (CNNs) learn kernels via backpropagation, automatically discovering optimal feature detectors rather than hand-crafting them.

**Trade-offs:**
- Simple but assumes linear relationships.
- Boundary handling (padding) affects edge behavior.
- Computational cost scales with kernel size and image resolution.

---

### Q2. What are padding and stride in convolution, and how do they affect output size?

**A:** Padding adds extra rows/columns around the image boundary. Stride controls how many pixels the kernel moves at each step.

**Beginner intuition:**
- **Padding:** imagine surrounding a puzzle with extra border pieces to prevent the kernel from falling off edges.
- **Stride:** if stride=1, the kernel moves 1 pixel at a time; if stride=2, it jumps 2 pixels.

**Output size formula:**
$$
\text{Output size} = \left\lfloor \frac{\text{Input size} - \text{Kernel size} + 2 \times \text{Padding}}{\text{Stride}} \right\rfloor + 1
$$

**Common padding types:**
1. **Valid (no padding):** output shrinks. For 32×32 image and 3×3 kernel with stride=1: output is 30×30.
2. **Same padding:** output size equals input size. Requires padding = (kernel_size - 1) / 2.
3. **Full padding:** output expands. Useful in deconvolution.

**Effect on border regions:**
- No padding: edge pixels have fewer neighbors, possibly leading to edge bias.
- Zero padding: artificially introduces zero values, can affect edge detection.
- Reflect/replicate padding: mimic image at boundaries, more natural.

**Stride effects:**
- Stride=1: dense feature maps (high resolution, more computation).
- Stride>1: downsampling (reduce resolution, fewer parameters, fewer computations).
- Stride in both dimensions creates 2D downsampling.

**Expert view:**
From signal processing perspective, stride is decimation (downsampling). Without proper anti-alias filtering before decimation, aliasing can appear. In practice, pooling layers after convolution provide anti-aliasing implicitly.

In deep learning, stride is often used for efficiency (replacing pooling), but multi-scale feature hierarchies (FPN) are often superior to naive stride-based reduction.

---

### Q3. What are pooling operations and why are they used in CNNs?

**A:** Pooling is a downsampling operation that reduces spatial dimensions by aggregating neighboring pixels. Common types are max pooling and average pooling.

**Beginner level:**
- **Max pooling:** in each window, keep the maximum value.
- **Average pooling:** in each window, average all values.
- Both slide a window (usually 2×2) with stride 2, reducing size by 2× in each dimension.

**Why pooling matters:**
1. **Computational efficiency:** fewer pixels to process downstream.
2. **Receptive field growth:** stacking pooling expands the effective region each neuron sees.
3. **Translation invariance:** max pooling provides robustness to small shifts (if an object moves slightly, the same max value is still selected).
4. **Feature selection:** max pooling picks the most activated response, focusing on strong signals.

**Max vs Average:**
- **Max pooling:** selective, keeps strongest features, better for detecting specific patterns.
- **Average pooling:** averages neighborhood, smoother, retains more information but less selective.

**Expert perspective:**
Pooling is lossy compression. The information discarded is typically high-frequency detail and noise, which is often not essential for classification. However, for tasks requiring fine spatial localization (for example, segmentation), pooling is harmful. This led to:
1. **Atrous/dilated convolution:** increase receptive field without downsampling.
2. **Deconvolution/transposed convolution:** upsample and recover spatial detail.
3. **Skip connections (U-Net):** bypass pooling to preserve resolution.

**Information-theoretic view:**
Pooling is a form of dimensionality reduction. In max pooling, we retain the maximum activation, sacrificing all sub-maximum activations. This is justified by the assumption that small spatial shifts should not change the output (invariance), but this assumption fails for tasks requiring precise localization.

---

### Q4. What are activation functions and why is ReLU preferred in modern CNNs?

**A:** Activation functions introduce non-linearity, allowing networks to learn complex patterns beyond linear transformations. ReLU (Rectified Linear Unit) is the most common choice.

**Common activation functions:**

1. **Sigmoid:** $\sigma(x) = \frac{1}{1 + e^{-x}}$
   - Maps input to (0, 1).
   - Smooth, interpretable as probability.
   - Problem: vanishing gradient for large |x| (gradient → 0), slowing learning.

2. **Tanh:** $\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$
   - Maps input to (-1, 1).
   - Zero-centered, slightly better gradient than sigmoid.
   - Still suffers from vanishing gradient.

3. **ReLU:** $\text{ReLU}(x) = \max(0, x)$
   - Zero for $x < 0$, linear for $x > 0$.
   - Fast to compute, gradient is 0 or 1 (no vanishing).
   - Became standard in 2012 (AlexNet) because it enables training of very deep networks.

4. **Leaky ReLU:** $\text{LeakyReLU}(x) = \max(\alpha x, x)$ where $\alpha$ small (e.g., 0.01)
   - Allows small negative gradient, avoiding "dead ReLU" problem (neurons that output 0 for all inputs).
   - Slightly better than ReLU in some cases.

5. **ELU, SELU:** smoother variants with better properties for very deep networks.

**Why ReLU works so well:**
1. **No vanishing gradient:** gradient is constant (1) for positive inputs.
2. **Sparsity:** many activations are exactly 0, leading to sparse representations.
3. **Computational simplicity:** just a threshold operation.
4. **Good empirical performance:** enables training of 100+ layer networks.

**Expert view:**
ReLU's success is tied to the gradient flow problem. In deep networks with many sigmoid/tanh layers, gradients are multiplied across layers:
$$
\frac{\partial L}{\partial w^{(1)}} = \frac{\partial L}{\partial w^{(L)}} \cdot \frac{\partial w^{(L)}}{\partial w^{(L-1)}} \cdots \frac{\partial w^{(2)}}{\partial w^{(1)}}
$$

If each factor is < 1 (as with sigmoid gradient), the product vanishes exponentially with depth. ReLU's gradient of 1 for positive inputs avoids this. However, dead neurons (always outputting 0) are a real problem, motivating variants like Leaky ReLU and GELU.

**Gating and normalization:**
Modern networks often combine ReLU with batch normalization, which stabilizes activations and makes the network less sensitive to initialization and activation saturation.

---

### Q5. What is backpropagation and how does it compute gradients?

**A:** Backpropagation is the algorithm used to train neural networks. It computes gradients of the loss function with respect to all parameters by applying the chain rule, propagating errors backward from output to input.

**Beginner intuition:**
- Forward pass: compute predictions layer by layer.
- Loss: measure error between prediction and ground truth.
- Backward pass: for each layer, compute how much each weight contributes to the error, then adjust weights to reduce error.
- Chain rule: gradient of a composed function is the product of local gradients.

**Chain rule in calculus:**
If $z = f(g(x))$, then $\frac{dz}{dx} = \frac{dz}{dg} \cdot \frac{dg}{dx}$.

In a neural network with layers $h^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$ and activation $a^{(l)} = \sigma(h^{(l)})$:

$$
\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial a^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial h^{(L)}} \cdots \frac{\partial a^{(l)}}{\partial h^{(l)}} \cdot \frac{\partial h^{(l)}}{\partial W^{(l)}}
$$

**Algorithm steps:**
1. **Forward pass:** compute $a^{(1)}, a^{(2)}, \ldots, a^{(L)}$.
2. **Compute loss:** $L = \text{Loss}(a^{(L)}, y)$.
3. **Backward pass:** compute $\frac{\partial L}{\partial a^{(l)}}$ for each layer from $l=L$ down to $l=1$.
4. **Weight update:** $W^{(l)} \leftarrow W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}$ where $\alpha$ is learning rate.

**Expert perspective:**
Backpropagation is elegant but computationally expensive. Key insights:
1. **Dynamic programming:** reuse intermediate gradients to avoid redundant computation (hence the term "backprop").
2. **Computational graph:** modern frameworks (PyTorch, TensorFlow) build a computation graph and differentiate it automatically via automatic differentiation (autograd).
3. **Vanishing/exploding gradients:** in deep networks, gradients can become very small or very large during backpropagation, causing training to stall or diverge. Solutions include:
   - ReLU activations (gradient of 1 in positive region).
   - Skip connections (ResNets) allow gradients to flow directly.
   - Gradient clipping: cap gradient magnitude.
   - Weight initialization strategies (Xavier, He initialization).

**Computational complexity:**
Backpropagation has the same cost as forward pass (same order of operations). This is why training is roughly 2-3× slower than inference (forward + backward vs forward only).

---

### Q6. What is batch normalization and why does it improve training?

**A:** Batch normalization (BatchNorm) standardizes activations within mini-batches, making training faster and more stable.

**How it works:**
For a batch of $m$ samples, at each layer:
1. Compute mean $\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i$ and variance $\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$.
2. Normalize: $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$ (small $\epsilon$ for numerical stability).
3. Scale and shift: $y_i = \gamma \hat{x}_i + \beta$ (learnable parameters $\gamma, \beta$).

**Benefits:**
1. **Stabilized activation distribution:** activations stay roughly in the linear region of ReLU, avoiding saturation.
2. **Faster convergence:** learning rate can be higher without divergence.
3. **Regularization effect:** noise from mini-batch statistics acts as regularization.
4. **Reduced internal covariate shift:** reduces the phenomenon where distribution of layer inputs changes during training.

**Expert view:**
BatchNorm is empirically very powerful, but the theoretical understanding is still evolving. Recent work suggests:
1. Internal covariate shift is not the main driver of its benefit.
2. The normalization effect (Lipschitz smoothing) is more important—it makes the loss landscape smoother, improving optimization.
3. Batch statistics introduce noise, providing a regularization effect similar to dropout.

**Test-time behavior:**
At test time, batch statistics are unavailable (only one or a few samples). Instead, running estimates (exponential moving average) from training are used:
$$
\mu_{test} = \text{EMA}(\mu_B), \quad \sigma_{test} = \text{EMA}(\sigma_B)
$$

This decoupling between train and test can cause domain shift if training and test distributions differ.

**Modern variants:**
- **Layer normalization:** normalize over features instead of batch (useful for RNNs, Transformers).
- **Instance normalization:** normalize each sample-channel pair (useful for style transfer).
- **Group normalization:** normalize over groups of channels (useful when batch size is small).

---

### Q7. What is dropout and how does it regularize neural networks?

**A:** Dropout is a regularization technique that randomly "drops" (sets to zero) a fraction of activations during training, preventing co-adaptation of neurons and reducing overfitting.

**How it works:**
1. During training: randomly set each activation to 0 with probability $p$ (typically $p=0.5$). Scale remaining activations by $\frac{1}{1-p}$ to maintain expected value.
2. During testing: use all activations without dropping.

**Intuition:**
- Co-adaptation problem: neurons can become highly dependent on specific other neurons, memorizing training data.
- Dropout forces each neuron to be useful on its own and in different contexts (different subsets of other neurons).
- Think of it as training an ensemble of thinned networks simultaneously, then averaging at test time.

**Why it works:**
1. **Ensemble effect:** each dropout configuration trains a different sub-network. At test time, using all neurons approximates averaging over this ensemble.
2. **Feature co-adaptation prevention:** neurons must learn general, robust features that work even if some inputs are missing.
3. **Reduces sensitivity:** the network becomes less sensitive to small perturbations in layer inputs.

**Expert perspective:**
Dropout is a form of regularization with theoretical connections to:
1. **Bayesian interpretation:** dropout approximates variational inference in a Bayesian neural network.
2. **Noise injection:** dropout adds multiplicative noise to activations, which smooths the loss landscape.
3. **Ensemble learning:** the test-time prediction without dropout approximates averaging predictions from many randomly thinned networks (explicit ensemble would be much more expensive).

However, modern batch normalization often provides sufficient regularization, so dropout is less critical in modern architectures. Dropout works best with:
- Fully connected layers (less useful after ReLU which already provides sparsity).
- Small datasets (to prevent overfitting).
- Models without batch normalization.

---

### Q8. What is a convolutional neural network (CNN) and what are the key architectural components?

**A:** A CNN is a neural network with specialized layers designed to process images by exploiting spatial structure. Key components include convolution, pooling, and fully connected layers.

**Basic architecture:**
```
Input → Conv → ReLU → Pool → Conv → ReLU → Pool → ... → FC → FC → Output
```

**Component purposes:**
1. **Convolutional layers:** detect local features (edges, corners, textures) via learned kernels.
2. **Pooling layers:** downsample, reduce computation, provide translation invariance.
3. **Fully connected layers:** combine features for classification.
4. **Activation functions:** introduce non-linearity.

**Why CNNs work for images:**
1. **Local connectivity:** pixels are correlated with nearby pixels, not distant ones.
2. **Weight sharing:** same features (edges, corners) appear everywhere in the image.
3. **Hierarchical features:** early layers detect simple patterns (edges), later layers combine them into complex patterns (faces, objects).

**Classic architectures:**
1. **LeNet-5 (1998):** pioneering CNN for digit recognition.
2. **AlexNet (2012):** deep CNN that won ImageNet; proved deep learning works for large-scale image classification.
3. **VGGNet (2014):** very deep, simple architecture; showed depth is crucial.
4. **ResNet (2015):** skip connections allow training of 100+ layer networks.
5. **Inception (2014):** multi-scale parallel convolutions.

**Expert view:**
Modern CNNs are much more sophisticated:
1. **Bottleneck layers:** reduce channels before expensive operations.
2. **Depthwise separable convolutions:** decompose into spatial (depthwise) and channel-wise (pointwise) convolutions for efficiency.
3. **Attention mechanisms:** weight different spatial regions or channels differently.
4. **Compound scaling:** balance depth, width, and resolution (EfficientNet).
5. **Vision Transformers (ViT):** replace convolution entirely with self-attention.

---

### Q9. What is transfer learning and why is it essential for practical computer vision?

**A:** Transfer learning means using a model pre-trained on a large dataset (ImageNet, COCO) and fine-tuning it on a smaller target task. It is essential because:
1. Large annotated datasets are expensive to collect.
2. Training from scratch requires huge computational resources.
3. Pre-trained features are often good for many tasks.

**Approaches:**
1. **Feature extraction:** freeze pre-trained weights, train only the final classification layer.
2. **Fine-tuning:** unfreeze some or all pre-trained weights and train with lower learning rate.
3. **Domain adaptation:** align source and target distributions before fine-tuning.

**Why it works:**
Early layers of CNNs learn generic features (edges, textures, shapes) applicable to many tasks. Deeper layers learn task-specific features. By leveraging the learned early layers, we can train on smaller datasets.

**Learning rate considerations:**
- Lower learning rate for pre-trained layers (learning rate smaller by 10×).
- Higher learning rate for newly added layers.
- Discriminative fine-tuning: different learning rates for different layers.

**Expert view:**
Transfer learning has become fundamental in computer vision:
1. **ImageNet era:** pre-training on ImageNet (1.2M images, 1000 classes) became standard.
2. **Self-supervised pre-training:** methods like SimCLR, MoCo, BYOL pre-train on unlabeled data, often outperforming ImageNet.
3. **Domain shift:** models trained on ImageNet may not work well on medical images or satellite imagery. Domain adaptation techniques address this.
4. **Few-shot learning:** meta-learning approaches enable learning from very few examples.

---

### Q10. What is the difference between data augmentation and regularization in the context of overfitting?

**A:** Both combat overfitting but work differently. Data augmentation expands the effective training set; regularization constrains model complexity.

**Data augmentation:**
Artificially create new training samples by applying realistic transformations:
- **Geometric:** rotation, scaling, translation, crop, flip.
- **Photometric:** brightness, contrast, color jitter, noise.
- **Mixed:** mixup (blend images), cutmix (crop-paste regions).

**Why it helps:**
- Increases training data diversity.
- Teaches model invariance to realistic transformations.
- Reduces variance of learned model.

**Regularization:**
Directly constrain model capacity:
- **L2 regularization (weight decay):** penalize large weights.
- **L1 regularization:** encourage sparse weights.
- **Dropout:** randomly drop activations.
- **Early stopping:** stop training when validation loss stops improving.
- **Batch normalization:** provides implicit regularization.

**Key difference:**
- **Augmentation:** increases effective data, making model see more diversity.
- **Regularization:** reduces capacity, forcing model to learn simpler patterns.

**Ideal practice:**
Use both. Augmentation is cheap (just transformations); regularization is necessary to constrain capacity even with augmented data.

**Expert perspective:**
Modern augmentation strategies (AutoAugment, RandAugment, CutMix) are learned or searched rather than hand-crafted, yielding better results. Regularization strength should be tuned based on available data and model capacity; over-regularization hurts performance as much as under-regularization.

---

## Deep Learning Architectures

### Q11. What are skip connections (residual connections) and why did ResNet revolutionize deep learning?

**A:** Skip connections allow gradients and information to flow directly across layers, enabling training of very deep networks (100+ layers). ResNet's key innovation was:
$$
y = F(x) + x
$$

where $F(x)$ is a residual block and the addition creates the skip connection.

**Problem ResNet solves:**
Deep networks trained with backpropagation suffer from vanishing gradients—gradients become exponentially smaller in early layers. With 100+ layers, training becomes intractable.

**Why skip connections help:**
1. **Gradient flow:** the identity connection ($+x$) allows gradients to flow directly backward without multiplication chains.
2. **Easier optimization:** instead of learning $y = F(x)$, the network learns $y = F(x) + x$, i.e., the residual $\Delta y = F(x)$. Residuals are typically smaller, easier to learn.
3. **Implicit regularization:** encouraging small residuals is a form of regularization.

**Architectural details:**
A typical residual block:
```
Input x
  ↓
Conv(3×3, 64) → BatchNorm → ReLU
  ↓
Conv(3×3, 64) → BatchNorm
  ↓
ReLU(x + output)  ← Skip connection
```

If input and output channels differ, a 1×1 convolution projects the skip:
$$
y = F(x) + W_s x
$$

**Impact:**
ResNet enabled:
1. Training of 152-layer networks (vs. 22-layer VGGNet).
2. Winning ImageNet 2015.
3. Spawning variants: ResNeXt, DenseNet, EfficientNet, etc.

**Expert view:**
Skip connections are now ubiquitous (Transformers, diffusion models, etc.). Modern interpretation:
1. **Residual learning:** networks learn incremental changes rather than absolute mappings.
2. **Ensemble view:** ResNet approximates an ensemble of shallow paths.
3. **Signal propagation:** skip connections are highways for information, ensuring every neuron receives useful gradients.

Modern networks balance depth and skip density to optimize gradient flow.

---

### Q12. What are attention mechanisms and how do they differ from convolution?

**A:** Attention mechanisms learn to focus on relevant parts of input by computing weighted combinations of features, allowing the network to dynamically ignore irrelevant regions.

**Convolution vs. Attention:**
- **Convolution:** fixed receptive field (kernel size), same weights applied everywhere.
- **Attention:** dynamic receptive field (can look at any position), position-dependent weights.

**Scaled dot-product attention:**
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where:
- $Q$ (Query): what are we looking for?
- $K$ (Key): where can we find it?
- $V$ (Value): what information to aggregate?

**Intuition:**
1. For each query position, compute similarity (dot product) to all key positions.
2. Normalize with softmax to get attention weights.
3. Use weights to aggregate (mix) values.

**Advantages over convolution:**
1. **Global receptive field:** can attend to any position, not just local neighborhood.
2. **Dynamic weights:** different positions are weighted differently based on content.
3. **Interpretability:** attention weights show which parts were important.

**Disadvantages:**
1. **Computational cost:** $O(n^2)$ where $n$ is sequence/image length (convolution is $O(n)$).
2. **Requires large datasets:** self-attention is data-hungry, works best with lots of training data.

**Multi-head attention:**
Use multiple attention heads in parallel, each focusing on different aspects:
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

Each head learns different representations.

**Expert view:**
Vision Transformers (ViT) replace all convolutions with attention, proving pure attention can work for images. Key insights:
1. **Patch embedding:** divide image into patches, treat as tokens (like NLP).
2. **Positional encoding:** add position information to patch embeddings (since attention is permutation-invariant).
3. **Hierarchical attention:** local attention in early layers, global later.
4. **Hybrid models:** combine convolution and attention for efficiency.

---

### Q13. What is a Transformer architecture and how does it apply to vision?

**A:** A Transformer is a sequence-to-sequence model based entirely on self-attention, with no recurrence or convolution. For vision, images are divided into patches and treated as sequences.

**Vision Transformer (ViT) architecture:**
```
Image → Patch Embedding → Transformer Encoder → Classification Head
```

**Key components:**
1. **Patch embedding:** split $H \times W \times 3$ image into $(HW/P^2) \times (P^2 \times 3)$ patches (typically $P=16$), then linearly project to embedding dimension.
2. **Positional encoding:** add learnable or sinusoidal positional information to each patch.
3. **Transformer encoder:** stack of multi-head self-attention + feed-forward blocks.
4. **Classification token ([CLS]):** special token prepended to sequence; its output after encoding is used for classification.

**Comparison to CNN:**
| Aspect | CNN | ViT |
|---|---|---|
| Inductive bias | Local connectivity, translation equivariance | None (learned from data) |
| Data efficiency | Good (works with limited data) | Poor (needs large datasets) |
| Computational cost | Linear in spatial size ($O(HW)$) | Quadratic in spatial size ($O((HW)^2)$) |
| Interpretability | Receptive field grows gradually | Attention weights show direct dependencies |

**Why ViT works:**
1. **Inductive biases not always needed:** with enough data (ImageNet-21k), the network can learn useful spatial structures without hard-coded biases.
2. **Global context:** attention can look at entire image, useful for tasks requiring global reasoning.
3. **Scalability:** ViT scales better with compute and data (follows scaling laws better than CNNs).

**Expert perspective:**
ViT sparked a paradigm shift:
1. **Hybrid models:** combine convolution and attention (e.g., ConvNeXt, Swin Transformer).
2. **Multimodal models:** CLIP combines image ViT with text transformer, enabling zero-shot recognition.
3. **Scaling laws:** emergent abilities appear at large scales (like language models).
4. **Self-supervised learning:** ViT benefits greatly from self-supervised pre-training (DINO, MAE).

---

### Q14. What is a Generative Adversarial Network (GAN) and how do the generator and discriminator compete?

**A:** A GAN has two networks competing: a Generator (creates fake images) and a Discriminator (classifies real vs. fake). The generator tries to fool the discriminator; the discriminator tries to catch the generator.

**Game theory formulation:**
$$
\min_G \max_D \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

The generator minimizes the discriminator's ability to distinguish real from fake; the discriminator maximizes this ability.

**Training dynamics:**
1. **Generator:** learns to produce images close to real data distribution.
2. **Discriminator:** learns to distinguish real from fake.
3. Ideally, at equilibrium, generator produces indistinguishable images and discriminator is at chance ($50\%$ accuracy).

**Applications:**
1. **Image generation:** generate realistic images from random noise.
2. **Image-to-image translation:** pix2pix, CycleGAN for unpaired translation.
3. **Super-resolution:** generate high-resolution from low-resolution.
4. **Style transfer:** transfer artistic style between images.

**Challenges:**
1. **Mode collapse:** generator produces limited variety, only fooling discriminator on a subset of data.
2. **Training instability:** discriminator can be too strong or too weak, destabilizing training.
3. **Evaluation difficulty:** no single metric for image quality; Inception Score (IS) and Fréchet Inception Distance (FID) are proxies.

**Expert view:**
Modern GANs address instability:
1. **Spectral normalization:** constrain discriminator weights to prevent sudden jumps.
2. **Wasserstein loss:** use Wasserstein distance instead of JS divergence, providing better gradients.
3. **Progressive growing:** grow both networks gradually from low to high resolution.
4. **StyleGAN:** learned affine transformations at different layers allow fine-grained control.

Diffusion models (DDPM, Stable Diffusion) have recently outperformed GANs in many tasks, providing more stable training.

---

### Q15. What is batch size and how does it affect training and generalization?

**A:** Batch size is the number of samples processed before updating weights. It affects training speed, memory usage, and generalization.

**Trade-offs:**
1. **Large batch size (e.g., 256, 512):**
   - Pro: stable gradients, better GPU utilization, faster wall-clock training.
   - Con: may converge to sharp minima (poor generalization), require careful learning rate tuning.

2. **Small batch size (e.g., 8, 16):**
   - Pro: noisy gradients act as regularization, converge to flatter minima (better generalization).
   - Con: slower training (more gradient updates for same data), noisier gradients can cause divergence.

**Learning rate scaling:**
When batch size increases, gradient estimates are more stable, so learning rate can be increased proportionally (learning rate scaling rule: $\text{LR} \propto \sqrt{\text{Batch Size}}$ empirically works well).

**Expert perspective:**
Recent research (Bengio et al.) shows:
1. **Sharpness vs. flatness:** sharp minima (small batch) generalize better (contradicting earlier beliefs).
2. **Label noise:** large batches can be harmed by noisy labels (minority class noise is averaged out in large batches).
3. **Optimal batch size:** empirically, moderate batch sizes (32-256) often work best; very large (>1024) can degrade generalization.
4. **Gradient accumulation:** simulating large batches by accumulating gradients over multiple forward/backward passes allows large effective batch size with small memory.

---

## Object Detection & Localization

### Q16. What is object detection and how does it differ from classification?

**A:** Classification assigns a single label to an entire image. Object detection finds multiple objects in an image, predicting both class and bounding box for each.

**Output format:**
- **Classification:** single label (e.g., "dog").
- **Object detection:** list of (class, x, y, width, height) tuples, one per object.

**Key challenges:**
1. **Variable number of objects:** image may have 0, 1, or many objects.
2. **Localization accuracy:** bounding box must be precise.
3. **Class imbalance:** background (negative examples) vastly outnumber objects.
4. **Scale variation:** objects vary in size.

**Evaluation metrics:**
- **mAP (mean Average Precision):** average precision across all classes, computed at IoU (Intersection over Union) threshold (typically 0.5 or 0.5:0.95).
- **IoU:** $\text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}}$.

**Two main paradigms:**
1. **Two-stage detectors (Faster R-CNN):** first propose candidate regions, then refine.
2. **One-stage detectors (YOLO, SSD):** directly predict all boxes in one pass.

---

### Q17. What is R-CNN and how do Faster R-CNN and Mask R-CNN improve upon it?

**A:** R-CNN introduced the idea of extracting regions of interest (RoIs), extracting features, and classifying them. Faster and Mask improve efficiency and extend functionality.

**R-CNN pipeline (2014):**
1. Extract ~2000 region proposals using selective search.
2. Warp each region to fixed size.
3. Pass through CNN to extract features.
4. Classify with SVM.
5. Refine bounding box with linear regression.

**Problem:** slow (2000 forward passes per image).

**Fast R-CNN (2015):**
1. Extract features once for entire image (single CNN forward pass).
2. Extract RoI features via RoI pooling (crop feature map, max-pool to fixed size).
3. Classify and refine box in one step per RoI.

**Speedup:** 10× faster, more efficient region-wise processing.

**Faster R-CNN (2015):**
Add Region Proposal Network (RPN):
1. Slide a small network over feature map.
2. Predict objectness and box adjustments at multiple scales (anchors).
3. Feed RPN proposals to RoI pooling + classifier.

**Major impact:** end-to-end trainable, no external proposal method needed.

**Mask R-CNN (2017):**
Extend Faster R-CNN for instance segmentation:
1. Add branch that predicts mask (pixel-level classification) for each RoI.
2. Use RoIAlign instead of RoI pooling (bilinear interpolation, preserves small objects).
3. Train mask classification loss in parallel with box and class losses.

**Applications:** object detection, instance segmentation, person keypoint detection.

**Expert view:**
R-CNN family established the two-stage paradigm (proposal → refinement), which is still powerful. Modern improvements:
1. **Cascade R-CNN:** sequential stages of detectors with increasing IoU thresholds, improving high-quality detections.
2. **Deformable RoI:** adapt RoI shape to object, handling non-rectangular objects.
3. **Sparse R-CNN:** treat detection as set prediction problem (inspired by DETR).

---

### Q18. What is YOLO and what advantages do one-stage detectors have?

**A:** YOLO (You Only Look Once) divides image into a grid and predicts bounding boxes and class probabilities for each grid cell, achieving real-time detection.

**YOLO pipeline:**
1. Divide image into $S \times S$ grid.
2. For each grid cell, predict:
   - $B$ bounding boxes (x, y, width, height, confidence).
   - Class probabilities for $C$ classes.
3. Total output: $S \times S \times (5B + C)$ tensor.
4. Non-max suppression removes duplicate/overlapping boxes.

**Speed advantage:**
No separate region proposal step; entire detection is one forward pass. YOLO is real-time (45-155 FPS depending on version).

**Trade-offs vs. two-stage:**
| Aspect | YOLO | Faster R-CNN |
|---|---|---|
| Speed | Real-time | Slower |
| Accuracy | Slightly lower mAP | Higher mAP |
| Small objects | Struggles (grid cell can't have two boxes) | Better |
| Training | Simpler | More complex |

**YOLO variants:**
- **YOLOv2:** batch norm, multi-scale training, anchor boxes.
- **YOLOv3:** multi-scale predictions (3 scales), better for small objects.
- **YOLOv4:** CSPDarknet backbone, mosaic augmentation, IoU-based loss.
- **YOLOv5:** PyTorch implementation, improved training pipeline.

**Modern one-stage detectors:**
- **SSD:** multi-scale feature maps for detections (like FPN).
- **RetinaNet:** focal loss to handle class imbalance.
- **FCOS:** dense prediction (predict at every pixel, not grid cells).
- **EfficientDet:** compound scaling for efficiency.

**Expert view:**
One-stage detectors have become competitive with two-stage after addressing limitations:
1. **Focal loss:** handles foreground-background imbalance (99% of predictions are background).
2. **Anchor-free methods:** FCOS, CornerNet avoid hand-designed anchors.
3. **Transformer-based:** DETR (Detection Transformer) treats detection as set prediction.

For real-time applications, one-stage is dominant.

---

### Q19. What is the difference between anchor-based and anchor-free object detection?

**A:** Anchor-based detectors use predefined box templates; anchor-free detectors directly predict object properties without anchors.

**Anchor-based (YOLO, Faster R-CNN, RetinaNet):**
- Define anchors: predefined boxes at multiple scales and aspect ratios.
- For each anchor, predict class and box offset (tx, ty, tw, th).
- Advantages: simple, well-established.
- Disadvantages: requires careful anchor design, many hyperparameters (scales, ratios, strides).

**Anchor-free (FCOS, CenterNet, CornerNet):**
- Predict object center and size directly at each spatial location.
- Variants:
  - **FCOS:** predict center, width, height per pixel.
  - **CenterNet:** predict center and size regression.
  - **CornerNet:** detect corner points, infer box from corners.

**Advantages of anchor-free:**
1. **Fewer hyperparameters:** no need to tune anchor scales/ratios.
2. **Better for small objects:** anchors may not cover all small object scales.
3. **Flexibility:** can handle objects of any aspect ratio without predefined anchors.

**Disadvantages:**
1. **Training complexity:** positive samples are sparse (only near object center).
2. **Performance:** on standard benchmarks (COCO), anchor-based often still wins.

**Expert perspective:**
Anchor-free gained traction with attention-based detectors (DETR, Deformable DETR), which treat detection as set prediction. The field is moving toward transformer-based, anchor-free approaches, but anchor-based still dominates efficient networks (MobileNets, EfficientDet).

---

### Q20. What is the focal loss and why is it effective for imbalanced object detection?

**A:** Focal loss down-weights easy (background) samples and focuses training on hard (object) examples, addressing the class imbalance problem in object detection.

**Standard cross-entropy:**
$$
\text{CE}(p, y) = -y \log(p) - (1-y) \log(1-p)
$$

For an image with 10 objects and 10,000 background pixels, 99.9% of samples are background. The loss is dominated by easy negatives.

**Focal loss:**
$$
\text{FL}(p, y) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

where:
- $p_t = p$ if $y=1$ (object), $p_t = 1-p$ if $y=0$ (background).
- $\gamma$ (focusing parameter, typically 2): down-weights easy samples. If $p_t$ is high (easy), $(1-p_t)^\gamma$ is small, reducing loss.
- $\alpha_t$ (balancing parameter): class weighting.

**Effect:**
- Hard samples ($p_t \approx 0.5$): $(1-p_t)^\gamma \approx 1$, loss unchanged.
- Easy samples ($p_t \approx 1$): $(1-p_t)^\gamma \approx 0$, loss heavily reduced.

**Impact on training:**
Focus on hard samples accelerates learning and improves final accuracy, especially for imbalanced datasets.

**Expert view:**
Focal loss inspired broader research on loss functions:
1. **Generalized focal loss:** improved parameterization for both classification and localization.
2. **Weighted variants:** learnable sample weights instead of fixed focal loss.
3. **Hard negative mining:** older approach, similar effect but more explicit.

For imbalanced data, focal loss or careful sampling strategies are essential.

---

## Semantic & Instance Segmentation

### Q21. What is semantic segmentation and how does it differ from object detection?

**A:** Semantic segmentation assigns a class label to every pixel. Unlike detection (which gives a bounding box), segmentation provides pixel-level classifications.

**Key difference:**
- **Object detection:** bounding box + class (instance-level).
- **Semantic segmentation:** class label per pixel (semantic-level, doesn't distinguish instances).
- **Instance segmentation:** class label + instance ID per pixel (pixel-level + instance distinction).

**Applications:**
- Autonomous driving (segment road, sidewalk, vehicle, pedestrian).
- Medical imaging (segment organs, tumors).
- Scene understanding (segment furniture, walls, floor).

**Evaluation metrics:**
- **IoU (Jaccard Index):** $\text{IoU} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$ per class.
- **mIoU (mean IoU):** average IoU across classes.
- **Pixel accuracy:** percentage of correctly classified pixels.

**Challenges:**
1. **Pixel-level detail:** need to preserve fine spatial information.
2. **Class imbalance:** some classes (small objects) are rare.
3. **Boundary accuracy:** hard to get precise class boundaries.

---

### Q22. What is a Fully Convolutional Network (FCN) and why was it revolutionary for segmentation?

**A:** FCN replaces fully connected layers with convolutions, enabling variable-size input and pixel-level prediction. It was the first end-to-end trainable network for dense prediction.

**Key innovation:**
Replace final FC layers with convolutions of the same functionality but preserving spatial dimensions.

**Upsampling challenge:**
Convolution reduces spatial resolution via pooling. To recover resolution for pixel-level prediction:
1. **Naive upsampling:** bilinear interpolation (simple but blurry).
2. **Skip connections:** combine low-resolution semantic features with high-resolution spatial details.
   - $\text{predict}_2 = \text{upsample}(\text{conv5}) + \text{conv4}$
   - Repeat for multiple scales.

**Architecture:**
```
Input → (Conv + Pool)* 5 → Conv → Upsample + Skip → Output (same size as input)
```

**Why it works:**
- Early layers preserve spatial detail but lack semantic understanding.
- Deep layers have semantic information but low resolution.
- Skip connections combine both.

**Impact:**
FCN established the encoder-decoder paradigm, which became standard for all dense prediction tasks (segmentation, depth estimation, etc.).

---

### Q23. What is U-Net and how does it improve upon FCN?

**A:** U-Net is an encoder-decoder architecture with skip connections connecting features at the same resolution level, designed for medical image segmentation with limited training data.

**Architecture:**
- **Encoder:** successive conv + pooling, reducing resolution.
- **Decoder:** successive upsampling + conv, increasing resolution.
- **Skip connections:** concatenate encoder features with decoder features at matching resolutions.

The architecture looks like a "U" (hence the name).

**Key improvements over FCN:**
1. **Symmetric encoder-decoder:** preserves spatial information better.
2. **Concatenation (not addition):** concatenate features rather than add, preserving more information.
3. **Better for small datasets:** designed for medical imaging where data is limited.

**Advantages:**
1. **Simple and effective:** works well with small datasets.
2. **Fast training:** relatively few parameters.
3. **Versatile:** extended to 3D (volumetric data), instance segmentation, etc.

**Applications:**
- Medical image segmentation (organs, lesions).
- Microscopy image analysis.
- Satellite image analysis.
- Any dense prediction task with limited data.

**Expert view:**
U-Net's success lies in:
1. **Skip connections:** similar to ResNet, enable deep networks.
2. **Data efficiency:** design choices favor limited data.
3. **Generalization:** performs well across domains without task-specific tuning.

Modern variants:
- **Attention U-Net:** add attention gates to guide skip connections.
- **Dense U-Net:** DenseNet-style dense connections.
- **Transformer U-Net:** replace convolutions with attention.

---

### Q24. What is instance segmentation and how do Mask R-CNN or YOLACT achieve it?

**A:** Instance segmentation assigns both class and instance ID to each pixel. Unlike semantic segmentation (one class per pixel), instance segmentation distinguishes separate objects of the same class.

**Approaches:**

1. **Mask R-CNN (two-stage):**
   - Extend Faster R-CNN with mask branch.
   - For each detected object, predict a mask.
   - High accuracy, slower inference.

2. **YOLACT (one-stage):**
   - Predict prototype masks (shared masks for all instances).
   - Predict mask coefficients per object (linear combination of prototypes).
   - Fast real-time inference, slightly lower accuracy.

3. **SOLO (segmentation by looking at one instance):**
   - Predict masks directly per spatial location without bounding boxes.
   - Anchor-free, simple, effective.

**Evaluation metrics:**
- **AP (Average Precision) at IoU=0.5 or 0.5:0.95** (same as detection but for mask IoU).
- **Mask IoU:** IoU of predicted vs. ground-truth instance mask.

**Challenges:**
1. **Object occlusion:** overlapping objects are hard to segment separately.
2. **Small objects:** masks must be precise, hard with limited resolution.
3. **Boundary accuracy:** need sub-pixel precision at object edges.

**Expert view:**
Recent paradigm: treat instance segmentation as end-to-end set prediction (like detection). MaskFormer uses transformer to predict masks directly (maskformer2, per-pixel classification with transformer cross-attention).

---

### Q25. What is the difference between boundary refinement and dense prediction in segmentation?

**A:** Dense prediction assigns a label to every pixel. Boundary refinement refers to improving accuracy specifically at object boundaries, where most errors occur.

**Dense prediction:**
Predict class label for every output pixel using fully convolutional architecture (no fully connected layers).

**Boundary challenges:**
1. **Recep field limitation:** small receptive field can't see full object context.
2. **Detail loss:** multiple pooling layers lose fine spatial detail.
3. **Aliasing artifacts:** boundary can be staircase-like due to discrete sampling.

**Boundary refinement techniques:**

1. **CRF (Conditional Random Field):** post-process predictions to smooth boundaries while respecting edges.
   - Unary potential: per-pixel class prediction.
   - Pairwise potential: penalize neighboring pixels having different classes.
   - Iterative refinement improves boundaries.

2. **Atrous convolution (dilated convolution):** increase receptive field without reducing resolution, preserving detail.
   - Use dilated kernels (spacing between kernel elements).
   - Cascade atrous convolutions at different rates.

3. **Boundary-aware loss:**
   - Weight boundary pixels higher in loss function.
   - Encourage model to focus on boundary accuracy.

4. **Post-processing:**
   - Morphological operations (erosion, dilation).
   - Active contours or level sets.

**Expert view:**
Modern approaches (e.g., SegFormer, Mask2Former) treat boundaries implicitly through better architectures and training rather than explicit post-processing. Transformer-based methods with cross-attention can directly model long-range dependencies crucial for boundary coherence.

---

## Optical Flow & Motion

### Q26. What is optical flow and how is it estimated?

**A:** Optical flow is the apparent motion of pixels between consecutive frames. It is estimated by finding correspondence: for each pixel in frame 1, find its location in frame 2.

**Brightness constancy assumption:**
A pixel's intensity doesn't change between frames (or changes predictably):
$$
I(x, y, t) = I(x + u, y + v, t + \Delta t)
$$

where $(u, v)$ is the optical flow.

**Lucas-Kanade method:**
Assume all pixels in a small neighborhood have the same flow. Solve:
$$
\begin{bmatrix} I_x & I_y \\ I_x & I_y \\ \vdots & \vdots \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = -\begin{bmatrix} I_t \\ I_t \\ \vdots \end{bmatrix}
$$

Using least squares: $(A^T A)^{-1} A^T b$ where $A$ contains derivatives and $b$ contains temporal derivative.

**Limitations:**
1. **Aperture problem:** only detects motion perpendicular to edges (parallel motion is ambiguous).
2. **Brightness assumption:** fails with illumination changes.
3. **Large displacements:** linearization assumes small motion.

**Deep learning approach (FlowNet, PWCNet, RAFT):**
End-to-end CNN trained to predict optical flow.
- Input: two frames.
- Output: displacement field (u, v).
- Train with synthetic data (FlyingChairs, FlyingThings3D) where ground truth is available.
- Generalize to real video.

**Applications:**
- Video interpolation/synthesis.
- Motion segmentation.
- 3D reconstruction.
- Video action recognition.

---

### Q27. What are the challenges in estimating optical flow and how are they addressed?

**A:** Optical flow estimation faces several challenges due to occlusions, motion boundaries, and large displacements.

**Challenge 1: Occlusions**
When an object moves, new regions appear (disocclusions) and regions disappear (occlusions). No corresponding pixel exists in the other frame.

**Solution:**
- Predict occlusion masks explicitly.
- Use bidirectional flow (forward and backward) to detect inconsistencies.
- Ignore occluded regions in loss.

**Challenge 2: Motion boundaries**
Sharp transitions where neighboring pixels have very different motion. Traditional methods blur at boundaries.

**Solution:**
- Edge-aware filtering (preserve boundaries).
- Boundary-specific losses or post-processing.
- Deep learning automatically learns to preserve boundaries.

**Challenge 3: Large displacements**
For large motion, linear assumptions break down. Searching a large spatial area is expensive.

**Solution:**
- Coarse-to-fine strategy: estimate motion at low resolution first, refine at high resolution.
- Hierarchical approaches: multi-scale pyramids.
- Recurrent refinement: iterative improvement (RAFT).

**Challenge 4: Fast motion / motion blur**
Motion blur violates brightness constancy.

**Solution:**
- Robust cost functions (less sensitive to outliers).
- Temporal window larger than 1 frame.

**Challenge 5: Textureless regions**
Regions without texture (plain walls) have ambiguous motion.

**Solution:**
- Spatial regularization (smooth field, enforce consistency with neighbors).
- Semantic guidance (use object information).

**Expert view:**
RAFT (Recurrent All-Pairs Field Transforms) achieves state-of-the-art by:
1. Computing all-pairs correlations (expensive but complete information).
2. Iterative update with GRU cell (like traditional iteration but learnable).
3. Training on diverse data improves generalization.

Optical flow remains challenging due to fundamental ambiguities in imaging.

---

### Q28. What is scene flow and how does it extend optical flow?

**A:** Scene flow is 3D motion in a real-world scene, extending 2D optical flow (which is in image space) to 3D. It assigns a 3D velocity vector to each point.

**Optical flow vs. Scene flow:**
- **Optical flow:** (u, v) displacement in image space (2D).
- **Scene flow:** (vx, vy, vz) 3D velocity in world space.

**Scene flow includes:**
1. **Camera motion:** ego-motion.
2. **Object motion:** independent motion of objects.

**Estimation methods:**

1. **From two images:**
   - Use optical flow as input.
   - Add depth (from stereo or monocular depth estimation).
   - Recover 3D motion via camera geometry.

2. **From stereo + temporal:**
   - Compute depth from stereo.
   - Compute optical flow between frames.
   - Combine to get 3D velocity.

3. **Point cloud-based (3D convolution):**
   - For LiDAR/3D data, use 3D convolution on point clouds.
   - Learn 3D motion directly.

4. **Deep learning (FlowNet3D, PointRCNN):**
   - End-to-end learning from point clouds.
   - Predict 3D displacement per point.

**Applications:**
- Autonomous driving (understand surrounding motion).
- 3D reconstruction.
- Action recognition.
- Robot navigation.

**Expert view:**
Scene flow is crucial for autonomous driving and robotics. Recent work combines:
1. **Multi-modal fusion:** RGB, stereo, LiDAR.
2. **Uncertainty estimation:** important for safety-critical systems.
3. **Temporal consistency:** enforce smoothness across time.

Scene flow in dynamic, multi-object scenes with occlusions remains an open challenge.

---

## 3D Vision & Geometry

### Q29. What is the epipolar geometry and why is it fundamental to stereo vision?

**A:** Epipolar geometry describes the geometric relationship between two views of a scene. It constrains where a point in one image can appear in another image, reducing the search space for correspondence.

**Key concepts:**
- **Epipole:** projection of one camera center onto the other camera's image plane.
- **Epipolar line:** line in one image along which corresponding point must lie.
- **Fundamental matrix (F):** encodes epipolar geometry (relates pixel coordinates in two images).

**Epipolar constraint:**
If point $p$ in image 1 corresponds to point $p'$ in image 2:
$$
p'^T F p = 0
$$

This says $p'$ lies on the epipolar line $F p$ (or equivalently, $p$ lies on $F^T p'$).

**Computing F:**
Use 8-point algorithm:
1. Select 8 corresponding point pairs.
2. Set up system $AF = 0$.
3. Solve using SVD, take singular vector for smallest singular value.
4. Enforce rank-2 constraint via SVD.

**Benefits:**
1. **Reduced search space:** instead of searching entire 2D image, search along 1D epipolar line.
2. **Robustness:** reject outliers whose point correspondences violate epipolar constraint.
3. **Rectification:** reproject images so epipolar lines are horizontal, simplifying stereo matching.

**Expert view:**
Epipolar geometry is elegant but assumes:
1. **Known intrinsics:** if camera calibration unknown, use essential matrix (E) instead of fundamental matrix.
2. **Pinhole camera model:** assumes perfect perspective projection.
3. **No radial distortion:** if present, must undistort first.

Modern deep learning approaches (SuperGlue, LoFTR) learn correspondences end-to-end, partly avoiding need for explicit epipolar geometry, but it remains theoretically fundamental.

---

### Q30. What is structure from motion (SfM) and how does it reconstruct 3D from images?

**A:** Structure from Motion reconstructs a 3D scene from multiple 2D images by solving for 3D point positions and camera poses simultaneously.

**Pipeline:**
1. **Feature matching:** find corresponding points across images.
2. **Camera pose estimation:** compute relative camera positions (R, t).
3. **Triangulation:** compute 3D point positions from multiple views.
4. **Bundle adjustment:** jointly refine all parameters (3D points + camera poses) to minimize reprojection error.

**Mathematical formulation:**
Given image point $x_{ij}$ in image $i$ corresponding to 3D point $X_j$:
$$
x_{ij} = P_i X_j
$$

where $P_i$ is the projection matrix (camera matrix). Solve for $X_j$ and $P_i$.

**Bundle adjustment:**
Minimize reprojection error:
$$
\sum_i \sum_j \| x_{ij} - P_i X_j \|^2
$$

This is the gold standard for high-quality reconstruction.

**Challenges:**
1. **Scale ambiguity:** SfM recovers scale up to a factor (can't distinguish 100m with 10m camera vs. 1000m with 100m camera if images are identical after scaling).
2. **Initialization:** first pose pair must be reliable; wrong initialization leads to wrong reconstruction.
3. **Occlusion:** points visible in some views but not others; tracking consistency is hard.
4. **Computationally expensive:** bundle adjustment is non-linear optimization on large systems.

**Applications:**
- 3D scene reconstruction.
- Large-scale mapping (Google Earth 3D).
- Virtual reality content creation.
- Autonomous vehicles (localization and mapping).

**Expert view:**
Modern SfM uses:
1. **Incremental SfM:** add one image at a time, refining as you go (more stable).
2. **Global SfM:** estimate all poses first, then refine (faster for large sets).
3. **Feature learning:** SuperPoint, SuperGlue learn better features/matches than hand-crafted (SIFT).
4. **Neural radiance fields (NeRF):** alternative approach, learn implicit 3D representation and render novel views.

---

### Q31. What is depth estimation and what are monocular vs. stereo approaches?

**A:** Depth estimation assigns a depth value (distance from camera) to each pixel. Monocular uses single image; stereo uses two synchronized images.

**Monocular depth:**
Single image → depth map. Underdetermined problem (infinitely many 3D scenes project to same image), requires learning priors.

**Approach:**
1. **Self-supervised learning:** use photometric loss (warping consistency between views) instead of ground truth.
   - Predict depth and ego-motion from consecutive frames.
   - Warp frame $t$ to frame $t-1$ using predicted depth/motion.
   - Minimize photometric loss (brightness difference).

2. **Supervised learning:** train on depth datasets (KITTI, NYU Depth).
   - CNN encoder-decoder predicting depth map.
   - Loss: L1 or L2 distance to ground truth.

**Stereo depth:**
Two images from synchronized cameras, known baseline distance.

**Approach:**
1. **Correspondence:** find matching pixels in left and right images.
2. **Disparity:** compute horizontal displacement (disparity = baseline × focal_length / depth).
3. **Depth:** depth = baseline × focal_length / disparity.

**Stereo matching methods:**
- **Block matching:** correlate small windows across images, select best match.
- **Deep learning (GCNet, PSMNet):** learn matching cost via CNN, then aggregate via post-processing.

**Trade-offs:**

| Aspect | Monocular | Stereo |
|---|---|---|
| **Requirement** | Single image | Two synchronized cameras |
| **Accuracy** | Lower (ambiguous) | Higher (geometric) |
| **Baseline dependency** | None | Baseline affects accuracy |
| **Computational cost** | Moderate | High (many candidates per pixel) |
| **Real-time feasibility** | Yes | Yes (with efficient hardware) |

**Expert view:**
Recent hybrid approaches:
1. **Stereo + learning:** combine geometric constraints with learned cues.
2. **Multi-view:** more than two views improve robustness.
3. **Active stereo:** project structured light to aid matching in textureless regions.
4. **Event cameras:** capture brightness changes asynchronously, useful for fast motion.

---

### Q32. What is point cloud processing and why is it important for 3D vision?

**A:** Point clouds are sets of 3D points, typically from LiDAR or depth cameras. Processing point clouds is fundamental for 3D object detection, segmentation, and autonomous driving.

**Challenges with point clouds:**
1. **Unordered:** no natural ordering like images (which have rows/columns).
2. **Unstructured:** sparse and irregular sampling.
3. **Scale variation:** different regions have different densities.
4. **Rotation/translation invariance:** network should be insensitive to global pose.

**Processing approaches:**

1. **Voxelization:** convert point cloud to 3D grid, apply 3D convolution.
   - Advantage: use standard 3D CNN.
   - Disadvantage: memory intensive, loses fine detail if voxel size large.

2. **PointNet:** process points directly without voxelization.
   - Apply MLP to each point independently.
   - Aggregate (max-pool) to get global feature.
   - Permutation invariant (order doesn't matter).

3. **Graph-based (PointNet++, DGCNN):** build k-NN graph, convolve on graph.
   - Respect local structure.
   - More expressive than flat PointNet.

4. **Transformer-based (Point Transformer):** self-attention on points.
   - Each point attends to all other points.
   - Learns which points are relevant.

**Applications:**
- 3D object detection (autonomous driving).
- Semantic segmentation (scene understanding).
- 3D shape completion (inpainting missing regions).
- Point cloud registration (alignment).

**Expert view:**
Point cloud research is active:
1. **Efficiency:** real-time processing for autonomous driving requires efficient networks.
2. **Generalization:** models trained on one sensor (LiDAR) often fail on another (depth camera).
3. **Sparse convolution:** specialized kernels for sparse 3D data.
4. **Fusion:** combining LiDAR + image for better 3D understanding.

---

## Advanced Topics

### Q33. What is adversarial robustness and why are neural networks vulnerable to adversarial examples?

**A:** Adversarial examples are inputs with tiny perturbations that fool neural networks but are imperceptible to humans. Robustness is the network's resistance to such attacks.

**Example:**
Add small noise to an image of a dog, and the network predicts "cat" with 99% confidence, even though the image looks unchanged to humans.

**Why this happens:**
1. **Linear hypothesis:** networks are surprisingly linear in high dimensions; tiny perturbations can cause large decision boundary crossings.
2. **High-dimensional vulnerability:** in high dimensions, lots of room for adversarial directions.
3. **Overfitting to clean data:** networks learn decision boundaries fit to training distribution, not robust to out-of-distribution perturbations.

**Adversarial attack methods:**

1. **FGSM (Fast Gradient Sign Method):**
   - Compute gradient $\nabla_x L(x, y)$.
   - Perturb: $x' = x + \epsilon \cdot \text{sign}(\nabla_x L)$.
   - Fast but not always effective against defended models.

2. **PGD (Projected Gradient Descent):**
   - Iteratively apply FGSM-like updates.
   - More powerful, closer to optimal adversarial example (given $\epsilon$ bound).

3. **C&W (Carlini & Wagner):**
   - Optimize perturbation magnitude while ensuring misclassification.
   - Very effective but computationally expensive.

**Defense mechanisms:**

1. **Adversarial training:**
   - Train on adversarial examples alongside clean examples.
   - Improves robustness but may hurt clean accuracy.

2. **Certified defenses:**
   - Provably bound robustness radius (any perturbation $< r$ is correctly classified).
   - Trade-off: lower clean accuracy.

3. **Detection:**
   - Detect adversarial examples and reject them.
   - May be circumvented by adaptive attacks.

**Expert view:**
Adversarial robustness is fundamental to deploying models in safety-critical systems (autonomous driving, medical diagnosis). Recent insights:
1. **Accuracy-robustness trade-off:** improving robustness often hurts clean accuracy.
2. **Certified robustness:** randomized smoothing can provide provable guarantees.
3. **Threat model matters:** robustness against FGSM is not same as robustness against PGD or C&W.
4. **Transferability:** adversarial examples often transfer across models, raising security concerns.

---

### Q34. What is domain adaptation and why is it important for practical computer vision?

**A:** Domain adaptation addresses the gap between training data distribution (source) and test data distribution (target). A model trained on one domain (synthetic) may fail on another (real).

**Problem example:**
Train detector on synthetic 3D graphics → test on real photos. Large domain gap causes accuracy drop (30-50% mAP drop typical).

**Unsupervised domain adaptation:**
1. **Distribution matching:**
   - Minimize distance between source and target feature distributions (MMD, CORAL).
   - Assumption: same decision boundary works in both domains.

2. **Adversarial adaptation:**
   - Train feature extractor and domain classifier adversarially.
   - Classifier tries to distinguish source/target features.
   - Extractor tries to fool classifier, making features domain-invariant.

3. **Self-training:**
   - Use model's own predictions on unlabeled target data as pseudo-labels.
   - Iteratively refine.
   - Risk: label drift (errors compound).

**Semi-supervised domain adaptation:**
- Combine few labeled target samples + unlabeled target + labeled source.
- Often gives best results.

**Multi-source domain adaptation:**
- Leverage multiple source domains.
- Learn which source is most relevant for each target sample.

**Applications:**
- Synthetic-to-real (simulation for training, real world for deployment).
- Cross-dataset generalization (ImageNet-trained model on custom data).
- Medical imaging (different hospitals, devices).
- Autonomous driving (different cities, weather).

**Expert view:**
Domain adaptation is active research:
1. **Open-set adaptation:** target domain has classes not in source.
2. **Universal adaptation:** single model works across all target domains.
3. **Test-time adaptation:** adapt at test time using unlabeled test data.
4. **Partial domain adaptation:** only some target classes are in source.

Realistic deployment requires handling domain shift, not just optimizing on clean data.

---

### Q35. What is zero-shot and few-shot learning, and how do they enable learning with limited data?

**A:** Zero-shot learning recognizes unseen classes using semantic information (attributes, descriptions). Few-shot learning learns from very few labeled examples (1-10 per class).

**Zero-shot learning:**
- **Problem:** recognize objects not in training set.
- **Solution:** use auxiliary semantic information.
  - **Attributes:** "has feathers", "lays eggs" → recognize unseen birds.
  - **Word embeddings:** semantic similarity between unseen class description and seen classes.
  - **Example:** if trained on dog, can recognize wolf using description similarity.

**Advantages:**
- Scale to arbitrary new classes without retraining.
- Use semantic knowledge (language) to guide recognition.

**Disadvantages:**
- Semantic information may be incomplete or incorrect.
- Performance lower than supervised learning on seen classes.

**Few-shot learning:**
- **Problem:** learn from 1-5 examples per class.
- **Approach 1 (Metric learning):**
  - Learn embedding space where same-class examples are close, different-class examples are far.
  - At test time, classify based on distance to few support examples.
  - Example: Siamese networks, Prototypical Networks.

- **Approach 2 (Meta-learning):**
  - Train on many few-shot tasks (each task: support set + query set).
  - Learn an algorithm that quickly adapts to new tasks.
  - Example: MAML (Model-Agnostic Meta-Learning).

- **Approach 3 (Transductive):**
  - Use unlabeled test examples to refine predictions.
  - Often outperforms inductive (label-only) approaches.

**MAML algorithm:**
1. Sample task (e.g., distinguish cats vs dogs with 5 examples).
2. Perform few gradient steps on support set.
3. Evaluate on query set.
4. Backprop error through gradient steps to update meta-parameters.
5. Repeat on new tasks.

**Applications:**
- **One-shot learning:** recognize person from single photo (face recognition).
- **Few-shot object detection:** detect new classes with few examples.
- **Medical imaging:** diseases are rare, few labeled examples.

**Expert view:**
Few-shot learning is important for:
1. **Rapid adaptation:** new product recognition without retraining.
2. **Personalization:** adapt to individual user preferences.
3. **Fairness:** enable minority classes to be recognized despite fewer training examples.

Recent work combines few-shot with foundation models (CLIP) for improved transfer.

---

### Q36. What is self-supervised learning and how does it leverage unlabeled data?

**A:** Self-supervised learning trains networks on unlabeled data by creating supervised tasks from the data itself, then using learned representations for downstream tasks.

**Key idea:**
Use the data as its own label. For example, predict future frames from past frames, or predict one part of image from another.

**Common self-supervised approaches:**

1. **Contrastive learning (SimCLR, MoCo):**
   - Create two augmented views of same image (positive pair).
   - Create pairs with different images (negative pairs).
   - Train to maximize similarity of positive pairs, minimize negative pairs.
   - Force network to learn invariances to augmentation.

   Loss (NT-Xent):
   $$
   \mathcal{L}_i = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \exp(\text{sim}(z_i, z_k) / \tau)}
   $$

2. **Masked autoencoding (MAE):**
   - Mask random patches of image.
   - Predict masked patches from visible ones.
   - Forces network to understand structure.

3. **Momentum contrast (MoCo):**
   - Maintain momentum-updated dictionary of past encodings.
   - Use large-scale (100k+) dictionary for better negative samples.
   - Improves contrastive learning efficiency.

4. **BYOL (Bootstrap Your Own Latent):**
   - No explicit negatives (only positive pairs).
   - Uses exponential moving average (EMA) of network as target.
   - Prevents representation collapse via EMA, not negative mining.

**Why it works:**
- Augmentation invariance: network learns that rotated/cropped images are "same".
- Semantic learning: to predict masked patches, network must understand semantics.
- No label noise: self-supervised signals are noise-free.

**Downstream use:**
1. **Linear evaluation:** freeze pre-trained features, train linear classifier on small labeled dataset.
2. **Fine-tuning:** unfreeze and fine-tune on downstream task.

**Results:**
- SimCLR on ImageNet: 69.3% top-1 with 1% labels (vs. 25% supervised baseline).
- MAE: 84% top-1 ImageNet after fine-tuning (competitive with supervised).

**Expert view:**
Self-supervised learning is transforming CV:
1. **Scale matters:** improvements are proportional to model/data scale (like language models).
2. **Foundation models:** pre-train once, fine-tune on many tasks (DINO, OpenAI CLIP).
3. **Multimodal learning:** combine image + text (CLIP pre-trained image features useful for zero-shot classification).
4. **Temporal learning:** video provides richer supervision than static images.

---

### Q37. What is image inpainting (completion) and what approaches are used?

**A:** Image inpainting fills in missing or corrupted image regions. It is useful for removing objects, repairing damage, or content creation.

**Traditional approaches:**

1. **Exemplar-based:**
   - Find similar patches elsewhere in image or database.
   - Copy and blend patches into missing region.
   - Fast, simple, works when similar content exists nearby.
   - Fails when missing region is unique.

2. **Optimization-based (total variation, Poisson blending):**
   - Formulate as optimization: minimize difference with neighbors, match gradients at boundary.
   - Smooth, but often blurry.

**Deep learning approaches:**

1. **GANs:**
   - Generator produces inpainted region.
   - Discriminator ensures realism.
   - Context Encoder, Pix2Pix variants.
   - Fast inference, but may hallucinate detail.

2. **VAE (Variational Autoencoder):**
   - Encoder compresses image.
   - Decoder reconstructs.
   - For inpainting, condition decoder on visible parts.
   - Smoother results than GAN, less sharp.

3. **Diffusion models:**
   - Reverse process: iteratively add noise to image (forward), then denoise in reverse.
   - Condition on visible region during reverse process.
   - State-of-the-art quality, sharp and realistic.

4. **Transformer-based:**
   - Treat inpainting as sequence completion.
   - Attend to surrounding context.

**Loss functions:**

1. **Reconstruction loss:** $L_1(\hat{I}, I)$ (predict known regions accurately).
2. **Adversarial loss:** GAN loss (make inpainting indistinguishable from real).
3. **Perceptual loss:** VGG feature similarity (preserve high-level structure).
4. **Style loss:** Gram matrix similarity (match texture).

**Challenges:**
- **Ambiguity:** multiple plausible completions.
- **Context understanding:** must understand global scene structure.
- **Realistic details:** fine textures are hard to generate.

**Expert view:**
Inpainting quality has improved dramatically with diffusion models and large foundation models. Recent work:
1. **Instruction-based inpainting:** use text descriptions to guide what to generate.
2. **Interactive inpainting:** user provides hints, model refines.
3. **Probabilistic inpainting:** estimate distribution of completions, not just single prediction.

---

### Q38. What is 3D object reconstruction and what are common approaches?

**A:** 3D object reconstruction generates 3D geometry from 2D images or point clouds. It is used in AR/VR, e-commerce, 3D printing, etc.

**Approaches:**

1. **Shape from X:**
   - **Shape from shading:** recover 3D from shading/lighting.
   - **Shape from texture:** recover from texture distortion.
   - **Shape from motion:** recover from video.

2. **Multi-view reconstruction:**
   - Capture object from multiple views.
   - Use structure from motion or depth from stereo.
   - Integrate depth maps into single model.
   - Produce mesh or voxel representation.

3. **Learning-based (3D-R2N2, Pix3D):**
   - Single image → 3D shape.
   - Train on synthetic data (ShapeNet 3D model database).
   - Predict voxel grid or point cloud.
   - Low resolution, but learns realistic shapes.

4. **Neural implicit representations (NeRF, SIREN):**
   - Represent 3D as continuous function: position → color/density.
   - MLP network learns this function.
   - Render novel views by querying function.
   - High quality, but computationally expensive.

5. **Mesh reconstruction (Kaolin, Neural-PIL):**
   - Directly predict mesh vertices + faces.
   - Differentiable rendering enables end-to-end training.
   - Faster rendering than implicit.

**Representations:**
1. **Voxel grids:** simple but memory-intensive.
2. **Point clouds:** sparse, efficient.
3. **Meshes:** standard for graphics, parameterizable.
4. **Implicit functions:** compact, arbitrary resolution.
5. **Multi-plane images:** compromise between efficiency and quality.

**Challenges:**
- **Ambiguity:** multiple 3D shapes project to same image.
- **Occlusion:** hidden parts not visible.
- **Topology:** hard to predict correct genus (number of holes).

**Expert view:**
3D reconstruction is advancing:
1. **NeRF variants:** faster, better quality, dynamic scenes.
2. **Generalization:** methods that work on diverse object categories.
3. **Texture recovery:** jointly reconstruct geometry and appearance.
4. **4D reconstruction:** dynamic scenes (objects moving).

---

### Q39. What is visual question answering (VQA) and how does it combine vision and language?

**A:** VQA takes an image and natural language question, then generates an answer. It requires both visual understanding and language reasoning.

**Example:**
- Image: dog playing fetch
- Question: "What is the dog holding?"
- Answer: "ball"

**Architecture:**
1. **Image encoder:** CNN extracting visual features.
2. **Question encoder:** RNN/Transformer encoding text.
3. **Fusion:** combine image and text features.
4. **Decoder:** generate answer (or select from candidates).

**Approaches:**

1. **Attention-based:**
   - Image features at each spatial location.
   - Attention weights based on question.
   - Aggregate attended features to answer.

2. **Graph-based:**
   - Scene graphs: objects + relationships.
   - Question traverses graph to answer.
   - Better for compositional questions.

3. **Transformer-based:**
   - Vision Transformer for image.
   - Text Transformer for question.
   - Cross-attention between image and text tokens.
   - Example: ViLBERT, LXMERT.

**Evaluation metrics:**
- **Accuracy:** exact match percentage.
- **Balanced accuracy:** account for answer distribution imbalance.
- **GQA metric:** relaxed accuracy (synonyms accepted).

**Challenges:**
- **Compositional reasoning:** "Does the dog's ball have the same color as the fence?"
- **Counting:** "How many people are in the image?"
- **Visual relationships:** "Is the dog to the left of the cat?"
- **Common sense:** requires external knowledge beyond image.

**Expert view:**
VQA is advancing toward:
1. **Video QA:** questions about video (temporal reasoning).
2. **Embodied QA:** robot must navigate to answer question (grounding).
3. **Interleaved reasoning:** ask follow-up questions, refine understanding.
4. **Multimodal foundation models:** GPT-4V, Gemini Pro use unified model for image+text.

---

### Q40. What is semantic scene understanding and how do we parse complex scenes?

**A:** Scene understanding assigns semantic meaning to entire images: objects, relationships, layout, attributes. It is more complex than object detection (which only localizes).

**Components of scene understanding:**
1. **Object detection:** "what" objects exist and "where".
2. **Semantic relationships:** "person riding on bike" (spatial + action).
3. **Scene type:** "outdoor beach scene" (global context).
4. **Attributes:** "wooden table", "red car" (object properties).
5. **Physics:** "person standing on ground" (gravity, support).

**Approaches:**

1. **Multi-task learning:**
   - Single encoder, multiple decoders for different tasks.
   - Share learned features across tasks.

2. **Scene graphs:**
   - Nodes: objects.
   - Edges: relationships.
   - Represent structure of scene.
   - Predict: "person [RIDE] on bike [HAS] wheel".

3. **Graph neural networks:**
   - Objects as nodes, relationships as edges.
   - Message passing to refine predictions.

4. **Vision-language models (CLIP, BLIP):**
   - Pre-train on image-text pairs.
   - Can describe scenes with natural language.
   - Flexible: ask arbitrary questions about scene.

**Applications:**
- Autonomous driving: understand traffic, pedestrians, lanes.
- Robotics: navigate and manipulate objects based on scene understanding.
- Content moderation: automatically flag inappropriate content.
- Image search: search by semantic properties, not just visual similarity.

**Expert view:**
Scene understanding is moving toward:
1. **End-to-end reasoning:** from raw pixels to high-level semantic statements.
2. **Open-vocabulary:** recognize any object described in language, not just predefined classes.
3. **3D scene understanding:** 3D scene graphs with spatial relationships.
4. **Temporal consistency:** understand how scenes change over time (video).

---

### Q41. What is panoptic segmentation and why is it useful?

**A:** Panoptic segmentation combines semantic segmentation ("stuff": uncountable regions like sky, road) and instance segmentation ("things": countable objects like person, car).

**Traditional approaches:**
- **Semantic segmentation:** label every pixel with class (e.g., road, sky, person).
- **Instance segmentation:** label countable objects with instance IDs.
- **Panoptic:** unified output combining both.

**Why combine:**
- Autonomous driving needs both: road (stuff) and vehicles (things).
- Scene understanding requires complete picture.
- Single unified metric (PQ: Panoptic Quality) better reflects practical needs than separate metrics.

**Panoptic Quality (PQ):**
$$
\text{PQ} = \frac{\text{TP}}{\text{TP} + 0.5 \cdot \text{FP} + 0.5 \cdot \text{FN}} \times \text{Mean IoU}
$$

Accounts for both detection rate and IoU (quality).

**Panoptic architectures:**
1. **Mask R-CNN variant:** instance branch for "things", semantic branch for "stuff", combine.
2. **Panoptic FCN:** single FCN predicting panoptic labels.
3. **Transformer-based (DETR, Mask2Former):** unified set prediction for all segments (things + stuff).

**Mask2Former advantages:**
- Single decoder for both stuff and things.
- Learned masking queries per class.
- Significant accuracy improvements over prior art.

**Expert view:**
Panoptic segmentation is becoming standard in scene understanding:
1. **4D panoptic:** panoptic + temporal (video panoptic segmentation).
2. **Open vocabulary:** recognize any stuff/thing described in natural language.
3. **Real-time panoptic:** needed for autonomous driving (30+ FPS).

---

### Q42. What is video action recognition and how do you capture temporal information?

**A:** Video action recognition classifies activities in videos (e.g., "playing basketball", "running"). It requires understanding temporal dynamics, not just individual frames.

**Approaches:**

1. **2D CNN + temporal pooling:**
   - Apply 2D CNN to each frame.
   - Pool (max or average) across frames.
   - Simple but ignores temporal order.

2. **3D CNN (C3D):**
   - Extend convolution to temporal dimension.
   - Kernel shape: $(t, h, w)$ (time, height, width).
   - Captures spatiotemporal patterns directly.
   - Computationally expensive.

3. **Two-stream networks:**
   - **Spatial stream:** RGB frames (what).
   - **Temporal stream:** optical flow (how moving).
   - Fuse predictions from both streams.
   - Interpretable, effective.

4. **RNN / LSTM:**
   - Frame features → RNN → action label.
   - Capture temporal dependencies.
   - Better for long-range temporal reasoning.

5. **Transformer (TimeSformer, ViViT):**
   - Self-attention across time.
   - Process video as sequence of tokens.
   - Flexible: can model long-range dependencies.

**Input formats:**
1. **Frame sampling:** use every $k$-th frame to reduce computation.
2. **Clips:** use short clips (8-32 frames) around action.
3. **Dense sampling:** use all frames (computationally expensive).

**Evaluation metrics:**
- **Top-1 accuracy:** most probable action is correct.
- **Top-5 accuracy:** ground truth in top 5 predictions.
- **Mean accuracy per class:** average across classes (handles imbalance).

**Challenges:**
- **Temporal scale variation:** some actions last seconds, others minutes.
- **Untrimmed videos:** action may span only part of video.
- **Partial visibility:** action may start/end outside video.

**Expert view:**
Video action recognition is evolving:
1. **Efficient models:** real-time action recognition.
2. **Self-supervised temporal learning:** learn from unlabeled video.
3. **Long-range temporal reasoning:** handle actions spanning many seconds.
4. **Open-vocabulary:** recognize actions described in language, not predefined classes.

---

### Q43. What is visual tracking and what are the main challenges?

**A:** Visual tracking follows an object across video frames. Given initial bounding box in frame 1, predict box in frames 2, 3, ... It is fundamental for surveillance, autonomous driving, sports analytics.

**Challenges:**

1. **Appearance change:** object changes due to lighting, pose, scale, deformation.
2. **Motion blur:** fast motion causes blur.
3. **Occlusion:** object disappears behind obstacles.
4. **Background clutter:** similar textures nearby.
5. **Scale variation:** object gets closer/farther.
6. **Out-of-plane rotation:** object rotates in 3D.

**Tracking methods:**

1. **Correlation filtering (KCF):**
   - Train discriminative classifier to distinguish object from background.
   - Use fast frequency-domain operations.
   - Fast but struggles with drastic appearance changes.

2. **Deep learning (Siamese networks, SiamFC):**
   - Siamese architecture: two identical branches for template and search region.
   - Compute similarity map: areas most similar to template.
   - Peak indicates target location.
   - Faster than detection-based methods, but may drift.

3. **Attention-based (SiamRPN, SiamBAN):**
   - Combine Siamese with RPN (region proposal network).
   - Attention modules refine predictions.
   - Better handling of scale changes.

4. **Transformer-based (TransT, KeepTrack):**
   - Self/cross-attention between template and search region.
   - Effective feature matching.
   - State-of-the-art accuracy.

**Online vs. offline:**
- **Online:** update model during tracking (adapt to appearance changes).
- **Offline:** fixed model (faster, more stable).

**Evaluation metrics:**
- **Center error:** distance between predicted and ground-truth center.
- **IoU:** bounding box overlap.
- **Success rate:** % of frames where IoU > threshold.

**Expert view:**
Tracking applications differ:
1. **Short-term tracking:** frames are temporally close, appearance stable.
2. **Long-term tracking:** frames may be far apart, appearance may change drastically.
3. **Multi-object tracking:** track multiple objects simultaneously, handle occlusion, ID switches.
4. **Efficient tracking:** real-time on embedded devices (mobile, edge).

---

### Q44. What is image retrieval and how do we find similar images?

**A:** Image retrieval finds images similar to a query image in a large database. Applications: visual search, copyright detection, deduplication, recommendation.

**Approaches:**

1. **Feature-based:**
   - Extract features (SIFT, ORB, learned embeddings).
   - Compare features (Euclidean distance, cosine similarity).
   - Fast, simple.

2. **Learning to rank:**
   - Train metric learning model to embed images.
   - Similar images have close embeddings.
   - Methods: triplet loss, contrastive loss, angular loss.

3. **Hashing:**
   - Quantize embeddings to binary codes.
   - Compare via Hamming distance (very fast).
   - Trade accuracy for speed.

4. **Graph-based:**
   - Build similarity graph of database.
   - Perform nearest neighbor search on graph.
   - Approximate NN (ANN) enables billion-scale search.

**Deep metric learning:**
1. **Siamese networks:** twin branches, learn to distinguish same/different pairs.
2. **Triplet loss:** push positive pair together, negative pair apart.
   $$
   \mathcal{L} = \max(d(a, p) - d(a, n) + \alpha, 0)
   $$

3. **Hard negative mining:** focus on difficult (hard) negatives.
4. **Multi-scale retrieval:** use features from multiple layers for robustness.

**Evaluation metrics:**
- **Precision@K:** fraction of top-K results that are correct.
- **mAP (mean Average Precision):** average precision across all queries.
- **Recall:** percentage of true positives found.

**Challenges:**
- **Query ambiguity:** "apple" could be fruit or company logo.
- **Domain shift:** model trained on ImageNet may not work on medical images.
- **Scale:** billion-image databases require efficient indexing.

**Expert view:**
Image retrieval is evolving:
1. **Cross-modal retrieval:** find images matching text description (CLIP).
2. **Fine-grained retrieval:** distinguish subtle differences (different car models).
3. **Adversarial robustness:** robust to adversarial perturbations in query.
4. **Efficient indexing:** sublinear search time with learned indexes.

---

### Q45. What is style transfer and how does it separate content from style?

**A:** Style transfer applies the style (texture, color, brush strokes) of one image to the content of another. It requires separating and recombining content and style.

**Neural style transfer (Gatys et al.):**
1. **Content representation:** deep CNN features capture semantic content (objects, layout).
2. **Style representation:** Gram matrix of features captures style (texture, color statistics).
3. **Optimization:** minimize content loss (match content features) + style loss (match Gram matrices).

**Content loss:**
$$
L_{\text{content}} = \frac{1}{2} \sum_{i,j} (F^l_{ij} - P^l_{ij})^2
$$

where $F, P$ are feature maps of generated image and content image.

**Style loss:**
$$
L_{\text{style}} = \sum_l w_l \sum_{j,k} (G^l_{jk} - A^l_{jk})^2
$$

where $G, A$ are Gram matrices of generated and style images.

**Total loss:**
$$
L_{\text{total}} = \alpha L_{\text{content}} + \beta L_{\text{style}}
$$

**Advantages:**
- Captures semantic content and visual style separately.
- Works across different domains.
- Interpretable: Gram matrix is good texture descriptor.

**Disadvantages:**
- Slow (optimization on every image).
- May introduce artifacts (distorted shapes if style is strong).
- Limited to visual style, not semantic style.

**Fast style transfer (feed-forward networks):**
- Train CNN to perform style transfer in one forward pass (instead of iterative optimization).
- Much faster but less flexible (one network per style).
- Trade-off: speed vs. flexibility.

**Advanced techniques:**
1. **Adaptive instance normalization (AdaIN):** align feature statistics of content/style.
2. **Perceptual losses:** use pre-trained features as loss (better than pixel-level L2).
3. **Photo-realistic style transfer:** preserve photorealism while transferring color/texture.

**Expert view:**
Style transfer applications:
1. **Artistic rendering:** convert photo to painting style.
2. **Domain transfer:** convert photo style (e.g., sunny to rainy).
3. **Augmentation:** augment training data with style variations.
4. **Consistent style in video:** extend to temporal dimension, maintain consistency across frames.

---

### Q46. What is super-resolution and what are the trade-offs between quality and speed?

**A:** Super-resolution reconstructs high-resolution (HR) images from low-resolution (LR) inputs. It is fundamental for enhancing image quality, medical imaging, surveillance.

**Challenges:**
- **Ill-posed problem:** many HR images could generate same LR image (multiple valid solutions).
- **Aliasing:** information is lost during downsampling; recovery requires assumptions.
- **Realistic detail:** most important: hallucinate plausible details, not just interpolate.

**Approaches:**

1. **Interpolation-based:**
   - Bilinear, bicubic, Lanczos.
   - Fast but blurry (no high-frequency reconstruction).

2. **Sparse coding:**
   - Learn dictionaries of LR/HR patch pairs.
   - Represent LR patch using sparse code, map to HR via dictionary.
   - Better than interpolation, slower than CNN.

3. **Deep CNN (SRCNN, VDSR):**
   - CNN learns residual (HR - upsampled LR).
   - Combine upsampling (bilinear) + CNN refinement.
   - Fast, good quality (PSNR 25-35 dB).

4. **Generative (SRGAN, ESRGAN):**
   - Use GAN to generate realistic, perceptually pleasing details.
   - Perceptual loss (VGG features) + adversarial loss.
   - Higher perceptual quality but may hallucinate (LPIPS better than PSNR).

5. **Diffusion models:**
   - Iteratively denoise from Gaussian noise.
   - Can leverage multiple scales of information.
   - High quality but slow (100+ iterations).

**Trade-offs:**

| Method | Speed | PSNR | Perception |
|---|---|---|---|
| **Bilinear** | Very fast | Low | Low (blurry) |
| **SRCNN** | Fast | Good | Moderate |
| **SRGAN** | Moderate | Lower | High |
| **Diffusion** | Slow | Varies | Very high |

**Expert view:**
Super-resolution is evolving:
1. **Real-world SR:** train on realistic degradation (not just downsampling).
2. **Blind SR:** don't know degradation model, must infer from LR image.
3. **Lightweight SR:** mobile/edge devices (distillation, pruning).
4. **Face SR:** specialized for faces (better priors available).
5. **Video SR:** temporal consistency across frames.

Perceptual quality (LPIPS, FID) increasingly favored over PSNR in research.

---

### Q47. What is video prediction and why is it difficult?

**A:** Video prediction generates future frames given past frames. It requires understanding physics, object interactions, and temporal consistency.

**Example:**
- Input: frames 1-5 of a ball rolling.
- Output: frames 6-10 (predict where ball goes).

**Challenges:**

1. **Stochasticity:** same situation can have multiple futures (person may go left or right).
2. **Long-range dependencies:** physics unfolds over many frames.
3. **Compounding error:** errors accumulate (error at frame 6 affects frame 7, etc.).
4. **Computational cost:** predicting many frames is expensive.

**Approaches:**

1. **Convolutional (ConvLSTM):**
   - LSTM with convolutional operations (instead of matrix multiplication).
   - Preserve spatial structure.
   - Simple, fast, but often blurry.

2. **Adversarial (GAN-based):**
   - Generator predicts frames, discriminator judges realism.
   - Can produce sharp details.
   - May have mode collapse (predict average, blurry frame).

3. **Variational (VAE-based):**
   - Learn distribution of possible futures.
   - Sample from distribution to generate diverse predictions.
   - More realistic, handles stochasticity.

4. **Transformer:**
   - Treat frames as sequence of tokens.
   - Self-attention models dependencies.
   - Flexible, good for understanding motion.

5. **Physical models:**
   - Learn physics (forces, collisions).
   - More interpretable, generalizes better.
   - Harder to learn.

**Evaluation:**
- **PSNR/SSIM:** pixel-level metrics (punish small shifts heavily, blurry predictions often score high).
- **Perceptual metrics (LPIPS, FID):** better correlation with human perception.
- **Diversity:** for stochastic prediction, measure coverage of modes.

**Expert view:**
Video prediction is moving toward:
1. **Latent space prediction:** predict in compressed space (faster, better).
2. **Hierarchical prediction:** predict at multiple timescales.
3. **Physics-informed:** incorporate known physics (gravity, collisions).
4. **Infinite-horizon:** predict arbitrarily far into future (currently limited to short-term).

---

### Q48. What is person re-identification (ReID) and what makes it challenging?

**A:** Person re-identification matches a person across multiple camera views. It is crucial for surveillance, tracking, and forensics.

**Challenges:**

1. **Appearance change:** person may wear different clothes, change pose, or have different lighting.
2. **Camera variation:** different camera angles, resolutions, lighting.
3. **Occlusion:** parts of person hidden.
4. **Similar appearance:** multiple people may look similar.

**Approaches:**

1. **Metric learning:**
   - Train CNN to embed person into space where same person is close, different people far.
   - Triplet loss, center loss, margin-based losses.

2. **Part-based:**
   - Divide person into parts (head, torso, legs).
   - Compare parts separately, aggregate.
   - More robust to occlusion.

3. **Temporal:**
   - Use multiple frames (tracklet) rather than single image.
   - Aggregate features across frames.
   - Improve robustness.

4. **Cross-domain adaptation:**
   - Model trained on one dataset may not work on another (domain gap).
   - Use adversarial adaptation or self-training.

**Evaluation:**
- **Rank-1 accuracy:** top match is correct person.
- **mAP (mean Average Precision):** area under precision-recall curve.
- **CMC (Cumulative Matching Characteristic):** rank-K accuracy for K=1, 2, ..., N.

**Datasets:**
- **Market-1501:** 1501 identities, 32,668 images from 6 cameras.
- **CUHK03:** 1,360 identities, 13,164 images from 6 cameras.
- **DukeMTMC:** 1,406 identities, 36,411 images from 8 cameras.

**Expert view:**
ReID is advancing toward:
1. **Open-set ReID:** recognize identities not in training set.
2. **Partial ReID:** match full person to partial body.
3. **Text-based ReID:** find person matching text description ("wearing red shirt").
4. **Video ReID:** track person in video, use temporal information.
5. **Occluded ReID:** match people despite occlusion.

---

### Q49. What is facial recognition and what are the key components?

**A:** Facial recognition verifies (1-to-1) or identifies (1-to-many) a person from face image. It involves face detection, alignment, embedding, and matching.

**Pipeline:**
1. **Face detection:** locate faces in image (e.g., RetinaFace, MTCNN).
2. **Face alignment:** align face to canonical pose (rotate, warp so eyes/mouth are at expected positions).
3. **Embedding:** extract feature vector (512D or 128D) via CNN.
4. **Matching:** compare embeddings via distance (Euclidean or cosine).

**Key architectures:**
1. **VGGFace:** first CNN-based face recognition.
2. **FaceNet:** triplet loss for learning face embeddings. Goal: same person close, different people far.
3. **ArcFace:** angular margin loss. Directly optimize angular distance in hypersphere.
   $$
   L = -\log \frac{e^{s(\cos(\theta_i + m))}}{e^{s(\cos(\theta_i + m))} + \sum_{j \neq i} e^{s \cos \theta_j}}
   $$

**Loss functions:**
1. **Triplet loss:** $(D(a,p) - D(a,n) + \alpha)^+$.
2. **Center loss:** pull samples toward class center.
3. **Additive margin losses (ArcFace, CosFace):** large margin in feature space.

**Challenges:**
1. **Pose variation:** side-view face vs. front-view.
2. **Illumination:** bright/dark lighting.
3. **Aging:** face changes over years.
4. **Occlusion:** face covered by mask, glasses, etc.
5. **Deepfakes:** synthetic faces that fool recognition.

**Evaluation:**
- **LFW (Labeled Faces in the Wild):** benchmark, 13,233 face pairs. Accuracy ~99.8% (state-of-the-art).
- **VoxCeleb:** large-scale speaker verification dataset, commonly used for face recognition too.
- **MegaFace:** benchmark with 1M distractors, more realistic (harder).

**Privacy concerns:**
- Accuracy is very high, raises surveillance/privacy concerns.
- Bias: models perform worse on certain demographics (darker skin tone, women).
- Regulations (EU GDPR, US state laws) restrict facial recognition use.

**Expert view:**
Facial recognition is mature but evolving:
1. **Mask-robust:** recognize faces with/without masks (COVID impact).
2. **Synthetic faces:** detect and handle deepfakes.
3. **Fairness:** reduce demographic bias.
4. **Privacy-preserving:** federated learning, on-device processing.
5. **Age-invariant:** handle aging over time.

---

### Q50. What is multi-task learning and why is it important in computer vision?

**A:** Multi-task learning trains a single model on multiple tasks simultaneously, leveraging shared representations to improve generalization.

**Example:**
Train single CNN for:
- Semantic segmentation (per-pixel class).
- Instance segmentation (per-pixel instance ID).
- Edge detection (per-pixel binary edge).
- Depth estimation (per-pixel depth).

**Why it helps:**
1. **Shared features:** early layers extract general features useful for all tasks.
2. **Regularization:** multitask provides implicit regularization (harder to overfit).
3. **Improved generalization:** more training signal (loss from multiple tasks).
4. **Efficiency:** single model vs. multiple models.

**Architecture:**
1. **Shared encoder:** extract features once.
2. **Task-specific decoders:** output predictions for each task.

**Loss combination:**
$$
L_{\text{total}} = \alpha_1 L_1 + \alpha_2 L_2 + \ldots + \alpha_n L_n
$$

Weights $\alpha_i$ balance tasks (harder tasks may need higher weight).

**Challenges:**

1. **Task conflation:** tasks may not benefit each other equally.
2. **Weighting:** finding good loss weights is important and difficult.
3. **Negative transfer:** sometimes one task hurts another.

**Solutions:**
1. **Task weighting:** learnable weights (uncertainty weighting).
   $$
   L = \sum_i \frac{1}{\sigma_i^2} L_i + \lambda \log \sigma_i
   $$
   Higher uncertainty → lower weight.

2. **Curriculum learning:** train easy tasks first, then hard tasks.
3. **Task-specific layers:** allow some divergence between tasks.

**Successful applications:**
- **Panoramic multi-task (Taskonomy):** studied 26 vision tasks, found structure and transfer learning potential.
- **Autonomous driving:** predict bounding boxes, segmentation, depth, optical flow.
- **Medical imaging:** segment organs, detect lesions, estimate risk.

**Expert view:**
Multi-task learning is fundamental to modern vision:
1. **Foundation models:** pre-train on many tasks, fine-tune on downstream task.
2. **Self-supervised multi-task:** combine self-supervised losses (contrastive + reconstruction).
3. **Meta-learning:** learn to weight tasks automatically.
4. **Continual multi-task:** handle new tasks arriving over time without forgetting old ones.

---

## Traditional Image Processing Techniques
### Foundation Techniques Interviewers Test for Core Understanding

---

### Q51. What is image filtering and what are the main types of filters?

**A:** Image filtering applies a kernel to image pixels, computing weighted combinations to enhance or extract features. It is fundamental to all image processing.

**Types:**

1. **Linear filters:**
   - **Gaussian blur:** smooths image, reduces noise.
   - **Sobel, Prewitt:** compute gradients (edge detection).
   - **Laplacian:** second derivative, edge detection and enhancement.

2. **Non-linear filters:**
   - **Median filter:** replace pixel with median of neighborhood (removes salt-and-pepper noise).
   - **Bilateral filter:** smooth while preserving edges (edge-aware).
   - **Morphological filters:** erosion, dilation (binary image processing).

**Convolution formula:**
$$
g(x, y) = \sum_{i=-n}^{n} \sum_{j=-n}^{n} f(x+i, y+j) \cdot h(i, j)
$$

where $f$ is image, $h$ is filter kernel, $g$ is output.

**Frequency domain view:**
Convolution in spatial domain = multiplication in frequency domain (Fourier convolution theorem). This insight enables FFT-based fast filtering.

**Expert perspective:**
Filtering is ubiquitous: every vision algorithm uses it (downsampling, edge detection, noise reduction, feature extraction). Understanding filters deeply is critical.

---

### Q52. What is the Sobel operator and how does it detect edges?

**A:** The Sobel operator computes image gradients (directional derivatives) using 3×3 kernels, enabling edge detection.

**Sobel kernels:**
$$
S_x = \left[
\begin{array}{ccc}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{array}
\right],
\quad
S_y = \left[
\begin{array}{ccc}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{array}
\right]
$$

$S_x$ detects vertical edges (horizontal gradient); $S_y$ detects horizontal edges (vertical gradient).

**Gradient magnitude and direction:**
$$
G = \sqrt{S_x^2 + S_y^2}, \quad \theta = \operatorname{atan2}(S_y, S_x)
$$

**Why Sobel works:**
1. **Smoothing:** averaging reduces noise before taking derivative.
2. **Derivative:** central difference approximation to gradient.
3. **Edge detection:** edges appear as high-gradient regions.

**Advantages:**
- Simple, computationally efficient.
- Works well for strong, clean edges.

**Disadvantages:**
- Sensitive to noise.
- Thick edges (multiple pixels).
- Directional bias (prefers horizontal/vertical over diagonal).

**Comparison to alternatives:**
- **Prewitt:** similar to Sobel but different weighting.
- **Canny edge detector:** more sophisticated (non-maximum suppression, hysteresis thresholding).
- **Laplacian:** second derivative, sensitive to noise but produces thinner edges.

---

### Q53. What is the Canny edge detector and why is it superior to simple Sobel?

**A:** Canny is an optimal edge detector with sophisticated post-processing (non-maximum suppression, hysteresis thresholding), producing thin, well-localized edges.

**Canny pipeline:**

1. **Gaussian blur:** reduce noise.
2. **Gradient computation:** Sobel or other edge operators.
3. **Non-maximum suppression:** thin edges.
   - For each pixel, check if it's a local maximum along gradient direction.
   - Keep only local maxima, suppressing thick edges.
4. **Hysteresis thresholding:**
   - Two thresholds: $T_{high}$ and $T_{low}$ (e.g., 100, 50).
   - Pixels with gradient > $T_{high}$: strong edges (keep).
   - Pixels with gradient < $T_{low}$: not edges (discard).
   - Pixels in between: edges if connected to strong edges (connectivity).

**Why hysteresis helps:**
- Avoids breaking edges due to noise (use lower threshold for connected weak pixels).
- Removes disconnected weak edges (connected component check).

**Advantages over Sobel:**
1. **Thin edges:** non-maximum suppression produces single-pixel-wide edges.
2. **Connected edges:** hysteresis connects nearby edge segments.
3. **Fewer false positives:** strict criteria for edge inclusion.

**Expert perspective:**
Canny is optimal in sense of:
1. **Maximizing detection:** minimize false negatives.
2. **Maximizing localization:** edges precisely located.
3. **Minimizing response:** few spurious edges.

Hyperparameter tuning (threshold values) is crucial for good results.

---

### Q54. What is image thresholding and what are threshold selection methods?

**A:** Thresholding converts grayscale image to binary (black/white) by comparing pixel intensity to threshold. It is fundamental for image segmentation and preprocessing.

**Simple thresholding:**
$$
B(x, y) = \begin{cases} 255 & \text{if } I(x, y) > T \\ 0 & \text{otherwise} \end{cases}
$$

**Threshold selection methods:**

1. **Manual/fixed:** choose threshold based on domain knowledge.
   - Pros: simple, interpretable.
   - Cons: doesn't adapt to image variations.

2. **Otsu's method (automatic):**
   - Find threshold that minimizes within-class variance.
   - Assumes bimodal histogram (two clear peaks: foreground and background).
   - Widely used, no parameters.
   $$
   \sigma_w^2(t) = w_0(t) \sigma_0^2(t) + w_1(t) \sigma_1^2(t)
   $$
   Minimize over all thresholds $t$.

3. **Adaptive thresholding:**
   - Different threshold for each region (neighborhood).
   - Handles illumination variations.
   $$
   T(x, y) = \text{mean}(\text{neighborhood}) - C
   $$
   where $C$ is constant offset.

4. **Multi-level thresholding:**
   - Use multiple thresholds to create multiple classes.
   - Extension of Otsu.

**Challenges:**
- Assumes distinct foreground/background (fails for gradual transitions).
- Sensitive to illumination.
- Noise can create spurious regions.

**Expert view:**
Thresholding is simple but often effective. In practice:
1. Combine with preprocessing (blur, histogram equalization) for robustness.
2. Use adaptive thresholding for non-uniform lighting.
3. Post-process with morphology to clean up.

---

### Q55. What are morphological operations and how do they process binary images?

**A:** Morphological operations (erosion, dilation, opening, closing) process binary images using structuring elements, enabling shape-based filtering and noise removal.

**Basic operations:**

1. **Erosion:**
   - Output pixel = 1 if all pixels in structuring element neighborhood are 1.
   - Shrinks white regions, removes small objects.
   $$
   E(x, y) = \min_{(i,j) \in SE} I(x+i, y+j)
   $$

2. **Dilation:**
   - Output pixel = 1 if any pixel in structuring element neighborhood is 1.
   - Expands white regions, fills holes.
   $$
   D(x, y) = \max_{(i,j) \in SE} I(x+i, y+j)
   $$

3. **Opening:** Erosion followed by dilation.
   - Removes small objects, smooths boundaries.
   $$
   \text{Open} = D(E(I))
   $$

4. **Closing:** Dilation followed by erosion.
   - Fills small holes, smooths boundaries.
   $$
   \text{Close} = E(D(I))
   $$

**Structuring elements:**
- Disk, square, line shapes.
- Size determines extent of operation.

**Applications:**
1. **Noise removal:** opening removes small noise blobs.
2. **Hole filling:** closing fills small holes.
3. **Boundary smoothing:** alternating open/close.
4. **Shape analysis:** erosion to remove thin structures.

**Expert perspective:**
Morphological operations are shape-aware filtering:
- Unlike linear filters (blur, gradient), morphology preserves sharp boundaries.
- Sequence of operations (e.g., close-then-open) achieves sophisticated effects.
- Generalizations: gradient (dilation - erosion), skeleton (medial axis).

---

### Q56. What is the Hough transform and how does it detect lines and circles?

**A:** The Hough transform detects geometric shapes (lines, circles) by voting in parameter space. It is robust to occlusion and noise.

**Line Hough transform:**
- **Image space:** $(x, y)$ coordinates.
- **Parameter space:** line parameters $(r, \theta)$ or $(m, c)$.

**Polar line equation:**
$$
r = x \cos\theta + y \sin\theta
$$

**Algorithm:**
1. For each edge point $(x, y)$, compute $(r, \theta)$ for all possible lines through it.
2. Increment a 2D accumulator array at $(\theta, r)$ for each computed value.
3. Peaks in accumulator array correspond to lines in image.

**Circle detection:**
- **Parameter space:** $(c_x, c_y, r)$ (3D).
- For each edge point, accumulate votes for all circles passing through it.
- More expensive (3D array), but powerful.

**Why it works:**
- All points on same line accumulate votes at same $(\theta, r)$.
- Occlusion: points are individual voters, missing some doesn't eliminate line.
- Noise: outliers don't get enough votes to form peaks.

**Advantages:**
- Robust to occlusion, noise, broken edges.
- Detects multiple shapes in single pass.
- Elegant formulation.

**Disadvantages:**
- Computationally expensive (2D or 3D accumulation).
- Requires preprocessing (edge detection).
- Sensitive to accumulator resolution (discretization).

**Expert view:**
Hough transform is classical but still used (autonomous driving: lane detection). Generalizations: arbitrary shapes, 3D (Hough sphere), probabilistic variants.

---

### Q57. What is histogram equalization and what problem does it solve?

**A:** Histogram equalization stretches pixel intensity distribution to use full dynamic range, improving contrast and visibility of details.

**Problem:**
Images may have limited tonal range (e.g., mostly dark), wasting available intensity levels.

**Algorithm:**
1. Compute histogram $h(i)$ (count of pixels at intensity $i$).
2. Compute cumulative histogram: $H(i) = \sum_{j=0}^{i} h(j)$.
3. Normalize: $H'(i) = H(i) / N$ (where $N$ is image size).
4. Map each pixel: $I'(x, y) = \text{round}(255 \cdot H'(I(x, y)))$.

**Result:**
Flattened histogram, utilizing full range [0, 255].

**Adaptive histogram equalization (AHE):**
- Apply equalization to local regions instead of globally.
- Preserves local contrast without excessive brightening.
- Useful for medical imaging.

**Disadvantages:**
- Can over-amplify noise (uniformly distributed across range).
- May produce unnatural colors in color images.
- Sensitive to outliers (single very bright/dark pixel stretches entire range).

**Expert view:**
Histogram equalization is preprocessing tool:
- Simple, fast, effective for low-contrast images.
- Use CLAHE (Contrast Limited AHE) in practice for better results.
- Modern approach: deep learning for contrast enhancement (more flexible).

---

### Q58. What are image histograms and how are they used in computer vision?

**A:** Histograms count pixel intensities (or color channels), revealing image properties and enabling histogram-based processing.

**Applications:**

1. **Image analysis:**
   - Histogram shape reveals distribution: bimodal (two objects), uniform (noise).
   - Detect over-exposure, under-exposure, low contrast.

2. **Image segmentation:**
   - Bimodal histogram → Otsu thresholding.
   - Multi-modal → multiple thresholds.

3. **Histogram matching:**
   - Given target histogram, adjust image to match.
   - Enables style transfer, color correction.

4. **Image quality assessment:**
   - Histogram distance (chi-square, Bhattacharyya) compares images.

5. **Color-based segmentation:**
   - 3D histogram in color space (R, G, B).
   - Find dominant colors.

**Histogram distance metrics:**
- **Chi-square:** $\chi^2 = \sum_i \frac{(h_1(i) - h_2(i))^2}{h_1(i) + h_2(i)}$.
- **Bhattacharyya:** $D = -\ln \sum_i \sqrt{h_1(i) h_2(i)}$.
- **Hellinger:** $D = \sqrt{1 - \sum_i \sqrt{h_1(i) h_2(i)}}$.

**Limitations:**
- Loses spatial information (histogram is global).
- Two very different images can have same histogram.

**Expert perspective:**
Histograms are fast, interpretable, but low-level. Often combined with spatial information (color + texture) for robustness.

---

### Q59. What is the Harris corner detector and why are corner features useful?

**A:** Harris corner detector finds corner points (high curvature, distinctive landmarks) in images using local image derivatives.

**Intuition:**
- Edges have high gradient in one direction, low in perpendicular.
- Corners have high gradient in multiple directions.

**Algorithm:**
1. Compute image derivatives $I_x, I_y$.
2. Compute structure tensor (second moment matrix):
   $$
   M = \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}
   $$
   (average over neighborhood).

3. Compute Harris response:
   $$
   R = \det(M) - \alpha \cdot \text{trace}(M)^2 = \lambda_1 \lambda_2 - \alpha(\lambda_1 + \lambda_2)^2
   $$
   where $\lambda_1, \lambda_2$ are eigenvalues of $M$.

4. Peaks in $R$ are corner locations.

**Interpretation:**
- $\lambda_1, \lambda_2$ small: flat region (no edges).
- $\lambda_1$ large, $\lambda_2$ small: edge (gradient in one direction).
- $\lambda_1, \lambda_2$ both large: corner (high curvature).

**Advantages:**
- Fast, efficient.
- Rotation invariant.
- Well-established.

**Disadvantages:**
- Scale-dependent (corners at different scales may not be detected).
- Sensitive to parameter $\alpha$.

**Comparison to other detectors:**
- **FAST:** faster, less accurate.
- **SIFT:** multi-scale, invariant to rotation and scale.

---

### Q60. What is SIFT (Scale-Invariant Feature Transform) and why is it powerful?

**A:** SIFT detects and describes keypoints that are invariant to scale, rotation, and illumination changes. It is among the most influential feature descriptors in CV.

**SIFT pipeline:**

1. **Scale-space construction:**
   - Build Gaussian pyramids (multi-scale representations).
   - Difference of Gaussians (DoG) at multiple scales for keypoint detection.

2. **Keypoint localization:**
   - Find local extrema in DoG space (3D peaks).
   - Refine to sub-pixel accuracy.

3. **Orientation assignment:**
   - Compute dominant gradient orientation at each keypoint.
   - Use this orientation for rotation invariance.

4. **Descriptor computation:**
   - 4×4 grid of orientation histograms (8 bins each).
   - 128-dimensional descriptor.
   - Rotation invariant (relative to dominant orientation).

**Why SIFT works:**

1. **Scale invariance:** multi-scale detection via DoG.
2. **Rotation invariance:** orientation normalization.
3. **Illumination robustness:** gradient-based (invariant to additive lighting).
4. **Distinctiveness:** 128D descriptor discriminative enough for reliable matching.

**Applications:**
- Image registration, panoramic stitching.
- 3D reconstruction.
- Object recognition.
- Image retrieval.

**Disadvantages:**
- Computationally expensive (multi-scale pyramid).
- Patented (royalty issues), motivating alternatives (SURF, ORB).

**Expert perspective:**
SIFT is foundational. Even with deep learning, SIFT principles (scale-space, multi-scale) remain relevant. Modern learned features (SuperPoint) follow similar ideas.

---

### Q61. What is feature matching and correspondence and how is it done robustly?

**A:** Feature matching finds correspondence between keypoints in two images. It is critical for 3D reconstruction, image registration, and object recognition.

**Matching pipeline:**

1. **Detect keypoints:** SIFT, SURF, ORB in both images.
2. **Compute descriptors:** feature vectors for each keypoint.
3. **Match descriptors:** find pairs with similar descriptors.
4. **Verify correspondences:** remove outliers via geometric constraints.

**Matching metrics:**
- **Euclidean distance:** $L_2$ norm (for real-valued descriptors).
- **Hamming distance:** XOR count (for binary descriptors, fast).

**Robust matching (handle outliers):**

1. **Ratio test (Lowe's criterion):**
   - Compute distances to first and second nearest neighbors.
   - Keep match only if $\frac{d_1}{d_2} < \alpha$ (typically 0.7).
   - Rejects ambiguous matches.

2. **RANSAC (Random Sample Consensus):**
   - Sample subset of matches.
   - Fit geometric model (homography, fundamental matrix).
   - Count inliers (matches consistent with model).
   - Repeat, keep model with most inliers.
   - Robust to outliers (even if 50% are outliers, RANSAC finds inliers).

3. **Bundle adjustment:**
   - Jointly refine all matched points and camera poses.
   - Minimizes reprojection error globally.

**Challenges:**
- Ambiguous matches (multiple candidates similar distance).
- Occlusion (feature not visible in second image).
- Scale/rotation changes.

**Expert perspective:**
Robust matching is crucial for real-world applications. RANSAC is standard, but modern approaches combine deep learning (learned features) + geometric verification.

---

### Q62. What is image warping and what are common geometric transformations?

**A:** Image warping applies geometric transformations (rotation, scaling, perspective) to images. It is essential for image registration, rectification, and content-based image manipulation.

**Common transformations:**

1. **Affine transformation:**
   - 2×3 matrix, preserves parallel lines.
   $$
   \begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a & b & c \\ d & e & f \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
   $$
   - Includes: rotation, scaling, translation, skewing.
   - Requires 3 corresponding points to estimate.

2. **Perspective (homography):**
   - 3×3 matrix, handles 3D viewpoint changes.
   $$
   \begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
   $$
   - Requires 4 corresponding points.

3. **Similarity:** translation + rotation + uniform scaling (preserves angles).
4. **Rigid:** translation + rotation (no scaling).

**Interpolation during warping:**
- Forward mapping: for each destination pixel, where did it come from? (prone to holes)
- Backward mapping: for each destination pixel, look back at source. (preferred)
- Use bilinear/bicubic interpolation at non-integer source locations.

**Applications:**
- Image registration (align two images).
- Panoramic stitching.
- Document scanning (perspective correction).
- Face alignment (warp to canonical pose).

**Expert view:**
Warping is fundamental to image geometry. Key insight: backward mapping with interpolation avoids holes and aliasing.

---

### Q63. What is image convolution and correlation, and what is the difference?

**A:** Convolution and correlation are similar operations but differ in kernel orientation, affecting symmetry properties.

**Convolution:**
$$
(f * g)(x, y) = \sum_{i,j} f(i, j) \cdot g(x-i, y-j)
$$
Note: kernel $g$ is flipped (negative indices).

**Correlation:**
$$
(f \star g)(x, y) = \sum_{i,j} f(i, j) \cdot g(x+i, y+j)
$$
Note: kernel $g$ is not flipped.

**Key difference:**
- **Convolution:** commutative ($f * g = g * f$), associative, has algebraic properties.
- **Correlation:** measures similarity, not commutative.

**In practice:**
- For symmetric kernels (Gaussian, Laplacian), convolution = correlation.
- For asymmetric kernels (directional), they differ.
- Deep learning uses correlation (no flip), despite calling it convolution (misnomer).

**Applications:**
- **Convolution:** filtering theory, signal processing, PDEs.
- **Correlation:** template matching, feature similarity, cross-correlation.

**Template matching via correlation:**
Find location of template $T$ in image $I$:
$$
C(x, y) = \sum_{i,j} I(x+i, y+j) \cdot T(i, j)
$$
Peak at $(x, y)$ indicates template location.

**Expert perspective:**
Understanding convolution vs. correlation is fundamental to signal processing. Most CV practitioners conflate them, but understanding the distinction is important for theoretical depth.

---

### Q64. What is the Laplacian operator and what is the Laplacian of Gaussian (LoG)?

**A:** The Laplacian is a second-derivative operator detecting edges and blobs. LoG combines Laplacian with Gaussian smoothing for robust multi-scale blob detection.

**Laplacian operator:**
$$
\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}
$$

**2D Laplacian kernel:**
$$
L = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}
$$

**Properties:**
- High response at edges and blobs (rapid intensity changes).
- Zero-crossing (sign change) at precise edge locations.
- Sensitive to noise (amplifies high-frequency).

**Laplacian of Gaussian (LoG):**
$$
\nabla^2 G(x, y, \sigma) = -\frac{1}{\pi \sigma^4} (1 - \frac{x^2 + y^2}{2\sigma^2}) \exp(-\frac{x^2 + y^2}{2\sigma^2})
$$

First smooth with Gaussian, then apply Laplacian.

**Advantages:**
1. **Smoothing first:** reduces noise.
2. **Multi-scale:** different $\sigma$ values detect blobs at different scales.
3. **Zero-crossings:** precise edge localization.

**Applications:**
- Blob detection (SIFT uses DoG, which approximates LoG).
- Edge detection (Marr-Hildreth edge detector).
- Texture analysis.

**Relation to DoG (Difference of Gaussians):**
DoG approximates LoG:
$$
\nabla^2 G \approx \frac{G(x, y, \sigma_1) - G(x, y, \sigma_2)}{\sigma_1 - \sigma_2}
$$

SIFT uses DoG for efficiency (only Gaussian smoothing, no Laplacian).

---

### Q65. What is image binarization and what challenges arise?

**A:** Binarization converts grayscale to binary (0 or 255), used for preprocessing, object isolation, and shape analysis.

**Challenges:**

1. **Threshold selection:** choosing right value.
2. **Illumination variation:** uneven lighting requires adaptive approach.
3. **Noise:** salt-and-pepper noise creates spurious pixels.
4. **Object touching:** objects merge in binary image, hard to separate.
5. **Thin structures:** thin lines may disappear after thresholding.

**Solutions:**

1. **Preprocessing:**
   - Histogram equalization improves contrast.
   - Gaussian blur reduces noise.

2. **Post-processing:**
   - Morphology (opening, closing) cleans up.
   - Connected component analysis removes small blobs.

3. **Adaptive thresholding:** local threshold accounts for illumination.

4. **Multi-level thresholding:** multiple thresholds for complex scenes.

**Expert perspective:**
Binary image processing is fundamental to document analysis (OCR), industrial inspection. Simple but effective when combined with good preprocessing/postprocessing.

---

### Q66. What is connected component analysis and how is it used?

**A:** Connected component analysis (CCA) identifies connected regions of pixels in binary images, labeling them uniquely.

**Algorithm (two-pass):**
1. **First pass:** scan image, assign provisional labels to 1-pixels.
2. **Union-find:** merge labels of adjacent pixels.
3. **Second pass:** relabel with final labels.

**Connectivity:**
- **4-connectivity:** pixel connected to 4 neighbors (up, down, left, right).
- **8-connectivity:** pixel connected to 8 neighbors (including diagonals).

**Output:**
- Label image where each connected region has unique ID.
- Statistics per region: size, centroid, bounding box, moments.

**Applications:**
1. **Noise removal:** small connected components are noise, remove them.
2. **Object separation:** disconnected objects get different labels.
3. **Shape analysis:** compute statistics of connected regions.
4. **Document analysis:** separate characters/words.

**Computational complexity:**
- Single pass: $O(n)$ where $n$ is number of pixels.
- Very efficient.

**Expert view:**
CCA is fast, simple, foundational to binary image processing. Combined with thresholding and morphology, solves many real-world problems.

---

### Q67. What are image moments and what information do they capture?

**A:** Moments are statistical properties of pixel distributions, capturing shape information (area, centroid, orientation, roundness).

**Moments:**
$$
m_{ij} = \sum_{x,y} x^i y^j I(x, y)
$$

**Central moments** (relative to centroid):
$$
\mu_{ij} = \sum_{x,y} (x - \bar{x})^i (y - \bar{y})^j I(x, y)
$$

**Common moments:**

1. **$m_{00}$:** total intensity (object area).
2. **$m_{10}, m_{01}$:** first moments (compute centroid).
3. **$\mu_{20}, \mu_{02}, \mu_{11}$:** second moments (covariance matrix, orientation, scale).

**Shape descriptors from moments:**

1. **Centroid:** $(\bar{x}, \bar{y}) = (m_{10}/m_{00}, m_{01}/m_{00})$.
2. **Orientation:** $\theta = 0.5 \arctan(2\mu_{11} / (\mu_{20} - \mu_{02}))$.
3. **Eccentricity/aspect ratio:** $e = \sqrt{1 - \lambda_{min} / \lambda_{max}}$ where $\lambda$ are eigenvalues of covariance.

**Hu moments:** invariant to translation, scale, and rotation.
$$
M_i = f(m_{ij})
$$

Seven Hu moments are rotation/scale/translation invariant, useful for shape matching.

**Applications:**
- Shape classification.
- Object centroid and orientation.
- Deformable shape analysis.

**Expert perspective:**
Moments are interpretable, fast, but low-level. Modern deep learning replaces hand-crafted moments with learned features.

---

### Q68. What is color space and why are different color spaces used?

**A:** Color spaces represent color as combination of channels. Different spaces suit different tasks.

**Common color spaces:**

1. **RGB:**
   - Red, Green, Blue (monitor native).
   - Intuitive for display.
   - High correlation between channels (not efficient for compression).

2. **HSV:**
   - Hue (color), Saturation (purity), Value (brightness).
   - Intuitive for color-based segmentation.
   - Hue invariant to lighting changes (robust).

3. **YCbCr:**
   - Y (luminance), Cb, Cr (chrominance).
   - Used in video/JPEG (humans sensitive to luminance, less sensitive to chroma).
   - Efficient for compression.

4. **LAB:**
   - L (luminance), A, B (color opponent axes).
   - Perceptually uniform (Euclidean distance ≈ perceived difference).
   - Used in color correction, image quality assessment.

5. **Grayscale:**
   - Single channel (luminance).
   - Efficient, works for many algorithms.

**Color space conversion:**
- **RGB to Grayscale:** $Gray = 0.299R + 0.587G + 0.114B$ (weighted by human sensitivity).
- **RGB to HSV:** complex nonlinear transformation.
- **RGB to YCbCr:** linear transformation.

**When to use each:**
- **RGB:** display, general purpose.
- **HSV:** color-based segmentation.
- **YCbCr:** video, compression.
- **LAB:** perceptually meaningful operations.
- **Grayscale:** efficiency, robustness to illumination.

**Expert perspective:**
Color space choice can significantly impact algorithm robustness. HSV for color, LAB for perceptual operations, YCbCr for video.

---

### Q69. What is color quantization and what is its purpose?

**A:** Color quantization reduces the number of colors in an image, compressing it while maintaining visual quality.

**Methods:**

1. **Uniform quantization:**
   - Divide color space into equal-sized bins.
   - Simple, fast.
   - May not match actual color distribution (wastes bins).

2. **K-means clustering:**
   - Cluster pixels into $k$ colors.
   - Optimal in least-squares sense.
   - Iterative: assign pixels to nearest centroid, update centroids.

3. **Octree quantization:**
   - Build octree of color space.
   - Prune leaves if too many nodes.
   - Efficient, hierarchical.

4. **Median cut:**
   - Recursively partition color space by cutting at median.
   - Produces balanced tree.
   - Fast, good results.

**Dithering:**
- After quantization, neighboring pixels differ more (banding).
- Dithering adds patterns to simulate intermediate colors.
- Options: ordered dithering (Bayer), error diffusion (Floyd-Steinberg).

**Applications:**
- GIF creation (256-color limit).
- Embedded systems (limited memory).
- Artistic effects.

**Expert perspective:**
Color quantization is simple but effective. K-means provides good quality. Modern approach: use deep learning for image compression (neural codecs).

---

### Q70. What is the Fourier transform and how does it help in image processing?

**A:** The Fourier transform converts image from spatial domain to frequency domain, enabling frequency-based analysis and filtering.

**2D Fourier transform:**
$$
F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) e^{-j2\pi(ux/M + vy/N)}
$$

**Interpretation:**
- Low frequencies: smooth variations (overall shape, lighting).
- High frequencies: rapid changes (edges, textures, noise).

**Frequency-domain filtering:**

1. **Compute FFT:** $F(u, v) = \text{FFT}(f(x, y))$.
2. **Apply filter:** $G(u, v) = H(u, v) \cdot F(u, v)$ (multiplication in frequency domain).
3. **Inverse FFT:** $g(x, y) = \text{IFFT}(G(u, v))$.

**Efficiency:**
- Spatial convolution: $O(M N K^2)$ for $M \times N$ image, $K \times K$ kernel.
- Frequency filtering: $O(MN \log(MN))$ for FFT + multiplication.
- Fast for large kernels.

**Applications:**
1. **Lowpass filter:** remove high-frequency noise.
2. **Highpass filter:** enhance edges.
3. **Bandpass:** extract specific frequency ranges (texture analysis).
4. **Homomorphic filtering:** separate illumination from reflectance.

**Visualization:**
- Magnitude spectrum: $|F(u, v)|$ (brightness).
- Phase spectrum: $\arg(F(u, v))$ (color).
- Log magnitude useful (wide dynamic range).

**Expert perspective:**
Frequency domain is powerful for understanding image properties and designing filters. However, local operations (spatial domain) often more intuitive. Use frequency domain for theoretical understanding, spatial for implementation.

---

### Q71. What is the power spectrum and what does it reveal about images?

**A:** Power spectrum is squared magnitude of Fourier transform, revealing energy distribution across frequencies. It reveals image characteristics.

**Power spectrum:**
$$
P(u, v) = |F(u, v)|^2
$$

**Interpretation:**

1. **DC component (center):** $P(0, 0)$ is average image intensity.
2. **Radial distribution:** decreases with frequency (natural images have more low-frequency energy).
3. **Directional patterns:** anisotropy indicates dominant orientations (textures).

**Image properties:**

1. **Smooth image:** energy concentrated at low frequencies.
2. **Textured image:** significant high-frequency energy.
3. **Edges:** peaks in radial frequency (edge sharpness).

**Texture analysis via power spectrum:**
- Compute power spectrum of texture region.
- Frequency signature (which bands have high energy) characterizes texture.
- Enables texture classification and synthesis.

**Natural image statistics:**
- Average power spectrum of natural images follows $1/f$ law (energy ~ $1/\text{frequency}$).
- Fundamental property, observed across datasets.
- Useful for image quality assessment, compression.

**Expert perspective:**
Power spectrum provides compact, interpretable image descriptor. Combined with orientation (Fourier analysis in multiple directions), characterizes texture and image properties.

---

### Q72. What is template matching and what are its challenges?

**A:** Template matching finds regions in image most similar to template, used for object detection, localization, and tracking.

**Algorithm:**
1. Slide template over image.
2. Compute similarity at each location (correlation, sum of squared differences).
3. Find peaks (highest correlation = best matches).

**Similarity metrics:**

1. **Cross-correlation:**
   $$
   CC(x, y) = \sum_{i,j} I(x+i, y+j) \cdot T(i, j)
   $$

2. **Sum of squared differences:**
   $$
   SSD(x, y) = \sum_{i,j} (I(x+i, y+j) - T(i, j))^2
   $$

3. **Normalized cross-correlation:**
   $$
   NCC(x, y) = \frac{\sum_{i,j} (I(x+i, y+j) - \bar{I})(T(i, j) - \bar{T})}{\sqrt{\sum (I - \bar{I})^2 \sum (T - \bar{T})^2}}
   $$
   Invariant to illumination changes.

**Challenges:**

1. **Scale variation:** template fixed size, objects vary.
   - Solution: multi-scale matching (template pyramid).

2. **Rotation:** template not rotation invariant.
   - Solution: rotate template multiple angles.

3. **Illumination:** intensity changes affect correlation.
   - Solution: normalized cross-correlation.

4. **Occlusion:** template partially hidden.
   - Solution: accept matches above threshold, not just peak.

5. **Computational cost:** slide template is $O(MN)$ for image search.
   - Solution: FFT-based for large kernels.

**Modern alternative:** use learned features + matching instead of raw pixel correlation.

---

### Q73. What is optical character recognition (OCR) and what are the main stages?

**A:** OCR converts scanned documents/images to machine-readable text. It combines image processing, feature extraction, and pattern recognition.

**OCR pipeline:**

1. **Image preprocessing:**
   - Binarization: convert to black/white.
   - Skew correction: straighten tilted text.
   - Noise removal: morphology, median filter.

2. **Text localization:**
   - Find text regions (connected components, text detection networks).
   - Separate lines and words.

3. **Character segmentation:**
   - Divide lines into individual characters.
   - Challenging: touching characters, variable spacing.

4. **Character recognition:**
   - Classify each character.
   - Traditional: template matching, feature-based classifiers (SVM, HMM).
   - Modern: deep learning (CNN).

5. **Post-processing:**
   - Language models: correct obviously wrong characters.
   - Spell checking.

**Challenges:**

1. **Variable fonts:** different typefaces, sizes, styles.
2. **Degraded documents:** noise, blur, low contrast.
3. **Handwriting:** high variability, cursive connecting characters.
4. **Layout:** complex documents with tables, columns.

**Modern approach:**
End-to-end deep learning:
- CNN extract features from image.
- RNN (LSTM) predicts sequence of characters.
- Attention mechanisms align recognition to text regions.

**Expert perspective:**
OCR is mature task (>99% accuracy for clean printed text). Challenges remain for handwriting, degraded documents, complex layouts.

---

### Q74. What is image registration and what methods are used?

**A:** Image registration aligns two or more images to same coordinate system. Crucial for change detection, multi-modal fusion, image stitching.

**Applications:**
- Medical imaging: compare scans before/after treatment.
- Satellite: detect changes over time.
- Panoramic stitching: align overlapping images.

**Registration types:**

1. **Rigid (rotation + translation):**
   - Preserves distances.
   - Suitable for small misalignments.

2. **Affine (rotation + scaling + translation + skew):**
   - More flexible.
   - 6 DOF (degrees of freedom).

3. **Deformable (non-rigid):**
   - Warps image to align with target.
   - High DOF, powerful but requires good initialization.

**Methods:**

1. **Feature-based:**
   - Detect features in both images (SIFT, Harris).
   - Match features.
   - Estimate transformation (RANSAC).
   - Pros: fast, robust to occlusion.
   - Cons: needs distinctive features.

2. **Intensity-based (direct):**
   - Minimize photometric error (difference between images).
   - Optimization: gradient descent.
   - Pros: uses all pixels, accurate.
   - Cons: requires good initialization, slow.

3. **Phase correlation:**
   - Fourier-based, very fast.
   - Finds translation.
   - Suitable for global alignment.

**Similarity metrics:**
- **Sum of squared differences (SSD):** $\sum (I_1 - I_2)^2$.
- **Cross-correlation:** $\sum I_1 I_2$.
- **Mutual information:** information shared between images (robust to intensity changes).

**Expert perspective:**
Registration is challenging in practice:
- Multi-modal fusion (CT + MRI) requires normalized mutual information.
- Large deformations need hierarchical approach (coarse-to-fine).
- Modern: combine feature-based + deep learning for robustness.

---

### Q75. What is the pyramid data structure in image processing?

**A:** Image pyramid represents image at multiple scales, fundamental for multi-scale algorithms (blur, edge detection, feature detection, object detection).

**Types:**

1. **Gaussian pyramid:**
   - Successively blur and downsample image.
   - Each level is $1/4$ area of previous (if downsample by 2 in each dimension).
   - Smooth, reduces noise.

2. **Laplacian pyramid:**
   - Difference between Gaussian pyramid levels.
   - Captures edge information at each scale.
   - Useful for: image reconstruction, edge detection, multi-scale analysis.

3. **Image pyramid construction:**
   - Level 0: original image.
   - Level $i$: blur level $i-1$ with Gaussian, downsample by 2.
   - Typically 4-5 levels before image becomes tiny.

**Applications:**

1. **SIFT:** multi-scale feature detection via Gaussian/Laplacian pyramid.
2. **Panoramic stitching:** match at coarse level first, refine at fine levels.
3. **Image blending:** seamless blending at multiple scales.
4. **Object detection:** detect objects at different scales.

**Laplacian pyramid for image blending:**
- Build Laplacian pyramids of both images.
- Blend at each level.
- Reconstruct from blended pyramid.
- Result: seamless blending (smooth transitions, no artifacts).

**Expert perspective:**
Coarse-to-fine via pyramids is fundamental optimization strategy:
- Fast coarse matching.
- Refined fine-level details.
- Robust initialization (coarse level finds rough solution).

Modern deep learning incorporates multi-scale features (FPN: Feature Pyramid Networks).

---

### Q76. What is seeded region growing and how is it used for segmentation?

**A:** Seeded region growing segments image by iteratively growing regions from seed points, useful for object segmentation with manual or automatic seeding.

**Algorithm:**

1. **Initialize:** select seed points (manual or automatic).
2. **Grow:** iteratively add neighboring pixels if similarity to region exceeds threshold.
3. **Stop:** region stops growing when no more neighbors meet criterion.

**Similarity criteria:**
- **Intensity:** pixel intensity within range of region mean.
- **Color:** RGB distance within threshold.
- **Texture:** local texture matches region texture.

**Advantages:**
1. **Interactive:** user can place seeds, guides segmentation.
2. **Fast:** linear in image size.
3. **Intuitive:** similar concept to paint bucket tool.

**Disadvantages:**
1. **Seed-dependent:** results depend on seed placement.
2. **Leakage:** grows across weak boundaries.
3. **Manual seeding:** requires user intervention.

**Improvements:**
1. **Morphological reconstruction:** prevents leakage via control markers.
2. **Adaptive threshold:** region-specific thresholds.
3. **Multi-seed:** automatic seed detection (local minima).

**Expert perspective:**
Region growing is simple but effective for specific objects with clear boundaries. Combined with preprocessing (thresholding) and postprocessing (morphology), handles many real-world cases.

---

### Q77. What is the watershed algorithm and how does it separate touching objects?

**A:** Watershed treats image as topographic surface, finding watershed lines that separate regions. It is powerful for separating touching objects.

**Algorithm:**
1. **Gradient:** compute magnitude of image gradient (uphill = high gradient).
2. **Find minima:** local intensity minima are markers.
3. **Flood:** progressively raise water level, starting from each minimum.
4. **Watershed:** where water from two minima meet is watershed line.
5. **Result:** regions separated by watershed lines.

**Intuition:**
Imagine topographic map. Water flows downhill, collects at valleys (minima). Watershed is the ridge where water from different valleys would meet.

**Marker-controlled watershed:**
- Manually or automatically place markers (seed regions).
- Enforce only marked minima, preventing over-segmentation.
- More control than unsupervised watershed.

**Advantages:**
1. **Separates touching objects:** watershed lines naturally separate objects.
2. **Automatic:** no parameter tuning (compared to threshold).
3. **Connected regions:** guaranteed connected components.

**Disadvantages:**
1. **Over-segmentation:** many small regions if image has noise.
   - Solution: marker control.
2. **Gradient computation:** sensitive to noise.
   - Solution: preprocessing (morphological smoothing).

**Applications:**
- Cell segmentation (biology).
- Coin/object counting.
- Document image segmentation.

**Expert perspective:**
Watershed is classical but still used (especially with markers). Deep learning alternatives: semantic segmentation, instance segmentation. But watershed's interpretability and speed remain valuable for some applications.

---

### Q78. What is the distance transform and what is it used for?

**A:** Distance transform computes distance from each pixel to nearest non-zero pixel, useful for shape analysis, thinning, and geometrical operations.

**Definition:**
$$
D(x, y) = \min_{(x', y') \in \text{non-zero}} \text{distance}((x, y), (x', y'))
$$

**Distance metrics:**
1. **Euclidean:** $\sqrt{(x-x')^2 + (y-y')^2}$ (true distance).
2. **Manhattan (L1):** $|x-x'| + |y-y'|$ (fast, chess-king distance).
3. **Chebyshev (L∞):** $\max(|x-x'|, |y-y'|)$ (fast).

**Efficient computation:**
- Chamfer distance: approximate with separable passes.
- Multiple passes over image: $O(n)$ for $n$ pixels.

**Applications:**

1. **Medial axis (skeleton):** local maxima of distance transform.
   - Thinned representation of shape.
   - Used for shape matching, recognition.

2. **Morphological operations:** erosion = distance < radius, dilation = distance < radius from complement.

3. **Shape analysis:** distance values reveal object thickness.

4. **Generalized cylinder representation:** axis + radius at each point.

**Expert perspective:**
Distance transform is fundamental geometric operator. Combined with other tools (morphology, connected components), solves many shape analysis problems.

---

### Q79. What is image thinning and how is it different from erosion?

**A:** Thinning extracts skeleton (medial axis) of objects, reducing them to single-pixel-wide structures while preserving connectivity.

**Erosion vs. Thinning:**
- **Erosion:** shrinks object, removes small structures, stops when object disappears.
- **Thinning:** iteratively removes boundary pixels while preserving topology, continues until single-pixel-wide skeleton remains.

**Thinning algorithm:**
1. Repeat:
   - Remove boundary pixels satisfying conditions (don't break connectivity).
   - Conditions: pixel is background neighbor to multiple foreground components, OR pixel is not endpoint of line.
2. Until no more pixels can be removed.

**Connectivity preservation:**
- 8-connectivity: 8 neighbors.
- Check: after removing pixel, connectivity should remain (no isolated components created).

**Applications:**
1. **Character recognition:** thinned characters are scale-invariant, simpler to recognize.
2. **Fingerprint analysis:** ridge thinning for minutiae extraction.
3. **Vascular segmentation:** vessel networks represented as lines.
4. **Shape analysis:** skeleton captures shape structure.

**Related: Medial axis transform:**
- Thicker version: radius at each skeleton point.
- Represents shape as axis + thickness.
- Invertible (can reconstruct original shape).

**Expert perspective:**
Thinning is important preprocessing for line-based feature extraction. Modern approaches use distance transform (local maxima = skeleton) as alternative to iterative thinning.

---

### Q80. What is histogram backprojection and how is it used for color-based tracking?

**A:** Histogram backprojection maps each pixel to probability of belonging to target color distribution, enabling robust color-based tracking and segmentation.

**Algorithm:**

1. **Target histogram:** compute histogram of target object (or region) in HSV color space.
   - More robust than RGB (hue invariant to brightness).

2. **Backprojection:** for each image pixel, look up histogram bin, assign probability:
   $$
   \text{BP}(x, y) = \text{histogram}[HSV(x, y)]
   $$

3. **Result:** probability map where high values indicate target color likelihood.

4. **Thresholding:** threshold probability map to get binary mask of target.

**Why it works:**
- Histogram captures color distribution of target.
- Backprojection answers: "is this pixel similar color to target?"
- Robust to illumination changes (hue-based).

**Tracking via backprojection:**
1. Initialize: target bounding box in first frame.
2. Each frame:
   - Compute backprojection in search region.
   - Find centroid of high-probability region.
   - Update target position.

**Advantages:**
1. **Fast:** single-pass algorithm.
2. **Robust to scale change:** size adapts via region size.
3. **Multiple targets:** backprojection for each target.

**Disadvantages:**
1. **Similar colored objects:** can confuse targets.
2. **Background clutter:** if background similar color.

**Modern variant: CAM Shift (Continuously Adaptive Meanshift):**
- Use backprojection as likelihood.
- Meanshift finds region centroid.
- Adaptive scale (track size changes).
- Better performance than simple centroid tracking.

---

### Q81. What is image denoising and what are the main denoising approaches?

**A:** Image denoising removes noise while preserving edges and details. It is fundamental preprocessing for many vision tasks.

**Noise types:**
1. **Gaussian:** additive, pixel-independent.
2. **Salt-and-pepper:** random black/white pixels.
3. **Poisson:** signal-dependent (from low photon count).
4. **Speckle:** multiplicative (radar, ultrasound).

**Denoising methods:**

1. **Spatial filters:**
   - **Gaussian blur:** simple, smooth.
   - **Bilateral filter:** blur edges less, preserve boundaries.
   - **Median filter:** effective for salt-and-pepper.

2. **Non-local methods:**
   - **Non-local means:** average similar patches (not just neighbors).
   - **Block matching 3D (BM3D):** collaborative filtering of 3D patch blocks.
   - Effective but computationally expensive.

3. **Variational methods:**
   - Minimize denoising energy: $E = ||I - I_0||^2 + \lambda TV(I)$ (total variation).
   - TV: sum of gradient magnitudes (encourages piecewise smooth solution).
   - Preserves edges while denoising.

4. **Deep learning:**
   - **Denoising autoencoder:** train network to map noisy→clean.
   - **Residual learning:** network learns noise, subtract from input.
   - **Generative models (VAE, diffusion):** learn clean data distribution.

**Trade-off:**
- Strong denoising: removes noise but blurs details.
- Weak denoising: preserves detail but retains noise.
- Edge-aware methods balance both.

**Expert perspective:**
Denoising quality depends on noise type and image content. No universal best method. Bilateral filter or BM3D good defaults. Deep learning enables learned denoisers that adapt to dataset.

---

### Q82. What is image enhancement and how does it improve visibility?

**A:** Image enhancement improves visibility and perceptual quality through contrast enhancement, sharpening, and artifact reduction.

**Enhancement techniques:**

1. **Contrast enhancement:**
   - Histogram equalization.
   - Adaptive histogram equalization (CLAHE).
   - Stretches pixel range to use full [0, 255].

2. **Sharpening:**
   - Unsharp masking: $\text{sharp} = I + \lambda(I - \text{blur}(I))$.
   - Emphasizes edges.
   - High-frequency emphasis in frequency domain.

3. **Illumination correction:**
   - Homomorphic filtering: separate illumination from reflectance.
   - Correct uneven lighting.

4. **Artifact reduction:**
   - Reduce compression artifacts (JPEG).
   - Remove noise.
   - Fill holes or inpaint.

**Perceptual metrics:**
- **Sharpness:** edge strength, texture detail.
- **Contrast:** range of pixel values.
- **Noise level:** high-frequency noise.
- **Artifacts:** visible distortions (banding, blockiness).

**Challenges:**
- Over-enhancement: artifacts, unnatural appearance.
- Parameter tuning: enhancement depends on image content.

**Modern approach:**
- Deep learning for automatic enhancement (learned from image pairs).
- Perceptual losses for natural appearance.

---

### Q83. What is camera calibration and why is it important?

**A:** Camera calibration determines intrinsic parameters (focal length, principal point, distortion) and extrinsic parameters (pose), enabling metric 3D reconstruction.

**Intrinsic parameters:**
- **Focal length** $f$: in pixels.
- **Principal point** $(c_x, c_y)$: image center offset.
- **Skew:** non-orthogonal pixel axes (usually 0).
- **Distortion:** radial and tangential lens distortion.

**Camera matrix:**
$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

**Extrinsic parameters:**
- **Rotation** $R$: orientation in world coordinates.
- **Translation** $t$: position in world coordinates.

**Calibration methods:**

1. **Checkerboard calibration:**
   - Print checkerboard pattern, photograph from multiple views.
   - Automatically detect corners.
   - Estimate parameters via optimization.

2. **Self-calibration:**
   - Exploit geometric constraints (known scene structure).
   - Estimate parameters from multiple views without calibration target.

3. **Vanishing points:**
   - Use parallel lines in scene.
   - Estimate focal length from vanishing point.

**Why important:**
1. **Metric reconstruction:** without calibration, 3D is up to scale/projective transformation.
2. **Undistortion:** remove lens distortion for geometric algorithms.
3. **Augmented reality:** accurately overlay virtual on real.

**Distortion correction:**
Apply inverse distortion model to undistort image:
$$
\text{undistorted} = K [R | t] \text{world\_point}
$$

---

### Q84. What is pose estimation and how does it relate to camera calibration?

**A:** Pose estimation determines camera position/orientation relative to object. It uses calibrated camera parameters to map 3D world to 2D image.

**Perspective-n-point (PnP) problem:**
- Given: $n$ 3D world points and their 2D image projections.
- Find: camera pose $[R | t]$ and distance scale.

**Solution methods:**

1. **Direct linear transformation (DLT):**
   - Set up linear system from projection equations.
   - Solve via SVD.
   - Requires 4+ points (6 DOF pose with 2 constraints per point).

2. **Iterative refinement:**
   - Initialize with linear solution.
   - Minimize reprojection error via Levenberg-Marquardt.
   - Nonlinear but converges to local minimum.

3. **RANSAC:**
   - Sample 4 points, estimate pose.
   - Count inliers (points with low reprojection error).
   - Keep pose with most inliers.
   - Robust to outliers.

**Applications:**
1. **AR/VR:** overlay virtual objects on camera feed.
2. **Robotic manipulation:** estimate object pose for grasping.
3. **Autonomous vehicles:** estimate vehicle pose relative to map.
4. **Object detection with 6D pose:** detect + orient objects.

**Expert perspective:**
Pose estimation is critical for robotics and AR. Challenges:
- Occlusion: hidden points.
- Ambiguity: symmetric objects have multiple valid poses.
- Depth ambiguity: far small object vs. near large object.

---

### Q85. What is image segmentation evaluation and what metrics are used?

**A:** Evaluation metrics assess segmentation quality, comparing predicted segmentation to ground truth.

**Pixel-level metrics:**

1. **Accuracy:** fraction of correctly classified pixels.
   - Prone to imbalance (ignore class > overall accuracy).

2. **IoU (Intersection over Union, Jaccard index):**
   $$
   \text{IoU} = \frac{|A \cap B|}{|A \cup B|}
   $$
   Per-class IoU, then average (mIoU).

3. **Dice coefficient:**
   $$
   \text{Dice} = \frac{2|A \cap B|}{|A| + |B|}
   $$
   Similar to IoU, slightly different weighting.

4. **F1 score:** harmonic mean of precision and recall.

**Region-level metrics:**

1. **Boundary metrics:** distance between predicted and ground-truth boundaries.
   - **Hausdorff distance:** max distance from one set to nearest point in other.
   - **Mean boundary distance.**

2. **Panoptic quality:** accounts for detection + IoU for things and stuff.

**Challenge:**
- **Ambiguity:** some pixels inherently ambiguous (edge).
- **User disagreement:** humans annotate differently.
- **Metrics bias:** some metrics favor recall, others precision.

**Best practice:**
- Use multiple metrics (IoU, F1, boundary distance).
- Report per-class and overall metrics.
- Discuss failure cases.

---

## Summary: Traditional Image Processing

This section covers ~35 questions on classical techniques:
- **Filtering:** Gaussian, Sobel, Laplacian, bilateral.
- **Edge detection:** Canny, Sobel, Laplacian.
- **Feature detection:** Harris, SIFT.
- **Shape analysis:** morphology, distance transform, skeleton.
- **Segmentation:** thresholding, region growing, watershed, connected components.
- **Geometric:** warping, homography, camera calibration, pose estimation.
- **Color:** color spaces, histograms, histogram equalization, backprojection.
- **Frequency domain:** Fourier, power spectrum, filtering.
- **Multi-scale:** image pyramids, scale-space.
- **Denoising, enhancement:** classical and modern approaches.
- **Evaluation:** metrics for assessing segmentation quality.

These techniques are **foundational knowledge every CV professional should understand deeply**, even with modern deep learning dominance. They form the conceptual basis for modern methods.

---

## Bonus Questions

### Q51. What is the difference between precision and recall, and when does each matter?

**A:** 
- **Precision:** fraction of predicted positives that are correct. $P = \frac{TP}{TP+FP}$.
- **Recall:** fraction of actual positives that are found. $R = \frac{TP}{TP+FN}$.

**Trade-off:**
- **High precision, low recall:** many false negatives (miss positives). Example: very conservative face detection (only report if highly confident). False negatives: missed faces.
- **Low precision, high recall:** many false positives (report positives that are wrong). Example: very liberal face detection. False positives: non-faces reported as faces.

**When each matters:**
1. **Medical diagnosis:** recall more important (miss cancer = bad). Precision secondary.
2. **Spam detection:** precision more important (false positive = user annoyance). Recall secondary.
3. **Autonomous driving:** both critical (crashes are bad, false alarms waste compute).

**F1 score:** harmonic mean of precision and recall.
$$
F_1 = 2 \cdot \frac{P \cdot R}{P + R}
$$

---

### Q52. What is batch normalization training vs. inference difference?

**A:** During training, BatchNorm uses batch statistics (mean, variance). At inference, it uses running averages computed during training.

**Why different:**
- Training: batch statistics provide regularization (noise from batch sampling).
- Inference: batch statistics unavailable (single sample or small batch). Must use pre-computed running statistics.

**Running average:**
$$
\mu_{\text{running}} \leftarrow (1-\alpha) \mu_{\text{running}} + \alpha \mu_{\text{batch}}
$$

Default $\alpha = 0.1$ (or based on batch count).

**Failure mode:**
If training and inference statistics differ (domain shift), BatchNorm performance degrades.

---

### Q53. How does gradient clipping help with training stability?

**A:** Gradient clipping caps gradient magnitude to prevent exploding gradients.

**Formula:**
$$
g' = g \cdot \min(1, \frac{\text{max\_norm}}{|g|})
$$

If gradient norm exceeds threshold, scale down to threshold.

**When needed:**
- **RNNs:** gradients can explode exponentially through many time steps.
- **GANs:** discriminator gradient can oscillate wildly.

**Effect:**
Stabilizes training, prevents NaN/Inf, allows higher learning rates.

---

### Q54. What is the vanishing gradient problem and how do ReLU activations help?

**A:** In deep networks with sigmoid/tanh activations, gradients become exponentially smaller in early layers due to repeated multiplication of numbers < 1 during backpropagation.

**Solution: ReLU:**
ReLU gradient is 1 for positive inputs (no multiplication by <1), avoiding exponential decay.

**Better solution: skip connections (ResNets):**
Allow gradients to flow directly across layers via identity connection, bypassing the nonlinear path.

---

### Q55. How do you select the right learning rate for training?

**A:** Learning rate is critical: too small → slow training, too large → divergence.

**Methods:**
1. **Learning rate scheduling:** start high, decay over time (step decay, cosine annealing).
2. **Learning rate ranges test:** try logarithmic range (1e-6 to 1e0), find loss decrease sweet spot.
3. **Adaptive learning rates (Adam, RMSprop):** algorithm adjusts per parameter (less sensitive to choice).
4. **Learning rate warmup:** start small, ramp up to full LR in first few epochs (helps with batch normalization).

**Rule of thumb:**
- Batch size 256: LR ≈ 0.1.
- Batch size increase 2×: LR increase by $\sqrt{2}$ ≈ 1.4×.

---

## Summary

This comprehensive Q&A guide covers **55+ foundation to intermediate computer vision topics** with **beginner-to-expert explanations**. 

**Key themes:**
1. **Fundamentals:** convolution, activation, backprop, normalization.
2. **Architectures:** CNNs, ResNets, Transformers, Vision Transformers.
3. **Detection/Segmentation:** object detection, semantic/instance segmentation, panoptic.
4. **Advanced:** optical flow, 3D vision, GANs, self-supervised learning.
5. **Practical:** domain adaptation, few-shot learning, adversarial robustness.
6. **Applications:** face recognition, action recognition, tracking, retrieval, inpainting.

**Interview tips:**
- **Answer confidently:** explain clearly from beginner level, then dive into expert details.
- **Show nuance:** discuss trade-offs (speed vs. accuracy, precision vs. recall).
- **Mention applications:** connect concepts to real-world problems.
- **Discuss challenges:** show awareness of limitations and open problems.
- **Ask clarifying questions:** "Are you asking about real-time constraints?" helps tailor answer.
- **Code examples:** if comfortable, mention implementation details (PyTorch, TensorFlow).

Good luck with your interview!

# 📑 Mathematical Foundations of BabyGPT

This document provides a deep dive into the mathematics driving the BabyGPT Engine.

## 1. Causal Self-Attention
The engine uses Multi-Head Causal Self-Attention, formulated as:

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- **Q (Query)**, **K (Key)**, and **V (Value)** are linear projections of the input.
- The **Causal Mask** ensures $QK^T$ is masked with $-\infty$ for future tokens, preserving the autoregressive property.

## 2. Layer Normalization (Pre-LN)
To ensure training stability, we apply normalization before each sub-layer:

$$\hat{x} = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} \cdot \gamma + \beta$$

## 3. Adam Optimization
The weight updates are handled by the Adam optimizer, which maintains adaptive learning rates for each parameter:

1. **First Moment:** $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
2. **Second Moment:** $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
3. **Weight Update:** $\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

## 4. Byte-Pair Encoding (BPE)
The tokenizer utilizes the BPE algorithm to iteratively merge the most frequent adjacent pairs of characters. This allows the model to:
- Minimize the sequence length (compression).
- Handle "Out-of-Vocabulary" words by breaking them into known sub-units.

## 5. Cross-Entropy Loss
The training objective is to minimize the negative log-likelihood of the target token:

$$L = -\sum_{i} y_i \log(\hat{y}_i)$$

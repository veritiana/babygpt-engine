/**
 * ARCHITECTURAL ROLE:
 * Layer Normalization (LN) is a critical stabilization component in the Transformer architecture.
 * In GPT-2 style models, it is used before the Multi-Head Attention and FeedForward blocks 
 * (Pre-LN) and as a final step before the LM Head.
 * * * Purpose:
 * 1. It re-centers and re-scales the activations for each token independently across the 
 * embedding dimension.
 * 2. It prevents gradients from exploding or vanishing during training, allowing for 
 * faster and more stable convergence.
 * 3. It provides two learnable parameters (gamma and beta) that allow the model to 
 * restore the optimal dynamic range of the data if needed.
 * * Hierarchy: TransformerBlock -> LayerNorm -> (Attention/FeedForward)
 */



/**
 * Parameter Initialization:
 * Creates the trainable scales (gamma) and shifts (beta).
 * Gamma starts at 1.0 (no scaling) and Beta starts at 0.0 (no shifting) 
 * so that the identity is preserved at the start of training.
 */
export function createLayerNormParams(embedDim) {
    const gamma = new Float32Array(embedDim);
    const beta  = new Float32Array(embedDim);

    for (let i = 0; i < embedDim; i++) {
        gamma[i] = 1.0;
        beta[i]  = 0.0;
    }

    return { gamma, beta };
}


/**
 * FORWARD PASS:
 * Normalizes the input vector 'x' for each position in the sequence.
 * Formula: LN(x) = ((x - mean) / sqrt(variance + eps)) * gamma + beta
 */
export function layernormForward(x, seqLen, embedDim, gamma, beta, eps = 1e-5) {
    const out = new Float32Array(x.length);

    // Arrays to store intermediate values needed for efficient backpropagation
    const mean = new Float32Array(seqLen);
    const invStd = new Float32Array(seqLen);

    for (let t = 0; t < seqLen; t++) {
        const off = t * embedDim;

        // 1. Calculate Mean: Average activation for the current token
        let m = 0;
        for (let j = 0; j < embedDim; j++) m += x[off + j];
        m /= embedDim;
        mean[t] = m;

        // 2. Calculate Variance: Average squared deviation from the mean
        let v = 0;
        for (let j = 0; j < embedDim; j++) {
            const dev = x[off + j] - m;
            v += dev * dev;
        }
        v /= embedDim;

        // 3. Inverse Standard Deviation: Cached for the backward pass
        const is = 1.0 / Math.sqrt(v + eps);
        invStd[t] = is;

        // 4. Transform: Center, Scale, and apply learnable gamma/beta
        for (let j = 0; j < embedDim; j++) {
            out[off + j] = (x[off + j] - m) * is * gamma[j] + beta[j];
        }
    }

    return { out, cache: { mean, invStd } };
}


/**
 * BACKWARD PASS (Training):
 * Computes gradients for the input (dx) and for the trainable parameters (dGamma, dBeta).
 * This involves complex calculus to propagate the error through the mean and variance 
 * calculations of the forward pass.
 */
export function layernormBackward(x, gradOut, seqLen, embedDim, gamma, cache) {
    const { mean, invStd } = cache;

    const dx = new Float32Array(x.length);
    const dGamma = new Float32Array(embedDim);
    const dBeta  = new Float32Array(embedDim);

    for (let t = 0; t < seqLen; t++) {
        const off = t * embedDim;

        // Step 1: Accumulate gradients for gamma and beta across the sequence
        for (let j = 0; j < embedDim; j++) {
            const norm = (x[off + j] - mean[t]) * invStd[t];
            dGamma[j] += gradOut[off + j] * norm;
            dBeta[j]  += gradOut[off + j];
        }

        // Step 2: Compute intermediate sums for the dx calculation (Chain Rule)
        let sumDy = 0;
        let sumDyX = 0;

        for (let j = 0; j < embedDim; j++) {
            const dy = gradOut[off + j] * gamma[j];
            sumDy += dy;
            sumDyX += dy * (x[off + j] - mean[t]);
        }

        const invN = 1 / embedDim;
        const is = invStd[t];
        const is3 = is * is * is;

        // Step 3: Compute final gradient for the input data (dx)
        for (let j = 0; j < embedDim; j++) {
            const dev = x[off + j] - mean[t];
            // Combining the gradient of the normalization formula
            dx[off + j] = (gradOut[off + j] * gamma[j] * is) - 
                          (sumDy * is * invN) - 
                          (dev * is3 * invN * sumDyX);
        }
    }

    return { dx, dGamma, dBeta };
}
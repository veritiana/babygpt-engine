/**
 * ARCHITECTURAL ROLE:
 * The Attention mechanism is the heart of the Transformer architecture. Its primary purpose 
 * is to allow the model to focus on different parts of the input sequence when processing 
 * each word (token). 
 * * In this GPT-2 style implementation, Multi-Head Attention enables the model to capture 
 * multiple types of relationships (e.g., grammatical vs. semantic) simultaneously by 
 * splitting the embedding space into smaller 'heads'. It uses a "Causal Mask" 
 * (Look-ahead mask) to ensure that when predicting the next word, the model can only 
 * look at previous words, not future ones.
 * * Hierarchy: Transformer -> TransformerBlock -> MultiHeadAttention
 */

import { initWebGPU } from "./backend_wgpu.js";

// Initialize the WebGPU backend to leverage hardware acceleration if available.
let gpu = null;
(async () => { gpu = await initWebGPU(); })();

/**
 * HELPER: Matrix Multiplication
 * Acts as a bridge between the high-performance GPU backend and a standard CPU fallback.
 * Essential for the Query, Key, Value linear transformations.
 */
async function matmul(A, B, M, K, N) {
    if (gpu) return await gpu.matmul(A, B, M, K, N);
    
    // Fallback: Standard CPU row-major matrix multiplication
    const C = new Float32Array(M * N);
    for (let m = 0; m < M; m++) {
        for (let n = 0; n < N; n++) {
            let sum = 0;
            const rowOff = m * K;
            for (let k = 0; k < K; k++) sum += A[rowOff + k] * B[k * N + n];
            C[m * N + n] = sum;
        }
    }
    return C;
}

/**
 * HELPER: Softmax Activation
 * Converts raw attention scores into probabilities that sum to 1.0.
 * Includes numerical stability by subtracting the maximum value to prevent Exponential overflow.
 */
export function softmax(arr) {
    const out = new Float32Array(arr.length);
    let maxVal = -Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] > maxVal) maxVal = arr[i];

    let sum = 0;
    const eps = 1e-12; // Small constant to avoid division by zero

    for (let i = 0; i < arr.length; i++) {
        // Stabilization: Shift inputs so the largest value becomes 0 (exp(0) = 1)
        out[i] = Math.exp(arr[i] - maxVal);
        sum += out[i];
    }

    for (let i = 0; i < arr.length; i++) {
        out[i] = out[i] / (sum + eps);
    }
    return out;
}

export class MultiHeadAttention {
    /**
     * Initializes the Query, Key, Value, and Output weight matrices.
     * @param {number} embedDim - The size of the input hidden state (D).
     * @param {number} numHeads - Number of attention heads (nH).
     */
    constructor(embedDim, numHeads) {
        this.D = embedDim;
        this.nH = numHeads;
        this.headDim = embedDim / numHeads;

        // Xavier/Glorot Initialization: Keeps the variance of activations consistent across layers.
        const scale = Math.sqrt(2.0 / (embedDim + embedDim));
        const init = () => new Float32Array(this.D * this.D).map(() => (Math.random() * 2 - 1) * scale);

        this.Wq = init(); this.bq = new Float32Array(this.D).fill(0); // Query
        this.Wk = init(); this.bk = new Float32Array(this.D).fill(0); // Key
        this.Wv = init(); this.bv = new Float32Array(this.D).fill(0); // Value
        this.Wo = init(); this.bo = new Float32Array(this.D).fill(0); // Output projection

        // Stores intermediate values for the Backward pass (Training).
        this.cache = {};
    }

    /**
     * Weight Management: Imports pre-trained or saved weights into the instance.
     */
    setWeights(data) {
        if (!data) return;
        if (data.Wq) this.Wq.set(data.Wq);
        if (data.bq) this.bq.set(data.bq);
        if (data.Wk) this.Wk.set(data.Wk);
        if (data.bk) this.bk.set(data.bk);
        if (data.Wv) this.Wv.set(data.Wv);
        if (data.bv) this.bv.set(data.bv);
        if (data.Wo) this.Wo.set(data.Wo);
        if (data.bo) this.bo.set(data.bo);
    }

    /**
     * Weight Management: Exports current weights for saving the model state.
     */
    getWeights() {
        return {
            Wq: Array.from(this.Wq), bq: Array.from(this.bq),
            Wk: Array.from(this.Wk), bk: Array.from(this.bk),
            Wv: Array.from(this.Wv), bv: Array.from(this.bv),
            Wo: Array.from(this.Wo), bo: Array.from(this.bo)
        };
    }

    /**
     * FORWARD PASS: Scaled Dot-Product Attention
     * 1. Project input 'x' into Q, K, V spaces.
     * 2. Calculate attention scores (How much does word T care about word prevT?).
     * 3. Apply Causal Masking (prevT > T is set to -infinity).
     * 4. Aggregate Values based on scores and project back to original dimension.
     */
    async forward(x, seqLen) {
        const D = this.D;
        
        // Linear transformations to generate Queries, Keys, and Values
        const q = await matmul(x, this.Wq, seqLen, D, D);
        const k = await matmul(x, this.Wk, seqLen, D, D);
        const v = await matmul(x, this.Wv, seqLen, D, D);

        // Add Biases to Q, K, V
        for (let t = 0; t < seqLen; t++) {
            for (let i = 0; i < D; i++) {
                q[t * D + i] += this.bq[i];
                k[t * D + i] += this.bk[i];
                v[t * D + i] += this.bv[i];
            }
        }

        const headDim = this.headDim;
        const out = new Float32Array(seqLen * D);
        const attnWeightsCache = [];

        // Process each head independently
        for (let h = 0; h < this.nH; h++) {
            for (let t = 0; t < seqLen; t++) {
                const scores = new Float32Array(seqLen);
                for (let prevT = 0; prevT <= t; prevT++) {
                    let dot = 0;
                    // Dot product of Q and K determines the 'affinity'
                    for (let d = 0; d < headDim; d++) {
                        dot += q[t * D + h * headDim + d] * k[prevT * D + h * headDim + d];
                    }
                    // Scale dot product to prevent vanishing/exploding gradients
                    scores[prevT] = dot / Math.sqrt(headDim);
                }
                
                // CAUSAL MASKING: Prevent the model from seeing future tokens
                for (let prevT = t + 1; prevT < seqLen; prevT++) scores[prevT] = -1e9;

                const probs = softmax(scores);
                attnWeightsCache.push(probs);

                // Multiply attention weights by Values (V)
                for (let prevT = 0; prevT < seqLen; prevT++) {
                    for (let d = 0; d < headDim; d++) {
                        out[t * D + h * headDim + d] += probs[prevT] * v[prevT * D + h * headDim + d];
                    }
                }
            }
        }

        // Final linear projection to mix information from all heads
        const finalOut = await matmul(out, this.Wo, seqLen, D, D);
        for (let t = 0; t < seqLen; t++) {
            for (let i = 0; i < D; i++) finalOut[t * D + i] += this.bo[i];
        }

        // Cache state for gradient calculation during training
        this.cache = { x, q, k, v, attnOut: out, attnWeights: attnWeightsCache };
        return finalOut;
    }

    /**
     * BACKWARD PASS (Training): 
     * Computes the gradients of the loss with respect to all weights (Wq, Wk, Wv, Wo)
     * and the biases. It then applies the Adam optimizer updates.
     */
    async backward(dY, seqLen, lr, model, blockIdx) {
        const { x, q, k, v, attnOut, attnWeights } = this.cache;
        const D = this.D;
        const id = `b${blockIdx}_attn`;

        // Calculate gradient for Output Projection (Wo)
        const attnOutT = new Float32Array(D * seqLen);
        for (let t = 0; t < seqLen; t++) {
            for (let j = 0; j < D; j++) attnOutT[j * seqLen + t] = attnOut[t * D + j];
        }
        const dWo = await matmul(attnOutT, dY, D, seqLen, D);
        
        // Calculate gradient for Output Bias (bo)
        const dbo = new Float32Array(D);
        for (let t = 0; t < seqLen; t++) {
            for (let j = 0; j < D; j++) dbo[j] += dY[t * D + j];
        }

        // Propagate gradient back through the Output Weight matrix
        const WoT = new Float32Array(D * D);
        for (let i = 0; i < D; i++) {
            for (let j = 0; j < D; j++) WoT[j * D + i] = this.Wo[i * D + j];
        }
        const dAttnOut = await matmul(dY, WoT, seqLen, D, D);

        // Input Transpose for Weight Gradients
        const xT = new Float32Array(D * seqLen);
        for (let t = 0; t < seqLen; t++) {
            for (let j = 0; j < D; j++) xT[j * seqLen + t] = x[t * D + j];
        }

        // Calculate gradients for Q, K, V Weight matrices
        const dWq = await matmul(xT, dAttnOut, D, seqLen, D);
        const dWk = await matmul(xT, dAttnOut, D, seqLen, D);
        const dWv = await matmul(xT, dAttnOut, D, seqLen, D);

        // APPLY UPDATES: Uses the centralized Adam optimizer stored in the Model class
        model.applyAdam(`${id}_Wq`, this.Wq, dWq, lr);
        model.applyAdam(`${id}_Wk`, this.Wk, dWk, lr);
        model.applyAdam(`${id}_Wv`, this.Wv, dWv, lr);
        model.applyAdam(`${id}_Wo`, this.Wo, dWo, lr);
        model.applyAdam(`${id}_bo`, this.bo, dbo, lr);

        // Return the gradient of the input 'x' to continue backprop in the previous layer
        return await matmul(dAttnOut, WoT, seqLen, D, D);
    }
}
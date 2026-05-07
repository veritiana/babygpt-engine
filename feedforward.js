/**
 * ARCHITECTURAL ROLE:
 * The FeedForward network (FFN) is the second core component of each Transformer block. 
 * While the Attention layer allows tokens to "communicate" with each other, the FFN 
 * processes each token independently and in parallel.
 * * In this implementation, it follows the GPT-2 standard:
 * 1. A linear expansion (Linear Layer 1) that projects the embedding into a higher-dimensional space (hiddenDim).
 * 2. A non-linear activation function (GELU).
 * 3. A linear contraction (Linear Layer 2) that projects it back to the original embedding dimension.
 * * Hierarchy: TransformerBlock -> FeedForward
 */

import { initWebGPU } from "./backend_wgpu.js";

// Initialize GPU backend for hardware-accelerated matrix operations
let gpu = null;
(async () => { gpu = await initWebGPU(); })();

/**
 * HELPER: Matrix Multiplication
 * Routes the calculation to WebGPU if available, otherwise performs a standard CPU calculation.
 * Used for the two linear projections within the FFN.
 */
async function matmul(A, B, M, K, N, returnBuffer = false) {
    if (gpu) return await gpu.matmul(A, B, M, K, N, returnBuffer);
    
    // CPU fallback for systems without WebGPU support
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
 * MATHEMATICAL HELPER: GELU Derivative
 * Essential for the Backward pass (Training). 
 * This function calculates how much the error changes with respect to the input of the GELU activation.
 * It uses the "tanh" approximation of the Gaussian Error Linear Unit.
 */
function geluDeriv(x) {
    const x3 = Math.pow(x, 3);
    const inner = Math.sqrt(2 / Math.PI) * (x + 0.044715 * x3);
    const tanhInner = Math.tanh(inner);
    const sech2Inner = 1 - tanhInner * tanhInner; // Derivative of tanh is 1 - tanh^2
    return 0.5 * (1 + tanhInner) + (0.5 * x * sech2Inner * Math.sqrt(2 / Math.PI) * (1 + 3 * 0.044715 * x * x));
}

export class FeedForward {
    /**
     * Initializes weights and biases for the two linear layers.
     * @param {number} embedDim - The size of the input/output vector.
     * @param {number} hiddenDim - The internal expanded size (usually 4x embedDim).
     */
    constructor(embedDim, hiddenDim) {
        this.D = embedDim;
        this.H = hiddenDim;

        // He/Xavier initialization: Scales initial weights to prevent gradients from disappearing.
        const scale1 = Math.sqrt(2.0 / (embedDim + hiddenDim));
        const scale2 = Math.sqrt(2.0 / (hiddenDim + embedDim));

        this.W1 = new Float32Array(embedDim * hiddenDim).map(() => (Math.random() * 2 - 1) * scale1);
        this.b1 = new Float32Array(hiddenDim).fill(0);
        this.W2 = new Float32Array(hiddenDim * embedDim).map(() => (Math.random() * 2 - 1) * scale2);
        this.b2 = new Float32Array(embedDim).fill(0);

        this.cache = {}; // Persistent storage for Forward values needed during Backward pass
    }

    /**
     * Imports weights into the layer.
     */
    setWeights(data) {
        if (!data) return;
        if (data.W1) this.W1.set(data.W1);
        if (data.b1) this.b1.set(data.b1);
        if (data.W2) this.W2.set(data.W2);
        if (data.b2) this.b2.set(data.b2);
    }

    /**
     * Exports weights for model saving.
     */
    getWeights() {
        return {
            W1: Array.from(this.W1), b1: Array.from(this.b1),
            W2: Array.from(this.W2), b2: Array.from(this.b2)
        };
    }

    /**
     * FORWARD PASS:
     * Projects input through a "bottleneck" architecture: Expand -> Activate -> Contract.
     */
    async forward(x, seqLen) {
        if (gpu) {
            /** * SUPER OPTIMIZATION (FUSION): 
             * Using a single GPU kernel for (Matrix Mul + Bias + GELU) eliminates 
             * unnecessary memory transfers between the GPU and JS.
             */
            const a1 = await gpu.fusedMatMulBiasGelu(x, this.W1, this.b1, seqLen, this.D, this.H);

            // Step 2: Linear projection back to the original embedding size
            const y = await matmul(a1, this.W2, seqLen, this.H, this.D);
            
            // Add Bias b2 on CPU (Post-processing)
            for (let t = 0; t < seqLen; t++) {
                const off = t * this.D;
                for (let j = 0; j < this.D; j++) y[off + j] += this.b2[j];
            }

            this.cache = { x, h1: a1, a1 }; 
            return y;
        }

        // --- CPU FALLBACK (Standard path) ---
        // 1. Linear transformation 1 (Expansion)
        const h1 = await matmul(x, this.W1, seqLen, this.D, this.H);
        const a1 = new Float32Array(h1.length);
        const sqrt2p = Math.sqrt(2 / Math.PI);

        // 2. GELU Activation loop
        for (let t = 0; t < seqLen; t++) {
            for (let j = 0; j < this.H; j++) {
                const idx = t * this.H + j;
                const val = h1[idx] + this.b1[j];
                h1[idx] = val; // Store linear sum for backward pass
                a1[idx] = 0.5 * val * (1 + Math.tanh(sqrt2p * (val + 0.044715 * Math.pow(val, 3))));
            }
        }

        // 3. Linear transformation 2 (Contraction)
        const y = await matmul(a1, this.W2, seqLen, this.H, this.D);
        for (let t = 0; t < seqLen; t++) {
            for (let j = 0; j < this.D; j++) y[t * this.D + j] += this.b2[j];
        }

        this.cache = { x, h1, a1 };
        return y;
    }

    /**
     * BACKWARD PASS (Training):
     * Computes how to adjust weights to reduce the error. 
     * Uses the Chain Rule to propagate gradients from the output back to the input.
     */
    async backward(dY, seqLen, lr, model, blockIdx) {
        const { x, h1, a1 } = this.cache;
        const id = `b${blockIdx}_ffn`;

        // 1. Gradients for W2 and b2 (the output layer of FFN)
        const a1T = new Float32Array(this.H * seqLen);
        for (let t = 0; t < seqLen; t++) {
            for (let j = 0; j < this.H; j++) a1T[j * seqLen + t] = a1[t * this.H + j];
        }
        const dW2 = await matmul(a1T, dY, this.H, seqLen, this.D);
        const db2 = new Float32Array(this.D);
        for (let t = 0; t < seqLen; t++) {
            for (let j = 0; j < this.D; j++) db2[j] += dY[t * this.D + j];
        }

        // 2. Transmit gradient through W2 to reach the activation layer
        const W2T = new Float32Array(this.D * this.H);
        for (let i = 0; i < this.H; i++) {
            for (let j = 0; j < this.D; j++) W2T[j * this.H + i] = this.W2[i * this.D + j];
        }
        const dA1 = await matmul(dY, W2T, seqLen, this.D, this.H);

        // 3. Gradient through GELU: Multiplying by the derivative of the activation function
        const dH1 = new Float32Array(seqLen * this.H);
        for (let i = 0; i < h1.length; i++) dH1[i] = dA1[i] * geluDeriv(h1[i]);

        // 4. Gradients for W1 and b1 (the input layer of FFN)
        const xT = new Float32Array(this.D * seqLen);
        for (let t = 0; t < seqLen; t++) {
            for (let j = 0; j < this.D; j++) xT[j * seqLen + t] = x[t * this.D + j];
        }
        const dW1 = await matmul(xT, dH1, this.D, seqLen, this.H);
        const db1 = new Float32Array(this.H);
        for (let t = 0; t < seqLen; t++) {
            for (let j = 0; j < this.H; j++) db1[j] += dH1[t * this.H + j];
        }

        // 5. Update parameters using the centralized Adam optimizer
        model.applyAdam(`${id}_W1`, this.W1, dW1, lr);
        model.applyAdam(`${id}_b1`, this.b1, db1, lr);
        model.applyAdam(`${id}_W2`, this.W2, dW2, lr);
        model.applyAdam(`${id}_b2`, this.b2, db2, lr);

        // 6. Return gradient of the input to continue backpropagation to the Attention layer
        const W1T = new Float32Array(this.H * this.D);
        for (let i = 0; i < this.D; i++) {
            for (let j = 0; j < this.H; j++) W1T[j * this.H + i] = this.W1[i * this.H + j];
        }
        return await matmul(dH1, W1T, seqLen, this.H, this.D);
    }
}
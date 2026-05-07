/**
 * ARCHITECTURAL ROLE:
 * This is the core engine class (BabyGPT) that implements the GPT-style Transformer.
 * * * Hierarchy:
 * 1. Token & Positional Embeddings (WTE + WPE)
 * 2. Sequential Transformer Blocks (Attention + MLP)
 * 3. Final LayerNorm
 * 4. Language Model Head (Linear Layer) to project back to Vocabulary size.
 * * * Features:
 * - Hybrid Execution: Uses WebGPU for heavy matrix operations with a CPU fallback.
 * - Training-Ready: Includes Cross-Entropy loss, Backpropagation, and Adam Optimizer.
 * - Persistence: Methods to save/load model state and tokenizer data.
 */

import { TransformerBlock } from "./transformer_block.js";
import { initWebGPU } from "./backend_wgpu.js";

let gpuBackend = null;
(async () => {
    gpuBackend = await initWebGPU();
})();

/**
 * UTILITY: Matrix Multiplication
 * Orchestrates where the math happens. If WebGPU is available, it offloads to the GPU.
 */
async function backendMatmul(A, B, M, K, N) {
    if (gpuBackend) {
        return await gpuBackend.matmul(A, B, M, K, N);
    }
    const out = new Float32Array(M * N);
    for (let m = 0; m < M; m++) {
        for (let n = 0; n < N; n++) {
            let sum = 0;
            const rowOff = m * K;
            for (let k = 0; k < K; k++) {
                sum += A[rowOff + k] * B[k * N + n];
            }
            out[m * N + n] = sum;
        }
    }
    return out;
}

/**
 * LOSS CALCULATION: Cross-Entropy
 * Measures the difference between the predicted probability distribution and the 
 * actual target token. It also computes the initial gradient (dLogits) for backprop.
 */
export function crossEntropyLoss(logits, targets, vocabSize) {
    const seqLen = targets.length;
    let totalLoss = 0;
    const dLogits = new Float32Array(logits.length);

    for (let t = 0; t < seqLen; t++) {
        const rowStart = t * vocabSize;
        const targetToken = targets[t];

        // 1. Find Max for Numerical Stability (Softmax trick)
        let maxVal = -Infinity;
        for (let i = 0; i < vocabSize; i++) {
            if (logits[rowStart + i] > maxVal) maxVal = logits[rowStart + i];
        }

        // 2. Compute Log-Sum-Exp
        let sumExp = 0;
        for (let i = 0; i < vocabSize; i++) {
            sumExp += Math.exp(logits[rowStart + i] - maxVal);
        }

        const logSumExp = maxVal + Math.log(sumExp);
        const probTarget = logits[rowStart + targetToken];

        // 3. Accumulate Loss
        totalLoss += (logSumExp - probTarget);

        // 4. Compute dLogits (Gradient of Cross-Entropy + Softmax)
        for (let i = 0; i < vocabSize; i++) {
            const p = Math.exp(logits[rowStart + i] - logSumExp);
            dLogits[rowStart + i] = (i === targetToken) ? (p - 1.0) : p;
        }
    }
    return { loss: totalLoss / seqLen, dLogits };
}

/**
 * GRADIENT CLIPPING:
 * Prevents "Exploding Gradients" by capping the maximum value of any gradient 
 * during training, ensuring numerical stability.
 */
function clipGrads(gradArray, limit = 5.0) {
    for (let i = 0; i < gradArray.length; i++) {
        if (gradArray[i] > limit) gradArray[i] = limit;
        if (gradArray[i] < -limit) gradArray[i] = -limit;
    }
}

export class BabyGPT {
    /**
     * MODEL CONSTRUCTOR:
     * Initializes all learnable weights: 
     * - wte: Token Embeddings (maps words to vectors)
     * - wpe: Positional Embeddings (tells the model where words are in a sentence)
     * - blocks: The hidden layers
     * - lmHead: The output projector
     * - Adam State: Buffers (m, v) for the optimizer
     */
    constructor(config) {
        this.config = config;
        const { vocabSize, embedDim, seqLen, numLayers } = config;

        this.wte = new Float32Array(vocabSize * embedDim);
        this.wpe = new Float32Array(seqLen * embedDim);
        this.lmHead = new Float32Array(embedDim * vocabSize);

        // Xavier/Kaiming initialization for better starting signal
        for (let i = 0; i < this.wte.length; i++) this.wte[i] = (Math.random() - 0.5) * 0.02;
        for (let i = 0; i < this.wpe.length; i++) this.wpe[i] = (Math.random() - 0.5) * 0.02;
        for (let i = 0; i < this.lmHead.length; i++) this.lmHead[i] = (Math.random() - 0.5) * 0.02;

        this.blocks = Array.from({ length: numLayers }, (_, i) => new TransformerBlock(embedDim, config, i));

        this.adamState = new Map();
        this.t = 0; // Iteration counter for Adam
        this.cache = {};
    }

    /**
     * FORWARD PASS:
     * Transforms input token IDs into predicted probability distributions (logits).
     */
    async forward(inputIds) {
        const T = inputIds.length;
        const D = this.config.embedDim;
        let x = new Float32Array(T * D);

        // 1. Embedding Layer: Add Token + Positional info
        for (let t = 0; t < T; t++) {
            const tok = inputIds[t];
            for (let i = 0; i < D; i++) {
                x[t * D + i] = this.wte[tok * D + i] + this.wpe[t * D + i];
            }
        }

        // 2. Transformer Blocks: Process through all hidden layers
        this.cache.blockInputs = [];
        for (let i = 0; i < this.blocks.length; i++) {
            this.cache.blockInputs[i] = new Float32Array(x);
            x = await this.blocks[i].forward(x, T);
        }

        this.cache.finalX = x;

        // 3. Output Head: Final projection to vocabulary
        return await backendMatmul(x, this.lmHead, T, D, this.config.vocabSize);
    }

    /**
     * OPTIMIZER: Adam
     * Updates weights using moving averages of gradients (m) and squared gradients (v).
     */
    applyAdam(key, params, grads, lr, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
        if (!this.adamState.has(key)) {
            this.adamState.set(key, {
                m: new Float32Array(params.length),
                v: new Float32Array(params.length)
            });
        }
        const { m, v } = this.adamState.get(key);
        const t = this.t + 1;

        for (let i = 0; i < params.length; i++) {
            m[i] = b1 * m[i] + (1 - b1) * grads[i];
            v[i] = b2 * v[i] + (1 - b2) * grads[i] * grads[i];
            const mHat = m[i] / (1 - Math.pow(b1, t));
            const vHat = v[i] / (1 - Math.pow(b2, t));
            params[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
        }
    }

    /**
     * BACKWARD PASS:
     * Computes the error for every parameter using the Chain Rule.
     */
    async backward(inputIds, dLogits) {
        const T = inputIds.length;
        const D = this.config.embedDim;
        const V = this.config.vocabSize;
        this.t++;

        // 1. Gradient for the Output Head (lmHead)
        const finalXT = new Float32Array(D * T);
        for (let t = 0; t < T; t++) {
            for (let i = 0; i < D; i++) finalXT[i * T + t] = this.cache.finalX[t * D + i];
        }
        const dLmHead = await backendMatmul(finalXT, dLogits, D, T, V);
        clipGrads(dLmHead);
        this.applyAdam('lmHead', this.lmHead, dLmHead, 0.001);

        // 2. Propagate error back through the final MatMul
        const lmHeadT = new Float32Array(V * D);
        for (let i = 0; i < D; i++) {
            for (let j = 0; j < V; j++) lmHeadT[j * D + i] = this.lmHead[i * V + j];
        }
        let dX = await backendMatmul(dLogits, lmHeadT, T, V, D);

        // 3. Backprop through each Transformer Block (Reverse Order)
        for (let i = this.blocks.length - 1; i >= 0; i--) {
            dX = await this.blocks[i].backward(this.cache.blockInputs[i], dX, T, 0.001);
        }

        // 4. Gradients for WTE and WPE
        const dWte = new Float32Array(this.wte.length);
        const dWpe = new Float32Array(this.wpe.length);
        for (let t = 0; t < T; t++) {
            const tok = inputIds[t];
            const off = t * D;
            const wteOff = tok * D;
            const wpeOff = t * D;
            for (let i = 0; i < D; i++) {
                const grad = dX[off + i];
                dWte[wteOff + i] += grad;
                dWpe[wpeOff + i] += grad;
            }
        }

        clipGrads(dWte);
        clipGrads(dWpe);
        this.applyAdam('wte', this.wte, dWte, 0.001);
        this.applyAdam('wpe', this.wpe, dWpe, 0.001);
    }

    computeLoss(logits, targets) {
        return crossEntropyLoss(logits, targets, this.config.vocabSize);
    }

    /**
     * PERSISTENCE:
     * Bundles the model configuration, weights, and tokenizer data into a single 
     * JSON file for local storage.
     */
    async saveToDisk(tokenizerData) {
        let data = {
            config: this.config,
            wte: Array.from(this.wte),
            wpe: Array.from(this.wpe),
            lmHead: Array.from(this.lmHead),
            t: this.t,
            blocks: this.blocks.map(b => b.getWeights())
        };

        if (tokenizerData && tokenizerData.tokenizerType === "bpe") {
            data.tokenizerType = "bpe";
            data.tokenizerData = tokenizerData.tokenizerData;
        } else {
            data.tokenizerType = "word";
            data.tokenizerVocab = tokenizerData || null;
        }

        const blob = new Blob([JSON.stringify(data)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `babygpt_model_${Date.now()}.json`;
        a.click();
    }

    async updateWeights(lr) {
        // Combined with backprop in this implementation
    }
}
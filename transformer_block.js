/**
 * ARCHITECTURAL ROLE:
 * The TransformerBlock is the fundamental repetitive unit of the GPT architecture.
 * It follows the "Pre-LayerNorm" design pattern used in modern Transformers (like GPT-2/3).
 * * * Internal Structure:
 * 1. Residual Connection 1: Input + Attention(LayerNorm(Input))
 * 2. Residual Connection 2: Input2 + FeedForward(LayerNorm(Input2))
 * * * Purpose:
 * - The Attention layer allows tokens to "communicate" and gather context from other tokens.
 * - The FeedForward layer allows the model to process the gathered information 
 * independently at each position.
 * - Residual connections (skip connections) ensure that deep networks can be trained 
 * without the gradient vanishing.
 */

import {
    layernormForward,
    layernormBackward
} from "./layernorm.js";

import { MultiHeadAttention } from "./attention.js";
import { FeedForward } from "./feedforward.js";

export class TransformerBlock {
    /**
     * Initialization:
     * Sets up the two sub-layers (Attention and FFN) and their respective 
     * normalization parameters (gamma and beta).
     */
    constructor(embedDim = 128, hiddenDim = 512, numHeads = 4, blockIdx = 0) {
        this.embedDim = embedDim;
        this.blockIdx = blockIdx;

        this.attn = new MultiHeadAttention(embedDim, numHeads);
        this.ffn  = new FeedForward(embedDim, hiddenDim);

        this.ln1 = {
            gamma: new Float32Array(embedDim).fill(1.0),
            beta:  new Float32Array(embedDim).fill(0.0)
        };
        this.ln2 = {
            gamma: new Float32Array(embedDim).fill(1.0),
            beta:  new Float32Array(embedDim).fill(0.0)
        };

        this.cache = {};
    }

    /**
     * WEIGHT LOADING:
     * Restores the state of the block from a saved model file, including 
     * sub-layers and normalization parameters.
     */
    setWeights(data) {
        if (!data) return;
        
        if (data.ln1) {
            this.ln1.gamma.set(data.ln1.gamma);
            this.ln1.beta.set(data.ln1.beta);
        }
        if (data.ln2) {
            this.ln2.gamma.set(data.ln2.gamma);
            this.ln2.beta.set(data.ln2.beta);
        }

        if (this.attn && data.attn) this.attn.setWeights(data.attn);
        if (this.ffn && data.ffn) this.ffn.setWeights(data.ffn);
    }

    /**
     * WEIGHT EXPORT:
     * Serializes all internal parameters into a plain object for saving to disk.
     */
    getWeights() {
        return {
            ln1: { gamma: Array.from(this.ln1.gamma), beta: Array.from(this.ln1.beta) },
            ln2: { gamma: Array.from(this.ln2.gamma), beta: Array.from(this.ln2.beta) },
            attn: this.attn.getWeights(),
            ffn: this.ffn.getWeights()
        };
    }

    /**
     * FORWARD PASS:
     * Implements the dual-residual logic:
     * x = x + Attention(LN(x))
     * x = x + FeedForward(LN(x))
     */
    async forward(x, seqLen) {
        const D = this.embedDim;

        // 1. First Sub-layer: LayerNorm + Multi-Head Attention
        const { out: ln1_out, cache: ln1_cache } = layernormForward(x, seqLen, D, this.ln1.gamma, this.ln1.beta);
        const attn_out = await this.attn.forward(ln1_out, seqLen, this.blockIdx);

        // Residual Connection 1
        const y = new Float32Array(x.length);
        for (let i = 0; i < x.length; i++) y[i] = x[i] + attn_out[i];

        // 2. Second Sub-layer: LayerNorm + FeedForward (MLP)
        const { out: ln2_out, cache: ln2_cache } = layernormForward(y, seqLen, D, this.ln2.gamma, this.ln2.beta);
        const ffn_out = await this.ffn.forward(ln2_out, seqLen, this.blockIdx);

        // Residual Connection 2
        const z = new Float32Array(y.length);
        for (let i = 0; i < y.length; i++) z[i] = y[i] + ffn_out[i];

        // Cache intermediate states for backpropagation
        this.cache = { x, y, ln1_cache, ln2_cache };
        return z;
    }

    /**
     * BACKWARD PASS:
     * Propagates gradients through the FFN and Attention sub-layers in reverse.
     * Gradients from the residual connections are summed.
     */
    async backward(dZ, seqLen, lr, model) {
        const D = this.embedDim;
        const { x, y, ln1_cache, ln2_cache } = this.cache;
        const bId = this.blockIdx;

        // Gradient for Residual Connection 2
        const dFFN_out = dZ; 
        
        // 1. Backprop through FFN and LN2
        const dLN2_out = await this.ffn.backward(dFFN_out, seqLen, lr, model, bId);
        const ln2Grad = layernormBackward(y, dLN2_out, seqLen, D, this.ln2.gamma, ln2_cache);

        // Update LN2 learnable parameters
        model.applyAdam(`b${bId}_ln2_g`, this.ln2.gamma, ln2Grad.dGamma, lr);
        model.applyAdam(`b${bId}_ln2_b`, this.ln2.beta, ln2Grad.dBeta, lr);

        // Gradient for Residual Connection 1
        const dY = new Float32Array(y.length);
        for (let i = 0; i < dY.length; i++) {
            dY[i] = ln2Grad.dx[i] + dZ[i]; // sum of branch gradient and skip gradient
        }

        // 2. Backprop through Attention and LN1
        const dAttn_out = dY;
        const dLN1_out = await this.attn.backward(dAttn_out, seqLen, lr, model, bId);
        const ln1Grad = layernormBackward(x, dLN1_out, seqLen, D, this.ln1.gamma, ln1_cache);

        // Update LN1 learnable parameters
        model.applyAdam(`b${bId}_ln1_g`, this.ln1.gamma, ln1Grad.dGamma, lr);
        model.applyAdam(`b${bId}_ln1_b`, this.ln1.beta, ln1Grad.dBeta, lr);

        // Final input gradient (dX) to be passed to the previous layer
        const dX = new Float32Array(x.length);
        for (let i = 0; i < dX.length; i++) {
            dX[i] = ln1Grad.dx[i] + dY[i]; // sum of branch gradient and skip gradient
        }

        return dX;
    }
}
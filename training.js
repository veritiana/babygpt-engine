/**
 * ARCHITECTURAL ROLE:
 * This module is the Training Engine of the BabyGPT project. It orchestrates the 
 * "learning" process where the model adjusts its internal weights to minimize prediction error.
 * * * Key Functions:
 * 1. Data Pipeline: Loading raw text and converting it into batches of token IDs.
 * 2. Optimization Loop: Iterating through epochs and batches, performing forward 
 * and backward passes.
 * 3. Learning Rate Management: Implementing adaptive learning rates (LR decay) 
 * to ensure stable convergence.
 * 4. Persistence: Handling the saving and loading of model states (checkpoints).
 * * This script connects the Transformer model, the Tokenizers, and the GPU backend 
 * into a functional training workflow.
 */

import { BabyGPT } from "./transformer.js";
import { WordTokenizer } from "./tokenizer.js";
import { BPETokenizer } from "./BPETokenizer.js";

// TOKENIZER SWITCH: Configuration to choose between simple Word-level or advanced BPE
export const TOKENIZER_TYPE = "bpe";

/**
 * Utility to prevent the browser UI from freezing during heavy 
 * synchronous JavaScript calculations by yielding control back to the main thread.
 */
const yieldToMain = () => new Promise(resolve => requestAnimationFrame(resolve));

/**
 * DATA LOADING: Fetches the raw text dataset from a URL or Local Blob.
 */
export async function loadTextDataset(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error("Dataset sa nepodarilo načítať.");
    return await resp.text();
}

/**
 * MODEL LOADING: Reads a JSON file containing model weights and configuration.
 */
export async function loadModelWeights(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(JSON.parse(e.target.result));
        reader.onerror = reject;
        reader.readAsText(file);
    });
}

/**
 * LOGGING: Saves the loss and time metrics of the training session for analysis.
 */
export function saveTrainingHistory(history, config) {
    const summary = {
        final_loss: history.length > 0 ? history[history.length - 1].loss : null,
        average_step_time: history.length > 0 ? parseFloat((history.reduce((acc, h) => acc + h.delta_sec, 0) / history.length).toFixed(4)) : null,
        config: config,
        history: history
    };
    const blob = new Blob([JSON.stringify(summary, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `training_history_${Date.now()}.json`;
    a.click();
}

/**
 * CORE TRAINING LOOP:
 * This is the main entry point for the training process.
 * 1. Initializes the Tokenizer and Model.
 * 2. Prepares Training Pairs (Input -> Target).
 * 3. Runs the Forward-Backward-Update cycle.
 */
export async function startTraining(datasetUrl, config, onStep, existingData = null) {
    const text = await loadTextDataset(datasetUrl);
    
    // Step 1: Tokenizer Setup
    let tokenizer;
    if (existingData && existingData.tokenizerType === "bpe") {
        tokenizer = new BPETokenizer();
        tokenizer.tokenToId = existingData.tokenizerData.tokenToId;
        tokenizer.idToToken = existingData.tokenizerData.idToToken;
        tokenizer.merges = existingData.tokenizerData.merges;
        tokenizer.vocabSize = Object.keys(tokenizer.tokenToId).length;
    } else {
        tokenizer = (TOKENIZER_TYPE === "bpe") ? new BPETokenizer(1500) : new WordTokenizer();
        tokenizer.fit(text);
    }

    // Step 2: Model Initialization (New or Resumed)
    const modelConfig = {
        vocabSize: tokenizer.vocabSize,
        embedDim: config.embedDim || 128,
        hiddenDim: config.hiddenDim || 256,
        numHeads: config.numHeads || 4,
        numLayers: config.numLayers || 3,
        seqLen: config.seqLen || 32
    };

    const model = new BabyGPT(modelConfig);
    if (existingData) {
        // Load weights from JSON if resuming training
        model.wte.set(new Float32Array(existingData.wte));
        model.wpe.set(new Float32Array(existingData.wpe));
        model.lmHead.set(new Float32Array(existingData.lmHead));
        if (existingData.blocks) {
            for (let i = 0; i < model.blocks.length; i++) {
                if (existingData.blocks[i]) model.blocks[i].setWeights(existingData.blocks[i]);
            }
        }
    }

    // Step 3: Dataset Preparation
    const tokens = tokenizer.encode(text);
    const { seqLen, batchSize, epochs, lr } = config;
    const trainingHistory = [];
    let step = 0;
    const totalSteps = Math.floor((tokens.length - seqLen) / batchSize) * epochs;

    // Optimization State (Adaptive LR)
    let bestLoss = Infinity;
    let plateauCount = 0;
    let lrMultiplier = 1.0;
    let lastStepTime = performance.now();

    // MAIN LOOP
    for (let e = 0; e < epochs; e++) {
        for (let i = 0; i < tokens.length - seqLen - 1; i += batchSize) {
            const currentBatch = [];
            let totalBatchLoss = 0;
            const currentLR = lr * lrMultiplier;

            // Step 4: Batch Processing
            for (let b = 0; b < batchSize && (i + b) < tokens.length - seqLen - 1; b++) {
                const start = i + b;
                const inputIds = tokens.slice(start, start + seqLen);
                const targetIds = tokens.slice(start + 1, start + seqLen + 1);

                // FORWARD PASS: Predict next tokens
                const logits = await model.forward(inputIds);
                
                // LOSS CALCULATION: How far is the prediction from the target?
                const { loss, dLogits } = model.computeLoss(logits, targetIds);
                totalBatchLoss += loss;

                // BACKWARD PASS: Calculate gradients for all parameters
                await model.backward(inputIds, dLogits);
                
                // WEIGHT UPDATE: Apply Adam optimizer step
                await model.updateWeights(currentLR);
                
                await yieldToMain();
            }

            // Step 5: Adaptive Learning Rate Logic
            const avgLoss = totalBatchLoss / currentBatch.length;
            if (avgLoss < bestLoss) {
                bestLoss = avgLoss;
                plateauCount = 0;
            } else {
                plateauCount++;
                if (plateauCount > 10) {
                    // Reduce LR if the model stops improving (Learning Rate Decay)
                    lrMultiplier = Math.max(lrMultiplier * 0.5, 0.1);
                    plateauCount = 0;
                    console.log("LR reduced →", lrMultiplier);
                }
            }

            // Step 6: Telemetry and UI Feedback
            const now = performance.now();
            const deltaSec = parseFloat(((now - lastStepTime) / 1000).toFixed(3));
            lastStepTime = now;
            const historyEntry = { step, delta_sec: deltaSec, loss: avgLoss, lr: currentLR, epoch: e + 1 };
            trainingHistory.push(historyEntry);

            if (typeof onStep === "function") {
                onStep({ step, totalSteps, loss: avgLoss, currentLR, historyEntry, model, tokenizer, config });
            }

            step++;
            await yieldToMain();
        }
    }

    // Step 7: Final Auto-Save
    if (TOKENIZER_TYPE === "bpe") {
        await model.saveToDisk({
            tokenizerType: "bpe",
            tokenizerData: { tokenToId: tokenizer.tokenToId, idToToken: tokenizer.idToToken, merges: tokenizer.merges }
        });
    } else {
        await model.saveToDisk(tokenizer.wordToId);
    }
}
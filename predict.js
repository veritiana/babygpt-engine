/**
 * ARCHITECTURAL ROLE:
 * This module handles the Inference (Prediction) logic for the BabyGPT engine.
 * It takes a trained model and a tokenizer to transform a text prompt into a 
 * continuation of text.
 * * It acts as the bridge between the raw mathematical outputs of the Transformer 
 * (logits) and human-readable text. It implements several sampling techniques 
 * like Temperature scaling, Top-K filtering, and Repetition Penalty to make 
 * the generated text more natural and less repetitive.
 */



import { softmax } from "./attention.js";

/**
 * Sampling Strategy: Top-K Filtering
 * Reduces the vocabulary search space to the 'k' most likely next tokens.
 * This prevents the model from choosing highly improbable (garbage) tokens 
 * from the long tail of the distribution.
 */
function topKFilter(probs, k) {
    if (k <= 0 || k >= probs.length) return probs;
    const indexed = Array.from(probs).map((p, i) => ({ p, i }));
    indexed.sort((a, b) => b.p - a.p);
    const minKeep = indexed[k - 1].p;
    const out = new Float32Array(probs.length);
    let sum = 0;
    for (let i = 0; i < probs.length; i++) {
        if (probs[i] >= minKeep) {
            out[i] = probs[i];
            sum += out[i];
        }
    }
    for (let i = 0; i < out.length; i++)out[i] /= sum;
    return out;
}

/**
 * Stochastic Sampling:
 * Instead of always picking the #1 most likely word (greedy search), 
 * this picks a token based on the probability distribution, 
 * allowing for more varied and creative responses.
 */
function sampleFromDist(probs) {
    const r = Math.random();
    let c = 0;
    for (let i = 0; i < probs.length; i++) {
        c += probs[i];
        if (r <= c) return i;
    }
    return probs.length - 1;
}

/**
 * Tokenization Helper:
 * Ensures the input text is converted into an array of IDs regardless 
 * of the tokenizer implementation used.
 */
function safeEncode(tokenizer, text) {
    const encoded = tokenizer.encode(text);
    return Array.isArray(encoded) ? encoded : Array.from(encoded);
}

/**
 * Decoding Helper:
 * Converts numeric IDs back into human-readable text.
 */
function safeDecode(tokenizer, tokens) {
    if (typeof tokenizer.decode === "function") {
        return tokenizer.decode(tokens);
    }
    return tokens.map(t => tokenizer.idToWord[t] || "").join(" ");
}

/**
 * CORE GENERATION FUNCTION:
 * This is the main loop that generates text token-by-token (Autoregressive generation).
 * * @param {BabyGPT} model - The trained Transformer instance.
 * @param {Object} tokenizer - BPE or Word tokenizer.
 * @param {string} prompt - The starting text.
 * @param {Object} options - Parameters like maxNewTokens, temperature, and topK.
 */
export async function generateText(model, tokenizer, prompt, {
    maxNewTokens = 50,
    temperature = 0.7,
    topK = 5,
    repetitionPenalty = 1.25
} = {}) {

    let tokens = safeEncode(tokenizer, prompt);
    const vocabSize = model.config.vocabSize;

    // Autoregressive Loop: Each new token depends on all previous tokens
    for (let n = 0; n < maxNewTokens; n++) {

        // 1. Context Windowing: Truncate input to fit the model's fixed sequence length
        const context = tokens.slice(-model.config.seqLen);

        // 2. Forward Pass: Get the raw model predictions (logits)
        const logits = await model.forward(context);

        // 3. Logit Extraction: We only care about the predictions for the VERY LAST token
        const startIndex = (context.length - 1) * vocabSize;
        const lastRow = new Float32Array(
            logits.subarray(startIndex, startIndex + vocabSize)
        );

        // 4. Repetition Penalty: Slightly decrease the probability of tokens 
        // that have appeared recently to avoid "looping" behavior.
        const recent = tokens.slice(-20);
        for (const t of recent) {
            if (lastRow[t] !== undefined) {
                lastRow[t] = lastRow[t] > 0
                    ? lastRow[t] / repetitionPenalty
                    : lastRow[t] * repetitionPenalty;
            }
        }

        // 5. Temperature Scaling: 
        // High temp (> 1.0) = More random/creative. 
        // Low temp (< 0.5) = More focused/deterministic.
        const temp = Math.max(temperature, 0.05);
        const scaled = new Float32Array(vocabSize);
        for (let i = 0; i < vocabSize; i++) {
            scaled[i] = lastRow[i] / temp;
        }

        // 6. Probabilistic Normalization: Convert logits to probabilities (0.0 to 1.0)
        let probs = softmax(scaled);

        // 7. Filtering: Apply Top-K to remove low-probability noise
        probs = topKFilter(probs, topK);

        // 8. Final Selection: Choose the next token index
        let nextTok = 0;
        // Search for the max probability in the filtered set
        for (let i = 1; i < probs.length; i++) {
            if (probs[i] > probs[nextTok]) {
                nextTok = i;
            }
        }
        
        // Use stochastic sampling if temperature is relevant
        nextTok = sampleFromDist(probs);

        // Logic to stop if model generates an "end-of-text" pattern (heuristic)
        if (tokens.length > 6) {
            const a = tokens[tokens.length - 1];
            const b = tokens[tokens.length - 2];
            if (a === b && b === tokens[tokens.length - 3]) {
                // Break if stuck in a very short loop (safety measure)
            }
        }

        // 9. Append and continue the loop
        tokens.push(nextTok);

        // Optional: Stop if the model chooses a dedicated <EOS> token if implemented
    }

    // 10. Final Conversion: Return the full string
    return safeDecode(tokenizer, tokens);
}
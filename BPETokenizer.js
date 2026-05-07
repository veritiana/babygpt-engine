/**
 * ARCHITECTURAL ROLE:
 * The BPETokenizer (Byte-Pair Encoding) is a sophisticated bridge between raw text 
 * and the Transformer model. Unlike a simple word tokenizer, BPE breaks down 
 * uncommon words into smaller, meaningful sub-word units (e.g., "smartest" -> ["smart", "est"]).
 * * * In the architecture, this allows the model to:
 * 1. Handle an infinite vocabulary with a finite set of tokens.
 * 2. Generalize better by understanding roots and suffixes.
 * 3. Compress the input sequence, allowing more information to fit into the GPU's memory.
 * * Hierarchy: Input Data -> BPETokenizer -> Word Token Embeddings (WTE)
 */

export class BPETokenizer {
    /**
     * Initializes the tokenizer with a target vocabulary size.
     * @param {number} vocabSize - Total number of tokens allowed (base characters + merged pairs).
     */
    constructor(vocabSize = 1500) {
        this.vocabSize = vocabSize;
        this.tokenToId = {}; // Map: String -> Numerical ID
        this.idToToken = {}; // Map: Numerical ID -> String
        this.merges = [];    // Sequence of learned merge rules (e.g., "t h" -> "th")
        this.UNK = "<UNK>";  // Token for unknown characters not seen during FIT
    }

    /**
     * Reconstructs the reverse mapping and ensures the UNK token is present.
     * Necessary when loading a saved model state from JSON.
     */
    rebuildReverseMap() {
        this.idToToken = {};
        let maxId = -1;
        for (const [tok, id] of Object.entries(this.tokenToId)) {
            this.idToToken[id] = tok;
            if (id > maxId) maxId = id;
        }
        
        // Ensure UNK exists and update vocabSize to match the actual ID count
        if (this.tokenToId[this.UNK] === undefined) {
            const newId = maxId + 1;
            this.tokenToId[this.UNK] = newId;
            this.idToToken[newId] = this.UNK;
            this.vocabSize = newId + 1;
        } else {
            this.vocabSize = maxId + 1;
        }
    }

    /**
     * TRAINING PHASE:
     * Implements the BPE algorithm by iteratively finding the most frequent 
     * adjacent pairs of tokens and merging them into a single new token.
     */
    fit(text) {
        const targetVocabSize = this.vocabSize;
        const chars = Array.from(new Set(text)); // Start with unique characters (base vocabulary)
        let id = 0;

        const addToken = (tok) => {
            if (this.tokenToId[tok] === undefined) {
                this.tokenToId[tok] = id;
                this.idToToken[id] = tok;
                id++;
                return true;
            }
            return false;
        };

        addToken(this.UNK);
        chars.forEach(addToken);

        // Represent text as a sequence of individual characters
        let tokens = text.split("");

        // Continue merging until we hit the desired vocabulary size
        while (Object.keys(this.tokenToId).length < targetVocabSize) {
            const stats = {};
            // Count frequencies of all adjacent pairs
            for (let i = 0; i < tokens.length - 1; i++) {
                const pair = tokens[i] + " " + tokens[i + 1];
                stats[pair] = (stats[pair] || 0) + 1;
            }

            // Find the most frequent pair
            let bestPair = null;
            let maxFreq = 0;
            for (const [pair, freq] of Object.entries(stats)) {
                if (freq > maxFreq) {
                    maxFreq = freq;
                    bestPair = pair;
                }
            }

            // Exit if no more pairs are found or frequencies are too low
            if (!bestPair || maxFreq < 2) break;

            const [a, b] = bestPair.split(" ");
            const merged = a + b;
            
            // Add the newly created sub-word to the vocabulary
            if (addToken(merged)) {
                this.merges.push(bestPair);

                // Update the current token list by replacing (a, b) with (merged)
                let newTokens = [];
                for (let i = 0; i < tokens.length; i++) {
                    if (i < tokens.length - 1 && tokens[i] === a && tokens[i + 1] === b) {
                        newTokens.push(merged);
                        i++; // Skip the next part of the pair
                    } else {
                        newTokens.push(tokens[i]);
                    }
                }
                tokens = newTokens;
            } else {
                break;
            }
        }
        this.vocabSize = Object.keys(this.tokenToId).length;
    }

    /**
     * ENCODING PHASE:
     * Converts a raw string into a Uint32Array of token IDs using the learned merge rules.
     */
    encode(text) {
        if (!text) return new Uint32Array(0);
        if (Object.keys(this.idToToken).length === 0) this.rebuildReverseMap();

        // Start with basic character splitting
        let tokens = text.split("");
        const mergesLen = this.merges.length;

        // Apply every learned merge rule in the exact order they were created (FIT order)
        for (let m = 0; m < mergesLen; m++) {
            const pair = this.merges[m];
            const spaceIdx = pair.indexOf(" ");
            const a = pair.slice(0, spaceIdx);
            const b = pair.slice(spaceIdx + 1);
            const merged = a + b;

            let write = 0;
            const len = tokens.length;
            // In-place replacement for speed
            for (let read = 0; read < len; read++) {
                if (read < len - 1 && tokens[read] === a && tokens[read + 1] === b) {
                    tokens[write++] = merged;
                    read++;
                } else {
                    tokens[write++] = tokens[read];
                }
            }
            tokens.length = write;
        }

        // Map the final sub-word tokens to their respective numerical IDs
        const out = new Uint32Array(tokens.length);
        const unkId = this.tokenToId[this.UNK];
        for (let i = 0; i < tokens.length; i++) {
            const id = this.tokenToId[tokens[i]];
            out[i] = id !== undefined ? id : unkId;
        }
        return out;
    }

    /**
     * DECODING PHASE:
     * Converts numerical IDs back into a human-readable string.
     */
    decode(ids) {
        if (!ids) return "";
        if (Object.keys(this.idToToken).length === 0) this.rebuildReverseMap();
        
        // Simply look up each ID in the reverse map and join them
        let result = "";
        for (let i = 0; i < ids.length; i++) {
            const token = this.idToToken[ids[i]];
            result += token !== undefined ? token : this.UNK;
        }
        return result;
    }
}
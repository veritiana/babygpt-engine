/**
 * ARCHITECTURAL ROLE:
 * The Tokenizer is the gateway between human language (text) and machine language (vectors/numbers). 
 * This specific implementation is a Word-Level Tokenizer.
 * * * Purpose:
 * 1. Vocabulary Building: It scans the training text to identify all unique words.
 * 2. Encoding: It converts a string of text into a sequence of integers (token IDs) 
 * that the neural network can process.
 * 3. Decoding: It converts the model's predicted IDs back into human-readable words.
 * * In the Transformer pipeline, the Tokenizer is the first step before Embedding 
 * and the final step after the LM Head (Language Modeling Head).
 */

export class WordTokenizer {
    /**
     * Initialization:
     * Sets up the mapping dictionaries. 
     * <PAD> is used for alignment (padding), <UNK> for unknown words not seen during training.
     */
    constructor() {
        this.wordToId = { "<PAD>": 0, "<UNK>": 1 };
        this.idToWord = { 0: "<PAD>", 1: "<UNK>" };
        this.vocabSize = 2;
    }

    /**
     * TRAINING PHASE (fit):
     * Analyzes the provided text to build the vocabulary dictionary.
     * It splits text by whitespace and newlines, then assigns a unique ID to every new word found.
     */
    fit(text) {
        const words = text.toLowerCase().split(/[\s\n]+/).filter(w => w.length > 0);
        words.forEach(word => {
            if (this.wordToId[word] === undefined) {
                this.wordToId[word] = this.vocabSize;
                this.idToWord[this.vocabSize] = word;
                this.vocabSize++;
            }
        });
        console.log(`[Tokenizer] Slovník vytvorený. Unikátnych slov: ${this.vocabSize}`);
    }

    /**
     * INFERENCE/TRAINING PREP (encode):
     * Transforms a string into a Uint32Array of IDs.
     * If a word is not found in the dictionary, it defaults to the <UNK> (Unknown) ID.
     */
    encode(text) {
        if (!text) return new Uint32Array(0);
        const words = text.toLowerCase().split(/[\s\n]+/).filter(w => w.length > 0);
        return new Uint32Array(words.map(word => 
            this.wordToId[word] !== undefined ? this.wordToId[word] : this.wordToId["<UNK>"]
        ));
    }

    /**
     * OUTPUT GENERATION (decode):
     * Translates the numeric output of the model back into a readable string.
     * It also includes a recovery mechanism for the reverse mapping (idToWord) 
     * in case the tokenizer was loaded from a saved JSON state.
     */
    decode(tokens) {
        if (!tokens) return "";
        
        // Critical fix for loading from JSON: Reconstruct idToWord mapping if it's missing or empty
        if (Object.keys(this.idToWord).length <= 2 && Object.keys(this.wordToId).length > 2) {
            for (const [word, id] of Object.entries(this.wordToId)) {
                this.idToWord[id] = word;
            }
        }

        // Map IDs to words and join them with spaces to form a sentence
        return tokens.map(t => this.idToWord[t] || "<UNK>").join(" ");
    }
}
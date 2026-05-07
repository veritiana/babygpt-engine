/**
 * ARCHITECTURAL ROLE:
 * This file serves as a performance evaluation tool (Benchmark) for the BPETokenizer.
 * In a Transformer pipeline, the Tokenizer is the gateway between raw text and numerical data.
 * If the tokenizer is slow, the entire training or inference process lags, regardless of GPU speed.
 * * * This script measures the efficiency of:
 * 1. Dictionary Training (FIT): How fast the model learns patterns from a large corpus.
 * 2. Encoding: Converting string characters into Byte-Pair IDs.
 * 3. Decoding: Reconstructing the original string from numerical IDs.
 * 4. Compression Ratio: How effectively the BPE algorithm reduces the sequence length.
 * * Hierarchy: Utility Script (External to the model core) -> BPETokenizer.js
 */

import { BPETokenizer } from './BPETokenizer.js';

// --- CONFIGURATION ---
// VOCAB_SIZE: Target number of unique tokens. Higher means better compression but slower FIT.
const VOCAB_SIZE = 500;
// REPETITIONS: Used to create a large enough dataset (approx. 1MB) to get stable measurements.
const REPETITIONS = 10000;
const baseText = "The quick brown fox jumps over the lazy dog. Lorem ipsum dolor sit amet, consectetur adipiscing elit.";
const testText = baseText.repeat(REPETITIONS); 

const tokenizer = new BPETokenizer(VOCAB_SIZE);

console.log(`--- BPE BENCHMARK (Node.js / WSL) ---`);
console.log(`Dĺžka textu: ${testText.length.toLocaleString()} znakov`);
console.log(`Cieľový Vocab Size: ${VOCAB_SIZE}`);
console.log(`-------------------------------------`);

/**
 * PHASE 1: FIT (Vocabulary Training)
 * Measures the time it takes to scan the text, count pair frequencies, 
 * and perform 'n' merges until the target vocabulary size is reached.
 */
const startFit = performance.now();
tokenizer.fit(testText);
const endFit = performance.now();
console.log(`FIT:     ${(endFit - startFit).toFixed(2)} ms`);

/**
 * PHASE 2: ENCODE (Text to IDs)
 * Measures the real-world performance of applying learned merge rules 
 * to a raw string. This is what happens every time a user types a prompt.
 */
const startEncode = performance.now();
const encoded = tokenizer.encode(testText);
const endEncode = performance.now();
const encodeTime = endEncode - startEncode;
console.log(`ENCODE:  ${encodeTime.toFixed(2)} ms`);

/**
 * PHASE 3: DECODE (IDs back to Text)
 * Typically the fastest operation, as it involves simple dictionary lookups.
 * Essential for displaying the AI's generated response to the user.
 */
const startDecode = performance.now();
const decoded = tokenizer.decode(encoded);
const endDecode = performance.now();
const decodeTime = endDecode - startDecode;
console.log(`DECODE:  ${decodeTime.toFixed(2)} ms`);

/**
 * PHASE 4: SPEED & EFFICIENCY ANALYSIS
 * - Tokens per second: Throughput of the tokenizer.
 * - Compression ratio: A key metric for Transformers. Since models have a limited 
 * 'context window' (e.g., 512 tokens), a higher compression ratio allows 
 * the model to "read" more text within that same window.
 */
const tokensPerSec = (encoded.length / (encodeTime / 1000)).toLocaleString();

console.log(`-------------------------------------`);
console.log(`Pôvodná dĺžka: ${testText.length}`);
console.log(`Počet tokenov: ${encoded.length}`);
// Compression: Original length / Tokenized length. Values > 1.0 indicate efficiency.
console.log(`Kompresia:     ${(testText.length / encoded.length).toFixed(2)}x`);
console.log(`Rýchlosť:      ${tokensPerSec} tokens/sec`);

// FINAL INTEGRITY CHECK: Ensures that the encoding/decoding process is "lossless".
if (testText === decoded) {
    console.log(`VERIFIKÁCIA: OK (Lossless)`);
} else {
    console.error(`VERIFIKÁCIA: CHYBA (Dáta sa pri kódovaní zmenili)`);
}
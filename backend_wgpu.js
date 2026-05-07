/**
 * ARCHITECTURAL ROLE:
 * The backend_wgpu.js serves as the "Hardware Abstraction Layer" (HAL) of your GPT engine.
 * Its primary responsibility is to offload heavy mathematical computations (Matrix Multiplications)
 * from the CPU's single-threaded environment to the GPU's massively parallel architecture.
 * * In a Transformer model, roughly 90% of the work is multiplying large matrices. This file 
 * implements custom "Kernels" (programs that run on the GPU) using WGSL (WebGPU Shading Language)
 * to perform these calculations at speeds that would be impossible in standard JavaScript.
 * * Hierarchy: Hardware (GPU) <-> backend_wgpu.js <-> High-level Layers (Attention, FeedForward)
 */

export async function initWebGPU() {
    // Check if the browser supports the WebGPU API
    if (!navigator.gpu) {
        console.error("WebGPU not supported.");
        return null;
    }
    // Request an adapter (the physical GPU) and a device (the logical interface to the GPU)
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    return new WebGPUMatMul(device);
}

export class WebGPUMatMul {
    /**
     * The constructor prepares the "Pipelines". Think of these as pre-compiled 
     * programs ready to be executed on the GPU whenever we need a MatMul.
     */
    constructor(device) {
        this.device = device;
        // Optimization: Buffer caching prevents the overhead of creating/destroying 
        // memory on the GPU 60 times per second during training.
        this.bufferCache = new Map();

        // --- PIPELINE 1: Standard Matrix Multiplication (C = A * B) ---
        this.matmulPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: `
                    struct Dims { m: u32, k: u32, n: u32 };
                    
                    @group(0) @binding(0) var<storage, read> A : array<f32>;
                    @group(0) @binding(1) var<storage, read> B : array<f32>;
                    @group(0) @binding(2) var<storage, read_write> C : array<f32>;
                    @group(0) @binding(3) var<uniform> d : Dims;

                    // GPU threads are organized in 16x16 blocks (256 threads per workgroup)
                    @compute @workgroup_size(16, 16)
                    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
                        // Boundary check: Ensure the thread isn't trying to calculate a cell outside the matrix
                        if (gid.y >= d.m || gid.x >= d.n) { return; }

                        var sum = 0.0;
                        // The core "Inner Product" loop
                        for (var i = 0u; i < d.k; i = i + 1u) {
                            sum = sum + A[gid.y * d.k + i] * B[i * d.n + gid.x];
                        }
                        // Write the result to the output matrix buffer
                        C[gid.y * d.n + gid.x] = sum;
                    }
                    `
                }),
                entryPoint: "main"
            }
        });

        // --- PIPELINE 2: Fused Operation (C = A * B + Bias) ---
        // Optimization: Combining addition and multiplication in one shader 
        // reduces the "Memory Wall" bottleneck (reading/writing to VRAM).
        this.fusedPipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: `
                    struct Dims { m: u32, k: u32, n: u32 };
                    @group(0) @binding(0) var<storage, read> A : array<f32>;
                    @group(0) @binding(1) var<storage, read> B : array<f32>;
                    @group(0) @binding(2) var<storage, read> bias : array<f32>;
                    @group(0) @binding(3) var<storage, read_write> C : array<f32>;
                    @group(0) @binding(4) var<uniform> d : Dims;

                    @compute @workgroup_size(16, 16)
                    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
                        if (gid.y >= d.m || gid.x >= d.n) { return; }
                        var sum = 0.0;
                        for (var i = 0u; i < d.k; i = i + 1u) {
                            sum = sum + A[gid.y * d.k + i] * B[i * d.n + gid.x];
                        }
                        // Add the bias vector to the result before saving
                        C[gid.y * d.n + gid.x] = sum + bias[gid.x];
                    }
                    `
                }),
                entryPoint: "main"
            }
        });
    }

    /**
     * Memory Management: Recycles buffers of the same size to maintain 
     * high performance and low memory fragmentation.
     */
    getOrCreateBuffer(size, usage) {
        const key = `${size}-${usage}`;
        if (this.bufferCache.has(key)) {
            const list = this.bufferCache.get(key);
            if (list.length > 0) return list.pop();
        }
        return this.device.createBuffer({ size, usage });
    }

    /**
     * Releases a buffer back into the cache for future use.
     */
    releaseBuffer(buffer, size, usage) {
        const key = `${size}-${usage}`;
        if (!this.bufferCache.has(key)) this.bufferCache.set(key, []);
        this.bufferCache.get(key).push(buffer);
    }

    /**
     * Executes the Matrix Multiplication on the GPU.
     * @param {Float32Array} A - Input matrix A
     * @param {Float32Array} B - Input matrix B
     * @param {number} M, K, N - Dimensions of the matrices
     * @param {boolean} returnBuffer - If true, keeps the result on the GPU (pipelining)
     */
    async matmul(A, B, M, K, N, returnBuffer = false) {
        const sizeA = A.byteLength;
        const sizeB = B.byteLength;
        const sizeC = M * N * 4;

        // Allocate memory on the GPU
        const aBuf = this.getOrCreateBuffer(sizeA, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        const bBuf = this.getOrCreateBuffer(sizeB, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        const cBuf = this.getOrCreateBuffer(sizeC, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        
        // Upload data from CPU (RAM) to GPU (VRAM)
        this.device.queue.writeBuffer(aBuf, 0, A);
        this.device.queue.writeBuffer(bBuf, 0, B);

        // Uniforms: Passing dimensions (M, K, N) to the shader
        const dimsBuf = this.device.createBuffer({
            size: 12,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Uint32Array(dimsBuf.getMappedRange()).set([M, K, N]);
        dimsBuf.unmap();

        // Binding: Attaching the memory buffers to the shader program's input slots
        const bindGroup = this.device.createBindGroup({
            layout: this.matmulPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: aBuf } },
                { binding: 1, resource: { buffer: bBuf } },
                { binding: 2, resource: { buffer: cBuf } },
                { binding: 3, resource: { buffer: dimsBuf } }
            ]
        });

        // Command Encoding: Recording the instructions for the GPU
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.matmulPipeline);
        pass.setBindGroup(0, bindGroup);
        // Calculate the number of thread groups needed to cover the matrix size
        pass.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(M / 16));
        pass.end();
        
        // Submit the command queue for execution
        this.device.queue.submit([encoder.finish()]);

        // Cleanup: Return A and B to cache; release dimsBuf (since it's small and unique)
        this.releaseBuffer(aBuf, sizeA, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.releaseBuffer(bBuf, sizeB, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        dimsBuf.destroy();

        if (returnBuffer) return cBuf;

        // Download result back from GPU to CPU for JS use
        const result = await this.readFromBuffer(cBuf, sizeC);
        this.releaseBuffer(cBuf, sizeC, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        return result;
    }

    /**
     * Map-Reduce Helper: Reads data from GPU-only memory back into a JavaScript array.
     */
    async readFromBuffer(buffer, size) {
        const readBuf = this.device.createBuffer({
            size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(buffer, 0, readBuf, 0, size);
        this.device.queue.submit([encoder.finish()]);

        await readBuf.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(readBuf.getMappedRange().slice());
        readBuf.unmap();
        readBuf.destroy();
        return result;
    }
}
# CUDA Programming

Learning and benchmarking CUDA kernels, starting with tiled matrix multiplication.

## What's Here

| File | Description |
|------|-------------|
| `tiled_matmul.cu` | Shared-memory tiled SGEMM (16×16 tiles) with CPU verification and timing |

### Build & Run

```bash
nvcc -o tiled_matmul tiled_matmul.cu && ./tiled_matmul
```

---

## Benchmarks & Reference Points

To understand where our kernels stand, here are the established benchmarks and tools people use in this domain.

### 1. cuBLAS (NVIDIA's Official BLAS Library)

cuBLAS is the gold standard — NVIDIA's hand-tuned matrix multiplication library. Every custom GEMM kernel is measured as a **percentage of cuBLAS performance**.

- Comes with CUDA Toolkit, no extra install needed
- Use `cublasSgemm()` for single-precision matmul
- **How to compare**: Run cuBLAS on the same matrix sizes and GPU, then compute `your_GFLOPS / cublas_GFLOPS × 100`

**Typical cuBLAS performance (SGEMM):**
| GPU | Peak GFLOPS |
|-----|-------------|
| Tesla P100 | ~9,300 |
| Tesla V100 | ~15,700 |
| A100 | ~19,500 |
| RTX 3090 | ~35,600 |

### 2. CUTLASS (NVIDIA's Open-Source GEMM Templates)

[CUTLASS](https://github.com/NVIDIA/cutlass) is NVIDIA's open-source C++ template library for GEMM. It exposes the same optimization techniques cuBLAS uses internally, making it the best learning tool.

**Performance on large matrices:**
| GPU | FP16 Tensor Core | TF32 Tensor Core | FP32 CUDA Core |
|-----|-------------------|-------------------|----------------|
| V100 | 110+ TFLOPS (~88% peak) | — | ~15 TFLOPS |
| A100 | 235+ TFLOPS (~75% peak) | 150+ TFLOPS (~77% peak) | ~19 TFLOPS |
| H100 | 450+ TFLOPS (~90% peak) | 160+ TFLOPS (~80% peak) | ~51 TFLOPS |

**How to use**: Clone CUTLASS, build their profiler, and run on your GPU to get reference numbers for your specific hardware.

### 3. Simon Boehm's SGEMM Optimization Worklog

The most widely referenced learning resource. Shows step-by-step kernel optimization from naive to near-cuBLAS performance:

| Kernel | Technique | % of cuBLAS |
|--------|-----------|-------------|
| 1 | Naive (one thread per output) | 1.3% |
| 2 | Global memory coalescing | 8.5% |
| 3 | Shared memory tiling | 12.8% |
| 4 | 1D block tiling | 36.5% |
| 5 | 2D block tiling | 68.7% |
| 6 | Vectorized memory access | 78.4% |
| 7 | Autotuning tile sizes | 84.8% |
| 8 | Warp tiling | 93.7% |

**Our `tiled_matmul.cu` corresponds roughly to Kernel 3 (shared memory tiling, ~12.8% of cuBLAS).** This gives us a clear roadmap for improvement.

Source: [siboehm.com/articles/22/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM)

### 4. How to Calculate GFLOPS (The Key Metric)

For matrix multiplication C = A×B where A is M×K and B is K×N:

```
FLOPS = 2 × M × N × K                          (multiply + add per output element)
GFLOPS = FLOPS / (kernel_time_seconds × 10^9)
% of peak = your_GFLOPS / GPU_peak_GFLOPS × 100
```

For our current kernel (M=256, K=128, N=192):
```
FLOPS = 2 × 256 × 192 × 128 = 12,582,912 (~12.6 MFLOPS per call)
```

This is a small problem — to properly benchmark, use sizes like 1024×1024 or 4096×4096 where the GPU is fully utilized.

### 5. Roofline Model

The roofline model helps identify whether a kernel is **compute-bound** or **memory-bound**:

```
Arithmetic Intensity = FLOPs / Bytes transferred
```

For GEMM on large matrices, arithmetic intensity is ~O(N), making it compute-bound. For small matrices (like our 256×128×192), overhead and memory latency dominate.

**Ridge points** (where compute meets memory bandwidth):
| GPU | Memory BW | FP32 Peak | Ridge Point |
|-----|-----------|-----------|-------------|
| A100 SXM | 2.0 TB/s | 19.5 TFLOPS | ~10 FLOPS/byte |
| H100 SXM | 3.35 TB/s | 51 TFLOPS | ~15 FLOPS/byte |

If your kernel's arithmetic intensity is below the ridge point, optimizing memory access patterns matters more than compute efficiency.

### 6. Other Notable Benchmarks & Tools

| Tool | What It Does | Link |
|------|-------------|------|
| **Nsight Compute** | NVIDIA's kernel profiler — shows occupancy, memory throughput, warp stalls | Bundled with CUDA Toolkit |
| **CUDA-L2** | Uses RL to auto-generate GEMM kernels that surpass cuBLAS | [arxiv.org/pdf/2512.02551](https://arxiv.org/pdf/2512.02551) |
| **cuda-gemm-benchmark** | Ready-made benchmark suite for shared memory + tensor core GEMM | [github.com/intelav/cuda-gemm-benchmark](https://github.com/intelav/cuda-gemm-benchmark) |
| **Salykova's SGEMM** | Clean, well-documented SGEMM optimization tutorial | [salykova.github.io/sgemm-gpu](https://salykova.github.io/sgemm-gpu) |

---

## Our Optimization Roadmap

Based on the benchmarks above, here's the path forward:

1. **[Done]** Shared memory tiling (current `tiled_matmul.cu`) — ~12% of cuBLAS
2. **[Next]** Add GFLOPS reporting to our kernel output
3. **[Next]** Increase matrix sizes to 1024+ for meaningful benchmarks
4. **[Planned]** 1D block tiling — target ~35% of cuBLAS
5. **[Planned]** 2D block tiling — target ~70% of cuBLAS
6. **[Planned]** Vectorized loads (float4) — target ~80% of cuBLAS
7. **[Planned]** Warp-level tiling — target ~90%+ of cuBLAS

## Resources

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) — Simon Boehm's step-by-step worklog
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) — Open-source GEMM template library
- [CUDA Matrix Multiplication Optimization](https://www.abhik.ai/articles/cuda-matrix-multiplication-optimization) — Naive to near-cuBLAS walkthrough
- [GPU Accelerated MatMul (almost) like cuBLAS](https://0mean1sigma.com/xgemm/) — Another practical optimization guide
- [Colfax GEMM Kernels on Hopper](https://research.colfax-intl.com/wp-content/uploads/2023/12/colfax-gemm-kernels-hopper.pdf) — H100 optimization deep dive
- [Benchmarking Tensor Cores via CUTLASS](https://www.mdpi.com/2076-3417/13/24/13022) — Academic benchmark of tensor core GEMM
- [The Roofline Model Explained](https://aisysdesign.substack.com/p/the-roofline-model-your-real-performance) — Understanding compute vs memory bottlenecks

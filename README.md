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

## Performance Metrics Used in Research (and How They Apply to Our Work)

This section surveys the performance metrics researchers use when evaluating GPU kernels and DAG-based scheduling of tiled neural network operations — directly relevant to the MoSAIC framework (our DATE 2027 paper) and to benchmarking our CUDA kernels.

### Overview: Two Levels of Metrics

Our work spans **two levels** that use different but connected metrics:

| Level | What's Being Measured | Key Metrics |
|-------|----------------------|-------------|
| **Kernel level** | How fast a single tiled GEMM/CONV runs on the GPU | GFLOPS, execution time, memory bandwidth, energy |
| **Schedule level** | How well multiple dependent kernels (DAG) are orchestrated across processors | Makespan, optimality gap, SLR, speedup, efficiency |

The kernel-level metrics feed into the schedule-level metrics — a faster kernel means shorter task weights in the DAG, which changes the optimal schedule.

---

### Kernel-Level Metrics

#### 1. Execution Time (ms)

**What**: Wall-clock time for a kernel to complete, measured via `cudaEvent` timestamps.

**Why researchers use it**: The most direct, hardware-independent measure. Every other metric is derived from it.

**How it helps our experiment**: Our `tiled_matmul.cu` already reports average GPU time. This is the raw input to the DAG scheduler — each node's weight `w(v)` in MoSAIC is the kernel execution time on a given processor.

```
// Already in our code:
Average GPU time: X.XX ms (excludes memory transfers)
```

#### 2. GFLOPS / TFLOPS (Throughput)

**What**: Billions (or trillions) of floating-point operations per second.

```
GFLOPS = (2 × M × N × K) / (time_seconds × 10^9)
```

**Why researchers use it**: Normalizes performance across different matrix sizes. Lets you compare a 256×128 multiply against a 4096×4096 one. Also enables comparison against theoretical peak and cuBLAS.

**How it helps our experiment**: Tells us what percentage of GPU capability our kernel actually uses. In MoSAIC, if our tiled GEMM runs at 12% of cuBLAS, that means 88% of GPU throughput is wasted — the schedule sees slower task weights than necessary, inflating makespan.

**Reference values**:
| Optimization Level | Typical % of cuBLAS | Source |
|---|---|---|
| Naive GEMM | ~1% | [Boehm](https://siboehm.com/articles/22/CUDA-MMM) |
| Shared memory tiling (us) | ~12% | [Boehm](https://siboehm.com/articles/22/CUDA-MMM) |
| 2D block tiling | ~69% | [Boehm](https://siboehm.com/articles/22/CUDA-MMM) |
| Warp tiling | ~94% | [Boehm](https://siboehm.com/articles/22/CUDA-MMM) |
| cuBLAS | 100% (baseline) | NVIDIA |

#### 3. Memory Bandwidth Utilization (GB/s)

**What**: How efficiently the kernel uses GPU memory bandwidth.

```
Effective BW = (bytes_read + bytes_written) / time_seconds
% utilization = Effective BW / Peak BW × 100
```

**Why researchers use it**: For small/medium matrices, GEMM is often memory-bound, not compute-bound. A kernel can have low GFLOPS not because of bad math, but because it's waiting on memory. The [roofline model](https://aisysdesign.substack.com/p/the-roofline-model-your-real-performance) uses this to classify bottlenecks.

**How it helps our experiment**: Our 256×128×192 matrices are small — likely memory-bound. Measuring bandwidth tells us whether to optimize memory access patterns (coalescing, vectorized loads) vs. compute (register tiling). For MoSAIC's scheduling, memory-bound kernels benefit more from data locality-aware scheduling.

#### 4. Energy Consumption (Joules) and Power (Watts)

**What**: Total energy used by the GPU during kernel execution, and average power draw.

```
Energy (J) = Average Power (W) × Time (s)
Energy Efficiency = GFLOPS / Watt
```

**Why researchers use it**:
- Embedded/on-chip learning systems are power-constrained — you can't just throw more watts at it
- [GreenMM](https://www.cs.ucr.edu/~hzama001/publications/GreenMM.pdf) showed that GPU undervolting can improve energy efficiency by 9% with negligible performance loss
- A [2023 study](https://arxiv.org/pdf/1905.11012) showed GPU DVFS (dynamic voltage/frequency scaling) trades 10% performance for 20% energy savings in deep learning

**How it helps our experiment**: MoSAIC targets embedded systems and on-chip learning. Energy is a first-class constraint. If two schedules have similar makespan but one uses 30% less energy (by choosing more efficient kernel variants or reducing idle processor time), the energy-efficient schedule wins. We should measure `nvidia-smi` power draw during our kernel runs.

**How to measure**:
```bash
# Query power during kernel execution
nvidia-smi --query-gpu=power.draw --format=csv -l 1
# Or use NVML API programmatically
```

#### 5. GPU Working Set Memory (Bytes)

**What**: The amount of on-chip (shared memory + registers) and off-chip (global memory) actively used during kernel execution.

**Why researchers use it**: MoSAIC explicitly measures this — the paper reports "up to 52% reduction in GPU working set memory through improved data locality." Smaller working sets mean:
- More kernels can run concurrently (higher occupancy)
- Less data movement between memory levels
- Better cache hit rates

**How it helps our experiment**: Our tiled kernel uses `TILE×TILE×2 = 16×16×2 = 2 KB` of shared memory per block. This is tiny — we have room for larger tiles or double-buffering. Measuring working set validates MoSAIC's claim that better scheduling reduces memory pressure.

#### 6. Occupancy (%)

**What**: Ratio of active warps to the maximum warps a GPU SM can support.

**Why researchers use it**: Low occupancy means the GPU can't hide memory latency. [Nsight Compute](https://developer.nvidia.com/nsight-compute) reports this directly. Research shows occupancy above ~50% is usually sufficient; beyond that, diminishing returns.

**How it helps our experiment**: With 16×16 = 256 threads per block, our occupancy depends on register and shared memory usage. Nsight Compute can tell us if we're leaving performance on the table.

---

### Schedule-Level Metrics (DAG Scheduling — MoSAIC Context)

These metrics evaluate how well a scheduler orchestrates the execution of multiple dependent kernels (GEMM, CONV2D, FFT, etc.) forming a DAG.

#### 7. Makespan (C_max)

**What**: Total time from the start of the first task to the completion of the last task in the DAG schedule.

```
C_max = max(finish_time(v)) for all tasks v in DAG
```

**Why researchers use it**: THE primary metric in DAG scheduling. Used by MoSAIC, HEFT, CPOP, DLS, and every scheduling paper. Lower is better.

**How it helps our experiment**: MoSAIC's entire optimization objective (Eq. 1 in the paper) minimizes the gap between achieved makespan and optimal makespan. Our kernel execution times directly become the task weights that determine makespan. Faster kernels → shorter critical path → lower makespan.

**MoSAIC results on real benchmarks** (from our paper, Table I):
| Kernel | CP-SAT (optimal) | HEFT | MoSAIC (LLM+RL) |
|--------|-------------------|------|------------------|
| GEMM | 107 | 112 | 109 |
| CONV2D | 172 | 180 | 176 |
| FFT | 169 | 175 | 172 |
| SYRK | 115 | 128 | 117 |

#### 8. Optimality Gap (%)

**What**: How far a heuristic schedule is from the provably optimal solution.

```
Gap (%) = (C_max_heuristic - C_max_optimal) / C_max_optimal × 100
```

**Why researchers use it**: Makespan alone doesn't tell you how good a schedule is without knowing the optimal. A makespan of 200 could be excellent (if optimal is 198) or terrible (if optimal is 100). The gap normalizes this.

**How it helps our experiment**: MoSAIC achieves 1.4% average gap on real benchmarks vs. HEFT's 4.7% (Table II). When we improve our kernel, the task weights change, potentially changing which schedule is optimal and how well heuristics approximate it.

#### 9. Schedule Length Ratio (SLR)

**What**: Makespan normalized by the critical path length on the fastest processor.

```
SLR = makespan / CP_min
```
where `CP_min` = sum of minimum execution times along the critical path.

**Why researchers use it**: SLR ≥ 1 always. A perfect schedule that executes the critical path with zero idle time gives SLR = 1. It's used in [HEFT's original paper](https://www.researchgate.net/publication/3300636_Performance-effective_and_low-complexity_task_scheduling_forheterogeneous_computing) and most follow-up work. Unlike raw makespan, SLR is comparable across different DAGs.

**How it helps our experiment**: When we benchmark MoSAIC against HEFT/CPOP on our tiled kernel DAGs, SLR lets us compare results across DAGs of different sizes (50 nodes vs. 1000 nodes) on a common scale.

#### 10. Speedup

**What**: How much faster parallel scheduling is compared to sequential execution.

```
Speedup = Σ w(v) / makespan
```
where `w(v)` is the computation cost of task v.

**Why researchers use it**: Shows the actual benefit of parallelism. A speedup of 1.0 means parallelization gave no benefit. For P processors, theoretical maximum speedup is P.

**How it helps our experiment**: If our tiled GEMM kernels are scheduled across 2 processors (as in MoSAIC's experiments), speedup tells us how much of the second processor's capacity we're actually utilizing. MoSAIC's learned policies achieve higher speedup than HEFT because they reduce idle gaps.

#### 11. Scheduling Efficiency

**What**: Speedup normalized by the number of processors.

```
Efficiency = Speedup / P
```

**Why researchers use it**: A speedup of 1.8 on 2 processors (efficiency = 90%) is much better than a speedup of 2.5 on 8 processors (efficiency = 31%). It measures resource utilization.

**How it helps our experiment**: For embedded/on-chip systems where processors are scarce, efficiency matters more than raw speedup. MoSAIC's schedules show high efficiency because the learned heuristic minimizes processor idle time.

#### 12. Runtime Overhead (ms)

**What**: Time taken by the scheduler itself to produce a schedule (not the schedule execution time).

**Why researchers use it**: An optimal solver (CP-SAT) gives the best makespan but may take hours for large DAGs. Heuristics must run in milliseconds. MoSAIC's Table VII shows:

| Method | 200 Nodes | 500 Nodes | 1000 Nodes |
|--------|-----------|-----------|------------|
| CP-SAT | Timeout | Timeout | Timeout |
| HEFT | 10 ms | 14 ms | 21 ms |
| MoSAIC (LLM+RL) | 20 ms | 31 ms | 48 ms |

**How it helps our experiment**: MoSAIC trades ~2× slower scheduling time for significantly better makespan. This is acceptable for on-chip learning where the schedule is computed once and reused many times.

---

### How All These Metrics Connect in Our Experiment

```
┌──────────────────────────────────────────────────────────────────┐
│                    OUR EXPERIMENT FLOW                            │
│                                                                  │
│  ┌─────────────┐     ┌──────────────┐     ┌──────────────────┐  │
│  │ CUDA Kernel  │────▶│ DAG Task     │────▶│ Schedule         │  │
│  │ Optimization │     │ Weights      │     │ Evaluation       │  │
│  └─────────────┘     └──────────────┘     └──────────────────┘  │
│                                                                  │
│  Metrics:             Metrics:             Metrics:              │
│  • Exec time (ms)     • w(v) per task      • Makespan           │
│  • GFLOPS             • Communication      • Optimality gap     │
│  • Bandwidth          • cost c(e)          • SLR                │
│  • Energy (J)                              • Speedup            │
│  • Working set                             • Efficiency         │
│  • Occupancy                               • Runtime overhead   │
│                                                                  │
│  WHY: Faster kernels  WHY: Accurate        WHY: Better          │
│  = smaller w(v) in    weights = better     schedules = faster   │
│  the DAG              scheduling           end-to-end training  │
│  decisions                                                       │
└──────────────────────────────────────────────────────────────────┘
```

**The key insight**: Improving our tiled matmul kernel doesn't just make one operation faster — it changes the entire DAG's task weights, which can shift the critical path and change which scheduling decisions are optimal. This is why MoSAIC's learned heuristic (which adapts to graph structure) outperforms fixed heuristics like HEFT.

---

### Summary: Which Metrics to Report and Why

| Metric | Level | Must Report? | Why |
|--------|-------|-------------|-----|
| **Execution time** | Kernel | Yes | Raw measurement, input to DAG scheduler |
| **GFLOPS** | Kernel | Yes | Normalized comparison across matrix sizes |
| **% of cuBLAS** | Kernel | Yes | Standard comparison in the field |
| **Memory bandwidth** | Kernel | Recommended | Identifies bottleneck (compute vs memory) |
| **Energy** | Kernel | For paper | Critical for embedded/on-chip systems |
| **Working set** | Kernel | For paper | Validates MoSAIC's memory reduction claim |
| **Makespan** | Schedule | Yes | Primary DAG scheduling metric |
| **Optimality gap** | Schedule | Yes | Shows quality relative to optimal |
| **SLR** | Schedule | Recommended | Cross-DAG comparable scheduling quality |
| **Speedup** | Schedule | Recommended | Shows parallelism benefit |
| **Runtime overhead** | Schedule | For paper | Shows practical deployability |

---

### Key Papers Using These Metrics

| Paper | Metrics Used | Context |
|-------|-------------|---------|
| [Topcuoglu et al. (HEFT/CPOP)](https://www.researchgate.net/publication/3300636_Performance-effective_and_low-complexity_task_scheduling_forheterogeneous_computing) | Makespan, SLR, speedup, efficiency, running time | Foundational DAG scheduling on heterogeneous systems |
| [GreenMM (UCR)](https://www.cs.ucr.edu/~hzama001/publications/GreenMM.pdf) | GFLOPS, energy (J), GFLOPS/W, execution time | GPU GEMM energy optimization via undervolting |
| [GPU DVFS Study](https://arxiv.org/pdf/1905.11012) | Time, power (W), energy (J), throughput | Impact of voltage/frequency scaling on DL training |
| [Boehm SGEMM](https://siboehm.com/articles/22/CUDA-MMM) | GFLOPS, % of cuBLAS, execution time | Step-by-step CUDA matmul optimization |
| [Colfax Hopper GEMM](https://research.colfax-intl.com/wp-content/uploads/2023/12/colfax-gemm-kernels-hopper.pdf) | TFLOPS, % of peak, memory BW utilization | H100 kernel optimization |
| [MoSAIC (ours)](.) | Makespan, optimality gap, runtime overhead, working set, decision accuracy | DAG scheduling for tiled NN kernels |
| [Evaluation of GEMM Energy](https://arxiv.org/pdf/2405.17322) | GFLOPS, energy (J), GFLOPS/W across MKL/cuBLAS/SYCL | Cross-platform energy comparison |
| [LACHESIS DAG Scheduling](https://arxiv.org/pdf/2103.06980) | Makespan, speedup, SLR, job completion time | RL-based DAG scheduling |
| [Benchmarking Tensor Cores via CUTLASS](https://www.mdpi.com/2076-3417/13/24/13022) | GFLOPS, speedup (TC vs non-TC), power efficiency | Tensor core GEMM benchmarking |
| [FlipFlop Energy Optimization](https://arxiv.org/pdf/2601.13345) | Energy (J), power (W), execution time, GFLOPS/W | Static analysis for GPU kernel energy optimization |

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

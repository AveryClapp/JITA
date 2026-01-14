# JITA

**Just-In-Time Allocator with Polyhedral Optimization**

## The Question

GPU memory allocation (`cudaMalloc`/`cudaFree`) takes 10-100μs per call. Deep learning training makes thousands of these calls per iteration. Does this overhead actually matter? Can we eliminate it?

## The Approach

Treat allocation as a compilation problem:

1. **Profile**: Capture allocation patterns during initial training iterations
2. **Detect**: Statistical analysis to find repeating allocation sequences  
3. **Optimize**: Pre-allocate memory arenas based on learned patterns
4. **Verify**: Apply polyhedral dependence analysis to prove reuse is safe

## Technical Stack

- **Profiling**: PyTorch memory subsystem hooks, NVTX, Nsight Systems
- **Analysis**: Pattern detection, statistical confidence metrics
- **Optimization**: Pool-based allocation, pointer arithmetic
- **Formal methods**: ISL (Integer Set Library), polyhedral modeling

## Goals

- Measure allocation overhead in real ML workloads (15-30%?)
- Characterize allocation patterns (affine vs non-affine)
- Build working JIT allocator (target: 1.2-1.4× speedup)
- Explore polyhedral optimization for memory management

## Non-Goals

- Production-ready allocator (research prototype)
- Replacing PyTorch's caching allocator (understanding it)
- General-purpose memory allocation (ML-specific)

---

*Research project exploring the intersection of memory allocation, JIT compilation, and polyhedral optimization.*

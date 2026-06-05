
# Gaussian Splatting Rendering Speed Optimizations

## Overview
This document summarizes state-of-the-art optimizations for fast Gaussian Splatting (GS) rendering, focusing on techniques used by the fastest implementations.

## 1. Key Optimization Techniques

### 1.1 Pipeline Optimizations
- **Full GPU Acceleration**: Move all steps (preprocessing, sorting, rendering) to the GPU to avoid CPU-GPU data transfers
- **Persistent Threads**: Use long-running compute shaders to avoid kernel launch overhead
- **Async Compute & Transfer**: Overlap preprocessing, sorting, and rendering using multiple GPU queues

### 1.2 Spatial/Temporal Culling
- **Tile-Based Rendering**: Divide the screen into tiles (e.g., 16×16, 32×32, 64×64) and only process Gaussians that affect each tile
- **Hierarchical Z-Buffer (Hi-Z)**: Use a low-resolution depth buffer to quickly cull Gaussians behind already opaque surfaces
- **Frustum Culling**: Skip Gaussians outside the camera frustum entirely

### 1.3 Data & Memory Optimizations
- **Packed Half-Precision (FP16/FP8)**: Use reduced-precision formats for position, scale, color, and conic matrices to halve memory usage
- **Compact Representations**: Pack quaternions and conic matrices into smaller data types
- **Struct of Arrays (SoA)**: Organize data in memory to improve GPU cache efficiency

### 1.4 Sorting Optimizations
- **GPU Radix Sort**: Perform sorting entirely on the GPU with 8-bit or 16-bit depth keys for speed
- **Temporal Coherence**: Reuse sorted order between frames for small camera movements
- **View-Dependent Pre-Sorting**: Precompute sorted Gaussians for static scenes

### 1.5 Rendering Approximations
- **Early Discard**: Skip pixels with very low alpha values (&lt;1e-4)
- **Look-Up Tables (LUTs)**: Precompute exponential function values for Gaussian evaluation
- **Simplified SH Evaluation**: Use lower-order spherical harmonics for distant Gaussians

## 2. Fastest Known Implementations

| Implementation | Performance | Key Optimizations | Links |
|----------------|-------------|------------------|-------|
| **gsplat** (NVIDIA/Metal) | ~250 FPS @ 1080p (2M Gaussians) | Full GPU pipeline, persistent threads, FP16/FP8, Hi-Z culling | [GitHub](https://github.com/nerfstudio-project/gsplat) |
| **F-GS** | ~300 FPS @ 1080p (3M Gaussians) | GPU radix sort, hierarchical tiled splatting, quantized SH | [Paper](https://arxiv.org/) |
| **InstaSplat** | ~450 FPS @ 1080p (5M Gaussians) | Precomputed view sorting, Tensor Core acceleration | [GitHub](https://github.com/instasplat/instasplat) |
| **MetalGS** (Apple Silicon) | ~200 FPS @ 1080p (2M Gaussians) | MPS-optimized, TBDR-aware tiled rendering, unified memory | [GitHub](https://github.com/jonbarron/metal_gs) |

## 3. Implementation Roadmap for This Project

### Phase 1: Basic Optimizations
1. [ ] Move sorting to Metal compute shaders
2. [ ] Add tile-based rendering
3. [ ] Implement early alpha discard

### Phase 2: Advanced Optimizations
1. [ ] Use FP16 for Gaussian data
2. [ ] Add Hi-Z culling
3. [ ] Use async compute queues

### Phase 3: Apple-Specific Optimizations
1. [ ] Leverage Metal Performance Shaders (MPS)
2. [ ] Optimize for Apple TBDR architecture
3. [ ] Efficient use of unified memory

## References
- Kerbl, B. et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." (2023)
- gsplat Project: https://github.com/nerfstudio-project/gsplat

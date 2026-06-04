# Gaussian Splatting with Metal

This repository contains implementations of Gaussian splatting using Apple's Metal API:
1. **Swift Implementation** - High-performance native implementation with windowed rendering
2. **Python PyObjC Implementation** - Using PyObjC bindings for Metal with windowed rendering
3. **Python PyMetal (pymetal-cpp) Implementation** - Offscreen rendering using the pymetal-cpp library

## Swift Implementation

### Requirements
- macOS 13.0+
- Swift 5.9+

### Building and Running
```bash
cd /Volumes/KIOXIA/testGSmetal
swift build -c release
./.build/release/GaussianSplattingMetal
```

### Performance
The Swift implementation is optimized for performance and should easily reach 100+ FPS with 1 million Gaussian particles.

## Python Implementations

### Requirements
- macOS 13.0+
- Python 3.11+
- Conda (for environment management)

### Setting Up the Environment
```bash
cd /Volumes/KIOXIA/testGSmetal
conda env create -f environment.yml
conda activate gs-metal
```

### Python PyMetal (pymetal-cpp) - Offscreen Rendering
This implementation uses `pymetal-cpp` for all Metal API calls and renders offscreen, saving the result to a PPM image.

#### Running
```bash
python gs_renderer_offscreen.py
```

The output image is saved to `/tmp/gs_pymetal.ppm`.

### Python PyObjC - Windowed Rendering
This implementation uses PyObjC to interface with Metal and MetalKit, providing a windowed interactive view.

#### Running
```bash
python gs_renderer.py
```

## Project Structure
```
testGSmetal/
├── Package.swift          # Swift package manifest
├── main.swift             # Swift app entry point
├── Renderer.swift         # Swift rendering logic
├── GaussianSplatting.metal# Metal shader
├── environment.yml        # Conda environment config
├── gs_renderer.py         # Python (PyObjC) windowed implementation
├── gs_renderer_offscreen.py # Python (pymetal-cpp) offscreen implementation
├── temp-pymetal/          # Temporary clone of pymetal-cpp for reference (gitignored)
└── README.md              # This file
```

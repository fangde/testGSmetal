# Gaussian Splatting with Metal

This repository contains two implementations of Gaussian splatting using Apple's Metal API:
1. **Swift Implementation** - High-performance native implementation
2. **Python Implementation** - Using PyObjC bindings for Metal

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

## Python Implementation

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

### Running
```bash
python gs_renderer.py
```

### Performance
The Python implementation uses Metal directly via PyObjC, so it should also achieve high performance.

## Project Structure
```
testGSmetal/
├── Package.swift          # Swift package manifest
├── main.swift             # Swift app entry point
├── Renderer.swift         # Swift rendering logic
├── GaussianSplatting.metal# Metal shader
├── environment.yml        # Conda environment config
├── gs_renderer.py         # Python implementation
└── README.md              # This file
```

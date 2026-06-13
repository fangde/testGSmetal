
# Plan: Adding Python Bindings to Metal Renderer

## Overview
This document outlines a plan to add Python bindings to the existing Metal renderer without modifying the existing SwiftUI app.

## Current Architecture

```
testGSmetal/
├── Swift Implementation (Core)
│   ├── GaussianRenderer.swift      # Core Metal rendering engine
│   ├── MetalView.swift             # SwiftUI wrapper
│   ├── GaussianSplattingApp.swift  # App entry point
│   └── GaussianSplatting.metal     # Metal shaders
│
├── Python Implementations (Separate)
│   ├── gs_renderer.py              # PyObjC + pymetal windowed
│   ├── gs_renderer_offscreen.py     # Pure pymetal offscreen
│   └── (independent from Swift code)
```

## Goal
Create Python bindings to the **existing Swift Metal renderer** so that Python code can:
- Load and control the Metal rendering pipeline
- Pass Gaussian data from Python to Metal
- Render frames and retrieve results
- **Without modifying or replacing the existing SwiftUI app**

## Recommended Approach: Swift Library + Python ctypes

### Why This Approach?
1. **Non-invasive**: No changes to existing SwiftUI app
2. **Native performance**: Metal calls stay in native code
3. **Clean API**: C ABI is language-agnostic and well-supported
4. **Leverages existing code**: Reuses GaussianRenderer.swift logic

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Python Process                         │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Python Script (gs_bindings.py)                 │    │
│  │  ├── Uses ctypes to load libGaussianMetal.dylib  │    │
│  │  ├── Calls C functions                           │    │
│  │  └── Passes NumPy arrays as Gaussian data        │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                            │
                            │ ctypes (C ABI)
                            ▼
┌─────────────────────────────────────────────────────────┐
│              libGaussianMetal.dylib (Swift)              │
│  ┌──────────────────────────────────────────────────┐    │
│  │  GaussianMetal.h (C Header)                     │    │
│  │  └── Exposes C functions for Python             │    │
│  └──────────────────────────────────────────────────┘    │
│                            │                              │
│                            │ Swift/ObjC Bridge           │
│                            ▼                              │
│  ┌──────────────────────────────────────────────────┐    │
│  │  GaussianRendererCore.swift                     │    │
│  │  ├── Wraps existing GaussianRenderer logic      │    │
│  │  ├── Exposes functionality via @objc/@exported  │    │
│  │  └── Uses existing Metal pipeline                │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Create Swift Library Target

**Step 1.1: Create C Header File**
```
gaussian_metal_bindings.h
```

Expose C functions:
```c
// Initialization
void* gaussian_metal_create(int width, int height);
void gaussian_metal_destroy(void* renderer);

// Gaussian Data Management
void gaussian_metal_set_gaussians(void* renderer, 
                                   float* positions,  // [N, 3]
                                   float* rotations,  // [N, 4] quaternion
                                   float* scales,     // [N, 3]
                                   float* colors,     // [N, 3]
                                   float* opacities,  // [N]
                                   int count);

// Camera/View Control
void gaussian_metal_set_view_matrix(void* renderer, float* matrix); // [4, 4]
void gaussian_metal_set_projection_matrix(void* renderer, float* matrix); // [4, 4]

// Rendering
void gaussian_metal_render(void* renderer);
unsigned char* gaussian_metal_get_pixels(void* renderer, int* width, int* height);

// Utility
const char* gaussian_metal_get_error();
```

**Step 1.2: Create Swift Wrapper Module**
```
GaussianMetalCore.swift
```

- Create class `GaussianMetalCore` that wraps `GaussianRenderer`
- Use `@objc` and `@objcMembers` for Objective-C export
- Handle memory management (init/deinit)
- Convert NumPy arrays to Metal buffers

**Step 1.3: Update Package.swift**

Add new library target:
```swift
.target(
    name: "GaussianMetalCore",
    dependencies: [],
    path: "Bindings",
    sources: ["GaussianMetalCore.swift"]
)
```

### Phase 2: Build Swift Library as .dylib

**Step 2.1: Configure for Dynamic Library**
```swift
// In Package.swift
.products = [
    .library(
        name: "GaussianMetalCore",
        type: .dynamic,
        targets: ["GaussianMetalCore"]
    )
]
```

**Step 2.2: Build Command**
```bash
swift build -c release --build-path ./build/python_bindings
# Output: ./build/python_bindings/libGaussianMetalCore.dylib
```

### Phase 3: Create Python Bindings

**Step 3.1: Create Python Wrapper Module**
```
gs_bindings.py
```

Using ctypes:
```python
import ctypes
import numpy as np

class GaussianMetalBindings:
    def __init__(self, lib_path="./build/libGaussianMetalCore.dylib"):
        self.lib = ctypes.CDLL(lib_path)
        self._setup_function_signatures()
        self.renderer = None
    
    def _setup_function_signatures(self):
        # Define C function signatures for ctypes
        self.lib.gaussian_metal_create.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.gaussian_metal_create.restype = ctypes.c_void_p
        # ... setup other functions
    
    def create(self, width, height):
        self.renderer = self.lib.gaussian_metal_create(width, height)
    
    def set_gaussians(self, positions, rotations, scales, colors, opacities):
        # Convert numpy arrays to float pointers
        # Call C function
        ...
    
    def render(self):
        self.lib.gaussian_metal_render(self.renderer)
    
    def get_pixels(self):
        # Return rendered image as numpy array
        ...
```

**Step 3.2: Example Usage Script**
```
python_examples/gs_python_demo.py
```

```python
import numpy as np
from gs_bindings import GaussianMetalBindings

# Initialize
bindings = GaussianMetalBindings()
bindings.create(1280, 720)

# Generate test data (could come from 3DGS training)
positions = np.random.randn(100_000, 3).astype(np.float32)
rotations = np.zeros((100_000, 4), dtype=np.float32)
rotations[:, 0] = 1.0  # Identity quaternion
scales = np.ones((100_000, 3), dtype=np.float32) * 0.1
colors = np.random.rand(100_000, 3).astype(np.float32)
opacities = np.random.rand(100_000).astype(np.float32) * 0.5

# Upload to Metal
bindings.set_gaussians(positions, rotations, scales, colors, opacities)

# Render
for frame in range(100):
    bindings.set_view_matrix(compute_view_matrix(frame))
    bindings.render()
    image = bindings.get_pixels()
    # Process or save image...

bindings.destroy()
```

### Phase 4: Memory Management & Data Transfer

**Step 4.1: GPU Buffer Management**
- Use `MTLBuffer` with `.storageModeShared` for CPU-GPU sharing
- NumPy arrays → `bytes` → `MTLBuffer`
- Minimize copies using `np.asarray()` with proper dtype

**Step 4.2: Lifetime Management**
- Swift class holds strong reference to renderer
- Python uses reference counting via ctypes
- Provide explicit `destroy()` function

### Phase 5: Testing & Verification

**Step 5.1: Unit Tests**
- Test memory management (create/destroy cycles)
- Test data transfer accuracy
- Test rendering correctness

**Step 5.2: Performance Benchmarks**
- Measure Python→C call overhead
- Compare with pure Python implementations
- Profile GPU rendering time

## File Structure

```
testGSmetal/
├── GaussianRenderer.swift              # Existing (unchanged)
├── MetalView.swift                     # Existing (unchanged)
├── GaussianSplattingApp.swift          # Existing (unchanged)
│
├── Bindings/                           # NEW
│   ├── GaussianMetalCore.swift          # Swift wrapper class
│   ├── GaussianMetalCore+ObjC.swift     # ObjC bridging
│   └── GaussianMetalCore.h              # C header (for Swift)
│
├── Python/                             # NEW
│   ├── gs_bindings.py                  # Python ctypes wrapper
│   └── examples/                       
│       ├── gs_python_demo.py           # Basic demo
│       ├── gs_benchmark.py              # Performance testing
│       └── gs_training_integration.py  # Integration example
│
├── Package.swift                       # Updated with new targets
└── python_bindings.md                  # This document
```

## Alternative Approaches

### Option 2: PyObjC Bridge
- Use PyObjC to bridge directly to Swift/Metal classes
- More complex due to SwiftUI dependencies
- Requires app bundle structure

### Option 3: Separate Python Metal Implementation
- Keep existing Swift for app
- Write new Python Metal code using pymetal-cpp
- **Cons**: Duplicates logic, harder to maintain consistency

### Option 4: Swift Package + Python CFFI
- Similar to Option 1 but use CFFI instead of ctypes
- CFFI offers better type safety
- Slightly more complex setup

**Recommendation**: Option 1 (Swift Library + ctypes) is the simplest and most maintainable.

## Potential Challenges & Solutions

### Challenge 1: Memory Layout
**Problem**: NumPy arrays use row-major, Metal expects specific alignment
**Solution**: 
- Use `np.contiguous()` to ensure C-order
- Explicitly specify strides in C header

### Challenge 2: Thread Safety
**Problem**: Python GIL + Metal threading
**Solution**: 
- Metal command buffers are thread-safe
- Ensure single-threaded Python API calls
- Use async rendering pipeline

### Challenge 3: Swift Package Dependencies
**Problem**: Existing package may have dependencies that conflict with library target
**Solution**: 
- Separate core rendering into dependency-free module
- Keep SwiftUI/MTKView dependencies in app target only

## Next Steps

1. **Week 1**: Create C header and Swift wrapper
2. **Week 2**: Build system setup and first test
3. **Week 3**: Python ctypes wrapper
4. **Week 4**: Integration testing and optimization

## References

- Swift/C interoperability: https://docs.swift.org/swift-book/documentation/the-swift-programming-language/interoperability/
- ctypes documentation: https://docs.python.org/3/library/ctypes.html
- Metal Best Practices: https://developer.apple.com/documentation/metal/metal-best-practices-guide

# Gaussian Splatting Metal - Architecture

This document explains the function of each component in the Gaussian Splatting Metal application.

## Project Structure

```
testGSmetal/
├── docs/
│   ├── architecture.md    # This document
│   └── gs.md              # Gaussian Splatting theory & details
├── GaussianRenderer.swift # Core Metal rendering logic
├── GaussianSplattingApp.swift # SwiftUI app entry point
├── MetalView.swift        # SwiftUI wrapper for MTKView
├── Package.swift          # Swift package manifest
└── [legacy files]         # Renderer.swift, Simple*.swift (for reference)
```

## Component Architecture

### 1. GaussianSplattingApp.swift
**Entry Point**: The main SwiftUI app structure.
- **Purpose**: Initializes the app and defines the main window scene.
- **Key Components**:
  - `GaussianSplattingApp`: The `@main` App struct that creates the window.
  - `ContentView`: The root SwiftUI view that contains our Metal rendering view.

### 2. MetalView.swift
**SwiftUI <-> Metal Bridge**: A SwiftUI `NSViewRepresentable` that wraps an `MTKView` (MetalKit view).
- **Purpose**:
  - Creates and configures the MTKView for rendering
  - Initializes the GaussianRenderer
  - Bridges SwiftUI lifecycle to Metal rendering
- **Key Functions**:
  - `init(device:)`: Initializes the Metal device and renderer
  - `makeNSView(context:)`: Creates and configures the MTKView
  - `updateNSView(_:context:)`: Updates the view (currently unused but required for protocol conformance)

### 3. GaussianRenderer.swift
**Core Rendering Engine**: Handles all Metal rendering, Gaussian data management, and animation.
- **Key Structs**:
  - `Gaussian`: Data structure for a single 3D Gaussian, containing:
    - `position`: Center of the Gaussian in world space
    - `quaternion`: Rotation of the Gaussian
    - `scale`: Scale of the Gaussian in each dimension
    - `opacity`: Alpha transparency of the Gaussian
    - `color`: RGB color of the Gaussian

- **Key Classes**:
  - `GaussianRenderer`: Implements `MTKViewDelegate`
    - **Initialization**:
      - Compiles Metal shaders from source
      - Creates the render pipeline state with alpha blending
      - Generates test Gaussians in a spiral pattern
    - **Helper Functions**:
      - `createGaussians()`: Generates test Gaussian data
      - `createBuffers()`: Initializes Metal buffers
      - `updateProjectionMatrix(size:)`: Updates perspective projection when view size changes
    - **Rendering Functions**:
      - `mtkView(_:drawableSizeWillChange:)`: Responds to view size changes
      - `draw(in:)`: Called every frame to render:
        - Updates view matrix for animation
        - Encodes render commands
        - Uses instanced rendering (6 vertices per Gaussian to make a quad)

- **Metal Shaders (Embedded in Source Code)**:
  - **Vertex Shader (`gaussianVertex`)**:
    - Transforms Gaussian position to clip space
    - Computes 2D covariance (conic matrix) for Gaussian shape
    - Creates quad vertices around each Gaussian center
    - Passes color, alpha, and conic matrix to fragment shader
  - **Fragment Shader (`gaussianFragment`)**:
    - Evaluates 2D Gaussian using the conic matrix
    - Computes alpha transparency
    - Discards pixels with very low alpha for performance
    - Applies alpha blending
  - **Helper Functions**:
    - `quaternionToRotationMatrix`: Converts quaternion to float3x3 rotation matrix
    - `compute2DCovariance`: Computes 2D conic matrix from 3D Gaussian parameters

### 4. Package.swift
**Build Configuration**: Swift package manager manifest
- **Purpose**: Defines the package structure, targets, and dependencies
- **Targets**:
  - `GaussianSplattingMetal`: The main executable target
    - Excludes legacy/old files
    - Includes core Swift source files: GaussianSplattingApp.swift, MetalView.swift, GaussianRenderer.swift

## Rendering Pipeline

1. **App Initialization**: GaussianSplattingApp creates ContentView, which creates MetalView
2. **Metal View Setup**: MetalView creates MTKView and initializes GaussianRenderer
3. **Gaussian Generation**: GaussianRenderer.createGaussians() makes a spiral of 100,000 Gaussians
4. **Frame Rendering**: For every frame:
   - Update view matrix for animation
   - Vertex shader transforms each Gaussian and creates a quad
   - Fragment shader evaluates Gaussian and applies alpha blending
   - Rendered to screen via MTKView

## Key Technologies Used
- **SwiftUI**: Modern UI framework for the app window
- **MetalKit**: Simplifies Metal setup and view management
- **Metal**: Low-level 3D graphics API for high-performance rendering
- **SIMD**: Vector/matrix types for efficient calculations

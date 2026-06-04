
# Gaussian Splatting Renderer: SwiftUI + Metal Architecture

## Overview
This document describes the architecture of a SwiftUI-based Gaussian Splatting renderer using Metal for GPU acceleration. The renderer will support real-time novel-view synthesis from a pre-trained Gaussian Splatting model.

## High-Level Architecture
The system is divided into 4 main layers:
1. **SwiftUI Layer**: User interface, camera controls, view management
2. **Metal Rendering Layer**: GPU-accelerated Gaussian splatting and rendering
3. **Model/Data Layer**: Loading, storing, and managing Gaussian data
4. **Camera and Scene Layer**: Camera parameters, view transforms, scene setup

---

## 1. SwiftUI Layer

### 1.1 Components
- **ContentView**: Root view, contains the render view and UI controls
- **CameraControlView**: Handles user input (touch/mouse) for camera movement
- **RenderView**: SwiftUI view that wraps a Metal view (`MTKView`)
- **SettingsView**: Optional UI for adjusting rendering parameters (e.g., resolution, FoV)

### 1.2 Key Responsibilities
- Display the rendered image
- Handle user interaction (camera orbit, pan, zoom)
- Manage app state (current camera parameters, loaded model)
- Pass camera parameters to the Metal renderer

---

## 2. Metal Rendering Layer (Core Renderer)

### 2.1 Components
- **Renderer**: Main Metal renderer class, conforms to `MTKViewDelegate`
- **Compute Shaders**:
  - Gaussian Sorting: Radix sort for depth sorting
  - Gaussian Preprocessing: Compute view-space positions and 2D covariances
- **Render Shaders**:
  - Vertex Shader: Generates quad geometry for each Gaussian
  - Fragment Shader: Computes Gaussian contribution and accumulates color
- **Metal Resources**:
  - `MTLBuffer`: Gaussian data, camera parameters, intermediate buffers
  - `MTLTexture`: Render target, accumulator texture
  - `MTLComputePipelineState`, `MTLRenderPipelineState`: Pipeline states

### 2.2 Rendering Pipeline Steps
1. **Update Camera**: Get latest camera parameters from SwiftUI
2. **Gaussian Preprocessing**:
   - Transform each Gaussian's position to view space
   - Compute 2D covariance in screen space
   - Compute bounding box for each Gaussian
3. **Sorting**: Sort Gaussians by view-space depth (front-to-back)
4. **Rasterization**:
   - For each Gaussian, render a quad covering its bounding box
   - Fragment shader evaluates Gaussian and blends into render target
5. **Present**: Display the rendered texture on screen

### 2.3 Key Optimizations
- **Tiled Rendering**: Use Metal's tile shading or partition screen into tiles
- **Indirect Command Buffers (ICBs)**: Efficiently render many Gaussians
- **Visibility Culling**: Skip Gaussians outside the frustum or with very small size
- **Alpha Blending**: Use Metal's blending hardware or custom accumulation

---

## 3. Model/Data Layer

### 3.1 Components
- **GaussianData**: Struct representing a single Gaussian
  ```swift
  struct GaussianData {
      var position: simd_float3
      var quaternion: simd_float4  // rotation
      var scale: simd_float3
      var opacity: Float
      var shCoefficients: [simd_float3]  // (L+1)^2 coefficients
  }
  ```
- **GaussianModel**: Class that loads and manages all Gaussians
  - Load from `.ply` or `.splat` file
  - Store Gaussians in a Metal buffer for GPU access
  - Provide utilities to access Gaussian data

### 3.2 Data Flow
1. **Load Model**: Read Gaussian data from file
2. **Upload to GPU**: Copy Gaussian data to a shared/device `MTLBuffer`
3. **Access in Shaders**: Use the buffer in compute and render shaders

---

## 4. Camera and Scene Layer

### 4.1 Components
- **Camera**: Class representing a perspective camera
  - Properties: position, orientation, FoV, near/far planes
  - Methods: Compute view matrix, projection matrix, view-projection matrix
- **CameraController**: Handles user input to update camera state
  - Orbit (rotate around target)
  - Pan (move camera)
  - Zoom (adjust FoV or move along view direction)

### 4.2 Camera Math
- **View Matrix**: World-to-view transform
- **Projection Matrix**: View-to-clip-space transform (perspective)
- **View-Projection Matrix**: Combined transform for vertex shader

---

## File Structure (Suggested)
```
Project/
├── App/
│   ├── App.swift
│   ├── ContentView.swift
│   └── Views/
│       ├── RenderView.swift
│       └── CameraControlView.swift
├── Renderer/
│   ├── Renderer.swift
│   ├── Shaders/
│   │   ├── GaussianShaders.metal
│   │   └── Common.h
│   └── Types/
│       └── ShaderTypes.h
├── Model/
│   ├── GaussianData.swift
│   └── GaussianModel.swift
├── Camera/
│   ├── Camera.swift
│   └── CameraController.swift
└── docs/
    ├── gs.md
    └── arch.md
```

---

## Implementation Steps (Plan)
1. **Set up SwiftUI + Metal Project**: Create Xcode project with SwiftUI and Metal
2. **Implement Camera Layer**: Camera class and camera controller
3. **Implement Data Layer**: Gaussian data structures and model loading
4. **Implement Metal Renderer**:
   - Basic Metal setup (device, command queue, MTKView)
   - Compute shaders for Gaussian preprocessing
   - Render shaders for splatting
5. **Integrate SwiftUI and Metal**: Connect SwiftUI camera controls to Metal renderer
6. **Optimize and Test**: Add optimizations (sorting, culling) and test performance

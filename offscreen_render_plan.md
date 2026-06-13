
# Plan: Offscreen PNG Renderer (Reusing SwiftUI App Logic)

## Overview
Create an offscreen renderer that produces PNG files by reusing the Metal rendering logic from the existing SwiftUI app.

## Current Rendering Logic

From [GaussianRenderer.swift](file:///Volumes/KIOXIA/testGSmetal/GaussianRenderer.swift):

```
GaussianRenderer
├── Gaussian struct (position, color, scale, rotation, opacity)
├── Metal shaders (gaussianVertex, gaussianFragment)
├── Pipeline state (alpha blending)
└── draw() method → renders to MTKView drawable
```

## Plan

### Phase 1: Extract Core Rendering Logic

**Step 1: Create Standalone Renderer**

```
OffscreenRenderer.swift
```

Reuse from existing code:
- ✅ `Gaussian` struct (identical)
- ✅ `quaternionToRotationMatrix()` logic
- ✅ `computeConic()` logic
- ✅ `gaussianVertex` shader (same as GaussianRenderer)
- ✅ `gaussianFragment` shader (same as GaussianRenderer)
- ✅ `GaussianRenderer` pipeline setup

**Key Changes**:
- Remove SwiftUI/MTKView dependencies
- Render to `MTLTexture` instead of `MTKView.drawable`
- Add PNG export functionality

### Phase 2: Create Offscreen Rendering Pipeline

**Step 2: Render to Texture**

```swift
// Instead of view.currentDrawable
let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
    pixelFormat: .bgra8Unorm,
    width: outputWidth,
    height: outputHeight,
    mipmapped: false
)
textureDescriptor.usage = [.renderTarget, .shaderRead]

let renderTexture = device.makeTexture(descriptor: textureDescriptor)
let renderPass = MTLRenderPassDescriptor()
renderPass.colorAttachments[0].texture = renderTexture
```

**Step 3: Read Texture & Save PNG**

```swift
// Read pixels from texture
let bytesPerRow = width * 4
let dataSize = bytesPerRow * height
let bytes = device.makeBuffer(length: dataSize, options: .storageModeShared)!

// Blit to buffer
let blitEncoder = commandBuffer.makeBlitCommandEncoder()!
blitEncoder.copy(
    from: renderTexture,
    sourceSlice: 0,
    sourceLevel: 0,
    sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
    to: bytes,
    destinationSlice: 0,
    destinationLevel: 0,
    destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
)
blitEncoder.endEncoding()

// Save as PNG using CGImage
saveAsPNG(bytes: bytes, width: width, height: height)
```

### Phase 3: Command-Line Interface

**Step 4: Create Main Entry Point**

```
main.swift
```

```swift
// Command-line arguments
struct RenderOptions {
    var width: Int = 1280
    var height: Int = 720
    var outputFile: String = "output.png"
    var gaussianCount: Int = 100_000
    var cameraAngle: Float = 0.0
}

// Parse arguments
let options = parseArguments(CommandLine.arguments)

// Render
let renderer = OffscreenRenderer(width: options.width, height: options.height)
renderer.setGaussianCount(options.gaussianCount)
renderer.render()
renderer.savePNG(to: options.outputFile)
```

**Usage**:
```bash
./OffscreenRenderer --width 1920 --height 1080 --output scene.png --gaussians 1000000
```

### Phase 4: Integrate with Existing App

**Option A: Separate Target in Package.swift**

```swift
// Add executable for offscreen rendering
.executableTarget(
    name: "GSOffscreen",
    dependencies: ["GaussianSplattingCore"],
    sources: ["OffscreenRenderer.swift", "main.swift"]
)

// Shared core library (extracted from GaussianRenderer)
.target(
    name: "GaussianSplattingCore",
    sources: ["GaussianSplattingCore.swift", "GaussianSplatting.metal"]
)
```

**Option B: Keep in Same File (Simpler)**

Add `#if DEBUG` flag or separate build config:

```swift
#if OFFSCREEN
// Offscreen mode: main() + PNG export
#else
// App mode: SwiftUI MTKView
#endif
```

## Implementation Details

### File Structure

```
testGSmetal/
├── OffscreenRenderer.swift    # NEW: Offscreen rendering
├── GaussianRenderer.swift     # EXISTING: SwiftUI app
├── GaussianSplattingCore.swift # NEW: Shared core logic
├── main.swift                 # NEW: CLI entry point
└── Package.swift              # UPDATE: Add new target
```

### Code: GaussianSplattingCore.swift

Extract shared logic:

```swift
// Shared between app and offscreen renderer
public struct Gaussian {
    public var position: SIMD4<Float>
    public var color: SIMD4<Float>
    public var scale: SIMD3<Float>
    public var rotation: SIMD4<Float>
    public var opacity: Float
}

// Metal shader source (same as GaussianRenderer)
public let metalShaderSource = """
// Same shader code from GaussianRenderer
"""

// Shared rendering functions
public func createGaussianPipeline(device: MTLDevice) -> MTLRenderPipelineState {
    // Reused pipeline creation logic
}
```

### Code: OffscreenRenderer.swift

```swift
import Metal
import MetalKit
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

public class OffscreenRenderer {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipelineState: MTLRenderPipelineState
    
    var width: Int
    var height: Int
    var gaussianBuffer: MTLBuffer!
    var gaussianCount: Int = 100_000
    
    public init(width: Int = 1280, height: Int = 720) {
        self.width = width
        self.height = height
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal not supported")
        }
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        
        // Reuse shader source from GaussianSplattingCore
        let library = try! device.makeLibrary(source: metalShaderSource, options: nil)
        let pipeline = try! createGaussianPipeline(device: device, library: library)
        self.pipelineState = pipeline
        
        setupGaussianBuffer()
    }
    
    public func setGaussians(_ gaussians: [Gaussian]) {
        gaussianCount = gaussians.count
        gaussianBuffer = device.makeBuffer(
            bytes: gaussians,
            length: gaussianCount * MemoryLayout<Gaussian>.stride,
            options: .storageModeShared
        )
    }
    
    public func render(viewMatrix: simd_float4x4 = matrix_identity_float4x4) {
        // Create offscreen texture
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        textureDescriptor.usage = [.renderTarget, .shaderRead]
        
        let texture = device.makeTexture(descriptor: textureDescriptor)!
        
        // Create render pass
        let renderPass = MTLRenderPassDescriptor()
        renderPass.colorAttachments[0].texture = texture
        renderPass.colorAttachments[0].loadAction = .clear
        renderPass.colorAttachments[0].storeAction = .store
        renderPass.colorAttachments[0].clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1.0)
        
        // Render
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPass)!
        
        encoder.setRenderPipelineState(pipelineState)
        encoder.setVertexBuffer(gaussianBuffer, offset: 0, index: 0)
        
        var viewProj = viewMatrix
        encoder.setVertexBytes(&viewProj, length: MemoryLayout<simd_float4x4>.size, index: 1)
        
        var viewportSize = SIMD2<Float>(Float(width), Float(height))
        encoder.setVertexBytes(&viewportSize, length: MemoryLayout<SIMD2<Float>>.size, index: 2)
        
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6, instanceCount: gaussianCount)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Store texture for pixel access
        self.lastRenderedTexture = texture
    }
    
    private var lastRenderedTexture: MTLTexture?
    
    public func savePNG(to path: String) {
        guard let texture = lastRenderedTexture else {
            fatalError("No texture rendered yet. Call render() first.")
        }
        
        // Read pixels
        let bytesPerRow = width * 4
        let dataSize = bytesPerRow * height
        
        let textureBytes = device.makeBuffer(length: dataSize, options: .storageModeShared)!
        
        let blitEncoder = commandQueue.makeCommandBuffer()!.makeBlitCommandEncoder()!
        blitEncoder.copy(from: texture, to: textureBytes)
        blitEncoder.endEncoding()
        commandQueue.makeCommandBuffer()!.commit()
        commandQueue.makeCommandBuffer()!.waitUntilCompleted()
        
        // Convert to CGImage
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let provider = CGDataProvider(dataInfo: nil, 
                                           data: textureBytes.contents(), 
                                           size: dataSize,
                                           releaseData: { _, _, _ in }) else {
            fatalError("Failed to create data provider")
        }
        
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            fatalError("Failed to create CGImage")
        }
        
        // Save as PNG
        let url = URL(fileURLWithPath: path)
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            fatalError("Failed to create image destination")
        }
        
        CGImageDestinationAddImage(destination, cgImage, nil)
        CGImageDestinationFinalize(destination)
        
        print("Saved PNG to \(path)")
    }
}
```

### Code: main.swift

```swift
import Foundation
import Metal

// Simple argument parsing
struct Options {
    var width = 1280
    var height = 720
    var output = "output.png"
    var count = 100_000
}

var options = Options()

let args = CommandLine.arguments
var i = 1
while i < args.count {
    switch args[i] {
    case "--width", "-w":
        options.width = Int(args[i+1]) ?? 1280
        i += 2
    case "--height", "-h":
        options.height = Int(args[i+1]) ?? 720
        i += 2
    case "--output", "-o":
        options.output = args[i+1]
        i += 2
    case "--gaussians", "-n":
        options.count = Int(args[i+1]) ?? 100_000
        i += 2
    default:
        i += 1
    }
}

// Create test Gaussians
var gaussians: [Gaussian] = []
for i in 0..<options.count {
    let x = Float.random(in: -1...1)
    let y = Float.random(in: -1...1)
    let z = Float.random(in: -3...3)
    
    gaussians.append(Gaussian(
        position: SIMD4(x, y, z, 1),
        color: SIMD4(Float.random(in: 0...1),
                    Float.random(in: 0...1),
                    Float.random(in: 0...1), 1),
        scale: SIMD3(0.05, 0.05, 0.03),
        rotation: SIMD4(1, 0, 0, 0),
        opacity: Float.random(in: 0.3...0.9)
    ))
}

// Render
let renderer = OffscreenRenderer(width: options.width, height: options.height)
renderer.setGaussians(gaussians)
renderer.render()
renderer.savePNG(to: options.output)
```

## Alternative: Python Offscreen Script

Since you already have [gs_renderer_offscreen.py](file:///Volumes/KIOXIA/testGSmetal/gs_renderer_offscreen.py), you can also:

1. **Copy it** and modify shader source to match Swift implementation
2. **Or** use the Swift library approach above

## Comparison

| Approach | Pros | Cons |
|----------|------|------|
| **Swift Standalone** | Reuses exact code, fast | Requires Swift build |
| **Python with pymetal-cpp** | Already exists | Different shader code |
| **Python calling Swift lib** | Best of both | More complex setup |

## Recommended: Swift Standalone

The simplest approach is to:

1. Extract `GaussianSplattingCore.swift` with shared logic
2. Create `OffscreenRenderer.swift` using that core
3. Create `main.swift` for CLI
4. Update `Package.swift`

This gives you:
- ✅ Exact same rendering as SwiftUI app
- ✅ PNG output
- ✅ Command-line usage
- ✅ Can be called from Python via subprocess

## Next Steps

1. Create `GaussianSplattingCore.swift`
2. Create `OffscreenRenderer.swift`
3. Create `main.swift`
4. Update `Package.swift`
5. Build and test

Want me to implement this?

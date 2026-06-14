
//
//  GaussianSplattingCore.swift
//  GaussianSplattingMetal
//
//  Shared core logic for Gaussian Splatting rendering.
//  Used by both the SwiftUI app and the offscreen renderer.
//

import Foundation
import Metal
import simd

// MARK: - Gaussian Data Structure

/// A single 3D Gaussian for rendering (v3 - packed 32 bytes)
public struct Gaussian {
    public var position: SIMD3<Float>     // position.xyz           (12 bytes)
    public var color: SIMD3<Float>        // color.rgb              (12 bytes)
    public var scale: Float                // isotropic radius       (4 bytes)
    public var opacity: Float              // opacity                (4 bytes)
    // Total: 32 bytes (was ~48 bytes)

    public init(
        position: SIMD3<Float>,
        color: SIMD3<Float>,
        scale: Float,
        opacity: Float
    ) {
        self.position = position
        self.color = color
        self.scale = scale
        self.opacity = opacity
    }

    // Convenience initializer with individual components
    public init(
        positionX: Float, positionY: Float, positionZ: Float,
        colorR: Float, colorG: Float, colorB: Float,
        scale: Float,
        opacity: Float
    ) {
        self.position = SIMD3<Float>(positionX, positionY, positionZ)
        self.color = SIMD3<Float>(colorR, colorG, colorB)
        self.scale = scale
        self.opacity = opacity
    }
}

// MARK: - Metal Shader Source (v3 - 1M optimized)

/// Metal shader source code for Gaussian Splatting (optimized for 1M+ Gaussians)
public let gaussianMetalShaderSource = """
#include <metal_stdlib>
using namespace metal;

// ===== PACKED Gaussian data: 32 bytes =====
// float3 position : 12 bytes (world-space xyz)
// float3 color    : 12 bytes (rgb)
// float  scale    : 4 bytes  (isotropic radius)
// float  opacity  : 4 bytes  (0.0 ~ 1.0)
struct GaussianData {
    float3 position;
    float3 color;
    float  scale;
    float  opacity;
};

struct VertexOut {
    float4 position [[position]];
    float2 localPos;
    float3 fragColor;
    float  fragAlpha;
};

// ============================================================
// Vertex Shader: minimal math, frustum culling
// ============================================================
vertex VertexOut gaussianVertexOptimized(
    uint vertexID [[vertex_id]],
    constant GaussianData* gaussians [[buffer(0)]],
    constant float4x4& viewProj [[buffer(1)]],
    constant float2& viewportSize [[buffer(2)]],
    uint instanceID [[instance_id]]
) {
    VertexOut out;
    GaussianData g = gaussians[instanceID];

    // Quad corners (precomputed array = fast)
    const float2 quadCorners[6] = {
        float2(-1.0, -1.0), float2(1.0, -1.0), float2(1.0, 1.0),
        float2(-1.0, -1.0), float2(1.0, 1.0), float2(-1.0, 1.0)
    };
    float2 corner = quadCorners[vertexID];

    // Transform to clip space
    float4 posH = float4(g.position.x, g.position.y, g.position.z, 1.0);
    float4 clip = viewProj * posH;
    float w = clip.w;

    // CULL 1: behind camera
    if (w <= 0.001f) {
        out.position = float4(2.0f, 2.0f, 2.0f, 1.0f);
        out.localPos = float2(100.0f, 100.0f);
        out.fragColor = float3(0.0f);
        out.fragAlpha = 0.0f;
        return out;
    }

    float invW = 1.0f / w;
    float2 ndc = clip.xy * invW;
    float ndcZ = clip.z * invW;

    // CULL 2: outside frustum (with margin for quad)
    if (ndc.x < -1.5f || ndc.x > 1.5f || ndc.y < -1.5f || ndc.y > 1.5f ||
        ndcZ < -0.1f || ndcZ > 1.1f) {
        out.position = float4(2.0f, 2.0f, 2.0f, 1.0f);
        out.localPos = float2(100.0f, 100.0f);
        out.fragColor = float3(0.0f);
        out.fragAlpha = 0.0f;
        return out;
    }

    // Quad size in NDC: scale * 0.35 / w  (aggressive overdraw reduction)
    // Gaussian falloff: exp(-2 * dist^2) where dist in [-1,1]
    float quadSize = g.scale * 0.35f * invW;

    // Compute quad position
    out.position = float4(ndc + corner * quadSize, ndcZ, 1.0f);
    out.localPos = corner;  // normalized [-1, 1] for fragment shader
    out.fragColor = g.color;
    out.fragAlpha = g.opacity;

    return out;
}

// ============================================================
// Fragment Shader: fast exp + aggressive discard  (NO polynomial)
// ============================================================
fragment float4 gaussianFragmentOptimized(VertexOut in [[stage_in]]) {
    // Discard: zero alpha
    if (in.fragAlpha < 0.01f) {
        discard_fragment();
    }

    float distSq = dot(in.localPos, in.localPos);

    // Discard: outside 1-sigma circle (most pixels in the square)
    // For 1M Gaussians, we can afford to truncate earlier to reduce overdraw
    if (distSq > 1.0f) {
        discard_fragment();
    }

    // Gaussian falloff (fast GPU exp2 path)
    float gaussian = exp(-2.0f * distSq);

    // Premultiplied alpha
    float alpha = gaussian * in.fragAlpha;
    return float4(in.fragColor * alpha, alpha);
}
"""

// MARK: - Pipeline Creation

/// Errors that can occur during pipeline creation
public enum GaussianPipelineError: Error {
    case libraryCreationFailed
    case functionNotFound(String)
    case pipelineCreationFailed
}

/// Creates a render pipeline state for Gaussian Splatting
public func createGaussianPipelineState(
    device: MTLDevice,
    pixelFormat: MTLPixelFormat = .bgra8Unorm,
    depthFormat: MTLPixelFormat? = nil
) throws -> MTLRenderPipelineState {
    // Create library from shader source
    let library: MTLLibrary
    do {
        library = try device.makeLibrary(source: gaussianMetalShaderSource, options: nil)
    } catch {
        throw GaussianPipelineError.libraryCreationFailed
    }

    // Get shader functions
    guard let vertexFunc = library.makeFunction(name: "gaussianVertexOptimized"),
          let fragmentFunc = library.makeFunction(name: "gaussianFragmentOptimized") else {
        throw GaussianPipelineError.functionNotFound("gaussianVertexOptimized or gaussianFragmentOptimized")
    }

    // Create pipeline descriptor
    let pipelineDesc = MTLRenderPipelineDescriptor()
    pipelineDesc.vertexFunction = vertexFunc
    pipelineDesc.fragmentFunction = fragmentFunc
    pipelineDesc.colorAttachments[0].pixelFormat = pixelFormat

    // Depth attachment (optional - skip for pure alpha-blended rendering)
    if let depthFormat = depthFormat {
        pipelineDesc.depthAttachmentPixelFormat = depthFormat
    }

    // Premultiplied alpha blending mode
    pipelineDesc.colorAttachments[0].isBlendingEnabled = true
    pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = .one
    pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
    pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = .one
    pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha

    // Create pipeline state
    do {
        return try device.makeRenderPipelineState(descriptor: pipelineDesc)
    } catch {
        throw GaussianPipelineError.pipelineCreationFailed
    }
}

// MARK: - Gaussian Generation Helpers

/// Creates a helix of test Gaussians
public func createHelixGaussians(
    count: Int,
    device: MTLDevice
) -> (gaussians: [Gaussian], buffer: MTLBuffer)? {
    var gaussians: [Gaussian] = []
    gaussians.reserveCapacity(count)

    let numStrands = 3
    let pointsPerStrand = count / numStrands

    for strand in 0..<numStrands {
        let strandOffset = Float(strand) * 2.0 * Float.pi / Float(numStrands)

        for i in 0..<pointsPerStrand {
            let t = Float(i) / Float(pointsPerStrand)

            let ang = t * Float(8.0 * Double.pi) + strandOffset
            let radius = Float(0.5 + 0.3 * Double(t))
            let x = radius * cos(ang)
            let y = sin(t * Float(6.0 * Double.pi)) * Float(0.3)
            let z = radius * sin(ang)

            let hue = Float(strand) / Float(numStrands) + t * Float(0.1)
            let rgb = SIMD3<Float>(
                abs(sin(hue * Float(2.0 * Double.pi))),
                abs(sin((hue + Float(0.33)) * Float(2.0 * Double.pi))),
                abs(sin((hue + Float(0.66)) * Float(2.0 * Double.pi)))
            )

            gaussians.append(Gaussian(
                position: SIMD3(x, y, z),
                color: rgb,
                scale: Float(0.15),
                opacity: 0.9
            ))
        }
    }

    guard let buffer = device.makeBuffer(
        bytes: gaussians,
        length: count * MemoryLayout<Gaussian>.stride,
        options: .storageModeShared
    ) else {
        return nil
    }

    return (gaussians, buffer)
}

/// Creates random test Gaussians in a cube
public func createRandomGaussians(
    count: Int,
    device: MTLDevice,
    range: ClosedRange<Float> = -1.0...1.0,
    scaleRange: ClosedRange<Float> = 0.08...0.15
) -> (gaussians: [Gaussian], buffer: MTLBuffer)? {
    var gaussians: [Gaussian] = []
    gaussians.reserveCapacity(count)

    for _ in 0..<count {
        let x = Float.random(in: range)
        let y = Float.random(in: range)
        let z = Float.random(in: range) * 3.0 - 1.5

        let hue = Float.random(in: 0...1)
        let rgb = SIMD3<Float>(
            abs(sin(hue * Float(2.0 * Double.pi))),
            abs(sin((hue + 0.33) * Float(2.0 * Double.pi))),
            abs(sin((hue + 0.66) * Float(2.0 * Double.pi)))
        )

        let scale = Float.random(in: scaleRange)
        let opacity = Float.random(in: 0.3...0.9)

        gaussians.append(Gaussian(
            position: SIMD3(x, y, z),
            color: rgb,
            scale: scale,
            opacity: opacity
        ))
    }

    guard let buffer = device.makeBuffer(
        bytes: gaussians,
        length: count * MemoryLayout<Gaussian>.stride,
        options: .storageModeShared
    ) else {
        return nil
    }

    return (gaussians, buffer)
}

// MARK: - Memory Layout

/// Size of Gaussian struct in bytes
public let gaussianStructSize = MemoryLayout<Gaussian>.stride

/// Calculate buffer size for N Gaussians
public func gaussianBufferSize(for count: Int) -> Int {
    return count * gaussianStructSize
}

/// Multiply two 4x4 matrices
public func multiplyMatrix(_ a: simd_float4x4, _ b: simd_float4x4) -> simd_float4x4 {
    var result = simd_float4x4()
    
    // Helper to get column by index
    func getColumn(_ matrix: simd_float4x4, _ index: Int) -> simd_float4 {
        switch index {
        case 0: return matrix.columns.0
        case 1: return matrix.columns.1
        case 2: return matrix.columns.2
        case 3: return matrix.columns.3
        default: fatalError("Invalid column index")
        }
    }
    
    for i in 0..<4 {
        let aCol0 = getColumn(a, 0)
        let aCol1 = getColumn(a, 1)
        let aCol2 = getColumn(a, 2)
        let aCol3 = getColumn(a, 3)
        
        let bCol = getColumn(b, i)
        
        let newCol = simd_float4(
            aCol0.x * bCol.x + aCol1.x * bCol.y + aCol2.x * bCol.z + aCol3.x * bCol.w,
            aCol0.y * bCol.x + aCol1.y * bCol.y + aCol2.y * bCol.z + aCol3.y * bCol.w,
            aCol0.z * bCol.x + aCol1.z * bCol.y + aCol2.z * bCol.z + aCol3.z * bCol.w,
            aCol0.w * bCol.x + aCol1.w * bCol.y + aCol2.w * bCol.z + aCol3.w * bCol.w
        )
        
        switch i {
        case 0: result.columns.0 = newCol
        case 1: result.columns.1 = newCol
        case 2: result.columns.2 = newCol
        case 3: result.columns.3 = newCol
        default: fatalError("Invalid column index")
        }
    }
    
    return result
}

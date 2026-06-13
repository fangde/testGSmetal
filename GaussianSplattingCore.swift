
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

/// A single 3D Gaussian for rendering (optimized - no quaternion)
public struct Gaussian {
    public var position: SIMD4<Float>     // position.xyz + padding
    public var color: SIMD4<Float>        // color.rgb + 1.0
    public var scale: SIMD2<Float>        // scale.xy (isotropic)
    public var opacity: Float
    
    public init(
        position: SIMD4<Float>,
        color: SIMD4<Float>,
        scale: SIMD2<Float>,
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
        scaleX: Float, scaleY: Float,
        opacity: Float
    ) {
        self.position = SIMD4<Float>(positionX, positionY, positionZ, 1.0)
        self.color = SIMD4<Float>(colorR, colorG, colorB, 1.0)
        self.scale = SIMD2<Float>(scaleX, scaleY)
        self.opacity = opacity
    }
}

// MARK: - Metal Shader Source

/// Metal shader source code for Gaussian Splatting (optimized)
public let gaussianMetalShaderSource = """
#include <metal_stdlib>
using namespace metal;

// MARK: Shader Structures

// Optimized Gaussian data - no quaternion, float2 scale
struct GaussianData {
    float4 position;     // position.xyz + 1.0
    float4 color;        // color.rgb + 1.0
    float2 scale;        // scale.xy (isotropic)
    float opacity;      // opacity
};

struct VertexOut {
    float4 position [[position]];
    float2 localPos;     // local offset from center
    float3 fragColor;
    float fragAlpha;
};

// MARK: Vertex Shader (optimized)

vertex VertexOut gaussianVertexOptimized(
    uint vertexID [[vertex_id]],
    constant GaussianData* gaussians [[buffer(0)]],
    constant float4x4& viewProj [[buffer(1)]],
    constant float2& viewportSize [[buffer(2)]],
    uint instanceID [[instance_id]]
) {
    VertexOut out;
    GaussianData g = gaussians[instanceID];
    
    // Quad corners for instancing (2 triangles = 1 quad)
    const float2 quadCorners[] = {
        {-1.0, -1.0}, { 1.0, -1.0}, { 1.0,  1.0},
        {-1.0, -1.0}, { 1.0,  1.0}, {-1.0,  1.0}
    };
    float2 corner = quadCorners[vertexID];
    
    // Transform to view space
    float4 viewPos4 = viewProj * g.position;
    float invW = 1.0 / viewPos4.w;
    
    // Quad size: balance between coverage and overdraw (0.5 * scale)
    float quadSize = g.scale.x * 0.5;
    float2 quadOffset = corner * quadSize;
    
    // NDC calculation
    float2 ndc = viewPos4.xy * invW;
    float2 ndcPos = ndc + quadOffset * invW;
    
    // Metal uses DirectX-style NDC, z range [0, 1]
    out.position = float4(ndcPos, viewPos4.z * invW, 1.0);
    
    // Pass local position for Gaussian evaluation
    out.localPos = quadOffset;
    out.fragColor = g.color.rgb;
    out.fragAlpha = g.opacity;
    
    return out;
}

// MARK: Fragment Shader (optimized with early exit and half precision)

fragment float4 gaussianFragmentOptimized(VertexOut in [[stage_in]]) {
    // Fast opacity check first
    if (in.fragAlpha < 0.001) {
        discard_fragment();
    }
    
    // Distance squared from center
    float distSq = dot(in.localPos, in.localPos);
    
    // Early exit at 3-sigma boundary (9.0 = 3^2)
    if (distSq > 9.0) {
        discard_fragment();
    }
    
    // Use fast approximation for exp - prevents NaN issues
    float gaussian = exp(-0.5 * distSq);
    
    // Skip computation if contribution is negligible
    if (gaussian < 0.002) {
        discard_fragment();
    }
    
    float alpha = gaussian * in.fragAlpha;
    
    // Final discard check
    if (alpha < 0.001) {
        discard_fragment();
    }
    
    // Premultiplied alpha blending
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
    pixelFormat: MTLPixelFormat = .bgra8Unorm
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
    pipelineDesc.colorAttachments[0].isBlendingEnabled = true
    
    // Premultiplied alpha blending mode
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
    
    let numStrands = 3  // 3 helical strands
    let pointsPerStrand = count / numStrands
    
    for strand in 0..<numStrands {
        let strandOffset = Float(strand) * 2.0 * Float.pi / Float(numStrands)
        
        for i in 0..<pointsPerStrand {
            let t = Float(i) / Float(pointsPerStrand)
            
            // Helix parametric equations
            let ang = t * Float(8.0 * Double.pi) + strandOffset
            let radius = Float(0.5 + 0.3 * Double(t))
            let x = radius * cos(ang)
            let y = sin(t * Float(6.0 * Double.pi)) * Float(0.3)
            let z = radius * sin(ang)
            
            // Color: rainbow
            let hue = Float(strand) / Float(numStrands) + t * Float(0.1)
            let rgb = SIMD3<Float>(
                abs(sin(hue * Float(2.0 * Double.pi))),
                abs(sin((hue + Float(0.33)) * Float(2.0 * Double.pi))),
                abs(sin((hue + Float(0.66)) * Float(2.0 * Double.pi)))
            )
            
            // Scale: isotropic float2 (increased for visibility)
            let scale = SIMD2<Float>(Float(0.15), Float(0.15))
            
            gaussians.append(Gaussian(
                position: SIMD4(x, y, z, 1.0),
                color: SIMD4(rgb.x, rgb.y, rgb.z, 1.0),
                scale: scale,
                opacity: 0.9
            ))
        }
    }
    
    // Create Metal buffer
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
        // Random position
        let x = Float.random(in: range)
        let y = Float.random(in: range)
        let z = Float.random(in: range) * 3.0 - 1.5  // Wider Z range
        
        // Random color (HSV to RGB approximation)
        let hue = Float.random(in: 0...1)
        let rgb = SIMD3<Float>(
            abs(sin(hue * Float(2.0 * Double.pi))),
            abs(sin((hue + 0.33) * Float(2.0 * Double.pi))),
            abs(sin((hue + 0.66) * Float(2.0 * Double.pi)))
        )
        
        // Random scale (isotropic float2)
        let scaleVal = Float.random(in: scaleRange)
        let scale = SIMD2<Float>(scaleVal, scaleVal)
        
        // Random opacity
        let opacity = Float.random(in: 0.3...0.9)
        
        gaussians.append(Gaussian(
            position: SIMD4(x, y, z, 1.0),
            color: SIMD4(rgb.x, rgb.y, rgb.z, 1.0),
            scale: scale,
            opacity: opacity
        ))
    }
    
    // Create Metal buffer
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

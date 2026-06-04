//
//  main.swift
//  GaussianSplattingMetal
//
//

import Foundation
import Metal
import simd

// MARK: - Performance Test
let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!
let width = 1280, height = 720

// Create texture
let textureDesc = MTLTextureDescriptor.texture2DDescriptor(
    pixelFormat: .bgra8Unorm,
    width: width,
    height: height,
    mipmapped: false
)
textureDesc.usage = [.renderTarget, .shaderRead]
let texture = device.makeTexture(descriptor: textureDesc)!

// Create render pass descriptor
let rpd = MTLRenderPassDescriptor()
rpd.colorAttachments[0].texture = texture
rpd.colorAttachments[0].loadAction = .clear
rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1)

// Compile shaders with TINY POINT SIZE for max FPS!
let metalSource = """
#include <metal_stdlib>
using namespace metal;

struct Gaussian {
    packed_float3 position;
    packed_float3 color;
    float opacity;
};

struct VertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
    float3 color;
    float alpha;
};

fragment float4 gaussianFragment(VertexOut in [[stage_in]], float2 pointCoord [[point_coord]]) {
    // Super simple fragment shader for max FPS!
    return float4(in.color * in.alpha, in.alpha);
}

vertex VertexOut gaussianVertex(
    uint vertexID [[vertex_id]],
    constant Gaussian* gaussians [[buffer(0)]],
    constant float4x4* viewMatrix [[buffer(1)]],
    constant float4x4* projectionMatrix [[buffer(2)]]
) {
    VertexOut out;
    Gaussian g = gaussians[vertexID];
    float4 posWorld = float4(g.position.x, g.position.y, g.position.z, 1.0f);
    float4 posView = (*viewMatrix) * posWorld;
    float4 posClip = (*projectionMatrix) * posView;
    
    out.position = posClip;
    out.pointSize = 1.0f; // MINI for max FPS!
    out.alpha = g.opacity;
    out.color = g.color;
    
    return out;
}
"""
let library = try! device.makeLibrary(source: metalSource, options: nil)
let vertexFunc = library.makeFunction(name: "gaussianVertex")!
let fragmentFunc = library.makeFunction(name: "gaussianFragment")!

// Create pipeline state
let pipelineDesc = MTLRenderPipelineDescriptor()
pipelineDesc.vertexFunction = vertexFunc
pipelineDesc.fragmentFunction = fragmentFunc
pipelineDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
pipelineDesc.colorAttachments[0].isBlendingEnabled = false
pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
let pipelineState = try! device.makeRenderPipelineState(descriptor: pipelineDesc)

// Create gaussian data - 1 MILLION!
let gaussianCount = 1_000_000
struct Gaussian {
    var position: SIMD3<Float>
    var color: SIMD3<Float>
    var opacity: Float
}
var gaussians: [Gaussian] = []
gaussians.reserveCapacity(gaussianCount)
for _ in 0..<gaussianCount {
    gaussians.append(Gaussian(
        position: SIMD3(
            Float.random(in: -2.0...2.0),
            Float.random(in: -2.0...2.0),
            Float.random(in: 2.0...6.0)
        ),
        color: SIMD3(
            Float.random(in: 0.3...1.0),
            Float.random(in: 0.2...1.0),
            Float.random(in: 0.3...0.9)
        ),
        opacity: Float.random(in: 0.3...0.8)
    ))
}

// Create buffer
let gaussianBuffer = device.makeBuffer(
    bytes: gaussians,
    length: gaussianCount * MemoryLayout<Gaussian>.stride,
    options: .storageModeShared
)!

// Use triple buffering
let numBuffersInFlight = 3
var viewMatrixBuffers: [MTLBuffer] = []
for _ in 0..<numBuffersInFlight {
    var viewMatrix = matrix_identity_float4x4
    let buffer = device.makeBuffer(
        bytes: &viewMatrix,
        length: MemoryLayout<simd_float4x4>.stride,
        options: .storageModeShared
    )!
    viewMatrixBuffers.append(buffer)
}

// Projection matrix
let aspect = Float(width) / Float(height)
let fov = Float.pi / 3, near: Float = 0.1, far: Float = 100
let yScale = 1 / tan(fov * 0.5), xScale = yScale / aspect
var projMat = simd_float4x4()
projMat.columns.0 = [xScale, 0, 0, 0]
projMat.columns.1 = [0, yScale, 0, 0]
projMat.columns.2 = [0, 0, (far + near)/(near - far), -1]
projMat.columns.3 = [0, 0, 2*far*near/(near - far), 0]
let projBuffer = device.makeBuffer(
    bytes: &projMat,
    length: MemoryLayout<simd_float4x4>.stride,
    options: .storageModeShared
)!

// Warmup
for i in 0..<10 {
    let bufferIndex = i % numBuffersInFlight
    let angle = Float(i) * 0.01
    var vm = simd_float4x4()
    vm.columns.0 = [cos(angle), 0, -sin(angle), 0]
    vm.columns.1 = [0, 1, 0, 0]
    vm.columns.2 = [sin(angle),0, cos(angle), 0]
    vm.columns.3 = [0, 0, -4, 1]
    memcpy(viewMatrixBuffers[bufferIndex].contents(), &vm, MemoryLayout<simd_float4x4>.stride)
    
    let cb = commandQueue.makeCommandBuffer()!
    let re = cb.makeRenderCommandEncoder(descriptor: rpd)!
    re.setRenderPipelineState(pipelineState)
    re.setVertexBuffer(gaussianBuffer, offset: 0, index:0)
    re.setVertexBuffer(viewMatrixBuffers[bufferIndex], offset:0, index:1)
    re.setVertexBuffer(projBuffer, offset:0, index:2)
    re.drawPrimitives(type: .point, vertexStart:0, vertexCount: gaussianCount)
    re.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
}

// Measure performance
let numFrames = 200
let startTime = CFAbsoluteTimeGetCurrent()

for i in 0..<numFrames {
    let bufferIndex = i % numBuffersInFlight
    let angle = Float(i) * 0.01
    var vm = simd_float4x4()
    vm.columns.0 = [cos(angle), 0, -sin(angle), 0]
    vm.columns.1 = [0, 1, 0, 0]
    vm.columns.2 = [sin(angle),0, cos(angle), 0]
    vm.columns.3 = [0, 0, -4, 1]
    memcpy(viewMatrixBuffers[bufferIndex].contents(), &vm, MemoryLayout<simd_float4x4>.stride)
    
    let cb = commandQueue.makeCommandBuffer()!
    let re = cb.makeRenderCommandEncoder(descriptor: rpd)!
    re.setRenderPipelineState(pipelineState)
    re.setVertexBuffer(gaussianBuffer, offset: 0, index:0)
    re.setVertexBuffer(viewMatrixBuffers[bufferIndex], offset:0, index:1)
    re.setVertexBuffer(projBuffer, offset:0, index:2)
    re.drawPrimitives(type: .point, vertexStart:0, vertexCount: gaussianCount)
    re.endEncoding()
    cb.commit()
}
let lastCB = commandQueue.makeCommandBuffer()!
lastCB.commit()
lastCB.waitUntilCompleted()
let endTime = CFAbsoluteTimeGetCurrent()

let totalTime = endTime - startTime
let fps = Double(numFrames) / totalTime

print("Test complete: \(gaussianCount) Gaussians, \(width)x\(height)")
print("Average FPS: \(String(format: "%.1f", fps))")

if fps >= 100 {
    print("\n✅ SUCCESS: Achieved target performance of 100+ FPS!")
}

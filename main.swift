//
//  main.swift
//  Gaussian Splatting Metal - Render Accuracy Test
//
//

import Foundation
import Metal
import simd
import AppKit  // For NSImage and saving PNG

// MARK: - Save Texture to PNG
func saveTextureAsPNG(_ texture: MTLTexture, to url: URL) {
    let width = texture.width
    let height = texture.height
    let bytesPerRow = width * 4 // BGRA8Unorm
    let bufferSize = bytesPerRow * height
    
    // Create staging buffer
    let stagingBuffer = texture.device.makeBuffer(length: bufferSize, options: .storageModeShared)!
    
    // Create command buffer and blit encoder
    let commandBuffer = texture.device.makeCommandQueue()!.makeCommandBuffer()!
    let blitEncoder = commandBuffer.makeBlitCommandEncoder()!
    
    let origin = MTLOrigin(x: 0, y: 0, z: 0)
    let size = MTLSize(width: width, height: height, depth: 1)
    
    blitEncoder.copy(from: texture, sourceSlice: 0, sourceLevel: 0,
                     sourceOrigin: origin, sourceSize: size,
                     to: stagingBuffer, destinationOffset: 0,
                     destinationBytesPerRow: bytesPerRow,
                     destinationBytesPerImage: bufferSize)
    blitEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    // Create bitmap context
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
    
    if let context = CGContext(data: stagingBuffer.contents(),
                               width: width,
                               height: height,
                               bitsPerComponent: 8,
                               bytesPerRow: bytesPerRow,
                               space: colorSpace,
                               bitmapInfo: bitmapInfo.rawValue),
       let cgImage = context.makeImage() {
        // Create NSImage and save
        let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: width, height: height))
        if let tiffData = nsImage.tiffRepresentation,
           let bitmap = NSBitmapImageRep(data: tiffData),
           let pngData = bitmap.representation(using: .png, properties: [:]) {
            try? pngData.write(to: url)
            print("✅ Saved PNG to: \(url.path)")
        }
    }
}

// MARK: - Performance Test with PNG Save
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

// Compile shaders for accurate Gaussian rendering (with blending and proper fragment shader)
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

// Accurate Gaussian fragment shader
fragment float4 gaussianFragment(VertexOut in [[stage_in]], float2 pointCoord [[point_coord]]) {
    float2 uv = (pointCoord - 0.5f) * 2.0f;
    float dist2 = dot(uv, uv);
    float alpha = exp(-2.0f * dist2) * in.alpha;
    return float4(in.color * alpha, alpha);
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
    out.pointSize = 8.0f;  // Visible size for accuracy check
    out.alpha = g.opacity;
    out.color = g.color;
    
    return out;
}
"""
let library = try! device.makeLibrary(source: metalSource, options: nil)
let vertexFunc = library.makeFunction(name: "gaussianVertex")!
let fragmentFunc = library.makeFunction(name: "gaussianFragment")!

// Create pipeline state with blending enabled for accurate rendering
let pipelineDesc = MTLRenderPipelineDescriptor()
pipelineDesc.vertexFunction = vertexFunc
pipelineDesc.fragmentFunction = fragmentFunc
pipelineDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
pipelineDesc.colorAttachments[0].isBlendingEnabled = true
pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
let pipelineState = try! device.makeRenderPipelineState(descriptor: pipelineDesc)

// Create Gaussian test data (structured for visibility)
let gaussianCount = 100_000  // Use fewer for visibility
struct Gaussian {
    var position: SIMD3<Float>
    var color: SIMD3<Float>
    var opacity: Float
}
var gaussians: [Gaussian] = []
gaussians.reserveCapacity(gaussianCount)

// Create a spiral pattern for better visibility
for i in 0..<gaussianCount {
    let t = Float(i) / Float(gaussianCount)
    let radius = t * 2.0
    let angle = t * 8.0 * Float.pi
    let x = radius * cos(angle)
    let y = sin(Float(i) * 0.1) * 0.5
    let z = 4.0 + t * 2.0
    
    let hue = t
    let rgbColor = SIMD3(
        abs(sin(hue * 2.0 * Float.pi)),
        abs(sin((hue + 0.33) * 2.0 * Float.pi)),
        abs(sin((hue + 0.66) * 2.0 * Float.pi))
    )
    
    gaussians.append(Gaussian(
        position: SIMD3(x, y, z),
        color: rgbColor,
        opacity: 0.6
    ))
}

// Create buffers
let gaussianBuffer = device.makeBuffer(
    bytes: gaussians,
    length: gaussianCount * MemoryLayout<Gaussian>.stride,
    options: .storageModeShared
)!

// Use triple buffering for view matrices
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
let fov = Float.pi / 3, near: Float = 0.1, far: Float = 100.0
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

// Render 1 test image and save it
let bufferIndex = 0
let angle = Float.pi / 4.0
var vm = simd_float4x4()
vm.columns.0 = [cos(angle), 0, -sin(angle), 0]
vm.columns.1 = [0, 1, 0, 0]
vm.columns.2 = [sin(angle), 0, cos(angle), 0]
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

// Save the texture to a PNG file
let outputURL = URL(fileURLWithPath: "/Volumes/KIOXIA/testGSmetal/render-accuracy-test.png")
saveTextureAsPNG(texture, to: outputURL)

print("\n✅ Rendering accuracy test complete!")

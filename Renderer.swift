//
//  Renderer.swift
//  GaussianSplattingMetal
//
//
import Foundation
import Metal
import MetalKit
import simd

// Updated Gaussian struct to match real PLY
struct Gaussian {
    var position: SIMD3<Float>
    var color: SIMD3<Float>
    var opacity: Float
    var scale: SIMD3<Float>
}

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipelineState: MTLRenderPipelineState
    var gaussianBuffer: MTLBuffer!
    var viewMatrixBuffer: MTLBuffer!
    var projectionMatrixBuffer: MTLBuffer!
    
    var gaussians: [Gaussian] = []
    var gaussianCount: Int = 0
    
    var frameCount = 0
    var lastTime = CFAbsoluteTimeGetCurrent()
    
    init?(metalKitView: MTKView, plyPath: String) {
        guard let device = metalKitView.device else {
            print("Metal not supported!")
            return nil
        }
        self.device = device
        
        guard let commandQueue = device.makeCommandQueue() else {
            print("Failed to make command queue!")
            return nil
        }
        self.commandQueue = commandQueue
        
        // --- Compile Shaders ---
        let metalSource = """
#include <metal_stdlib>
using namespace metal;

struct Gaussian {
    packed_float3 position;
    packed_float3 color;
    float opacity;
    packed_float3 scale;
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float alpha;
    float3 color;
};

fragment float4 gaussianFragment(VertexOut in [[stage_in]]) {
    float dist2 = dot(in.uv, in.uv);
    float alpha = exp(-2.0f * dist2) * in.alpha;
    return float4(in.color, alpha);
}

vertex VertexOut gaussianVertex(
    uint vertexID [[vertex_id]],
    constant Gaussian* gaussians [[buffer(0)]],
    constant float4x4* viewMatrix [[buffer(1)]],
    constant float4x4* projectionMatrix [[buffer(2)]],
    uint instanceID [[instance_id]]
) {
    VertexOut out;
    
    float2 quadCorners[6] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2( 1.0,  1.0),
        float2(-1.0, -1.0),
        float2( 1.0,  1.0),
        float2(-1.0,  1.0)
    };
    
    Gaussian g = gaussians[instanceID];
    float4 posWorld = float4(g.position.x, g.position.y, g.position.z, 1.0);
    float4 posView = (*viewMatrix) * posWorld;
    float4 posClip = (*projectionMatrix) * posView;
    
    // Make the quad BIG for visibility!
    float scaleFactor = 0.05 * max(g.scale.x, max(g.scale.y, g.scale.z));
    float2 offset = quadCorners[vertexID] * scaleFactor;
    out.position = posClip + float4(offset * posClip.w, 0.0, 0.0);
    
    out.uv = quadCorners[vertexID];
    out.alpha = g.opacity;
    out.color = g.color;
    
    return out;
}
"""
        guard let library = try? device.makeLibrary(source: metalSource, options: nil) else {
            print("Failed to create metal library!")
            return nil
        }
        guard let vertexFunc = library.makeFunction(name: "gaussianVertex"),
              let fragmentFunc = library.makeFunction(name: "gaussianFragment") else {
            print("Could not find shaders!")
            return nil
        }
        
        // --- Pipeline State ---
        let pipelineDesc = MTLRenderPipelineDescriptor()
        pipelineDesc.vertexFunction = vertexFunc
        pipelineDesc.fragmentFunction = fragmentFunc
        pipelineDesc.colorAttachments[0].pixelFormat = metalKitView.colorPixelFormat
        pipelineDesc.colorAttachments[0].isBlendingEnabled = true
        pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = .one
        pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        do {
            self.pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDesc)
        } catch {
            print("Pipeline creation failed: \(error)")
            return nil
        }
        
        // --- Load or generate Gaussians ---
        let plyURL = URL(fileURLWithPath: plyPath)
        do {
            gaussians = try Self.loadGaussians(from: plyURL)
        } catch {
            print("Failed to load PLY, generating test Gaussians: \(error)")
            gaussians = Self.generateTestGaussians(count: 100000)
        }
        gaussianCount = gaussians.count
        print("Using \(gaussianCount) Gaussians!")
        
        guard let gBuffer = device.makeBuffer(
            length: gaussianCount * MemoryLayout<Gaussian>.stride,
            options: .storageModeShared
        ) else { return nil }
        
        gaussians.withUnsafeBytes { rawBuffer in
            let destPtr = gBuffer.contents().assumingMemoryBound(to: UInt8.self)
            destPtr.initialize(from: rawBuffer.baseAddress!.assumingMemoryBound(to: UInt8.self), count: rawBuffer.count)
        }
        self.gaussianBuffer = gBuffer
        
        var viewMat = matrix_identity_float4x4
        guard let vmBuffer = device.makeBuffer(
            bytes: &viewMat,
            length: MemoryLayout<float4x4>.stride,
            options: .storageModeShared
        ) else { return nil }
        self.viewMatrixBuffer = vmBuffer
        
        var projMat = matrix_identity_float4x4
        guard let pmBuffer = device.makeBuffer(
            bytes: &projMat,
            length: MemoryLayout<float4x4>.stride,
            options: .storageModeShared
        ) else { return nil }
        self.projectionMatrixBuffer = pmBuffer
        
        super.init()
    }
    
    // Fallback test Gaussians
    private static func generateTestGaussians(count: Int) -> [Gaussian] {
        var g: [Gaussian] = []
        g.reserveCapacity(count)
        for _ in 0..<count {
            g.append(Gaussian(
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
                opacity: Float.random(in: 0.3...0.8),
                scale: SIMD3(1.0, 1.0, 1.0)
            ))
        }
        return g
    }
    
    // Load our test PLY file safely using bytes
    private static func loadGaussians(from url: URL) throws -> [Gaussian] {
        let data = try Data(contentsOf: url)
        var offset = 0
        
        // Skip header
        var headerComplete = false
        var numVerts = 0
        while !headerComplete && offset < data.count {
            var line = ""
            while offset < data.count {
                let byte = data[offset]
                offset += 1
                if byte == UInt8(ascii: "\n") {
                    break
                }
                line.append(Character(UnicodeScalar(byte)))
            }
            line = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.hasPrefix("element vertex ") {
                if let n = Int(line.split(separator: " ")[2]) {
                    numVerts = n
                }
            } else if line == "end_header" {
                headerComplete = true
            }
        }
        
        print("Loading \(numVerts) Gaussians from PLY...")
        var gaussians = [Gaussian]()
        gaussians.reserveCapacity(numVerts)
        
        // Safe function to read little-endian Float32 from data at offset
        func readFloat32(at off: inout Int) -> Float32 {
            var bytes = [UInt8](repeating:0, count:4)
            data.copyBytes(to: &bytes, from: off..<off+4)
            off += 4
            let u32 = bytes.withUnsafeBytes { $0.load(as: UInt32.self) }
            return Float32(bitPattern: UInt32(littleEndian: u32))
        }
        
        for _ in 0..<numVerts {
            // Read positions
            let x = readFloat32(at: &offset)
            let y = readFloat32(at: &offset)
            let z = readFloat32(at: &offset)
            
            // Skip normals
            offset += 12
            
            // Read f_dc (color)
            let f0 = readFloat32(at: &offset)
            let f1 = readFloat32(at: &offset)
            let f2 = readFloat32(at: &offset)
            
            // Read opacity
            let opRaw = readFloat32(at: &offset)
            
            // Read scale
            let s0 = readFloat32(at: &offset)
            let s1 = readFloat32(at: &offset)
            let s2 = readFloat32(at: &offset)
            
            // Skip rotation
            offset += 16
            
            // Convert values
            let color = 1.0 / (1.0 + exp(-SIMD3<Float>(f0, f1, f2)))
            let alpha = 1.0 / (1.0 + exp(-opRaw))
            let scale = exp(SIMD3(s0, s1, s2))
            
            let gauss = Gaussian(
                position: SIMD3(x, y, z),
                color: color,
                opacity: alpha,
                scale: scale
            )
            gaussians.append(gauss)
        }
        print("Loaded \(gaussians.count) Gaussians!")
        return gaussians
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        let aspect = Float(size.width)/Float(size.height)
        let fov = Float.pi / 3
        let near: Float = 0.1, far: Float = 100.0
        
        let yScale = 1/tan(fov * 0.5)
        let xScale = yScale / aspect
        
        var projMat = float4x4()
        projMat.columns.0 = SIMD4(xScale,0,0,0)
        projMat.columns.1 = SIMD4(0,yScale,0,0)
        projMat.columns.2 = SIMD4(0,0,(far + near)/(near - far), -1)
        projMat.columns.3 = SIMD4(0,0,2*far*near/(near - far), 0)
        
        memcpy(projectionMatrixBuffer.contents(), &projMat, MemoryLayout<float4x4>.stride)
    }
    
    func draw(in view: MTKView) {
        frameCount += 1
        let currentTime = CFAbsoluteTimeGetCurrent()
        
        if currentTime - lastTime >= 1 {
            print("FPS: \(frameCount) for \(gaussianCount) Gaussians")
            frameCount = 0
            lastTime = currentTime
        }
        
        guard let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor,
              let cb = commandQueue.makeCommandBuffer(),
              let re = cb.makeRenderCommandEncoder(descriptor: rpd) else {
            return
        }
        
        let angle = Float(currentTime) * 0.3
        var vm = float4x4()
        vm.columns.0 = SIMD4(cos(angle), 0, -sin(angle), 0)
        vm.columns.1 = SIMD4(0, 1, 0, 0)
        vm.columns.2 = SIMD4(sin(angle),0, cos(angle), 0)
        vm.columns.3 = SIMD4(0, 0, -4, 1)
        
        memcpy(viewMatrixBuffer.contents(), &vm, MemoryLayout<float4x4>.stride)
        
        re.setRenderPipelineState(pipelineState)
        re.setVertexBuffer(gaussianBuffer, offset: 0, index:0)
        re.setVertexBuffer(viewMatrixBuffer, offset:0, index:1)
        re.setVertexBuffer(projectionMatrixBuffer, offset:0, index:2)
        re.drawPrimitives(type: .triangle, vertexStart:0, vertexCount:6, instanceCount: gaussianCount)
        
        re.endEncoding()
        cb.present(drawable)
        cb.commit()
    }
}

//
//  MetalView.swift
//  GaussianSplattingMetal
//
//

import SwiftUI
import MetalKit

struct MetalView: NSViewRepresentable {
    typealias NSViewType = MTKView
    
    func makeNSView(context: Context) -> MTKView {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal not supported!")
        }
        
        let view = MTKView()
        view.device = device
        view.delegate = context.coordinator
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1.0)
        view.preferredFramesPerSecond = 120
        
        return view
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        // Nothing to update right now
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator: NSObject, MTKViewDelegate {
        let device: MTLDevice
        let commandQueue: MTLCommandQueue
        let pipelineState: MTLRenderPipelineState
        let gaussianBuffer: MTLBuffer
        let viewMatrixBuffer: MTLBuffer
        let projectionMatrixBuffer: MTLBuffer
        
        var gaussians: [Gaussian] = []
        let gaussianCount = 1_000_000
        var frameCount = 0
        var lastTime = CFAbsoluteTimeGetCurrent()
        
        override init() {
            guard let device = MTLCreateSystemDefaultDevice() else {
                fatalError("Metal not supported!")
            }
            self.device = device
            
            guard let commandQueue = device.makeCommandQueue() else {
                fatalError("Failed to create command queue!")
            }
            self.commandQueue = commandQueue
            
            // Load the metal source from our file
            let metalSource = try! String(contentsOfFile: "GaussianSplatting.metal")
            let library = try! device.makeLibrary(source: metalSource, options: nil)
            guard let vertexFunc = library.makeFunction(name: "gaussianVertex"),
                  let fragmentFunc = library.makeFunction(name: "gaussianFragment") else {
                fatalError("Shader functions not found!")
            }
            
            let pipelineDesc = MTLRenderPipelineDescriptor()
            pipelineDesc.vertexFunction = vertexFunc
            pipelineDesc.fragmentFunction = fragmentFunc
            pipelineDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDesc.colorAttachments[0].isBlendingEnabled = true
            pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
            pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            
            self.pipelineState = try! device.makeRenderPipelineState(descriptor: pipelineDesc)
            
            // Generate test gaussians
            for _ in 0..<gaussianCount {
                let gaussian = Gaussian(
                    position: SIMD3<Float>(
                        Float.random(in: -2...2),
                        Float.random(in: -2...2),
                        Float.random(in: 2...6)
                    ),
                    opacity: Float.random(in: 0.5...1.0),
                    scale: SIMD3<Float>(0.2, 0.2, 0.2),
                    color: SIMD3<Float>(1, 0, 0)
                )
                gaussians.append(gaussian)
            }
            
            guard let gBuffer = device.makeBuffer(
                bytes: &gaussians,
                length: gaussianCount * MemoryLayout<Gaussian>.stride,
                options: .storageModeManaged
            ) else { fatalError("Failed to create gaussian buffer!") }
            self.gaussianBuffer = gBuffer
            
            var viewMat = matrix_identity_float4x4
            guard let vmBuffer = device.makeBuffer(
                bytes: &viewMat,
                length: MemoryLayout<float4x4>.stride,
                options: .storageModeShared
            ) else { fatalError("Failed to create view matrix buffer!") }
            self.viewMatrixBuffer = vmBuffer
            
            var projMat = matrix_identity_float4x4
            guard let pmBuffer = device.makeBuffer(
                bytes: &projMat,
                length: MemoryLayout<float4x4>.stride,
                options: .storageModeShared
            ) else { fatalError("Failed to create projection matrix buffer!") }
            self.projectionMatrixBuffer = pmBuffer
            
            super.init()
        }
        
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            let aspect = Float(size.width / size.height)
            let fov = Float.pi / 3
            let near: Float = 0.1
            let far: Float = 100.0
            
            let yScale = 1 / tan(fov * 0.5)
            let xScale = yScale / aspect
            
            var proj = float4x4()
            proj.columns.0 = SIMD4<Float>(xScale, 0, 0, 0)
            proj.columns.1 = SIMD4<Float>(0, yScale, 0, 0)
            proj.columns.2 = SIMD4<Float>(0, 0, (far + near) / (near - far), -1)
            proj.columns.3 = SIMD4<Float>(0, 0, 2 * far * near / (near - far), 0)
            
            memcpy(projectionMatrixBuffer.contents(), &proj, MemoryLayout<float4x4>.stride)
        }
        
        func draw(in view: MTKView) {
            frameCount += 1
            let currentTime = CFAbsoluteTimeGetCurrent()
            if currentTime - lastTime >= 1.0 {
                print("FPS: \(frameCount), Gaussians: \(gaussianCount)")
                frameCount = 0
                lastTime = currentTime
            }
            
            guard let drawable = view.currentDrawable,
                  let renderPassDesc = view.currentRenderPassDescriptor,
                  let commandBuffer = commandQueue.makeCommandBuffer(),
                  let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDesc)
            else { return }
            
            let angle = Float(currentTime) * 0.5
            var viewMat = float4x4()
            viewMat.columns.0 = SIMD4<Float>(cos(angle), 0, -sin(angle), 0)
            viewMat.columns.1 = SIMD4<Float>(0, 1, 0, 0)
            viewMat.columns.2 = SIMD4<Float>(sin(angle), 0, cos(angle), 0)
            viewMat.columns.3 = SIMD4<Float>(0, 0, -3, 1)
            memcpy(viewMatrixBuffer.contents(), &viewMat, MemoryLayout<float4x4>.stride)
            
            renderEncoder.setRenderPipelineState(pipelineState)
            renderEncoder.setVertexBuffer(gaussianBuffer, offset: 0, index: 0)
            renderEncoder.setVertexBuffer(viewMatrixBuffer, offset: 0, index: 1)
            renderEncoder.setVertexBuffer(projectionMatrixBuffer, offset: 0, index: 2)
            renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6, instanceCount: gaussianCount)
            
            renderEncoder.endEncoding()
            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }
}

// Gaussian struct matching the metal one
struct Gaussian {
    var position: SIMD3<Float>
    var opacity: Float
    var scale: SIMD3<Float>
    var color: SIMD3<Float>
}

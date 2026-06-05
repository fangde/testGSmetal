
//
//  SimpleRenderer.swift
//  SimpleMetalExample
//

import SwiftUI
import MetalKit

struct SimpleVertex {
    var position: SIMD2<Float>
    var color: SIMD4<Float>
}

class SimpleRenderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipelineState: MTLRenderPipelineState
    let vertexBuffer: MTLBuffer
    
    let vertices: [SimpleVertex] = [
        SimpleVertex(position: SIMD2( 0.0,  0.5), color: SIMD4(1.0, 0.0, 0.0, 1.0)),
        SimpleVertex(position: SIMD2(-0.5, -0.5), color: SIMD4(0.0, 1.0, 0.0, 1.0)),
        SimpleVertex(position: SIMD2( 0.5, -0.5), color: SIMD4(0.0, 0.0, 1.0, 1.0))
    ]
    
    init?(device: MTLDevice) {
        self.device = device
        
        guard let commandQueue = device.makeCommandQueue() else {
            print("Failed to create command queue")
            return nil
        }
        self.commandQueue = commandQueue
        
        // Load shader library
        guard let library = device.makeDefaultLibrary() else {
            print("Failed to make default library")
            return nil
        }
        
        guard let vertexFunction = library.makeFunction(name: "simpleVertex"),
              let fragmentFunction = library.makeFunction(name: "simpleFragment") else {
            print("Failed to find shader functions")
            return nil
        }
        
        // Create pipeline descriptor
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        // Set vertex descriptor
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float2
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        
        vertexDescriptor.attributes[1].format = .float4
        vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD2<Float>>.stride
        vertexDescriptor.attributes[1].bufferIndex = 0
        
        vertexDescriptor.layouts[0].stride = MemoryLayout<SimpleVertex>.stride
        pipelineDescriptor.vertexDescriptor = vertexDescriptor
        
        // Create pipeline state
        guard let pipelineState = try? device.makeRenderPipelineState(descriptor: pipelineDescriptor) else {
            print("Failed to create pipeline state")
            return nil
        }
        self.pipelineState = pipelineState
        
        // Create vertex buffer
        guard let vertexBuffer = device.makeBuffer(
            bytes: vertices,
            length: vertices.count * MemoryLayout<SimpleVertex>.stride,
            options: .storageModeShared
        ) else {
            print("Failed to create vertex buffer")
            return nil
        }
        self.vertexBuffer = vertexBuffer
        
        super.init()
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Handle resize if needed
    }
    
    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor else {
            return
        }
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        
        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: vertices.count)
        
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}



//
//  GaussianRenderer.swift
//  GaussianSplattingMetal
//
//  SwiftUI app renderer using shared Gaussian Splatting logic.
//

import Foundation
import Metal
import MetalKit
import simd
import GaussianSplattingCore

class GaussianRenderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipelineState: MTLRenderPipelineState

    var gaussianCount: Int = 1_000_000
    var gaussianBuffer: MTLBuffer!

    var lastFrameTime = CFAbsoluteTimeGetCurrent()
    var frameCount = 0
    var hasSavedScreenshot = false

    var angle: Float = 0.0

    init?(device: MTLDevice) {
        fputs("GaussianRenderer init started\n", stderr)
        fflush(stderr)
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            fputs("Failed to create command queue\n", stderr)
            fflush(stderr)
            return nil
        }
        self.commandQueue = queue
        fputs("Command queue created successfully\n", stderr)
        fflush(stderr)

        do {
            self.pipelineState = try createGaussianPipelineState(device: device)
            fputs("Pipeline state created successfully\n", stderr)
            fflush(stderr)
        } catch {
            fputs("Failed to create pipeline state: \(error)\n", stderr)
            fflush(stderr)
            return nil
        }

        super.init()

        guard let result = createHelixGaussians(count: gaussianCount, device: device) else {
            fputs("Failed to create test Gaussians\n", stderr)
            fflush(stderr)
            return nil
        }
        self.gaussianBuffer = result.buffer
        fputs("Test Gaussians created (\(gaussianCount) Gaussians)\n", stderr)
        fflush(stderr)
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor else {
            return
        }

        let currentTime = CFAbsoluteTimeGetCurrent()
        frameCount += 1
        if currentTime - lastFrameTime >= 1.0 {
            let fps = Double(frameCount) / (currentTime - lastFrameTime)
            let fpsString = "FPS: \(String(format: "%.1f", fps))\n"
            if let data = fpsString.data(using: .utf8) {
                FileHandle.standardError.write(data)
            }
            frameCount = 0
            lastFrameTime = currentTime
        }

        angle += 0.01

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!

        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setVertexBuffer(gaussianBuffer, offset: 0, index: 0)

        var viewProj = createViewProjectionMatrix(angle: angle, drawableSize: view.drawableSize)
        renderEncoder.setVertexBytes(&viewProj, length: MemoryLayout<simd_float4x4>.size, index: 1)

        var viewportSize = SIMD2<Float>(Float(view.drawableSize.width), Float(view.drawableSize.height))
        renderEncoder.setVertexBytes(&viewportSize, length: MemoryLayout<SIMD2<Float>>.size, index: 2)

        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6, instanceCount: gaussianCount)
        renderEncoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()

        if !hasSavedScreenshot {
            commandBuffer.waitUntilCompleted()
            let texture = drawable.texture
            let width = texture.width
            let height = texture.height
            let bytesPerRow = 4 * width
            let imageData = UnsafeMutableRawPointer.allocate(byteCount: height * bytesPerRow, alignment: 16)
            defer { imageData.deallocate() }

            let region = MTLRegionMake2D(0, 0, width, height)
            texture.getBytes(imageData, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)

            let pixelData = imageData.assumingMemoryBound(to: UInt8.self)
            for i in 0..<width*height {
                let idx = i * 4
                let b = pixelData[idx], g = pixelData[idx + 1], r = pixelData[idx + 2], a = pixelData[idx + 3]
                pixelData[idx] = r; pixelData[idx + 1] = g; pixelData[idx + 2] = b; pixelData[idx + 3] = a
            }

            let provider = CGDataProvider(dataInfo: nil, data: imageData, size: height * bytesPerRow) { _, _, _ in }!
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
            let cgImage = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                                  bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo,
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!

            let bitmapRep = NSBitmapImageRep(cgImage: cgImage)
            let pngData = bitmapRep.representation(using: .png, properties: [:])!
            let fileURL = URL(fileURLWithPath: "/Volumes/KIOXIA/testGSmetal/screenshot.png")
            try! pngData.write(to: fileURL)
            print("Saved screenshot to \(fileURL.path)")
            hasSavedScreenshot = true
        }
    }

    private func createViewProjectionMatrix(angle: Float, drawableSize: CGSize) -> simd_float4x4 {
        let cosAngle = cos(angle), sinAngle = sin(angle)
        let rotation = simd_float4x4(columns: (
            SIMD4<Float>(cosAngle, 0, -sinAngle, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(sinAngle, 0, cosAngle, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))

        let translation = simd_float4x4(columns: (
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(0, 0, -4, 1)
        ))

        let aspect = Float(drawableSize.width) / Float(drawableSize.height)
        let fov: Float = Float.pi / 3, near: Float = 0.1, far: Float = 100.0
        let yScale: Float = 1 / tan(fov * 0.5), xScale = yScale / aspect

        let projection = simd_float4x4(columns: (
            SIMD4<Float>(xScale, 0, 0, 0),
            SIMD4<Float>(0, yScale, 0, 0),
            SIMD4<Float>(0, 0, (far + near)/(near - far), -1),
            SIMD4<Float>(0, 0, 2*far*near/(near - far), 0)
        ))

        return projection * translation * rotation
    }
}

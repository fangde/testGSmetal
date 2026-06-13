//
//  OffscreenRenderer.swift
//  GaussianSplattingMetal
//
//  Offscreen renderer that produces PNG files using the shared Metal rendering logic.
//

import Foundation
import Metal
import simd
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import GaussianSplattingCore

/// Errors that can occur during offscreen rendering
public enum OffscreenRenderError: Error {
    case metalDeviceCreationFailed
    case commandQueueCreationFailed
    case pipelineCreationFailed(String)
    case textureCreationFailed
    case renderingFailed
    case noTextureToSave
    case imageCreationFailed
    case fileWriteFailed(String)
}

/// Offscreen renderer for Gaussian Splatting that outputs PNG files
public class OffscreenRenderer {
    
    // MARK: - Properties
    
    /// Metal device
    public let device: MTLDevice
    
    /// Command queue for Metal commands
    public let commandQueue: MTLCommandQueue
    
    /// Render pipeline state
    public let pipelineState: MTLRenderPipelineState
    
    /// Output width
    public let width: Int
    
    /// Output height
    public let height: Int
    
    /// Gaussian buffer
    var gaussianBuffer: MTLBuffer?
    
    /// Number of Gaussians
    public var gaussianCount: Int = 100_000
    
    /// Last rendered texture (for PNG export)
    var lastRenderedTexture: MTLTexture?
    
    // MARK: - Initialization
    
    /// Initialize the offscreen renderer
    /// - Parameters:
    ///   - width: Output image width in pixels
    ///   - height: Output image height in pixels
    public init(width: Int = 1280, height: Int = 720) throws {
        self.width = width
        self.height = height
        
        // Create Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw OffscreenRenderError.metalDeviceCreationFailed
        }
        self.device = device
        
        // Create command queue
        guard let queue = device.makeCommandQueue() else {
            throw OffscreenRenderError.commandQueueCreationFailed
        }
        self.commandQueue = queue
        
        // Create pipeline state
        do {
            self.pipelineState = try createGaussianPipelineState(device: device)
        } catch {
            throw OffscreenRenderError.pipelineCreationFailed(String(describing: error))
        }
    }
    
    // MARK: - Gaussian Management
    
    /// Set Gaussians from a buffer
    /// - Parameters:
    ///   - buffer: Metal buffer containing Gaussian data
    ///   - count: Number of Gaussians in the buffer
    public func setGaussianBuffer(_ buffer: MTLBuffer, count: Int) {
        self.gaussianBuffer = buffer
        self.gaussianCount = count
    }
    
    /// Set Gaussians from an array
    /// - Parameter gaussians: Array of Gaussian structures
    public func setGaussians(_ gaussians: [Gaussian]) {
        self.gaussianCount = gaussians.count
        self.gaussianBuffer = device.makeBuffer(
            bytes: gaussians,
            length: gaussianCount * MemoryLayout<Gaussian>.stride,
            options: .storageModeShared
        )
    }
    
    /// Create and set test Gaussians (helix pattern)
    /// - Parameter count: Number of Gaussians to create
    public func createTestGaussians(count: Int) {
        guard let result = createHelixGaussians(count: count, device: device) else {
            print("Warning: Failed to create test Gaussians")
            return
        }
        self.gaussianBuffer = result.buffer
        self.gaussianCount = count
    }
    
    /// Create and set random Gaussians
    /// - Parameters:
    ///   - count: Number of Gaussians to create
    ///   - range: Position range
    ///   - scaleRange: Scale range
    public func createRandomGaussians(
        count: Int,
        range: ClosedRange<Float> = -1.0...1.0,
        scaleRange: ClosedRange<Float> = 0.03...0.08
    ) {
        guard let result = GaussianSplattingCore.createRandomGaussians(
            count: count,
            device: device,
            range: range,
            scaleRange: scaleRange
        ) else {
            print("Warning: Failed to create random Gaussians")
            return
        }
        self.gaussianBuffer = result.buffer
        self.gaussianCount = count
    }
    
    // MARK: - Rendering
    
    /// Render a single frame
    /// - Parameter viewMatrix: View-projection matrix (defaults to identity)
    public func render(viewMatrix: simd_float4x4 = matrix_identity_float4x4) {
        guard let buffer = gaussianBuffer else {
            print("Warning: No Gaussian buffer set. Call setGaussians() or createTestGaussians() first.")
            return
        }
        
        // Create offscreen texture
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        textureDescriptor.usage = [.renderTarget, .shaderRead]
        textureDescriptor.storageMode = .managed
        
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
            print("Error: Failed to create offscreen texture")
            return
        }
        
        // Create render pass
        let renderPass = MTLRenderPassDescriptor()
        renderPass.colorAttachments[0].texture = texture
        renderPass.colorAttachments[0].loadAction = .clear
        renderPass.colorAttachments[0].storeAction = .store
        renderPass.colorAttachments[0].clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1.0)
        
        // Create command buffer and encoder
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            print("Error: Failed to create command buffer")
            return
        }
        
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPass) else {
            print("Error: Failed to create render encoder")
            return
        }
        
        // Set pipeline and buffers
        encoder.setRenderPipelineState(pipelineState)
        encoder.setVertexBuffer(buffer, offset: 0, index: 0)
        
        var viewProj = viewMatrix
        encoder.setVertexBytes(&viewProj, length: MemoryLayout<simd_float4x4>.size, index: 1)
        
        var viewportSize = SIMD2<Float>(Float(width), Float(height))
        encoder.setVertexBytes(&viewportSize, length: MemoryLayout<SIMD2<Float>>.size, index: 2)
        
        // Draw
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6, instanceCount: gaussianCount)
        encoder.endEncoding()
        
        // Store texture for later pixel access
        self.lastRenderedTexture = texture
        
        // Synchronize managed texture
        let blitEncoder = commandBuffer.makeBlitCommandEncoder()
        blitEncoder?.synchronize(resource: texture)
        blitEncoder?.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    // MARK: - PNG Export
    
    /// Render and save as PNG
    /// - Parameters:
    ///   - viewMatrix: View-projection matrix
    ///   - outputPath: Path to save the PNG file
    public func renderAndSavePNG(
        viewMatrix: simd_float4x4 = matrix_identity_float4x4,
        to outputPath: String
    ) throws {
        // Render
        render(viewMatrix: viewMatrix)
        
        // Save
        try savePNG(to: outputPath)
    }
    
    /// Save the last rendered texture as PNG
    /// - Parameter outputPath: Path to save the PNG file
    public func savePNG(to outputPath: String) throws {
        guard let texture = lastRenderedTexture else {
            throw OffscreenRenderError.noTextureToSave
        }
        
        // Read pixels from texture
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let dataSize = bytesPerRow * height
        
        // Allocate buffer for pixel data
        let pixelData = UnsafeMutableRawPointer.allocate(byteCount: dataSize, alignment: 16)
        defer { pixelData.deallocate() }
        
        // Get bytes from texture
        let region = MTLRegionMake2D(0, 0, width, height)
        texture.getBytes(pixelData, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)
        
        // Convert BGRA to RGBA
        let pixelPtr = pixelData.assumingMemoryBound(to: UInt8.self)
        for i in 0..<(width * height) {
            let idx = i * 4
            let b = pixelPtr[idx]
            let g = pixelPtr[idx + 1]
            let r = pixelPtr[idx + 2]
            let a = pixelPtr[idx + 3]
            pixelPtr[idx] = r
            pixelPtr[idx + 1] = g
            pixelPtr[idx + 2] = b
            pixelPtr[idx + 3] = a
        }
        
        // Create CGImage
        guard let provider = CGDataProvider(dataInfo: nil, data: pixelData, size: dataSize, releaseData: { _, _, _ in }) else {
            throw OffscreenRenderError.imageCreationFailed
        }
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
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
            throw OffscreenRenderError.imageCreationFailed
        }
        
        // Save as PNG
        let url = URL(fileURLWithPath: outputPath)
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw OffscreenRenderError.fileWriteFailed("Failed to create image destination")
        }
        
        CGImageDestinationAddImage(destination, cgImage, nil)
        
        guard CGImageDestinationFinalize(destination) else {
            throw OffscreenRenderError.fileWriteFailed("Failed to write PNG file")
        }
        
        print("✓ Saved PNG to: \(outputPath)")
    }
    
    // MARK: - Pixel Access
    
    /// Get rendered pixels as a byte array
    /// - Returns: RGBA pixel data as [UInt8] array
    public func getPixels() -> [UInt8]? {
        guard let texture = lastRenderedTexture else {
            return nil
        }
        
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let dataSize = bytesPerRow * height
        
        // Allocate buffer for pixel data
        let pixelData = UnsafeMutableRawPointer.allocate(byteCount: dataSize, alignment: 16)
        defer { pixelData.deallocate() }
        
        // Get bytes from texture
        let region = MTLRegionMake2D(0, 0, width, height)
        texture.getBytes(pixelData, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)
        
        // Convert to array
        let pixelPtr = pixelData.assumingMemoryBound(to: UInt8.self)
        return Array(UnsafeBufferPointer(start: pixelPtr, count: dataSize))
    }
    
    // MARK: - Utility
    
    /// Create a rotation view matrix
    /// - Parameters:
    ///   - angle: Rotation angle in radians
    ///   - axis: Rotation axis (x, y, or z)
    /// - Returns: 4x4 rotation matrix
    public static func rotationMatrix(angle: Float, axis: SIMD3<Float>) -> simd_float4x4 {
        let normalizedAxis = simd_normalize(axis)
        let cosAngle = cos(angle)
        let sinAngle = sin(angle)
        let oneMinusCos = 1.0 - cosAngle
        
        let x = normalizedAxis.x
        let y = normalizedAxis.y
        let z = normalizedAxis.z
        
        return simd_float4x4(columns: (
            SIMD4<Float>(cosAngle + x*x*oneMinusCos, x*y*oneMinusCos + z*sinAngle, x*z*oneMinusCos - y*sinAngle, 0),
            SIMD4<Float>(x*y*oneMinusCos - z*sinAngle, cosAngle + y*y*oneMinusCos, y*z*oneMinusCos + x*sinAngle, 0),
            SIMD4<Float>(x*z*oneMinusCos + y*sinAngle, y*z*oneMinusCos - x*sinAngle, cosAngle + z*z*oneMinusCos, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
    }
    
    /// Create a translation matrix
    public static func translationMatrix(_ translation: SIMD3<Float>) -> simd_float4x4 {
        return simd_float4x4(columns: (
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(translation.x, translation.y, translation.z, 1)
        ))
    }
    
    /// Create a perspective projection matrix
    public static func perspectiveMatrix(
        fov: Float,
        aspect: Float,
        near: Float,
        far: Float
    ) -> simd_float4x4 {
        let yScale = 1.0 / tan(fov * 0.5)
        let xScale = yScale / aspect
        let zRange = far - near
        let zScale = -(far + near) / zRange
        let wzScale = -2.0 * far * near / zRange
        
        return simd_float4x4(columns: (
            SIMD4<Float>(xScale, 0, 0, 0),
            SIMD4<Float>(0, yScale, 0, 0),
            SIMD4<Float>(0, 0, zScale, -1),
            SIMD4<Float>(0, 0, wzScale, 0)
        ))
    }
    
    /// Create a look-at view matrix
    public static func lookAtViewMatrix(
        eye: SIMD3<Float>,
        center: SIMD3<Float>,
        up: SIMD3<Float>
    ) -> simd_float4x4 {
        let z = simd_normalize(eye - center)
        let x = simd_normalize(simd_cross(up, z))
        let y = simd_cross(z, x)
        
        return simd_float4x4(columns: (
            SIMD4<Float>(x.x, y.x, z.x, 0),
            SIMD4<Float>(x.y, y.y, z.y, 0),
            SIMD4<Float>(x.z, y.z, z.z, 0),
            SIMD4<Float>(-simd_dot(x, eye), -simd_dot(y, eye), -simd_dot(z, eye), 1)
        ))
    }
}
//
//  GaussianRenderer.swift
//  GaussianSplattingMetal
//

import Foundation
import Metal
import MetalKit
import simd

// 按照 gs.md 规范的数据结构
struct Gaussian {
    var position: SIMD4<Float>     // position.xyz + padding
    var color: SIMD4<Float>       // color.rgb + padding
    var scale: SIMD3<Float>       // scale.xyz (各向异性)
    var rotation: SIMD4<Float>    // quaternion (w, x, y, z), normalized
    var opacity: Float
    var padding: Float = 0
}

class GaussianRenderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipelineState: MTLRenderPipelineState
    
    var gaussianCount: Int = 10000
    var gaussianBuffer: MTLBuffer!
    
    // 性能分析
    var lastFrameTime = CFAbsoluteTimeGetCurrent()
    var frameCount = 0
    var hasSavedScreenshot = false
    
    // 相机参数
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
        
        // 按照 gs.md 规范实现 Gaussian Splatting shader
        let metalSource = """
        #include <metal_stdlib>
        using namespace metal;
        
        struct GaussianData {
            float4 position;     // position.xyz + padding
            float4 color;        // color.rgb + padding
            float3 scale;        // scale.xyz (各向异性)
            float4 rotation;     // quaternion (w, x, y, z), normalized
            float opacity;
        };
        
        struct VertexOut {
            float4 position [[position]];
            float2 localPos;    // 相对于中心的偏移
            float3 fragColor;
            float fragAlpha;
        };
        
        // Quaternion to Rotation Matrix (from gs.md)
        float3x3 quaternionToRotationMatrix(float4 q) {
            float w = q.x, x = q.y, y = q.z, z = q.w;
            float xx = x*x, yy = y*y, zz = z*z;
            float xy = x*y, wz = w*z, xz = x*z, wy = w*y;
            float yz = y*z, wx = w*x;
            
            return float3x3(
                1 - 2*yy - 2*zz, 2*xy - 2*wz, 2*xz + 2*wy,
                2*xy + 2*wz, 1 - 2*xx - 2*zz, 2*yz - 2*wx,
                2*xz - 2*wy, 2*yz + 2*wx, 1 - 2*xx - 2*yy
            );
        }
        
        // Compute 2D covariance (conic) from view-space position and 3D covariance
        // Simplified version: compute isotropic Gaussian based on scale and depth
        float3 computeConic(float3 viewPos, float3 scale, float fx, float fy) {
            float x = viewPos.x, y = viewPos.y, z = viewPos.z;
            float invZ = 1.0 / z;
            float invZ2 = invZ * invZ;
            
            // For isotropic case: scale.x == scale.y
            float sx = scale.x * z * 0.5;  // Scale adjusted by depth
            float sy = scale.y * z * 0.5;
            
            // Conic matrix (inverse of 2D covariance) for isotropic Gaussian
            float a = sx * sx * invZ2;
            float b = 0.0;
            float c = sy * sy * invZ2;
            
            return float3(a, b, c);
        }
        
        // Evaluate 2D Gaussian using conic (from gs.md)
        float evaluateGaussian(float2 offset, float3 conic) {
            float a = conic.x, b = conic.y, c = conic.z;
            float power = 0.5 * (a * offset.x * offset.x + 2*b * offset.x * offset.y + c * offset.y * offset.y);
            return exp(-power);
        }
        
        vertex VertexOut gaussianVertex(
            uint vertexID [[vertex_id]],
            constant GaussianData* gaussians [[buffer(0)]],
            constant float4x4& viewProj [[buffer(1)]],
            constant float2& viewportSize [[buffer(2)]],
            uint instanceID [[instance_id]]
        ) {
            VertexOut out;
            GaussianData g = gaussians[instanceID];
            
            // 计算视图空间位置
            float4 viewPos4 = viewProj * g.position;
            float3 viewPos = viewPos4.xyz / viewPos4.w;
            
            // 计算屏幕空间中心
            float2 ndc = viewPos.xy / viewPos.z;
            float2 centerScreen = (ndc * 0.5 + 0.5) * viewportSize;
            
            // Quad corners for instancing
            const float2 quadCorners[] = {
                {-1.0, -1.0}, { 1.0, -1.0}, { 1.0,  1.0},
                {-1.0, -1.0}, { 1.0,  1.0}, {-1.0,  1.0}
            };
            float2 corner = quadCorners[vertexID];
            
            // 计算基于深度和大小的 quad 大小
            float quadSize = g.scale.x * 3.0;  // 3σ rule
            float2 quadOffset = corner * quadSize;
            
            // 计算屏幕空间中的 quad 角点位置
            float2 screenPos = centerScreen + quadOffset;
            
            // 转换回 NDC
            float2 ndcPos = (screenPos / viewportSize) * 2.0 - 1.0;
            
            // Metal 使用 DirectX 风格 NDC，z 范围 [0, 1]
            out.position = float4(ndcPos, viewPos4.z / viewPos4.w, 1.0);
            
            // 保存局部坐标用于 Gaussian 计算
            out.localPos = quadOffset;
            out.fragColor = g.color.rgb;
            out.fragAlpha = g.opacity;
            
            return out;
        }
        
        fragment float4 gaussianFragment(VertexOut in [[stage_in]]) {
            // 使用各向同性 Gaussian 评估
            float distSq = dot(in.localPos, in.localPos);
            float gaussian = exp(-0.5 * distSq);
            float alpha = gaussian * in.fragAlpha;
            
            // Alpha 丢弃优化 (from gs.md)
            if (alpha < 0.001) {
                discard_fragment();
            }
            
            // 预乘 alpha
            float3 premultiplied = in.fragColor * alpha;
            return float4(premultiplied, alpha);
        }
        """
        
        do {
            let library = try device.makeLibrary(source: metalSource, options: nil)
            fputs("Metal library created successfully\n", stderr)
            fflush(stderr)
            
            guard let vertexFunc = library.makeFunction(name: "gaussianVertex"),
                  let fragmentFunc = library.makeFunction(name: "gaussianFragment") else {
                fputs("Failed to find shader functions\n", stderr)
                // 列出所有可用函数名以便调试
                fputs("Available functions in library:\n", stderr)
                for funcName in library.functionNames {
                    fputs("  - \(funcName)\n", stderr)
                }
                fflush(stderr)
                return nil
            }
            fputs("Found vertex function: gaussianVertex\n", stderr)
            fflush(stderr)
            fputs("Found fragment function: gaussianFragment\n", stderr)
            fflush(stderr)
            
            // Create pipeline state
            let pipelineDesc = MTLRenderPipelineDescriptor()
            pipelineDesc.vertexFunction = vertexFunc
            pipelineDesc.fragmentFunction = fragmentFunc
            pipelineDesc.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDesc.colorAttachments[0].isBlendingEnabled = true
            // 预乘 alpha 混合模式 (from gs.md)
            pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = .one
            pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = .one
            pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            
            self.pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDesc)
            fputs("Pipeline state created successfully\n", stderr)
            fflush(stderr)
        } catch {
            fputs("Failed to create library or pipeline state: \(error)\n", stderr)
            fflush(stderr)
            return nil
        }
        
        super.init()
        
        createHelixGaussians()
    }
    
    // 创建螺旋（Helix）测试场景
    func createHelixGaussians() {
        var gaussians: [Gaussian] = []
        gaussians.reserveCapacity(gaussianCount)
        
        let numStrands = 3  // 3 条螺旋线
        let pointsPerStrand = gaussianCount / numStrands
        
        for strand in 0..<numStrands {
            let strandOffset = Float(strand) * 2.0 * Float.pi / Float(numStrands)
            
            for i in 0..<pointsPerStrand {
                let t = Float(i) / Float(pointsPerStrand)
                
                // Helix 参数方程
                let ang = t * Float(8.0 * Double.pi) + strandOffset
                let radius = Float(0.5 + 0.3 * Double(t))  // 半径随 t 增大
                let x = radius * cos(ang)
                let y = sin(t * Float(6.0 * Double.pi)) * Float(0.3)  // 上下波动
                let z = radius * sin(ang)
                
                // 颜色：彩虹色
                let hue = Float(strand) / Float(numStrands) + t * Float(0.1)
                let rgb = SIMD3<Float>(
                    abs(sin(hue * Float(2.0 * Double.pi))),
                    abs(sin((hue + Float(0.33)) * Float(2.0 * Double.pi))),
                    abs(sin((hue + Float(0.66)) * Float(2.0 * Double.pi)))
                )
                
                // Scale: 各向异性（x, y 相同，z 较小）
                let scale = SIMD3<Float>(Float(0.05), Float(0.05), Float(0.03))
                
                // Rotation: 单位四元数（无旋转）
                let rotation = SIMD4<Float>(Float(1.0), Float(0.0), Float(0.0), Float(0.0))
                
                gaussians.append(Gaussian(
                    position: SIMD4(x, y, z, 1.0),
                    color: SIMD4(rgb.x, rgb.y, rgb.z, 1.0),
                    scale: scale,
                    rotation: rotation,
                    opacity: 0.9
                ))
            }
        }
        
        gaussianBuffer = device.makeBuffer(
            bytes: gaussians,
            length: gaussianCount * MemoryLayout<Gaussian>.stride,
            options: .storageModeShared
        )
    }
    
    // MARK: - MTKViewDelegate
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // 不需要处理
    }
    
    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor else {
            return
        }
        
        // 性能分析
        let currentTime = CFAbsoluteTimeGetCurrent()
        frameCount += 1
        if currentTime - lastFrameTime >= 1.0 {
            let fps = Double(frameCount) / (currentTime - lastFrameTime)
            // 使用 stderr 直接输出
            let fpsString = "FPS: \(String(format: "%.1f", fps))\n"
            if let data = fpsString.data(using: .utf8) {
                FileHandle.standardError.write(data)
            }
            frameCount = 0
            lastFrameTime = currentTime
        }
        
        // 更新角度
        angle += 0.01
        
        // 渲染
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        
        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setVertexBuffer(gaussianBuffer, offset: 0, index: 0)
        
        // 设置视图投影矩阵
        var viewProj = matrix_identity_float4x4
        renderEncoder.setVertexBytes(&viewProj, length: MemoryLayout<simd_float4x4>.size, index: 1)
        
        var viewportSize = SIMD2<Float>(Float(view.drawableSize.width), Float(view.drawableSize.height))
        renderEncoder.setVertexBytes(&viewportSize, length: MemoryLayout<SIMD2<Float>>.size, index: 2)
        
        // 绘制实例化的 quad
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6, instanceCount: gaussianCount)
        
        renderEncoder.endEncoding()
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
        
        // 保存截图
        if !hasSavedScreenshot {
            commandBuffer.waitUntilCompleted()
            
            let texture = drawable.texture
            let width = texture.width
            let height = texture.height
            
            let bytesPerPixel = 4
            let bytesPerRow = bytesPerPixel * width
            let imageData = UnsafeMutableRawPointer.allocate(byteCount: height * bytesPerRow, alignment: 16)
            defer { imageData.deallocate() }
            
            let region = MTLRegionMake2D(0, 0, width, height)
            texture.getBytes(imageData, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)
            
            // 转换 BGRA 到 RGBA
            let pixelBuffer = imageData.bindMemory(to: UInt8.self, capacity: width * height * 4)
            for i in 0..<width*height {
                let b = pixelBuffer[i*4 + 0]
                let g = pixelBuffer[i*4 + 1]
                let r = pixelBuffer[i*4 + 2]
                let a = pixelBuffer[i*4 + 3]
                pixelBuffer[i*4 + 0] = r
                pixelBuffer[i*4 + 1] = g
                pixelBuffer[i*4 + 2] = b
                pixelBuffer[i*4 + 3] = a
            }
            
            let provider = CGDataProvider(dataInfo: nil, data: imageData, size: height * bytesPerRow) { _, _, _ in }!
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
            let cgImage = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo, provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
            
            let bitmapRep = NSBitmapImageRep(cgImage: cgImage)
            let pngData = bitmapRep.representation(using: .png, properties: [:])!
            let fileURL = URL(fileURLWithPath: "/Volumes/KIOXIA/testGSmetal/screenshot.png")
            try! pngData.write(to: fileURL)
            print("Saved screenshot to \(fileURL.path)")
            
            hasSavedScreenshot = true
        }
    }
}

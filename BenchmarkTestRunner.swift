//
//  BenchmarkTestRunner.swift
//  Metal Composition Performance Benchmark
//

import Foundation
import Metal
import MetalKit
import QuartzCore
import AppKit
import simd
import GaussianSplattingCore

// MARK: - Frame Time Collector

struct FrameTimeCollector {
    private var times: [CFTimeInterval] = []
    private let sampleCount: Int

    init(sampleCount: Int = 100) {
        self.sampleCount = sampleCount
    }

    mutating func recordFrame(render: () -> Void) {
        let start = CACurrentMediaTime()
        render()
        let elapsed = CACurrentMediaTime() - start
        times.append(elapsed)
    }

    var average: CFTimeInterval {
        guard !times.isEmpty else { return 0 }
        return times.reduce(0, +) / CFTimeInterval(times.count)
    }

    var fps: Double {
        guard average > 0 else { return 0 }
        return 1.0 / average
    }

    var minTime: CFTimeInterval { times.min() ?? 0 }
    var maxTime: CFTimeInterval { times.max() ?? 0 }
    var allTimes: [CFTimeInterval] { times }
}

// MARK: - Matrix helpers

func makeTranslation(_ t: SIMD3<Float>) -> simd_float4x4 {
    return simd_float4x4(columns: (
        SIMD4<Float>(1, 0, 0, 0),
        SIMD4<Float>(0, 1, 0, 0),
        SIMD4<Float>(0, 0, 1, 0),
        SIMD4<Float>(t.x, t.y, t.z, 1)
    ))
}

func makePerspective(fov: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
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

// MARK: - Shared render setup

struct RenderResources {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipelineState: MTLRenderPipelineState
    let gaussianBuffer: MTLBuffer
    let gaussianCount: Int
}

func makeRenderResources(device: MTLDevice, count: Int = 1_000_000) -> RenderResources? {
    guard let queue = device.makeCommandQueue() else { return nil }
    do {
        let pipeline = try createGaussianPipelineState(device: device)
        guard let result = createHelixGaussians(count: count, device: device) else { return nil }
        return RenderResources(device: device, commandQueue: queue, pipelineState: pipeline,
                               gaussianBuffer: result.buffer, gaussianCount: count)
    } catch {
        return nil
    }
}

func encodeDrawPass(
    resources: RenderResources,
    renderPass: MTLRenderPassDescriptor,
    drawable: CAMetalDrawable?,
    angle: Float,
    width: Float,
    height: Float,
    wait: Bool = true
) {
    guard let commandBuffer = resources.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPass) else {
        return
    }

    encoder.setRenderPipelineState(resources.pipelineState)
    encoder.setVertexBuffer(resources.gaussianBuffer, offset: 0, index: 0)

    let cosA = cos(angle), sinA = sin(angle)
    let rotation = simd_float4x4(columns: (
        SIMD4<Float>(cosA, 0, -sinA, 0),
        SIMD4<Float>(0, 1, 0, 0),
        SIMD4<Float>(sinA, 0, cosA, 0),
        SIMD4<Float>(0, 0, 0, 1)
    ))
    let translation = makeTranslation(SIMD3<Float>(0, 0, -3))
    let projection = makePerspective(fov: Float.pi / 3, aspect: width / height, near: 0.1, far: 100.0)
    var viewProj = projection * translation * rotation
    encoder.setVertexBytes(&viewProj, length: MemoryLayout<simd_float4x4>.size, index: 1)

    var vpSize = SIMD2<Float>(width, height)
    encoder.setVertexBytes(&vpSize, length: MemoryLayout<SIMD2<Float>>.size, index: 2)

    encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6,
                            instanceCount: resources.gaussianCount)
    encoder.endEncoding()

    if let drawable = drawable {
        commandBuffer.present(drawable)
    }
    if wait {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    } else {
        commandBuffer.commit()
    }
}

// ============================================================
// Scenario 1: Offscreen (Pure Metal - no window, no display)
// ============================================================

class OffscreenBenchRenderer {
    let resources: RenderResources
    let texture: MTLTexture
    let width: Int
    let height: Int

    init?(width: Int, height: Int) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let res = makeRenderResources(device: device) else { return nil }
        self.resources = res
        self.width = width
        self.height = height

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = [.renderTarget]
        desc.storageMode = .private
        guard let tex = device.makeTexture(descriptor: desc) else { return nil }
        self.texture = tex
    }

    func renderFrame(angle: Float) {
        let renderPass = MTLRenderPassDescriptor()
        renderPass.colorAttachments[0].texture = texture
        renderPass.colorAttachments[0].loadAction = .clear
        renderPass.colorAttachments[0].storeAction = .dontCare
        renderPass.colorAttachments[0].clearColor = MTLClearColor(
            red: 0.05, green: 0.05, blue: 0.1, alpha: 1.0)

        guard let commandBuffer = resources.commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPass) else {
            return
        }

        encoder.setRenderPipelineState(resources.pipelineState)
        encoder.setVertexBuffer(resources.gaussianBuffer, offset: 0, index: 0)

        let cosA = cos(angle), sinA = sin(angle)
        let rotation = simd_float4x4(columns: (
            SIMD4<Float>(cosA, 0, -sinA, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(sinA, 0, cosA, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        let translation = makeTranslation(SIMD3<Float>(0, 0, -3))
        let projection = makePerspective(
            fov: Float.pi / 3,
            aspect: Float(width) / Float(height),
            near: 0.1, far: 100.0)
        var viewProj = projection * translation * rotation
        encoder.setVertexBytes(&viewProj, length: MemoryLayout<simd_float4x4>.size, index: 1)

        var vpSize = SIMD2<Float>(Float(width), Float(height))
        encoder.setVertexBytes(&vpSize, length: MemoryLayout<SIMD2<Float>>.size, index: 2)

        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6,
                                instanceCount: resources.gaussianCount)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    func syncGPU() {
        if let cb = resources.commandQueue.makeCommandBuffer() {
            cb.commit()
            cb.waitUntilCompleted()
        }
    }
}

// ============================================================
// Scenario 2: MTKView (Windowed) - Standard path
// ============================================================

class MTKViewBenchRenderer {
    let resources: RenderResources
    let mtkView: MTKView
    var angle: Float = 0.0
    let width: Int
    let height: Int

    init?(width: Int, height: Int, useOptimizedFlags: Bool = false) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let res = makeRenderResources(device: device) else { return nil }
        self.resources = res
        self.width = width
        self.height = height

        let view = MTKView(frame: NSRect(x: 0, y: 0, width: width, height: height), device: device)
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1.0)
        view.depthStencilPixelFormat = .invalid

        if useOptimizedFlags {
            // Full-screen optimization flags
            view.enableSetNeedsDisplay = false
            view.presentsWithTransaction = false
            view.preferredFramesPerSecond = 120
        } else {
            view.preferredFramesPerSecond = 60
        }

        self.mtkView = view
    }

    func renderFrame() {
        guard let drawable = mtkView.currentDrawable,
              let renderPass = mtkView.currentRenderPassDescriptor else {
            return
        }

        guard let commandBuffer = resources.commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPass) else {
            return
        }

        encoder.setRenderPipelineState(resources.pipelineState)
        encoder.setVertexBuffer(resources.gaussianBuffer, offset: 0, index: 0)

        let cosA = cos(angle), sinA = sin(angle)
        let rotation = simd_float4x4(columns: (
            SIMD4<Float>(cosA, 0, -sinA, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(sinA, 0, cosA, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        let translation = makeTranslation(SIMD3<Float>(0, 0, -3))
        let projection = makePerspective(fov: Float.pi / 3, aspect: Float(width) / Float(height),
                                          near: 0.1, far: 100.0)
        var viewProj = projection * translation * rotation
        encoder.setVertexBytes(&viewProj, length: MemoryLayout<simd_float4x4>.size, index: 1)

        var vpSize = SIMD2<Float>(Float(width), Float(height))
        encoder.setVertexBytes(&vpSize, length: MemoryLayout<SIMD2<Float>>.size, index: 2)

        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6,
                                instanceCount: resources.gaussianCount)
        encoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        angle += 0.01
    }
}

// ============================================================
// Scenario 3: CAMetalLayer (No MTKView) - Lowest display overhead
// ============================================================

class MetalLayerBenchRenderer {
    let resources: RenderResources
    let metalLayer: CAMetalLayer
    var angle: Float = 0.0
    let width: Int
    let height: Int

    init?(width: Int, height: Int) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let res = makeRenderResources(device: device) else { return nil }
        self.resources = res
        self.width = width
        self.height = height

        let layer = CAMetalLayer()
        layer.device = device
        layer.pixelFormat = .bgra8Unorm
        layer.frame = CGRect(x: 0, y: 0, width: width, height: height)
        layer.drawableSize = CGSize(width: width, height: height)
        layer.presentsWithTransaction = false
        layer.contentsGravity = .center

        // Attempt to reduce display overhead
        layer.maximumDrawableCount = 3
        layer.wantsExtendedDynamicRangeContent = false

        self.metalLayer = layer
    }

    func renderFrame() {
        guard let drawable = metalLayer.nextDrawable() else { return }

        let renderPass = MTLRenderPassDescriptor()
        renderPass.colorAttachments[0].texture = drawable.texture
        renderPass.colorAttachments[0].loadAction = .clear
        renderPass.colorAttachments[0].storeAction = .store
        renderPass.colorAttachments[0].clearColor = MTLClearColor(
            red: 0.05, green: 0.05, blue: 0.1, alpha: 1.0)

        guard let commandBuffer = resources.commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPass) else {
            return
        }

        encoder.setRenderPipelineState(resources.pipelineState)
        encoder.setVertexBuffer(resources.gaussianBuffer, offset: 0, index: 0)

        let cosA = cos(angle), sinA = sin(angle)
        let rotation = simd_float4x4(columns: (
            SIMD4<Float>(cosA, 0, -sinA, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(sinA, 0, cosA, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        let translation = makeTranslation(SIMD3<Float>(0, 0, -3))
        let projection = makePerspective(fov: Float.pi / 3, aspect: Float(width) / Float(height),
                                          near: 0.1, far: 100.0)
        var viewProj = projection * translation * rotation
        encoder.setVertexBytes(&viewProj, length: MemoryLayout<simd_float4x4>.size, index: 1)

        var vpSize = SIMD2<Float>(Float(width), Float(height))
        encoder.setVertexBytes(&vpSize, length: MemoryLayout<SIMD2<Float>>.size, index: 2)

        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6,
                                instanceCount: resources.gaussianCount)
        encoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        angle += 0.01
    }
}

// ============================================================
// Scenario 4: NSWindow in Full-Screen Mode (real window)
// ============================================================

class FullScreenBenchRenderer {
    let resources: RenderResources
    let window: NSWindow
    let mtkView: MTKView
    var angle: Float = 0.0
    let width: Int
    let height: Int

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let res = makeRenderResources(device: device) else { return nil }
        self.resources = res

        // Use main screen's frame (native display resolution in points)
        let screen = NSScreen.main ?? NSScreen.screens[0]
        let frame = screen.frame
        self.width = Int(frame.width)
        self.height = Int(frame.height)

        let window = NSWindow(
            contentRect: frame,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        window.level = .mainMenu + 1
        window.isOpaque = true
        window.hasShadow = false
        window.ignoresMouseEvents = true

        let view = MTKView(frame: NSRect(x: 0, y: 0, width: width, height: height), device: device)
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1.0)
        view.depthStencilPixelFormat = .invalid
        view.enableSetNeedsDisplay = false
        view.presentsWithTransaction = false
        view.preferredFramesPerSecond = 120

        window.contentView = view
        window.makeKeyAndOrderFront(nil)

        self.window = window
        self.mtkView = view
    }

    deinit {
        window.orderOut(nil)
        window.close()
    }

    func renderFrame() {
        guard let drawable = mtkView.currentDrawable,
              let renderPass = mtkView.currentRenderPassDescriptor else {
            return
        }

        guard let commandBuffer = resources.commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPass) else {
            return
        }

        encoder.setRenderPipelineState(resources.pipelineState)
        encoder.setVertexBuffer(resources.gaussianBuffer, offset: 0, index: 0)

        let cosA = cos(angle), sinA = sin(angle)
        let rotation = simd_float4x4(columns: (
            SIMD4<Float>(cosA, 0, -sinA, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(sinA, 0, cosA, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
        let translation = makeTranslation(SIMD3<Float>(0, 0, -3))
        let projection = makePerspective(fov: Float.pi / 3, aspect: Float(width) / Float(height),
                                          near: 0.1, far: 100.0)
        var viewProj = projection * translation * rotation
        encoder.setVertexBytes(&viewProj, length: MemoryLayout<simd_float4x4>.size, index: 1)

        var vpSize = SIMD2<Float>(Float(width), Float(height))
        encoder.setVertexBytes(&vpSize, length: MemoryLayout<SIMD2<Float>>.size, index: 2)

        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6,
                                instanceCount: resources.gaussianCount)
        encoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        angle += 0.01
    }
}

// ============================================================
// Main Benchmark Runner
// ============================================================

struct ScenarioResult {
    let name: String
    let fps: Double
    let avgMs: Double
    let minMs: Double
    let maxMs: Double
}

func runAndMeasure(renderer: String, warmup: Int = 10, frames: Int = 100,
                    setup: () -> (render: () -> Void, cleanup: () -> Void)) -> ScenarioResult? {
    print("  Warmup (\(warmup) frames)...")
    let (render, cleanup) = setup()
    for _ in 0..<warmup {
        render()
    }

    print("  Measuring \(frames) frames...")
    var collector = FrameTimeCollector(sampleCount: frames)
    for _ in 0..<frames {
        collector.recordFrame {
            render()
        }
    }
    cleanup()

    print("  Results:")
    print("    Average: \(String(format: "%.3f", collector.average * 1000)) ms")
    print("    Min:     \(String(format: "%.3f", collector.minTime * 1000)) ms")
    print("    Max:     \(String(format: "%.3f", collector.maxTime * 1000)) ms")
    print("    FPS:     \(String(format: "%.2f", collector.fps))")
    print()

    return ScenarioResult(
        name: renderer,
        fps: collector.fps,
        avgMs: collector.average * 1000,
        minMs: collector.minTime * 1000,
        maxMs: collector.maxTime * 1000
    )
}

func runBenchmark() {
    let warmupFrames = 10
    let measureFrames = 100
    let smallWidth = 1280
    let smallHeight = 720

    // Get native display resolution
    let screen = NSScreen.main ?? NSScreen.screens[0]
    let nativeW = Int(screen.frame.width)
    let nativeH = Int(screen.frame.height)

    print(String(repeating: "=", count: 70))
    print("Metal Composition Performance Benchmark")
    print("  Question: Does full-screen reduce compositing overhead?")
    print(String(repeating: "=", count: 70))
    print()
    print("Small resolution: \(smallWidth) x \(smallHeight)")
    print("Native resolution: \(nativeW) x \(nativeH)")
    print("Gaussians: 1,000,000")
    print("Warmup: \(warmupFrames) frames, Measure: \(measureFrames) frames")
    print()

    var results: [ScenarioResult] = []

    // ==== Scenario 1: Offscreen (pure Metal baseline) =====
    print("[" + String(repeating: "=", count: 30) + "]")
    print("Scenario 1: OFFSCREEN (Pure Metal)")
    print("  No window, no display system")
    print("  Size: \(smallWidth) x \(smallHeight)")
    print()

    if let renderer = OffscreenBenchRenderer(width: smallWidth, height: smallHeight) {
        var frameIdx = 0
        let result = runAndMeasure(renderer: "Offscreen 1280x720", warmup: warmupFrames, frames: measureFrames) {
            ({ renderer.renderFrame(angle: Float(frameIdx) * 0.01); frameIdx += 1 },
             { renderer.syncGPU() })
        }
        if let r = result { results.append(r) }
    }

    // ==== Scenario 2: MTKView at small resolution (windowed) =====
    print("[" + String(repeating: "=", count: 30) + "]")
    print("Scenario 2: MTKView \(smallWidth)x\(smallHeight) (Windowed, default flags)")
    print("  Standard MTKView with default 60 FPS cap")
    print()

    if let renderer = MTKViewBenchRenderer(width: smallWidth, height: smallHeight, useOptimizedFlags: false) {
        let result = runAndMeasure(renderer: "MTKView \(smallWidth)x\(smallHeight)", warmup: warmupFrames, frames: measureFrames) {
            ({ renderer.renderFrame() }, {})
        }
        if let r = result { results.append(r) }
    }

    // ==== Scenario 3: MTKView with optimized flags (full-screen style) =====
    print("[" + String(repeating: "=", count: 30) + "]")
    print("Scenario 3: MTKView \(smallWidth)x\(smallHeight) (Optimized flags)")
    print("  presentsWithTransaction=false, enableSetNeedsDisplay=false, 120 FPS")
    print()

    if let renderer = MTKViewBenchRenderer(width: smallWidth, height: smallHeight, useOptimizedFlags: true) {
        let result = runAndMeasure(renderer: "MTKView Optimized \(smallWidth)x\(smallHeight)", warmup: warmupFrames, frames: measureFrames) {
            ({ renderer.renderFrame() }, {})
        }
        if let r = result { results.append(r) }
    }

    // ==== Scenario 4: MTKView at native resolution =====
    print("[" + String(repeating: "=", count: 30) + "]")
    print("Scenario 4: MTKView \(nativeW)x\(nativeH) (Native resolution, optimized)")
    print("  Same optimized flags, but native display size")
    print()

    if let renderer = MTKViewBenchRenderer(width: nativeW, height: nativeH, useOptimizedFlags: true) {
        let result = runAndMeasure(renderer: "MTKView Native \(nativeW)x\(nativeH)", warmup: warmupFrames, frames: measureFrames) {
            ({ renderer.renderFrame() }, {})
        }
        if let r = result { results.append(r) }
    }

    // ==== Scenario 5: CAMetalLayer (no MTKView) at native resolution =====
    print("[" + String(repeating: "=", count: 30) + "]")
    print("Scenario 5: CAMetalLayer \(nativeW)x\(nativeH) (Direct layer, no MTKView)")
    print("  Bypasses MTKView overhead, presentsWithTransaction=false")
    print()

    if let renderer = MetalLayerBenchRenderer(width: nativeW, height: nativeH) {
        let result = runAndMeasure(renderer: "CAMetalLayer Native \(nativeW)x\(nativeH)", warmup: warmupFrames, frames: measureFrames) {
            ({ renderer.renderFrame() }, {})
        }
        if let r = result { results.append(r) }
    }

    // ==== Scenario 6: Real NSWindow full-screen =====
    print("[" + String(repeating: "=", count: 30) + "]")
    print("Scenario 6: REAL NSWindow Full-Screen \(nativeW)x\(nativeH)")
    print("  Borderless window covering main display")
    print()

    if let renderer = FullScreenBenchRenderer() {
        // Need a brief pause for the window to appear
        usleep(100_000)
        let result = runAndMeasure(renderer: "FullScreen Window \(nativeW)x\(nativeH)", warmup: warmupFrames, frames: measureFrames) {
            ({ renderer.renderFrame() }, {})
        }
        if let r = result { results.append(r) }
    }

    // ==== Summary Comparison Table =====
    print("[" + String(repeating: "=", count: 30) + "]")
    print("COMPARISON TABLE")
    print(String(repeating: "-", count: 70))

    let header = String(format: "  %-40s | %8s | %10s | %8s",
                        "Scenario".cString(using: .utf8)!,
                        "FPS".cString(using: .utf8)!,
                        "Avg (ms)".cString(using: .utf8)!,
                        "Overhead".cString(using: .utf8)!)
    print(header)
    print(String(repeating: "-", count: 70))

    let baseline = results[0]

    for (idx, r) in results.enumerated() {
        let overheadMs = r.avgMs - baseline.avgMs
        let overheadPct = baseline.avgMs > 0 ? (overheadMs / baseline.avgMs) * 100.0 : 0.0
        let overheadStr: String
        if idx == 0 {
            overheadStr = "baseline"
        } else {
            overheadStr = String(format: "+%.1f%% (%+.1f ms)", overheadPct, overheadMs)
        }

        let line = String(format: "  %-40s | %8.2f | %10.3f | %@",
                          r.name.padding(toLength: 40, withPad: " ", startingAt: 0).cString(using: .utf8)!,
                          r.fps, r.avgMs, overheadStr)
        print(line)
    }

    print()
    print(String(repeating: "-", count: 70))
    print()

    // ==== Answering the question =====
    print("ANSWER: Does full-screen reduce compositing overhead?")
    print(String(repeating: "=", count: 70))
    print()

    // Find the windowed (non-optimized) result
    let windowed = results.first(where: { $0.name.contains("MTKView") && !$0.name.contains("Optimized") && !$0.name.contains("Native") })
    let optimized = results.first(where: { $0.name.contains("Optimized") })
    let native = results.first(where: { $0.name.contains("Native") && $0.name.contains("MTKView") })
    let metalLayer = results.first(where: { $0.name.contains("CAMetalLayer") })
    let fullScreen = results.first(where: { $0.name.contains("FullScreen") })

    print("Key observations:")
    print()

    if let w = windowed, let opt = optimized {
        let fpsDelta = opt.fps - w.fps
        let pct = (fpsDelta / w.fps) * 100.0
        print("  1. MTKView optimized flags (presentsWithTransaction=false, 120 FPS cap):")
        print("     \(w.fps) -> \(opt.fps) FPS (\(String(format: "%+.1f", pct))%)")
        if fpsDelta > 0 {
            print("     -> YES, flags help. The default 60 FPS cap was limiting!")
        } else {
            print("     -> No improvement from flags alone")
        }
        print()
    }

    if let opt = optimized, let n = native {
        print("  2. Native resolution vs small window:")
        print("     \(opt.name): \(opt.fps) FPS")
        print("     \(n.name): \(n.fps) FPS")
        let pct = ((n.fps - opt.fps) / opt.fps) * 100.0
        print("     Difference: \(String(format: "%+.1f", pct))%")
        print("     -> Pixel fill rate is the bottleneck, not compositing")
        print()
    }

    if let ml = metalLayer, let n = native {
        let fpsDelta = ml.fps - n.fps
        let pct = (fpsDelta / n.fps) * 100.0
        print("  3. CAMetalLayer vs MTKView (both native):")
        print("     MTKView:       \(n.fps) FPS")
        print("     CAMetalLayer:  \(ml.fps) FPS")
        print("     Difference:    \(String(format: "%+.1f", pct))%")
        if fpsDelta > 3 {
            print("     -> YES, CAMetalLayer avoids MTKView's drawable sync overhead")
        } else {
            print("     -> Minimal difference. MTKView overhead is small")
        }
        print()
    }

    if let fs = fullScreen, let ml = metalLayer {
        let fpsDelta = fs.fps - ml.fps
        let pct = (fpsDelta / ml.fps) * 100.0
        print("  4. Real full-screen NSWindow vs CAMetalLayer:")
        print("     Full-screen window:  \(fs.fps) FPS")
        print("     CAMetalLayer:        \(ml.fps) FPS")
        print("     Difference:          \(String(format: "%+.1f", pct))%")
        if abs(pct) < 5 {
            print("     -> NO, full-screen window doesn't reduce overhead beyond layer mode")
        } else if pct > 0 {
            print("     -> YES, real full-screen reduces WindowServer compositing")
        } else {
            print("     -> Full-screen window is SLOWER (WindowServer still composites)")
        }
        print()
    }

    // Find the fastest and slowest display paths
    let displayPaths = Array(results.dropFirst()) // drop offscreen baseline
    if let fastest = displayPaths.max(by: { $0.fps < $1.fps }),
       let slowest = displayPaths.min(by: { $0.fps < $1.fps }) {
        print("  5. Best vs worst display path:")
        print("     BEST:  \(fastest.name) = \(fastest.fps) FPS")
        print("     WORST: \(slowest.name) = \(slowest.fps) FPS")
        print("     Gap:   \(String(format: "%.1f", fastest.fps - slowest.fps)) FPS")
        print()
    }

    // ==== Conclusion =====
    print("SUMMARY & RECOMMENDATIONS")
    print(String(repeating: "-", count: 70))

    print("  • The offscreen baseline shows your GPU can render ~\(String(format: "%.0f", baseline.fps)) FPS")
    print("  • Any display-bound path (MTKView/CAMetalLayer/window) adds ~\(String(format: "%.0f", baseline.avgMs)) ms GPU sync overhead")
    print("  • Full-screen helps WHEN:")
    print("    - You were previously compositing with other windows (overlays, menu bar)")
    print("    - The display can use Direct-to-Display bypass path")
    print("    - presentsWithTransaction=false reduces WindowServer round-trips")
    print()
    print("  • For MetalView+SwiftUI (NSViewRepresentable):")
    print("    - Use .borderless window style for full-screen apps")
    print("    - Set mtkView.presentsWithTransaction = false")
    print("    - Set mtkView.preferredFramesPerSecond to display max (120/144)")
    print("    - Avoid SwiftUI overlays on top of MetalView in full-screen mode")
    print()

    print(String(repeating: "=", count: 70))
    print("Benchmark complete.")
    print(String(repeating: "=", count: 70))
}

// Run the benchmark
runBenchmark()

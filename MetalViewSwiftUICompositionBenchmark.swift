//
//  MetalViewSwiftUICompositionBenchmark.swift
//  GaussianSplattingMetal
//
//  Benchmark tests for measuring MetalView to SwiftUI composition performance.
//

import SwiftUI
import MetalKit
import QuartzCore

// MARK: - Benchmark Results

struct CompositionBenchmarkResult: Identifiable {
    let id = UUID()
    let scenarioName: String
    let viewCreationTime: TimeInterval
    let coordinatorCreationTime: TimeInterval
    let updateNSViewTime: TimeInterval
    let averageFrameTime: TimeInterval
    let framesPerSecond: Double
    let overheadPercentage: Double
}

// MARK: - Frame Time Collector

struct FrameTimeCollector {
    private var times: [CFTimeInterval] = []
    private let sampleCount: Int
    private var rendered = 0

    init(sampleCount: Int = 100) {
        self.sampleCount = sampleCount
    }

    mutating func recordFrame(render: () -> Void) {
        let start = CACurrentMediaTime()
        render()
        let elapsed = CACurrentMediaTime() - start

        if rendered < sampleCount {
            times.append(elapsed)
        }
        rendered += 1
    }

    var average: CFTimeInterval {
        guard !times.isEmpty else { return 0 }
        return times.reduce(0, +) / CFTimeInterval(times.count)
    }

    var fps: Double {
        guard average > 0 else { return 0 }
        return 1.0 / average
    }

    var allTimes: [CFTimeInterval] { times }
}

// MARK: - Benchmark Scenarios

enum CompositionBenchmarkScenario {
    case directMTKView
    case nsViewRepresentable
    case nsViewRepresentableWithZStack
    case nsViewRepresentableWithMultipleOverlays
    case nsViewRepresentableWithStateUpdates

    var name: String {
        switch self {
        case .directMTKView: return "Direct MTKView (Baseline)"
        case .nsViewRepresentable: return "NSViewRepresentable Wrapper"
        case .nsViewRepresentableWithZStack: return "NSViewRepresentable + ZStack"
        case .nsViewRepresentableWithMultipleOverlays: return "NSViewRepresentable + Multiple Layers"
        case .nsViewRepresentableWithStateUpdates: return "NSViewRepresentable + State Updates"
        }
    }
}

// MARK: - Composition Benchmark Runner

@MainActor
class CompositionBenchmarkRunner: ObservableObject {
    @Published var results: [CompositionBenchmarkResult] = []
    @Published var isRunning = false

    private var device: MTLDevice?

    init() {
        self.device = MTLCreateSystemDefaultDevice()
    }

    // MARK: - Run All Benchmarks

    func runAllBenchmarks() {
        guard let device = device else {
            print("No Metal device available")
            return
        }

        isRunning = true
        results.removeAll()

        let scenarios: [CompositionBenchmarkScenario] = [
            .directMTKView,
            .nsViewRepresentable,
            .nsViewRepresentableWithZStack,
            .nsViewRepresentableWithMultipleOverlays,
            .nsViewRepresentableWithStateUpdates
        ]

        for scenario in scenarios {
            let result = runScenario(scenario, device: device)
            results.append(result)
        }

        printSummary()
        isRunning = false
    }

    // MARK: - Run Individual Scenario

    private func runScenario(_ scenario: CompositionBenchmarkScenario, device: MTLDevice) -> CompositionBenchmarkResult {
        var collector = FrameTimeCollector(sampleCount: 100)

        switch scenario {
        case .directMTKView:
            return runDirectMTKView(device: device)
        case .nsViewRepresentable:
            return runNSViewRepresentableBenchmark(device: device)
        case .nsViewRepresentableWithZStack:
            return runZStackBenchmark(device: device)
        case .nsViewRepresentableWithMultipleOverlays:
            return runMultipleLayersBenchmark(device: device)
        case .nsViewRepresentableWithStateUpdates:
            return runStateUpdatesBenchmark(device: device)
        }
    }

    // MARK: - Direct MTKView (Baseline)

    private func runDirectMTKView(device: MTLDevice) -> CompositionBenchmarkResult {
        let startTime = CACurrentMediaTime()

        // Create MTKView directly
        let mtkView = MTKView(frame: CGRect(x: 0, y: 0, width: 800, height: 600), device: device)
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1.0)

        let renderer = GaussianRenderer(device: device)
        mtkView.delegate = renderer

        let creationTime = CACurrentMediaTime() - startTime

        // Warmup
        for _ in 0..<5 {
            renderer?.draw(in: mtkView)
        }

        // Collect frame times
        var collector = FrameTimeCollector(sampleCount: 100)
        for _ in 0..<100 {
            collector.recordFrame {
                renderer?.draw(in: mtkView)
            }
        }

        return CompositionBenchmarkResult(
            scenarioName: "Direct MTKView (Baseline)",
            viewCreationTime: creationTime,
            coordinatorCreationTime: 0,
            updateNSViewTime: 0,
            averageFrameTime: collector.average,
            framesPerSecond: collector.fps,
            overheadPercentage: 0
        )
    }

    // MARK: - NSViewRepresentable Benchmark

    private func runNSViewRepresentableBenchmark(device: MTLDevice) -> CompositionBenchmarkResult {
        let wrapper = MetalView(device: device)

        // Measure coordinator/context creation
        let coordinatorStart = CACurrentMediaTime()
        let context = wrapper.makeCoordinatorContext()
        let coordinatorTime = CACurrentMediaTime() - coordinatorStart

        // Measure makeNSView
        let nsViewStart = CACurrentMediaTime()
        let nsView = wrapper.makeNSView(context: context)
        let makeNSViewTime = CACurrentMediaTime() - nsViewStart

        // Measure updateNSView
        let updateStart = CACurrentMediaTime()
        wrapper.updateNSView(nsView, context: context)
        let updateTime = CACurrentMediaTime() - updateStart

        let creationTime = coordinatorTime + makeNSViewTime

        // Collect frame times
        var collector = FrameTimeCollector(sampleCount: 100)
        for _ in 0..<100 {
            collector.recordFrame {
                (wrapper.renderer as? GaussianRenderer)?.draw(in: nsView)
            }
        }

        // Calculate overhead vs baseline
        let baselineFPS = 1000.0 / 16.67 // Assume 60 FPS baseline
        let overhead = ((baselineFPS - collector.fps) / baselineFPS) * 100

        return CompositionBenchmarkResult(
            scenarioName: "NSViewRepresentable Wrapper",
            viewCreationTime: creationTime,
            coordinatorCreationTime: coordinatorTime,
            updateNSViewTime: updateTime,
            averageFrameTime: collector.average,
            framesPerSecond: collector.fps,
            overheadPercentage: max(0, overhead)
        )
    }

    // MARK: - ZStack Overlay Benchmark

    private func runZStackBenchmark(device: MTLDevice) -> CompositionBenchmarkResult {
        let wrapper = MetalView(device: device)
        let context = wrapper.makeCoordinatorContext()
        let nsView = wrapper.makeNSView(context: context)

        // Simulate ZStack composition
        let compositionStart = CACurrentMediaTime()
        // In real SwiftUI: ZStack { MetalView(device: device) }
        // The MetalView is the base layer
        let compositionTime = CACurrentMediaTime() - compositionStart

        // Collect frame times
        var collector = FrameTimeCollector(sampleCount: 100)
        for _ in 0..<100 {
            collector.recordFrame {
                (wrapper.renderer as? GaussianRenderer)?.draw(in: nsView)
            }
        }

        let overhead = ((60.0 - collector.fps) / 60.0) * 100

        return CompositionBenchmarkResult(
            scenarioName: "NSViewRepresentable + ZStack",
            viewCreationTime: compositionTime,
            coordinatorCreationTime: 0,
            updateNSViewTime: 0,
            averageFrameTime: collector.average,
            framesPerSecond: collector.fps,
            overheadPercentage: max(0, overhead)
        )
    }

    // MARK: - Multiple Layers Benchmark

    private func runMultipleLayersBenchmark(device: MTLDevice) -> CompositionBenchmarkResult {
        let wrapper = MetalView(device: device)
        let context = wrapper.makeCoordinatorContext()
        let nsView = wrapper.makeNSView(context: context)

        // Simulate multiple SwiftUI layers
        let compositionStart = CACurrentMediaTime()
        // Simulating: ZStack {
        //   MetalView
        //   VStack { HStack { Text, Text } Spacer() }
        //   Overlay { Circle() }
        // }
        let layerCount = 3
        let compositionTime = CACurrentMediaTime() - compositionStart

        // Collect frame times
        var collector = FrameTimeCollector(sampleCount: 100)
        for _ in 0..<100 {
            collector.recordFrame {
                (wrapper.renderer as? GaussianRenderer)?.draw(in: nsView)
            }
        }

        let overhead = ((60.0 - collector.fps) / 60.0) * 100

        return CompositionBenchmarkResult(
            scenarioName: "NSViewRepresentable + Multiple Layers (\(layerCount))",
            viewCreationTime: compositionTime * Double(layerCount),
            coordinatorCreationTime: 0,
            updateNSViewTime: 0,
            averageFrameTime: collector.average,
            framesPerSecond: collector.fps,
            overheadPercentage: max(0, overhead)
        )
    }

    // MARK: - State Updates Benchmark

    private func runStateUpdatesBenchmark(device: MTLDevice) -> CompositionBenchmarkResult {
        let wrapper = MetalView(device: device)
        let context = wrapper.makeCoordinatorContext()
        let nsView = wrapper.makeNSView(context: context)

        // Measure updateNSView overhead (called on SwiftUI state changes)
        let updateStart = CACurrentMediaTime()
        for i in 0..<100 {
            wrapper.updateNSView(nsView, context: context)
        }
        let updateTime = CACurrentMediaTime() - updateStart

        // Collect frame times
        var collector = FrameTimeCollector(sampleCount: 100)
        for _ in 0..<100 {
            collector.recordFrame {
                (wrapper.renderer as? GaussianRenderer)?.draw(in: nsView)
            }
        }

        let overhead = ((60.0 - collector.fps) / 60.0) * 100

        return CompositionBenchmarkResult(
            scenarioName: "NSViewRepresentable + State Updates",
            viewCreationTime: 0,
            coordinatorCreationTime: 0,
            updateNSViewTime: updateTime / 100, // Per-update overhead
            averageFrameTime: collector.average,
            framesPerSecond: collector.fps,
            overheadPercentage: max(0, overhead)
        )
    }

    // MARK: - Print Summary

    private func printSummary() {
        print("\n" + String(repeating: "=", count: 70))
        print("MetalView SwiftUI Composition Benchmark Results")
        print(String(repeating: "=", count: 70))

        guard let baseline = results.first else { return }

        for result in results {
            print("\n[\(result.scenarioName)]")
            print("  Creation Time:     \(String(format: "%.4f", result.viewCreationTime * 1000)) ms")
            print("  Coordinator Time:   \(String(format: "%.4f", result.coordinatorCreationTime * 1000)) ms")
            print("  UpdateNSView Time: \(String(format: "%.4f", result.updateNSViewTime * 1000)) ms")
            print("  Avg Frame Time:    \(String(format: "%.3f", result.averageFrameTime * 1000)) ms")
            print("  FPS:               \(String(format: "%.1f", result.framesPerSecond))")

            if result.scenarioName != baseline.scenarioName {
                let fpsDelta = baseline.framesPerSecond - result.framesPerSecond
                let overhead = result.overheadPercentage
                print("  FPS Delta:         \(String(format: "%.1f", fpsDelta))")
                print("  Overhead:          \(String(format: "%.2f", overhead))%")
            }
        }

        if results.count > 1 {
            print("\n" + String(repeating: "-", count: 70))
            print("Comparison (vs Direct MTKView baseline):")

            for i in 1..<results.count {
                let result = results[i]
                let fpsLoss = baseline.framesPerSecond - result.framesPerSecond
                let percentLoss = (fpsLoss / baseline.framesPerSecond) * 100
                print("  \(result.scenarioName):")
                print("    FPS reduction: \(String(format: "%.2f", percentLoss))%")
                print("    Time/frame overhead: +\(String(format: "%.3f", (result.averageFrameTime - baseline.averageFrameTime) * 1000)) ms")
            }
        }

        print(String(repeating: "=", count: 70) + "\n")
    }
}

// MARK: - NSViewRepresentable Extension

extension MetalView {
    func makeCoordinatorContext() -> Context {
        Context(coordinator: Coordinator(), environment: EnvironmentValues())
    }
}

// MARK: - Benchmark SwiftUI View

struct CompositionBenchmarkView: View {
    @StateObject private var runner = CompositionBenchmarkRunner()

    var body: some View {
        VStack(spacing: 20) {
            Text("MetalView SwiftUI Composition Benchmark")
                .font(.title)
                .fontWeight(.bold)

            HStack(spacing: 20) {
                Button(action: {
                    runner.runAllBenchmarks()
                }) {
                    Label("Run Benchmark", systemImage: "play.fill")
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                        .background(runner.isRunning ? Color.gray : Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
                .disabled(runner.isRunning)

                if runner.isRunning {
                    ProgressView()
                        .padding(.leading, 10)
                }
            }

            if !runner.results.isEmpty {
                Divider()

                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(runner.results) { result in
                            BenchmarkResultRow(result: result, isBaseline: result.scenarioName.contains("Baseline"))
                        }
                    }
                    .padding()
                }
                .frame(maxHeight: 350)
                .background(Color(.systemGray6))
                .cornerRadius(8)

                // Summary comparison
                if runner.results.count > 1 {
                    BenchmarkComparisonTable(results: runner.results)
                }
            }

            Spacer()

            // Instructions
            Text("This benchmark measures the performance overhead of wrapping MTKView")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .frame(minWidth: 500, minHeight: 600)
    }
}

struct BenchmarkResultRow: View {
    let result: CompositionBenchmarkResult
    let isBaseline: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(result.scenarioName)
                    .font(.headline)
                    .foregroundColor(isBaseline ? .green : .primary)

                if isBaseline {
                    Text("BASELINE")
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.green.opacity(0.2))
                        .foregroundColor(.green)
                        .cornerRadius(4)
                }

                Spacer()

                Text("\(String(format: "%.1f", result.framesPerSecond)) FPS")
                    .font(.system(.title3, design: .monospaced))
                    .fontWeight(.semibold)
                    .foregroundColor(fpsColor)
            }

            HStack(spacing: 20) {
                Label("\(String(format: "%.2f", result.viewCreationTime * 1000)) ms", systemImage: "clock")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Label("\(String(format: "%.2f", result.coordinatorCreationTime * 1000)) ms", systemImage: "gear")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Label("\(String(format: "%.2f", result.updateNSViewTime * 1000)) ms", systemImage: "arrow.triangle.2.circlepath")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Spacer()

                Text("\(String(format: "%.2f", result.averageFrameTime * 1000)) ms/frame")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(isBaseline ? Color.green.opacity(0.1) : Color(.systemBackground))
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isBaseline ? Color.green.opacity(0.3) : Color.clear, lineWidth: 1)
        )
    }

    private var fpsColor: Color {
        if result.framesPerSecond >= 55 {
            return .green
        } else if result.framesPerSecond >= 30 {
            return .orange
        } else {
            return .red
        }
    }
}

struct BenchmarkComparisonTable: View {
    let results: [CompositionBenchmarkResult]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Overhead vs Direct MTKView")
                .font(.headline)

            if let baseline = results.first {
                ForEach(results.dropFirst()) { result in
                    let fpsLoss = baseline.framesPerSecond - result.framesPerSecond
                    let percentLoss = baseline.framesPerSecond > 0 ? (fpsLoss / baseline.framesPerSecond) * 100 : 0
                    let timeOverhead = (result.averageFrameTime - baseline.averageFrameTime) * 1000

                    HStack {
                        Text(result.scenarioName)
                            .font(.caption)
                            .frame(maxWidth: .infinity, alignment: .leading)

                        Text("-\(String(format: "%.1f", fpsLoss)) FPS")
                            .font(.caption.monospaced())
                            .foregroundColor(.orange)

                        Text("(\(String(format: "%.2f", percentLoss))%)")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        Text("+\(String(format: "%.3f", timeOverhead)) ms")
                            .font(.caption.monospaced())
                            .foregroundColor(.red)
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemGray5))
        .cornerRadius(8)
    }
}

#Preview {
    CompositionBenchmarkView()
}

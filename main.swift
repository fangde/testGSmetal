//
//  main.swift
//  GSOffscreen
//
//  Command-line interface for Gaussian Splatting offscreen rendering.
//

import Foundation
import Metal
import simd
import GaussianSplattingCore

// MARK: - Command Line Arguments

struct RenderOptions {
    var width: Int = 1280
    var height: Int = 720
    var outputFile: String = "output.png"
    var gaussianCount: Int = 100_000
    var animationFrames: Int = 0
    var framePrefix: String = "frame"
    var pattern: String = "helix"
    var showHelp: Bool = false
    var benchmark: Bool = false
    var warmupFrames: Int = 5
}

func parseArguments(_ args: [String]) -> RenderOptions {
    var options = RenderOptions()
    var i = 1

    while i < args.count {
        let arg = args[i]

        switch arg {
        case "--width", "-w":
            if i + 1 < args.count {
                options.width = Int(args[i + 1]) ?? 1280
                i += 2
            } else {
                i += 1
            }

        case "--height", "-h":
            if i + 1 < args.count {
                options.height = Int(args[i + 1]) ?? 720
                i += 2
            } else {
                i += 1
            }

        case "--output", "-o":
            if i + 1 < args.count {
                options.outputFile = args[i + 1]
                i += 2
            } else {
                i += 1
            }

        case "--gaussians", "-n":
            if i + 1 < args.count {
                options.gaussianCount = Int(args[i + 1]) ?? 100_000
                i += 2
            } else {
                i += 1
            }

        case "--frames", "-f":
            if i + 1 < args.count {
                options.animationFrames = Int(args[i + 1]) ?? 0
                i += 2
            } else {
                i += 1
            }

        case "--prefix", "-p":
            if i + 1 < args.count {
                options.framePrefix = args[i + 1]
                i += 2
            } else {
                i += 1
            }

        case "--pattern":
            if i + 1 < args.count {
                options.pattern = args[i + 1]
                i += 2
            } else {
                i += 1
            }

        case "--benchmark":
            options.benchmark = true
            i += 1

        case "--warmup":
            if i + 1 < args.count {
                options.warmupFrames = Int(args[i + 1]) ?? 5
                i += 2
            } else {
                i += 1
            }

        case "--help", "-help":
            options.showHelp = true
            i += 1

        default:
            i += 1
        }
    }

    return options
}

func printHelp() {
    print("""
    Gaussian Splatting Offscreen Renderer
    \(String(repeating: "=", count: 60))

    Usage: GSOffscreen [options]

    Options:
      -w, --width <pixels>       Output width (default: 1280)
      -h, --height <pixels>      Output height (default: 720)
      -o, --output <file>        Output PNG file (default: output.png)
      -n, --gaussians <count>    Number of Gaussians (default: 100000)
      -f, --frames <count>       Number of animation frames (default: 0)
      -p, --prefix <prefix>      Frame filename prefix (default: frame)
      --pattern <name>           Test pattern: helix, random (default: helix)
      --benchmark                Run GPU-only benchmark (no PNG save)
      --warmup <frames>          Warmup frames for benchmark (default: 5)
      --help, -help              Show this help message

    Examples:
      # Single frame render
      GSOffscreen --width 1920 --height 1080 --output scene.png --gaussians 150000

      # GPU benchmark (150k Gaussians, 200 frames)
      GSOffscreen --benchmark --gaussians 150000 --frames 200

      # Animation with PNG save
      GSOffscreen --frames 100 --prefix frame --gaussians 150000
    """)
}

// MARK: - Main

print(String(repeating: "=", count: 60))
print("Gaussian Splatting Offscreen Renderer")
print(String(repeating: "=", count: 60))
print()

// Parse command line arguments
let options = parseArguments(CommandLine.arguments)

if options.showHelp {
    printHelp()
    exit(0)
}

// Print configuration
print("Configuration:")
print("  Width:       \(options.width)px")
print("  Height:      \(options.height)px")
print("  Gaussians:  \(options.gaussianCount)")
print("  Pattern:     \(options.pattern)")
print()

if options.animationFrames > 0 {
    print("Animation:")
    print("  Frames:     \(options.animationFrames)")
    print("  Prefix:     \(options.framePrefix)")
    print()
}

// Initialize renderer
print("Initializing renderer...")
let renderer: OffscreenRenderer
do {
    renderer = try OffscreenRenderer(width: options.width, height: options.height)
    print("✓ Renderer initialized")
} catch {
    print("✗ Failed to initialize renderer: \(error)")
    exit(1)
}

// Create test Gaussians
print("Creating \(options.gaussianCount) Gaussians...")
switch options.pattern {
case "random":
    renderer.createRandomGaussians(count: options.gaussianCount)
default: // helix
    renderer.createTestGaussians(count: options.gaussianCount)
}
print("✓ Gaussians created")

// Create projection matrix
let aspect = Float(options.width) / Float(options.height)
let projection = OffscreenRenderer.perspectiveMatrix(
    fov: Float.pi / 3,  // 60 degrees
    aspect: aspect,
    near: 0.1,
    far: 100.0
)

// Precompute view-projection matrices for animation frames
func makeViewProj(forFrame frame: Int, projection: simd_float4x4) -> simd_float4x4 {
    let angle = Float(frame) * 0.1
    let rotation = OffscreenRenderer.rotationMatrix(angle: angle, axis: SIMD3<Float>(0, 1, 0))
    let translation = OffscreenRenderer.translationMatrix(SIMD3<Float>(0, 0, -3))
    let view = multiplyMatrix(translation, rotation)
    return multiplyMatrix(projection, view)
}

// ============================================================
// BENCHMARK MODE: GPU-only rendering (no PNG, no CPU sync per frame)
// ============================================================
if options.benchmark {
    print()
    print("BENCHMARK MODE (GPU-only, no PNG save)")
    print("  Warmup frames: \(options.warmupFrames)")
    print("  Measure frames: \(options.animationFrames > 0 ? options.animationFrames : 100)")
    print()

    let totalFrames = options.animationFrames > 0 ? options.animationFrames : 100

    // Warmup - initialize caches, GPU pipelines
    print("Warmup...")
    for frame in 0..<options.warmupFrames {
        let vp = makeViewProj(forFrame: frame, projection: projection)
        renderer.renderFast(viewMatrix: vp)
    }
    renderer.waitForGPU()
    print("  Warmup complete")

    // Timed rendering
    print("Rendering \(totalFrames) frames...")
    let startTime = Date()

    for frame in 0..<totalFrames {
        let vp = makeViewProj(forFrame: frame, projection: projection)
        renderer.renderFast(viewMatrix: vp)
    }
    // Wait for all GPU work to complete
    renderer.waitForGPU()

    let elapsed = Date().timeIntervalSince(startTime)
    let fps = Double(totalFrames) / elapsed
    let msPerFrame = (elapsed / Double(totalFrames)) * 1000.0

    print()
    print(String(repeating: "=", count: 60))
    print("BENCHMARK RESULTS")
    print(String(repeating: "=", count: 60))
    print("  Gaussians:    \(options.gaussianCount)")
    print("  Resolution:   \(options.width) x \(options.height)")
    print("  Frames:       \(totalFrames)")
    print("  Total time:   \(String(format: "%.3f", elapsed)) s")
    print("  Avg FPS:      \(String(format: "%.1f", fps))")
    print("  Avg ms/frame: \(String(format: "%.3f", msPerFrame))")
    print(String(repeating: "=", count: 60))
    print()

    // Also render & save one frame for visual verification
    print("Rendering verification frame...")
    let verifyVP = makeViewProj(forFrame: 0, projection: projection)
    renderer.render(viewMatrix: verifyVP)
    do {
        try renderer.savePNG(to: "benchmark_verify.png")
        print("  ✓ Saved to benchmark_verify.png")
    } catch {
        print("  ✗ Failed to save: \(error)")
    }

    exit(0)
}

// Render frames
if options.animationFrames > 0 {
    // Animation mode with PNG save
    print()
    print("Rendering \(options.animationFrames) frames...")

    let startTime = Date()

    for frame in 0..<options.animationFrames {
        let viewProj = makeViewProj(forFrame: frame, projection: projection)

        // Render
        renderer.render(viewMatrix: viewProj)

        // Save frame
        let frameNumber = String(format: "%04d", frame)
        let outputPath = "\(options.framePrefix)_\(frameNumber).png"

        do {
            try renderer.savePNG(to: outputPath)
        } catch {
            print("  ✗ Failed to save frame \(frame): \(error)")
        }

        if frame % 20 == 0 {
            print("  Frame \(frame)/\(options.animationFrames)")
        }
    }

    let elapsed = Date().timeIntervalSince(startTime)
    let fps = Double(options.animationFrames) / elapsed

    print()
    print("✓ Animation complete!")
    print("  Total time (render + PNG): \(String(format: "%.2f", elapsed))s")
    print("  Average FPS (including PNG): \(String(format: "%.1f", fps))")
    print()

} else {
    // Single frame mode
    print()
    print("Rendering frame...")

    let view = OffscreenRenderer.lookAtViewMatrix(
        eye: SIMD3<Float>(0, 0, 3),
        center: SIMD3<Float>(0, 0, 0),
        up: SIMD3<Float>(0, 1, 0)
    )
    let viewProj = multiplyMatrix(projection, view)

    renderer.render(viewMatrix: viewProj)

    print("Saving to \(options.outputFile)...")
    do {
        try renderer.savePNG(to: options.outputFile)
        print()
        print("✓ Render complete!")
        print("  Output: \(options.outputFile)")
        print("  Size: \(options.width)x\(options.height)")
    } catch {
        print("✗ Failed to save PNG: \(error)")
        exit(1)
    }
}

print()
print(String(repeating: "=", count: 60))
print("Done!")
print(String(repeating: "=", count: 60))

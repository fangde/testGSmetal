
// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "GaussianSplattingMetal",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "GaussianSplattingMetal", targets: ["GaussianSplattingMetal"]),
        .executable(name: "GSOffscreen", targets: ["GSOffscreen"]),
        .executable(name: "BenchmarkTest", targets: ["BenchmarkTest"])
    ],
    targets: [
        // ------- Shared rendering core (used by both executables) -------
        .target(
            name: "GaussianSplattingCore",
            path: ".",
            exclude: [
                // SwiftUI app files (not in core)
                "GaussianRenderer.swift",
                "GaussianSplattingApp.swift",
                "MetalView.swift",
                "MetalViewSwiftUICompositionBenchmark.swift",

                // CLI files (not in core)
                "OffscreenRenderer.swift",
                "main.swift",

                // Legacy / reference
                "Renderer.swift",
                "SimpleRenderer.swift",
                "SimpleShader.metal",
                "SimpleMetalView.swift",
                "SimpleMetalExampleView.swift",

                // Output files
                "output.png",

                // Python
                "gs_renderer.py",
                "gs_renderer_offscreen.py",
                "test_rendering.py",
                "environment.yml",

                // Docs / images
                "README.md",
                "GaussianSplatting.metal",
                "docs",
                "offscreen_render_plan.md",
                "python_bindings_plan.md",
                "screenshot.png",
                ".gitignore"
            ],
            sources: [
                "GaussianSplattingCore.swift"
            ]
        ),

        // ------- SwiftUI interactive windowed app -------
        .executableTarget(
            name: "GaussianSplattingMetal",
            dependencies: ["GaussianSplattingCore"],
            path: ".",
            exclude: [
                // CLI offscreen files
                "OffscreenRenderer.swift",
                "main.swift",

                // Legacy / reference
                "Renderer.swift",
                "SimpleRenderer.swift",
                "SimpleShader.metal",
                "SimpleMetalView.swift",
                "SimpleMetalExampleView.swift",

                // Output files
                "output.png",

                // Core (included via dependency)
                "GaussianSplattingCore.swift",

                // Python
                "gs_renderer.py",
                "gs_renderer_offscreen.py",
                "test_rendering.py",
                "environment.yml",

                // Docs / images
                "README.md",
                "GaussianSplatting.metal",
                "docs",
                "offscreen_render_plan.md",
                "python_bindings_plan.md",
                "screenshot.png",
                ".gitignore"
            ],
            sources: [
                "GaussianRenderer.swift",
                "GaussianSplattingApp.swift",
                "MetalView.swift",
                "MetalViewSwiftUICompositionBenchmark.swift"
            ]
        ),

        // ------- CLI offscreen renderer (PNG output) -------
        .executableTarget(
            name: "GSOffscreen",
            dependencies: ["GaussianSplattingCore"],
            path: ".",
            exclude: [
                // SwiftUI app files
                "GaussianRenderer.swift",
                "GaussianSplattingApp.swift",
                "MetalView.swift",
                "MetalViewSwiftUICompositionBenchmark.swift",

                // Core (included via dependency)
                "GaussianSplattingCore.swift",

                // Legacy / reference
                "Renderer.swift",
                "SimpleRenderer.swift",
                "SimpleShader.metal",
                "SimpleMetalView.swift",
                "SimpleMetalExampleView.swift",

                // Output files
                "output.png",

                // Python
                "gs_renderer.py",
                "gs_renderer_offscreen.py",
                "test_rendering.py",
                "environment.yml",

                // Docs / images
                "README.md",
                "GaussianSplatting.metal",
                "docs",
                "offscreen_render_plan.md",
                "python_bindings_plan.md",
                "screenshot.png",
                ".gitignore"
            ],
            sources: [
                "OffscreenRenderer.swift",
                "main.swift"
            ]
        ),

        // ------- Benchmark test runner -------
        .executableTarget(
            name: "BenchmarkTest",
            dependencies: ["GaussianSplattingCore"],
            path: ".",
            exclude: [
                // SwiftUI app files
                "GaussianRenderer.swift",
                "GaussianSplattingApp.swift",
                "MetalView.swift",
                "MetalViewSwiftUICompositionBenchmark.swift",

                // Core (included via dependency)
                "GaussianSplattingCore.swift",

                // CLI files
                "main.swift",

                // Legacy / reference
                "Renderer.swift",
                "SimpleRenderer.swift",
                "SimpleShader.metal",
                "SimpleMetalView.swift",
                "SimpleMetalExampleView.swift",

                // Output files
                "output.png",

                // Python
                "gs_renderer.py",
                "gs_renderer_offscreen.py",
                "test_rendering.py",
                "environment.yml",

                // Docs / images
                "README.md",
                "docs",
                "offscreen_render_plan.md",
                "python_bindings_plan.md",
                "screenshot.png",
                ".gitignore"
            ],
            sources: [
                "BenchmarkTestRunner.swift"
            ]
        )
    ]
)

// ===== Benchmark test target =====

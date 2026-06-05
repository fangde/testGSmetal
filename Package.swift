// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "GaussianSplattingMetal",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "GaussianSplattingMetal", targets: ["GaussianSplattingMetal"])
    ],
    targets: [
        .executableTarget(
            name: "GaussianSplattingMetal",
            dependencies: [],
            path: ".",
            exclude: [
                "Renderer.swift",
                "gs_renderer.py",
                "gs_renderer_offscreen.py",
                "test_rendering.py",
                "environment.yml",
                "README.md",
                "GaussianSplatting.metal",
                "docs",
                "render-accuracy-test.png",
                ".gitignore"
            ],
            sources: [
                "GaussianRenderer.swift",
                "MetalView.swift",
                "GaussianSplattingApp.swift"
            ]
        )
    ]
)

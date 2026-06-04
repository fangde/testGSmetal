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
                "environment.yml",
                "README.md",
                "GaussianSplatting.metal",
                "triangle-test-*.png"
            ],
            sources: [
                "main.swift"
            ]
        )
    ]
)

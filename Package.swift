// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "GaussianSplattingMetal",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "GaussianSplattingMetal",
            targets: ["GaussianSplattingMetal"]
        )
    ],
    targets: [
        .executableTarget(
            name: "GaussianSplattingMetal",
            dependencies: [],
            path: ".",
            exclude: [
                "environment.yml",
                "gs_renderer.py",
                "README.md"
            ],
            sources: [
                "main.swift",
                "Renderer.swift"
            ],
            resources: [
                .process("GaussianSplatting.metal")
            ]
        )
    ]
)

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
        .executableTarget(name: "GaussianSplattingMetal",
                         dependencies: [],
                         path: ".",
                         sources: [
                             "main.swift",
                             "Renderer.swift",
                             "SimpleTestRenderer.swift"
                         ]
        )
    ]
)

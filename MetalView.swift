//
//  MetalView.swift
//  GaussianSplattingMetal
//

import SwiftUI
import MetalKit

struct MetalView: NSViewRepresentable {
    let device: MTLDevice
    let renderer: GaussianRenderer
    
    init(device: MTLDevice) {
        self.device = device
        self.renderer = GaussianRenderer(device: device)!
    }
    
    func makeNSView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: .zero, device: device)
        mtkView.delegate = renderer
        mtkView.preferredFramesPerSecond = 60
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1.0)
        mtkView.depthStencilPixelFormat = .invalid  // 禁用深度测试
        return mtkView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        // Update if needed
    }
}


//
//  SimpleMetalView.swift
//  SimpleMetalExample
//

import SwiftUI
import MetalKit

struct SimpleMetalView: NSViewRepresentable {
    let device: MTLDevice
    let renderer: SimpleRenderer
    
    init(device: MTLDevice) {
        self.device = device
        self.renderer = SimpleRenderer(device: device)!
    }
    
    func makeNSView(context: Context) -> MTKView {
        let mtkView = MTKView(frame: .zero, device: device)
        mtkView.delegate = renderer
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
        return mtkView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        // Update if needed
    }
}


//
//  main.swift
//  GaussianSplattingMetal
//
//

import Cocoa
import MetalKit

class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var metalView: MTKView!
    var renderer: Any? = nil
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        print("App launched!")
        let windowSize = NSSize(width: 1280, height: 720)
        window = NSWindow(
            contentRect: NSRect(origin: .zero, size: windowSize),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.center()
        window.title = "Gaussian Splatting - Metal"
        print("Window created: \(window.frame)")
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal not supported!")
            NSApp.terminate(nil)
            return
        }
        print("Metal device: \(device.name)")
        
        metalView = MTKView(frame: window.contentRect(forFrameRect: window.frame), device: device)
        metalView.colorPixelFormat = .bgra8Unorm
        metalView.clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.2, alpha: 1.0) // Blue background
        metalView.preferredFramesPerSecond = 120
        print("MTKView created with blue clear color")
        
        guard let renderer = SimpleTestRenderer(metalKitView: metalView) else {
            print("Failed to create simple test renderer!")
            NSApp.terminate(nil)
            return
        }
        print("Simple test renderer created")
        
        self.renderer = renderer
        metalView.delegate = renderer
        print("Delegate set")
        
        window.contentView = metalView
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        print("Window shown")
    }
}

// Run app
let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
print("Running app!")
app.run()

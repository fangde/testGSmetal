//
//  GaussianSplattingApp.swift
//  GaussianSplattingMetal
//

import SwiftUI
import AppKit
import Metal

class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!

    func applicationDidFinishLaunching(_ aNotification: Notification) {
        let contentRect = NSRect(x: 200, y: 200, width: 1280, height: 720)
        window = NSWindow(
            contentRect: contentRect,
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.center()
        window.title = "Gaussian Splatting Metal"
        
        let device = MTLCreateSystemDefaultDevice()!
        let contentView = MetalView(device: device).edgesIgnoringSafeArea(.all)
        let hostingView = NSHostingView(rootView: contentView)
        hostingView.autoresizingMask = [.width, .height]
        hostingView.frame = window.contentRect(forFrameRect: contentRect)
        window.contentView = hostingView
        
        window.makeKeyAndOrderFront(nil)
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

@main
struct GaussianSplattingApp {
    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.delegate = delegate
        app.run()
    }
}

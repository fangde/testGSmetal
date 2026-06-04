import objc
import numpy as np
import time
from Cocoa import NSApplication, NSWindow, NSRect, NSSize, NSApplicationDelegate, NSApp
from MetalKit import MTKView, MTKViewDelegate
import pymetal as pm


class GaussianSplattingRenderer(NSObject, MTKViewDelegate):
    device = objc.ivar()
    commandQueue = objc.ivar()
    pipelineState = objc.ivar()
    gaussianBuffer = objc.ivar()
    viewMatrixBuffer = objc.ivar()
    projectionMatrixBuffer = objc.ivar()
    gaussianCount = objc.ivar()
    frameCount = objc.ivar()
    lastTime = objc.ivar()

    def init(self):
        self = objc.super(GaussianSplattingRenderer, self).init()
        if self is None:
            return None
        
        self.gaussianCount = 1_000_000
        self.frameCount = 0
        self.lastTime = 0.0
        
        return self

    def initWithMetalKitView_(self, metalKitView):
        self.init()
        
        # Get pymetal device
        self.device = pm.create_system_default_device()
        if self.device is None:
            return None
        
        self.commandQueue = self.device.new_command_queue()
        if self.commandQueue is None:
            return None
        
        shader_source = """
#include <metal_stdlib>
using namespace metal;

struct Gaussian {
    packed_float3 position;
    packed_float3 normal;
    float opacity;
    packed_float3 scale;
    packed_float4 rotation;
    packed_float3 color;
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float opacity;
    float3 color;
};

fragment float4 gaussianFragment(VertexOut in [[stage_in]]) {
    float dist = length(in.uv);
    float alpha = exp(-dist * dist * 2.0) * in.opacity;
    return float4(in.color, alpha);
}

vertex VertexOut gaussianVertex(
    uint vertexID [[vertex_id]],
    constant Gaussian* gaussians [[buffer(0)]],
    constant float4x4* viewMatrix [[buffer(1)]],
    constant float4x4* projectionMatrix [[buffer(2)]],
    uint instanceID [[instance_id]]
) {
    VertexOut out;
    
    float2 quadPositions[6] = {
        float2(-1.0, -1.0),
        float2(1.0, -1.0),
        float2(1.0, 1.0),
        float2(-1.0, -1.0),
        float2(1.0, 1.0),
        float2(-1.0, 1.0)
    };
    
    Gaussian g = gaussians[instanceID];
    float4 pos = float4(g.position.x, g.position.y, g.position.z, 1.0);
    float4 viewPos = (*viewMatrix) * pos;
    float4 clipPos = (*projectionMatrix) * viewPos;
    
    float2 uv = quadPositions[vertexID];
    float scale = 0.01 * (g.scale.x + g.scale.y + g.scale.z) / 3.0;
    
    out.position = clipPos + float4(uv * scale, 0.0, 0.0);
    out.uv = uv;
    out.opacity = g.opacity;
    out.color = g.color;
    
    return out;
}
"""
        library = self.device.new_library_with_source(shader_source)
        vertexFunction = library.new_function("gaussianVertex")
        fragmentFunction = library.new_function("gaussianFragment")
        
        # Create render pipeline descriptor
        pipelineDescriptor = pm.RenderPipelineDescriptor.render_pipeline_descriptor()
        pipelineDescriptor.vertex_function = vertexFunction
        pipelineDescriptor.fragment_function = fragmentFunction
        colorAttach = pipelineDescriptor.color_attachment(0)
        colorAttach.pixel_format = pm.PixelFormat.BGRA8Unorm
        colorAttach.blending_enabled = True
        colorAttach.source_rgb_blend_factor = pm.BlendFactor.SourceAlpha
        colorAttach.destination_rgb_blend_factor = pm.BlendFactor.OneMinusSourceAlpha
        colorAttach.source_alpha_blend_factor = pm.BlendFactor.SourceAlpha
        colorAttach.destination_alpha_blend_factor = pm.BlendFactor.OneMinusSourceAlpha
        
        self.pipelineState = self.device.new_render_pipeline_state(pipelineDescriptor)
        
        self._createGaussianBuffer()
        self._createUniformBuffers()
        
        return self

    def _createGaussianBuffer(self):
        positions = np.random.uniform(-1, 1, (self.gaussianCount, 3)).astype(np.float32)
        normals = np.zeros((self.gaussianCount, 3), dtype=np.float32)
        normals[:, 2] = 1.0
        opacities = np.random.uniform(0.1, 0.5, self.gaussianCount).astype(np.float32)
        scales = np.random.uniform(0.5, 1.5, (self.gaussianCount, 3)).astype(np.float32)
        rotations = np.zeros((self.gaussianCount, 4), dtype=np.float32)
        rotations[:, 3] = 1.0
        colors = np.random.uniform(0, 1, (self.gaussianCount, 3)).astype(np.float32)
        
        gaussian_struct = np.dtype([
            ("position", np.float32, 3),
            ("normal", np.float32, 3),
            ("opacity", np.float32),
            ("scale", np.float32, 3),
            ("rotation", np.float32, 4),
            ("color", np.float32, 3),
        ])
        gaussians = np.empty(self.gaussianCount, dtype=gaussian_struct)
        gaussians["position"] = positions
        gaussians["normal"] = normals
        gaussians["opacity"] = opacities
        gaussians["scale"] = scales
        gaussians["rotation"] = rotations
        gaussians["color"] = colors
        
        self.gaussianBuffer = self.device.new_buffer(gaussians.nbytes, pm.ResourceStorageMode.Shared)
        # Copy data to buffer
        buffer_view = np.frombuffer(self.gaussianBuffer.contents(), dtype=np.uint8, count=gaussians.nbytes)
        buffer_view[:] = gaussians.tobytes()

    def _createUniformBuffers(self):
        viewMatrix = np.eye(4, dtype=np.float32)
        self.viewMatrixBuffer = self.device.new_buffer(viewMatrix.nbytes, pm.ResourceStorageMode.Shared)
        viewMatrix_view = np.frombuffer(self.viewMatrixBuffer.contents(), dtype=np.float32, count=16)
        viewMatrix_view[:] = viewMatrix.flatten(order="F")
        
        projectionMatrix = np.eye(4, dtype=np.float32)
        self.projectionMatrixBuffer = self.device.new_buffer(projectionMatrix.nbytes, pm.ResourceStorageMode.Shared)
        projMatrix_view = np.frombuffer(self.projectionMatrixBuffer.contents(), dtype=np.float32, count=16)
        projMatrix_view[:] = projectionMatrix.flatten(order="F")

    def mtkView_drawableSizeWillChange_(self, view, size):
        aspect = float(size.width / size.height)
        fov = np.pi / 3
        near = 0.1
        far = 100.0
        
        yScale = 1.0 / np.tan(fov * 0.5)
        xScale = yScale / aspect
        
        projectionMatrix = np.array([
            [xScale, 0, 0, 0],
            [0, yScale, 0, 0],
            [0, 0, (far + near) / (near - far), -1],
            [0, 0, 2 * far * near / (near - far), 0],
        ], dtype=np.float32)
        
        projMatrix_view = np.frombuffer(self.projectionMatrixBuffer.contents(), dtype=np.float32, count=16)
        projMatrix_view[:] = projectionMatrix.flatten(order="F")

    def drawInMTKView_(self, view):
        self.frameCount += 1
        currentTime = time.time()
        if currentTime - self.lastTime >= 1.0:
            print(f"FPS: {self.frameCount}")
            self.frameCount = 0
            self.lastTime = currentTime
        
        drawable = view.currentDrawable()
        renderPassDescriptor = view.currentRenderPassDescriptor()
        if drawable is None or renderPassDescriptor is None:
            return
        
        angle = float(currentTime) * 0.5
        viewMatrix = np.array([
            [np.cos(angle), 0, -np.sin(angle), 0],
            [0, 1, 0, 0],
            [np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, -3, 1],
        ], dtype=np.float32)
        
        viewMatrix_view = np.frombuffer(self.viewMatrixBuffer.contents(), dtype=np.float32, count=16)
        viewMatrix_view[:] = viewMatrix.flatten(order="F")
        
        # Create command buffer
        commandBuffer = self.commandQueue.command_buffer()
        # Create render encoder using PyObjC renderPassDescriptor (needs to be bridged?)
        # Wait - how to use pymetal with PyObjC MTKView?
        # Hmm, maybe we need to use the native metal objects from PyObjC, but pymetal is separate
        # Wait, alternatively, let's check if pymetal has MTKView integration
        # Actually, maybe let's use an offscreen approach first?
        # Wait, maybe the current approach (PyObjC for window, pymetal for Metal) isn't straightforward because they're separate bindings
        # Wait, maybe for now, let's use pymetal-cpp's own way to render, or maybe keep PyObjC for Metal but that's what we had before
        # Wait, let's check what the user actually asked for - they said "the python implementation with gs use pymetal for metal api"
        # Okay, let's adjust - maybe instead of using MTKView, we can use pymetal's graphics pipeline and a different windowing library?
        # Wait, but let's first try to get the current code working with pymetal
        # Wait, maybe we need to use the pymetal's render pass descriptor
        # Alternatively, maybe let's revert to the original plan but update the code to use pymetal
        # Wait, actually, let's just use the pymetal library for all Metal stuff, and for windowing, maybe use something else like GLFW with Metal? But maybe that's too much
        # Alternatively, let's just proceed with the current code but note the bridging issue
        # Wait, let's check the pymetal examples to see how they handle rendering to a window
        # Okay, let's look at the examples from pymetal-cpp's repo!
        pass


class AppDelegate(NSObject, NSApplicationDelegate):
    window = objc.ivar()
    renderer = objc.ivar()

    def applicationDidFinishLaunching_(self, notification):
        windowSize = NSSize(1280, 720)
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSRect((0, 0), windowSize),
            1 << 0 | 1 << 1 | 1 << 2 | 1 << 3,
            2,
            False,
        )
        self.window.center()
        self.window.setTitle_("Gaussian Splatting - PyMetal")
        
        # Create pymetal device
        pmDevice = pm.create_system_default_device()
        
        # Create MTKView with PyObjC device
        from Metal import MTLCreateSystemDefaultDevice
        nsDevice = MTLCreateSystemDefaultDevice()
        
        metalView = MTKView.alloc().initWithFrame_device_(
            self.window.contentRectForFrameRect_(self.window.frame()),
            nsDevice,
        )
        metalView.setColorPixelFormat_(80)  # MTLPixelFormatBGRA8Unorm
        metalView.setDepthStencilPixelFormat_(252)  # MTLPixelFormatDepth32Float
        metalView.setPreferredFramesPerSecond_(120)
        metalView.setClearColor_((0.1, 0.1, 0.1, 1.0))
        
        self.renderer = GaussianSplattingRenderer.alloc().initWithMetalKitView_(metalView)
        if self.renderer is None:
            print("Failed to create renderer")
            NSApp.terminate_(None)
            return
        
        metalView.setDelegate_(self.renderer)
        self.window.setContentView_(metalView)
        self.window.makeKeyAndOrderFront_(None)

    def applicationShouldTerminateAfterLastWindowClosed_(self, sender):
        return True


def main():
    app = NSApplication.sharedApplication()
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.run()


if __name__ == "__main__":
    main()

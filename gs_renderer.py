import objc
import numpy as np
import time
from Cocoa import NSApplication, NSWindow, NSRect, NSSize, NSApplicationDelegate, NSApp
from MetalKit import MTKView, MTKViewDelegate
from Metal import (
    MTLCreateSystemDefaultDevice,
    MTLRenderPipelineDescriptor,
    MTLBlendFactor,
    MTLPrimitiveType,
    MTLStorageModeShared,
)


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
        
        self.device = metalKitView.device()
        if self.device is None:
            return None
        
        self.commandQueue = self.device.newCommandQueue()
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
        library, error = self.device.newLibraryWithSource_options_error_(
            shader_source, None, None
        )
        if library is None:
            print(f"Failed to create library: {error}")
            return None
        
        vertexFunction = library.newFunctionWithName_("gaussianVertex")
        fragmentFunction = library.newFunctionWithName_("gaussianFragment")
        
        pipelineDescriptor = MTLRenderPipelineDescriptor.alloc().init()
        pipelineDescriptor.setVertexFunction_(vertexFunction)
        pipelineDescriptor.setFragmentFunction_(fragmentFunction)
        pipelineDescriptor.colorAttachments().objectAtIndexedSubscript_(0).setPixelFormat_(metalKitView.colorPixelFormat())
        pipelineDescriptor.colorAttachments().objectAtIndexedSubscript_(0).setBlendingEnabled_(True)
        pipelineDescriptor.colorAttachments().objectAtIndexedSubscript_(0).setSourceRGBBlendFactor_(MTLBlendFactorSourceAlpha)
        pipelineDescriptor.colorAttachments().objectAtIndexedSubscript_(0).setDestinationRGBBlendFactor_(MTLBlendFactorOneMinusSourceAlpha)
        pipelineDescriptor.colorAttachments().objectAtIndexedSubscript_(0).setSourceAlphaBlendFactor_(MTLBlendFactorSourceAlpha)
        pipelineDescriptor.colorAttachments().objectAtIndexedSubscript_(0).setDestinationAlphaBlendFactor_(MTLBlendFactorOneMinusSourceAlpha)
        
        self.pipelineState, error = self.device.newRenderPipelineStateWithDescriptor_error_(
            pipelineDescriptor, None
        )
        if self.pipelineState is None:
            print(f"Failed to create pipeline state: {error}")
            return None
        
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
        
        self.gaussianBuffer = self.device.newBufferWithBytes_length_options_(
            gaussians.tobytes(),
            gaussians.nbytes,
            MTLStorageModeShared,
        )

    def _createUniformBuffers(self):
        viewMatrix = np.eye(4, dtype=np.float32)
        viewMatrixBytes = viewMatrix.tobytes(order="F")
        self.viewMatrixBuffer = self.device.newBufferWithBytes_length_options_(
            viewMatrixBytes,
            len(viewMatrixBytes),
            MTLStorageModeShared,
        )
        
        projectionMatrix = np.eye(4, dtype=np.float32)
        projectionMatrixBytes = projectionMatrix.tobytes(order="F")
        self.projectionMatrixBuffer = self.device.newBufferWithBytes_length_options_(
            projectionMatrixBytes,
            len(projectionMatrixBytes),
            MTLStorageModeShared,
        )

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
        projectionMatrixBytes = projectionMatrix.tobytes(order="F")
        self.projectionMatrixBuffer.contents().assign(projectionMatrixBytes)

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
        viewMatrixBytes = viewMatrix.tobytes(order="F")
        self.viewMatrixBuffer.contents().assign(viewMatrixBytes)
        
        commandBuffer = self.commandQueue.commandBuffer()
        renderEncoder = commandBuffer.renderCommandEncoderWithDescriptor_(renderPassDescriptor)
        
        renderEncoder.setRenderPipelineState_(self.pipelineState)
        renderEncoder.setVertexBuffer_offset_atIndex_(self.gaussianBuffer, 0, 0)
        renderEncoder.setVertexBuffer_offset_atIndex_(self.viewMatrixBuffer, 0, 1)
        renderEncoder.setVertexBuffer_offset_atIndex_(self.projectionMatrixBuffer, 0, 2)
        renderEncoder.drawPrimitives_vertexStart_vertexCount_instanceCount_(
            MTLPrimitiveTypeTriangle,
            0,
            6,
            self.gaussianCount,
        )
        
        renderEncoder.endEncoding()
        commandBuffer.presentDrawable_(drawable)
        commandBuffer.commit()


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
        self.window.setTitle_("Gaussian Splatting - Metal Python")
        
        device = MTLCreateSystemDefaultDevice()
        if device is None:
            print("Metal is not supported on this device")
            NSApp.terminate_(None)
            return
        
        metalView = MTKView.alloc().initWithFrame_device_(
            self.window.contentRectForFrameRect_(self.window.frame()),
            device,
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

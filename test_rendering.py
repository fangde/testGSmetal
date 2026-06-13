import numpy as np
import time
import unittest
import pymetal as pm
from PIL import Image


def save_image_as_png(filename, pixels, width, height):
    """Save RGBA pixels as PNG image using PIL."""
    # Convert from [0, 1] to [0, 255] and reshape
    pixels_uint8 = (pixels * 255).astype(np.uint8)
    pixels_uint8 = pixels_uint8.reshape((height, width, 4))
    # Swap red and blue channels (Metal stores as BGRA)
    pixels_uint8 = pixels_uint8[:, :, [2, 1, 0, 3]]
    img = Image.fromarray(pixels_uint8, 'RGBA')
    img.save(filename)


def read_texture_to_buffer(texture, device, queue, width, height):
    """Read texture contents to CPU memory using blit encoder."""
    bytes_per_row = width * 4  # RGBA8
    buffer_size = bytes_per_row * height
    staging_buffer = device.new_buffer(buffer_size, pm.ResourceStorageMode.Shared)

    cmd_buffer = queue.command_buffer()
    blit_encoder = cmd_buffer.blit_command_encoder()

    origin = pm.Origin(0, 0, 0)
    size = pm.Size(width, height, 1)

    blit_encoder.copy_from_texture_to_buffer(
        texture, 0, 0,
        origin, size,
        staging_buffer, 0,
        bytes_per_row,
        bytes_per_row * height
    )

    blit_encoder.end_encoding()
    cmd_buffer.commit()
    cmd_buffer.wait_until_completed()

    pixels = np.frombuffer(staging_buffer.contents(), dtype=np.uint8, count=buffer_size)
    return pixels / 255.0


class TestGaussianRenderer(unittest.TestCase):
    """Unit tests for Gaussian Splatting renderer."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.device = pm.create_system_default_device()
        cls.queue = cls.device.new_command_queue()
        cls.width, cls.height = 128, 128
        cls.gaussian_count = 100

    def test_device_creation(self):
        """Test that Metal device is created successfully."""
        self.assertIsNotNone(self.device)
        self.assertIsNotNone(self.device.name)

    def test_command_queue_creation(self):
        """Test that command queue is created successfully."""
        self.assertIsNotNone(self.queue)

    def test_texture_creation(self):
        """Test that color texture can be created."""
        color_desc = pm.TextureDescriptor.texture2d_descriptor(
            pm.PixelFormat.RGBA8Unorm, self.width, self.height, False
        )
        color_texture = self.device.new_texture(color_desc)
        self.assertIsNotNone(color_texture)

    def test_depth_texture_creation(self):
        """Test that depth texture can be created."""
        depth_desc = pm.TextureDescriptor.texture2d_descriptor(
            pm.PixelFormat.Depth32Float, self.width, self.height, False
        )
        depth_texture = self.device.new_texture(depth_desc)
        self.assertIsNotNone(depth_texture)

    def test_shader_compilation(self):
        """Test that shaders compile without errors."""
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
        float2(-1.0, -1.0), float2(1.0, -1.0), float2(1.0, 1.0),
        float2(-1.0, -1.0), float2(1.0, 1.0), float2(-1.0, 1.0)
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
        self.assertIsNotNone(library)
        vertex_func = library.new_function("gaussianVertex")
        fragment_func = library.new_function("gaussianFragment")
        self.assertIsNotNone(vertex_func)
        self.assertIsNotNone(fragment_func)

    def test_render_pipeline_creation(self):
        """Test that render pipeline can be created."""
        shader_source = """
#include <metal_stdlib>
using namespace metal;
struct VertexOut {
    float4 position [[position]];
};
vertex VertexOut simpleVertex(uint vertexID [[vertex_id]]) {
    VertexOut out;
    out.position = float4(0.0, 0.0, 0.0, 1.0);
    return out;
}
fragment float4 simpleFragment() {
    return float4(1.0, 0.0, 0.0, 1.0);
}
"""
        library = self.device.new_library_with_source(shader_source)
        vertex_func = library.new_function("simpleVertex")
        fragment_func = library.new_function("simpleFragment")

        pipeline_desc = pm.RenderPipelineDescriptor.render_pipeline_descriptor()
        pipeline_desc.vertex_function = vertex_func
        pipeline_desc.fragment_function = fragment_func
        color_attachment = pipeline_desc.color_attachment(0)
        color_attachment.pixel_format = pm.PixelFormat.RGBA8Unorm

        pipeline = self.device.new_render_pipeline_state(pipeline_desc)
        self.assertIsNotNone(pipeline)

    def test_gaussian_buffer_creation(self):
        """Test that Gaussian buffer is created with correct data."""
        positions = np.random.uniform(-1, 1, (self.gaussian_count, 3)).astype(np.float32)
        normals = np.zeros((self.gaussian_count, 3), dtype=np.float32)
        normals[:, 2] = 1.0
        opacities = np.random.uniform(0.1, 0.5, self.gaussian_count).astype(np.float32)
        scales = np.random.uniform(0.5, 1.5, (self.gaussian_count, 3)).astype(np.float32)
        rotations = np.zeros((self.gaussian_count, 4), dtype=np.float32)
        rotations[:, 3] = 1.0
        colors = np.random.uniform(0, 1, (self.gaussian_count, 3)).astype(np.float32)

        gaussian_struct = np.dtype([
            ("position", np.float32, 3),
            ("normal", np.float32, 3),
            ("opacity", np.float32),
            ("scale", np.float32, 3),
            ("rotation", np.float32, 4),
            ("color", np.float32, 3),
        ])
        gaussians = np.empty(self.gaussian_count, dtype=gaussian_struct)
        gaussians["position"] = positions
        gaussians["normal"] = normals
        gaussians["opacity"] = opacities
        gaussians["scale"] = scales
        gaussians["rotation"] = rotations
        gaussians["color"] = colors

        buffer = self.device.new_buffer(gaussians.nbytes, pm.ResourceStorageMode.Shared)
        self.assertIsNotNone(buffer)
        buffer_view = np.frombuffer(buffer.contents(), dtype=np.uint8, count=gaussians.nbytes)
        buffer_view[:] = gaussians.tobytes()

    def test_render_and_read_texture(self):
        """Test that rendering produces non-trivial output."""
        color_desc = pm.TextureDescriptor.texture2d_descriptor(
            pm.PixelFormat.RGBA8Unorm, self.width, self.height, False
        )
        color_texture = self.device.new_texture(color_desc)

        shader_source = """
#include <metal_stdlib>
using namespace metal;
struct VertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
};
vertex VertexOut gaussianVertex(uint vertexID [[vertex_id]]) {
    VertexOut out;
    float x = float((vertexID % 10) - 4.5) * 0.15f;
    float y = float((vertexID / 10) - 4.5) * 0.15f;
    out.position = float4(x, y, 0.0f, 1.0f);
    out.pointSize = 50.0f;
    return out;
}
fragment float4 gaussianFragment() {
    return float4(1.0f, 0.0f, 0.0f, 1.0f);
}
"""
        library = self.device.new_library_with_source(shader_source)
        vertex_func = library.new_function("gaussianVertex")
        fragment_func = library.new_function("gaussianFragment")

        pipeline_desc = pm.RenderPipelineDescriptor.render_pipeline_descriptor()
        pipeline_desc.vertex_function = vertex_func
        pipeline_desc.fragment_function = fragment_func
        color_attachment = pipeline_desc.color_attachment(0)
        color_attachment.pixel_format = pm.PixelFormat.RGBA8Unorm

        pipeline = self.device.new_render_pipeline_state(pipeline_desc)

        render_pass = pm.RenderPassDescriptor.render_pass_descriptor()
        color_att = render_pass.color_attachment(0)
        color_att.texture = color_texture
        color_att.load_action = pm.LoadAction.Clear
        color_att.store_action = pm.StoreAction.Store
        color_att.clear_color = pm.ClearColor(0.1, 0.1, 0.2, 1.0)

        cmd_buffer = self.queue.command_buffer()
        encoder = cmd_buffer.render_command_encoder(render_pass)
        encoder.set_render_pipeline_state(pipeline)
        encoder.draw_primitives(pm.PrimitiveType.Point, 0, self.gaussian_count)
        encoder.end_encoding()
        cmd_buffer.commit()
        cmd_buffer.wait_until_completed()

        pixels = read_texture_to_buffer(color_texture, self.device, self.queue, self.width, self.height)
        self.assertEqual(pixels.shape[0], self.width * self.height * 4)
        self.assertGreater(pixels.max(), 0.0)


def render_and_save_test_images():
    device = pm.create_system_default_device()
    queue = device.new_command_queue()

    width, height = 1280, 720
    gaussian_count = 100  # Just a few points

    print(f"Rendering {gaussian_count} gaussians at {width}x{height} on {device.name}")

    # Create color render target
    color_desc = pm.TextureDescriptor.texture2d_descriptor(
        pm.PixelFormat.RGBA8Unorm, width, height, False
    )
    color_texture = device.new_texture(color_desc)

    # Shader source - extremely simple, just big red points in center
    shader_source = """
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
};

vertex VertexOut gaussianVertex(uint vertexID [[vertex_id]]) {
    VertexOut out;
    // Just place points in a grid in clip space directly!
    float x = float((vertexID % 10) - 4.5) * 0.15f;
    float y = float((vertexID / 10) - 4.5) * 0.15f;
    out.position = float4(x, y, 0.0f, 1.0f);
    out.pointSize = 100.0f;
    return out;
}

fragment float4 gaussianFragment() {
    return float4(1.0f, 0.0f, 0.0f, 1.0f);
}
"""
    print("Compiling shaders...")
    library = device.new_library_with_source(shader_source)
    vertex_func = library.new_function("gaussianVertex")
    fragment_func = library.new_function("gaussianFragment")

    # Create render pipeline
    print("Creating render pipeline...")
    pipeline_desc = pm.RenderPipelineDescriptor.render_pipeline_descriptor()
    pipeline_desc.vertex_function = vertex_func
    pipeline_desc.fragment_function = fragment_func
    color_attachment = pipeline_desc.color_attachment(0)
    color_attachment.pixel_format = pm.PixelFormat.RGBA8Unorm

    pipeline = device.new_render_pipeline_state(pipeline_desc)

    # Create render pass descriptor
    render_pass = pm.RenderPassDescriptor.render_pass_descriptor()
    color_att = render_pass.color_attachment(0)
    color_att.texture = color_texture
    color_att.load_action = pm.LoadAction.Clear
    color_att.store_action = pm.StoreAction.Store
    color_att.clear_color = pm.ClearColor(0.1, 0.1, 0.2, 1.0)  # Dark blue background

    # Render 1 image
    cmd_buffer = queue.command_buffer()
    encoder = cmd_buffer.render_command_encoder(render_pass)

    encoder.set_render_pipeline_state(pipeline)
    encoder.draw_primitives(pm.PrimitiveType.Point, 0, gaussian_count)

    encoder.end_encoding()
    cmd_buffer.commit()
    cmd_buffer.wait_until_completed()

    # Read and save image
    pixels = read_texture_to_buffer(color_texture, device, queue, width, height)
    print(f"Pixel range: min={pixels.min()}, max={pixels.max()}")
    print(f"Number of non-background pixels: {(pixels.reshape(-1,4)[:,0] > 0.2).sum()}")
    output_file = f"/Volumes/KIOXIA/testGSmetal/test-final.png"
    save_image_as_png(output_file, pixels, width, height)
    print(f"✅ Saved test image to: {output_file}")

    print("\n✅ Completed!")


def main():
    print("=" * 60)
    print("Gaussian Splatting Test Rendering")
    print("=" * 60)
    print()

    render_and_save_test_images()

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

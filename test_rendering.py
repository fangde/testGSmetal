import numpy as np
import time
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
    staging_buffer = device.new_buffer(buffer_size, pm.ResourceStorageModeShared)

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

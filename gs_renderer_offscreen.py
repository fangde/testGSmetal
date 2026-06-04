import numpy as np
import time
import pymetal as pm


def save_image_as_ppm(filename, pixels, width, height):
    """Save RGBA pixels as PPM image (simple format, no dependencies)."""
    with open(filename, 'w') as f:
        f.write(f'P3\n{width} {height}\n255\n')
        for y in range(height):
            for x in range(width):
                idx = (y * width + x) * 4
                r = int(pixels[idx] * 255)
                g = int(pixels[idx + 1] * 255)
                b = int(pixels[idx + 2] * 255)
                f.write(f'{r} {g} {b} ')
            f.write('\n')


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


def render_gaussians():
    device = pm.create_system_default_device()
    queue = device.new_command_queue()

    width, height = 1280, 720
    gaussian_count = 1_000_000

    print(f"Rendering {gaussian_count} gaussians at {width}x{height} on {device.name}")

    # Create color and depth render targets
    color_desc = pm.TextureDescriptor.texture2d_descriptor(
        pm.PixelFormat.RGBA8Unorm, width, height, False
    )
    color_texture = device.new_texture(color_desc)

    depth_desc = pm.TextureDescriptor.texture2d_descriptor(
        pm.PixelFormat.Depth32Float, width, height, False
    )
    depth_texture = device.new_texture(depth_desc)

    # Shader source
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
    print("Compiling shaders...")
    library = device.new_library_with_source(shader_source)
    vertex_func = library.new_function("gaussianVertex")
    fragment_func = library.new_function("gaussianFragment")

    # Create depth/stencil state
    depth_stencil_desc = pm.DepthStencilDescriptor.depth_stencil_descriptor()
    depth_stencil_desc.depth_compare_function = pm.CompareFunction.Less
    depth_stencil_desc.depth_write_enabled = True
    depth_stencil_state = device.new_depth_stencil_state(depth_stencil_desc)

    # Create render pipeline
    print("Creating render pipeline...")
    pipeline_desc = pm.RenderPipelineDescriptor.render_pipeline_descriptor()
    pipeline_desc.vertex_function = vertex_func
    pipeline_desc.fragment_function = fragment_func
    color_attachment = pipeline_desc.color_attachment(0)
    color_attachment.pixel_format = pm.PixelFormat.RGBA8Unorm
    color_attachment.blending_enabled = True
    color_attachment.source_rgb_blend_factor = pm.BlendFactor.SourceAlpha
    color_attachment.destination_rgb_blend_factor = pm.BlendFactor.OneMinusSourceAlpha
    color_attachment.source_alpha_blend_factor = pm.BlendFactor.SourceAlpha
    color_attachment.destination_alpha_blend_factor = pm.BlendFactor.OneMinusSourceAlpha
    pipeline_desc.depth_attachment_pixel_format = pm.PixelFormat.Depth32Float

    pipeline = device.new_render_pipeline_state(pipeline_desc)

    # Create Gaussian data
    print(f"Creating {gaussian_count} gaussians...")
    positions = np.random.uniform(-1, 1, (gaussian_count, 3)).astype(np.float32)
    normals = np.zeros((gaussian_count, 3), dtype=np.float32)
    normals[:, 2] = 1.0
    opacities = np.random.uniform(0.1, 0.5, gaussian_count).astype(np.float32)
    scales = np.random.uniform(0.5, 1.5, (gaussian_count, 3)).astype(np.float32)
    rotations = np.zeros((gaussian_count, 4), dtype=np.float32)
    rotations[:, 3] = 1.0
    colors = np.random.uniform(0, 1, (gaussian_count, 3)).astype(np.float32)

    gaussian_struct = np.dtype([
        ("position", np.float32, 3),
        ("normal", np.float32, 3),
        ("opacity", np.float32),
        ("scale", np.float32, 3),
        ("rotation", np.float32, 4),
        ("color", np.float32, 3),
    ])
    gaussians = np.empty(gaussian_count, dtype=gaussian_struct)
    gaussians["position"] = positions
    gaussians["normal"] = normals
    gaussians["opacity"] = opacities
    gaussians["scale"] = scales
    gaussians["rotation"] = rotations
    gaussians["color"] = colors

    # Create buffers
    gaussian_buffer = device.new_buffer(gaussians.nbytes, pm.ResourceStorageMode.Shared)
    gaussian_view = np.frombuffer(gaussian_buffer.contents(), dtype=np.uint8, count=gaussians.nbytes)
    gaussian_view[:] = gaussians.tobytes()

    view_matrix = np.eye(4, dtype=np.float32)
    view_buffer = device.new_buffer(view_matrix.nbytes, pm.ResourceStorageMode.Shared)
    view_view = np.frombuffer(view_buffer.contents(), dtype=np.float32, count=16)
    view_view[:] = view_matrix.flatten(order="F")

    # Create projection matrix
    aspect = width / height
    fov = np.pi / 3
    near = 0.1
    far = 100.0
    y_scale = 1.0 / np.tan(fov * 0.5)
    x_scale = y_scale / aspect
    proj_matrix = np.array([
        [x_scale, 0, 0, 0],
        [0, y_scale, 0, 0],
        [0, 0, (far + near) / (near - far), -1],
        [0, 0, 2 * far * near / (near - far), 0],
    ], dtype=np.float32)
    proj_buffer = device.new_buffer(proj_matrix.nbytes, pm.ResourceStorageMode.Shared)
    proj_view = np.frombuffer(proj_buffer.contents(), dtype=np.float32, count=16)
    proj_view[:] = proj_matrix.flatten(order="F")

    # Create render pass descriptor
    render_pass = pm.RenderPassDescriptor.render_pass_descriptor()
    color_att = render_pass.color_attachment(0)
    color_att.texture = color_texture
    color_att.load_action = pm.LoadAction.Clear
    color_att.store_action = pm.StoreAction.Store
    color_att.clear_color = pm.ClearColor(0.1, 0.1, 0.1, 1.0)

    depth_att = render_pass.depth_attachment
    depth_att.texture = depth_texture
    depth_att.load_action = pm.LoadAction.Clear
    depth_att.store_action = pm.StoreAction.Store
    depth_att.clear_depth = 1.0

    # Render loop - let's render a few frames to measure FPS
    print("Rendering frames...")
    frame_count = 0
    last_time = time.time()
    num_frames = 100
    for i in range(num_frames):
        angle = time.time() * 0.5
        view_matrix = np.array([
            [np.cos(angle), 0, -np.sin(angle), 0],
            [0, 1, 0, 0],
            [np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, -3, 1],
        ], dtype=np.float32)
        view_view[:] = view_matrix.flatten(order="F")

        cmd_buffer = queue.command_buffer()
        encoder = cmd_buffer.render_command_encoder(render_pass)

        encoder.set_render_pipeline_state(pipeline)
        encoder.set_depth_stencil_state(depth_stencil_state)
        encoder.set_vertex_buffer(gaussian_buffer, 0, 0)
        encoder.set_vertex_buffer(view_buffer, 0, 1)
        encoder.set_vertex_buffer(proj_buffer, 0, 2)
        encoder.draw_primitives(pm.PrimitiveType.Triangle, 0, 6, gaussian_count)

        encoder.end_encoding()
        cmd_buffer.commit()
        cmd_buffer.wait_until_completed()

        frame_count += 1
        if time.time() - last_time >= 1.0:
            print(f"FPS: {frame_count}")
            frame_count = 0
            last_time = time.time()

    print("\nRendering complete!")
    return color_texture, width, height


def main():
    print("=" * 60)
    print("PyMetal Gaussian Splatting Demo (Offscreen)")
    print("=" * 60)
    print()

    color_texture, width, height = render_gaussians()

    print("Reading back rendered image...")
    device = pm.create_system_default_device()
    queue = device.new_command_queue()
    pixels = read_texture_to_buffer(color_texture, device, queue, width, height)

    output_file = "/tmp/gs_pymetal.ppm"
    print(f"Saving image to {output_file}...")
    save_image_as_ppm(output_file, pixels, width, height)
    print(f"\n✓ Image saved to: {output_file}")
    print(f"  You can view it with: open {output_file}")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

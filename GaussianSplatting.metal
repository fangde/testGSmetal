//
//  GaussianSplatting.metal
//  GaussianSplattingMetal
//
//

#include <metal_stdlib>
using namespace metal;

struct Gaussian {
    packed_float3 position;
    float opacity;
    packed_float3 scale;
    packed_float3 color;
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float alpha;
    float3 color;
};

fragment float4 gaussianFragment(VertexOut in [[stage_in]]) {
    float dist2 = dot(in.uv, in.uv);
    float alpha = exp(-2.0f * dist2) * in.alpha;
    return float4(in.color * alpha, alpha);
}

struct VertexIn {
    float2 position [[attribute(0)]];
};

vertex VertexOut gaussianVertex(
    uint vertexID [[vertex_id]],
    constant Gaussian* gaussians [[buffer(0)]],
    constant float4x4* viewMatrix [[buffer(1)]],
    constant float4x4* projectionMatrix [[buffer(2)]],
    uint instanceID [[instance_id]]
) {
    VertexOut out;
    
    float2 quadCorners[6] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2( 1.0,  1.0),
        float2(-1.0, -1.0),
        float2( 1.0,  1.0),
        float2(-1.0,  1.0)
    };
    
    Gaussian g = gaussians[instanceID];
    
    float4 posWorld = float4(g.position.x, g.position.y, g.position.z, 1.0);
    float4 posView = (*viewMatrix) * posWorld;
    float4 posClip = (*projectionMatrix) * posView;
    
    // Fixed size quad in NDC for testing
    float2 ndcOffset = quadCorners[vertexID] * 0.05f;
    
    out.position = posClip + float4(ndcOffset * posClip.w, 0.0f, 0.0f);
    out.uv = quadCorners[vertexID];
    out.alpha = g.opacity;
    out.color = g.color;
    
    return out;
}

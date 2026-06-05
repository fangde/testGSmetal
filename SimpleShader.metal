
//
//  SimpleShader.metal
//  SimpleMetalExample
//

#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float2 position [[attribute(0)]];
    float4 color [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
};

vertex VertexOut simpleVertex(
    const device VertexIn* vertices [[buffer(0)]],
    uint vertexID [[vertex_id]]
) {
    VertexOut out;
    out.position = float4(vertices[vertexID].position, 0.0, 1.0);
    out.color = vertices[vertexID].color;
    return out;
}

fragment float4 simpleFragment(VertexOut in [[stage_in]]) {
    return in.color;
}


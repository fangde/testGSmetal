
# Gaussian Splatting: 3D Gaussian Splatting for Real-Time Radiance Field Rendering

## Overview
3D Gaussian Splatting is a method for real-time, high-quality novel-view synthesis that combines the best of neural radiance fields (NeRF) and traditional rasterization. Instead of using implicit representations, it uses a set of anisotropic 3D Gaussians to represent the scene, allowing for fast GPU-accelerated rendering.

## 1. 3D Gaussians

### 1.1 Gaussian Definition
A 3D Gaussian is defined by:
- **Mean (Position):** $\boldsymbol{\mu} \in \mathbb{R}^3$
- **Covariance Matrix:** $\boldsymbol{\Sigma} \in \mathbb{R}^{3 \times 3}$, symmetric positive definite
- **Opacity:** $\alpha \in [0, 1]$
- **Spherical Harmonic Coefficients:** $\mathbf{c} \in \mathbb{R}^{(L+1)^2 \times 3}$ for color (where $L$ is the SH order, typically $L=3$)

The 3D Gaussian probability density function (PDF) is:
$$
G(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{\sqrt{(2\pi)^3 \det(\boldsymbol{\Sigma})}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
$$

### 1.2 Covariance Representation
To ensure $\boldsymbol{\Sigma}$ is positive definite and efficiently optimize it, we represent it using a quaternion (rotation) and scaling vector:
- Quaternion: $\mathbf{q} = [w, x, y, z] \in \mathbb{R}^4$ (normalized)
- Scaling: $\mathbf{s} = [s_x, s_y, s_z] \in \mathbb{R}^3$ ($s_i > 0$)

First, convert quaternion to rotation matrix $\mathbf{R}$:
$$
\mathbf{R} = \begin{bmatrix}
1 - 2y^2 - 2z^2 & 2xy - 2wz & 2xz + 2wy \\
2xy + 2wz & 1 - 2x^2 - 2z^2 & 2yz - 2wx \\
2xz - 2wy & 2yz + 2wx & 1 - 2x^2 - 2y^2
\end{bmatrix}
$$

Then, the covariance matrix is:
$$
\boldsymbol{\Sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^\top \mathbf{R}^\top
$$
where $\mathbf{S} = \text{diag}(s_x, s_y, s_z)$.

## 2. View Transform and Projection

### 2.1 World-to-View Transform
Given camera parameters:
- Camera position (translation): $\mathbf{t} \in \mathbb{R}^3$
- Camera orientation (rotation): $\mathbf{R}_c \in \mathbb{R}^{3 \times 3}$
- Field of view (FoV)

The world-to-view transformation is:
$$
\mathbf{T}_{\text{w2v}} = \begin{bmatrix}
\mathbf{R}_c & \mathbf{t} \\
\mathbf{0} & 1
\end{bmatrix}
$$
For a point $\mathbf{x}_w$ in world space, view space point:
$$
\mathbf{x}_v = \mathbf{T}_{\text{w2v}} \cdot \begin{bmatrix} \mathbf{x}_w \\ 1 \end{bmatrix}
$$

### 2.2 View-to-Screen Projection (Perspective)
Projection matrix $\mathbf{P}$ for perspective camera (OpenGL convention):
$$
\mathbf{P} = \begin{bmatrix}
\frac{1}{\tan(\text{FoV}/2)} & 0 & 0 & 0 \\
0 & \frac{1}{\tan(\text{FoV}/2)} & 0 & 0 \\
0 & 0 & -\frac{z_{\text{far}} + z_{\text{near}}}{z_{\text{far}} - z_{\text{near}}} & -\frac{2 z_{\text{far}} z_{\text{near}}}{z_{\text{far}} - z_{\text{near}}} \\
0 & 0 & -1 & 0
\end{bmatrix}
$$

The Gaussian's mean in view space: $\boldsymbol{\mu}_v = \mathbf{R}_c (\boldsymbol{\mu} - \mathbf{t})$

### 2.3 2D Covariance (Efficient Approximation)
We approximate the 3D Gaussian's projection onto the 2D image plane using the Jacobian of the projection function. For a view-space point $(x, y, z)$, the screen space $(u, v)$ is:
$$
u = f_x \frac{x}{z}, \quad v = f_y \frac{y}{z}
$$
where $f_x, f_y$ are the focal lengths.

The Jacobian $\mathbf{J}$ of the projection is:
$$
\mathbf{J} = \begin{bmatrix}
\frac{f_x}{z} & 0 & -\frac{f_x x}{z^2} \\
0 & \frac{f_y}{z} & -\frac{f_y y}{z^2}
\end{bmatrix}
$$

The 2D covariance $\boldsymbol{\Sigma}'$ in screen space is:
$$
\boldsymbol{\Sigma}' = \mathbf{J} \boldsymbol{\Sigma}_v \mathbf{J}^\top
$$
where $\boldsymbol{\Sigma}_v = \mathbf{R}_c \boldsymbol{\Sigma} \mathbf{R}_c^\top$ is the 3D covariance in view space.

To compute this efficiently:
1. Compute $\boldsymbol{\Sigma}_v = \mathbf{R}_c \boldsymbol{\Sigma} \mathbf{R}_c^\top$ (only upper triangle due to symmetry)
2. Apply the Jacobian to get $\boldsymbol{\Sigma}'$

## 3. Rendering Pipeline

### 3.1 Sorting Gaussians
First, we sort all Gaussians by depth (view-space $z$-coordinate) in front-to-back order for alpha blending.

### 3.2 Rendering Each Gaussian
For each Gaussian (after sorting):
1. **Frustum Culling:** Check if Gaussian is inside the camera frustum
2. **Ellipse Bounding Box:** Compute the 2D ellipse's bounding box on screen
3. **Splatting:** For each pixel in the bounding box, compute the Gaussian's contribution and accumulate color and alpha.

### 3.3 Alpha Blending
Front-to-back alpha blending formula:
$$
C_{\text{out}} = C_{\text{in}} (1 - \alpha_{\text{acc}}) + C_{\text{gauss}} \alpha_{\text{gauss}} (1 - \alpha_{\text{acc}})
$$
$$
\alpha_{\text{acc}} = \alpha_{\text{acc}} + \alpha_{\text{gauss}} (1 - \alpha_{\text{acc}})
$$

## 4. Spherical Harmonics for View-Dependent Color

### 4.1 Spherical Harmonics (SH)
Spherical harmonics are orthonormal basis functions defined on the unit sphere. For $L$th-order SH, we have $(L+1)^2$ coefficients.

### 4.2 Color Evaluation
Given a view direction $\mathbf{d}$ (unit vector in world space), convert to spherical coordinates $(\theta, \phi)$ and evaluate the SH basis functions $Y_l^m(\theta, \phi)$. The color is:
$$
C(\mathbf{d}) = \sum_{l=0}^L \sum_{m=-l}^l c_l^m Y_l^m(\mathbf{d})
$$
where $c_l^m$ are the RGB SH coefficients for the Gaussian.

## 5. GPU Implementation Details with Pseudo-Code

### 5.1 Data Structures on GPU

#### Pseudo-Code: GPU Memory Layout (Struct of Arrays or Array of Structs)
```glsl
// Array of Structs (AoS) - common for simplicity
struct GaussianData {
    vec3 position;         // Mean (μ)
    vec4 quaternion;       // Rotation (w, x, y, z), normalized
    vec3 scale;            // Scaling (s_x, s_y, s_z), positive
    float opacity;         // α, [0, 1]
    vec3 shCoeffs[16];     // Spherical Harmonic coefficients (L=3: 16 coeffs)
};

// Buffers
layout(std430, binding = 0) buffer GaussianBuffer {
    GaussianData gaussians[];
};

// Framebuffer data
layout(binding = 0, rgba32f) uniform image2D colorImage;
layout(binding = 1, r32f) uniform image2D alphaImage;
```

---

### 5.2 Helper Functions

#### Pseudo-Code: Quaternion to Rotation Matrix
```glsl
mat3 quaternionToRotationMatrix(vec4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, wz = w*z, xz = x*z, wy = w*y;
    float yz = y*z, wx = w*x;
    
    return mat3(
        1 - 2*yy - 2*zz, 2*xy - 2*wz, 2*xz + 2*wy,
        2*xy + 2*wz, 1 - 2*xx - 2*zz, 2*yz - 2*wx,
        2*xz - 2*wy, 2*yz + 2*wx, 1 - 2*xx - 2*yy
    );
}
```

#### Pseudo-Code: Compute 3D Covariance in View Space
```glsl
mat3 computeViewSpaceCovariance(vec3 scale, mat3 rotationWorldToView, mat3 rotationGaussian) {
    // Scale matrix S
    mat3 S = mat3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
    // World-space covariance: R * S² * Rᵀ
    mat3 covWorld = rotationGaussian * S * S * transpose(rotationGaussian);
    // View-space covariance: R_c * cov_world * R_cᵀ
    return rotationWorldToView * covWorld * transpose(rotationWorldToView);
}
```

#### Pseudo-Code: Compute 2D Covariance (Screen Space)
```glsl
vec3 compute2DCovariance(vec3 viewPos, mat3 covView, float fx, float fy) {
    float x = viewPos.x, y = viewPos.y, z = viewPos.z;
    float invZ = 1.0 / z;
    float invZ2 = invZ * invZ;
    
    // Jacobian J
    mat2x3 J = mat2x3(
        fx * invZ, 0, -fx * x * invZ2,
        0, fy * invZ, -fy * y * invZ2
    );
    
    // Compute Σ' = J * Σ_v * Jᵀ (upper triangle only, since symmetric)
    float a = covView[0][0], b = covView[0][1], c = covView[0][2];
    float d = covView[1][1], e = covView[1][2];
    float f = covView[2][2];
    
    float j00 = J[0][0], j02 = J[0][2];
    float j11 = J[1][1], j12 = J[1][2];
    
    // Return [conic.x, conic.y, conic.z] = [Σ'[0][0], Σ'[0][1], Σ'[1][1]]
    return vec3(
        j00*a*j00 + 2*j00*c*j02 + j02*f*j02,
        j00*b*j11 + j00*e*j12 + j11*c*j02 + j12*f*j02,
        j11*d*j11 + 2*j11*e*j12 + j12*f*j12
    );
}
```

#### Pseudo-Code: Evaluate 2D Gaussian
```glsl
float evaluateGaussian(vec2 offset, vec3 conic) {
    float a = conic.x, b = conic.y, c = conic.z;
    float power = 0.5 * (a * offset.x * offset.x + 2*b * offset.x * offset.y + c * offset.y * offset.y);
    return exp(-power);
}
```

---

### 5.3 Compute Shader: Preprocessing and Sorting
First, compute view-space position and depth for each Gaussian, then sort by depth.

```glsl
// Compute shader: Preprocess Gaussians
layout(local_size_x = 256) in;

uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform vec2 viewportSize;
uniform float fx, fy; // Focal lengths

struct PreprocessedGaussian {
    vec3 viewPos;
    float depth;
    vec3 conic; // Upper triangle of 2D covariance: [a, b, c]
    vec3 color;
    float alpha;
    vec2 centerScreen;
    vec2 bboxMin;
    vec2 bboxMax;
};

layout(std430, binding = 1) buffer PreprocessedBuffer {
    PreprocessedGaussian preprocessed[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= gaussians.length()) return;
    
    GaussianData g = gaussians[idx];
    
    // 1. Compute view-space position
    vec4 worldPos = vec4(g.position, 1.0);
    vec4 viewPos4 = viewMatrix * worldPos;
    vec3 viewPos = viewPos4.xyz / viewPos4.w;
    
    // 2. Convert quaternion to rotation matrix
    mat3 rotGaussian = quaternionToRotationMatrix(g.quaternion);
    mat3 rotWorldToView = mat3(viewMatrix);
    
    // 3. Compute view-space covariance and 2D conic
    mat3 covView = computeViewSpaceCovariance(g.scale, rotWorldToView, rotGaussian);
    vec3 conic = compute2DCovariance(viewPos, covView, fx, fy);
    
    // 4. Compute screen center and bounding box
    vec4 clipPos = projectionMatrix * vec4(viewPos, 1.0);
    vec2 ndcPos = clipPos.xy / clipPos.w;
    vec2 centerScreen = (ndcPos + 1.0) * 0.5 * viewportSize;
    
    // Bounding box (3σ rule)
    float sqrtA = sqrt(max(conic.x, 1e-6));
    float sqrtC = sqrt(max(conic.z, 1e-6));
    vec2 bboxSize = vec2(3.0 * sqrtA, 3.0 * sqrtC);
    vec2 bboxMin = centerScreen - bboxSize;
    vec2 bboxMax = centerScreen + bboxSize;
    
    // 5. Evaluate SH for color (simplified for L=0)
    vec3 color = g.shCoeffs[0]; // L=0: base color
    float alpha = clamp(g.opacity, 0.0, 1.0);
    
    // Store preprocessed data
    preprocessed[idx] = PreprocessedGaussian(
        viewPos, -viewPos.z, // depth (negative z for OpenGL convention)
        conic, color, alpha,
        centerScreen, bboxMin, bboxMax
    );
}
```

---

### 5.4 Rendering: Vertex and Fragment Shaders (or Compute Shader)
For rasterization, we can use instanced rendering (1 instance = 1 Gaussian, 6 vertices = 1 quad) or compute shaders. Here's both approaches:

#### Option A: Vertex + Fragment Shaders (Rasterization)
```glsl
// Vertex Shader
layout(location = 0) in uint instanceID; // Instanced rendering

out vec2 fragUV;
out vec3 fragColor;
out float fragAlpha;
out vec3 fragConic;
out vec2 fragCenter;

uniform vec2 viewportSize;

const vec2 quadCorners[6] = vec2[6](
    vec2(-1, -1), vec2(1, -1), vec2(1, 1),
    vec2(-1, -1), vec2(1, 1), vec2(-1, 1)
);

void main() {
    PreprocessedGaussian g = preprocessed[instanceID];
    
    // 1. Compute quad corners in screen space
    vec2 corner = g.centerScreen + quadCorners[gl_VertexID] * (g.bboxMax - g.bboxMin) * 0.5;
    vec2 ndc = (corner / viewportSize) * 2.0 - 1.0;
    
    gl_Position = vec4(ndc, 0.0, 1.0);
    fragUV = quadCorners[gl_VertexID] * (g.bboxMax - g.bboxMin) * 0.5;
    fragConic = g.conic;
    fragCenter = g.centerScreen;
    fragColor = g.color;
    fragAlpha = g.alpha;
}

// Fragment Shader
in vec2 fragUV;
in vec3 fragColor;
in float fragAlpha;
in vec3 fragConic;

layout(location = 0) out vec4 outColor;

void main() {
    vec2 offset = fragUV;
    float gaussianVal = evaluateGaussian(offset, fragConic);
    float alpha = fragAlpha * gaussianVal;
    
    if (alpha < 1e-4) discard;
    
    outColor = vec4(fragColor * alpha, alpha);
}
```

#### Option B: Compute Shader (Tiled Rendering)
```glsl
// Compute shader: Tiled Gaussian Splatting
layout(local_size_x = 16, local_size_y = 16) in;

uniform vec2 viewportSize;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= int(viewportSize.x) || pixel.y >= int(viewportSize.y)) return;
    
    vec3 accumColor = vec3(0.0);
    float accumAlpha = 0.0;
    
    // Iterate over Gaussians (already sorted front-to-back)
    for (uint i = 0; i < preprocessed.length() && accumAlpha < 0.999; i++) {
        PreprocessedGaussian g = preprocessed[i];
        
        // 1. Bounding box culling
        if (pixel.x < int(g.bboxMin.x) || pixel.x > int(g.bboxMax.x) ||
            pixel.y < int(g.bboxMin.y) || pixel.y > int(g.bboxMax.y)) continue;
        
        // 2. Compute offset and Gaussian value
        vec2 offset = vec2(pixel) - g.centerScreen;
        float gaussianVal = evaluateGaussian(offset, g.conic);
        float alpha = g.alpha * gaussianVal;
        
        // 3. Front-to-back blending
        float blend = alpha * (1.0 - accumAlpha);
        accumColor += g.color * blend;
        accumAlpha += blend;
    }
    
    // Store to framebuffer
    imageStore(colorImage, pixel, vec4(accumColor, 1.0));
    imageStore(alphaImage, pixel, vec4(accumAlpha));
}
```

---

### 5.5 State-of-the-Art Performance Optimizations

Modern Gaussian Splatting implementations use several key optimizations to reach hundreds of FPS with millions of Gaussians:

1. **Adaptive Tile Size and Tile Culling**
   - Sort Gaussians into screen-space tiles using a spatial data structure (e.g., hash grid, bounding volume hierarchy)
   - Process only tiles affected by visible Gaussians
   - Pseudo-code:
     ```glsl
     // Compute tile index for Gaussian
     ivec2 tileSize = ivec2(64, 64);
     ivec2 tileMin = ivec2(floor(g.bboxMin / vec2(tileSize)));
     ivec2 tileMax = ivec2(ceil(g.bboxMax / vec2(tileSize)));
     // For each tile in tileMin to tileMax, add Gaussian to that tile's list
     ```

2. **Hierarchical Z-Buffer (Hi-Z) Culling**
   - Use a low-resolution depth buffer to quickly cull Gaussians behind already opaque surfaces
   - Can reduce the number of Gaussians processed by 50-80% in practice

3. **Precomputed Sorting and Persistent Threads**
   - Keep Gaussians in sorted order between frames (for small camera movements)
   - Use persistent threads in compute shaders to avoid kernel launch overhead

4. **Packed Data Formats**
   - Use half-precision (fp16) for most data (position, scale, color)
   - Pack quaternions and conic matrices into smaller data types
   - Example:
     ```glsl
     struct PackedGaussian {
         f16vec3 position;  // fp16 instead of fp32
         f16vec4 quaternion;
         f16vec3 scale;
         f16 opacity;
         f16vec3 shCoeffs[8]; // Lower SH order if possible
     };
     ```

5. **Approximate Gaussian Evaluation**
   - Use look-up tables (LUTs) for exponential functions
   - Clamp small Gaussians to 0 early (e.g., α < 1e-4 → discard)

6. **Hardware Acceleration Features**
   - Use rasterizer discard to skip invisible fragments
   - Use hardware blending units for accumulation
   - Use async compute to overlap preprocessing, sorting, and rendering

7. **Recent SOTA Advances**
   - **2D Gaussian Splatting:** Project Gaussians to 2D early to reduce memory usage
   - **Gaussian Pruning:** Dynamically remove Gaussians that contribute less than a threshold
   - **Level-of-Detail (LOD) Gaussians:** Use smaller/fewer Gaussians for distant objects
   - **Neural Acceleration:** Use tiny neural networks to accelerate Gaussian evaluation or sorting

## 6. Optimization (Training)
To optimize the Gaussian parameters, we minimize the photometric loss between rendered images and training images. The key steps are:
- Initialize Gaussians from a point cloud (e.g., from COLMAP)
- Use stochastic gradient descent (SGD) or Adam
- Optimize: position, quaternion, scaling, opacity, SH coefficients
- Use differentiable rendering (autograd) to compute gradients
- Apply density control (split/merge Gaussians) to adaptively increase/decrease number of Gaussians

## References
- Kerbl, Bernhard, et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." ACM Transactions on Graphics (TOG) 42.4 (2023): 1-14.

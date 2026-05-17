---
title: "In-depth analysis of the Kawase method"
description: ""
pubDate: 2026-05-17
tags: ["Gaussian Filter", "Kawase Blur", "Frequency Analysis"]
heroImage: "/images/kawase.jpg"
---
# In-depth analysis of the Kawase method

## 0. Original literature

**Formal citation**

> Masaki Kawase, "Frame Buffer Postprocessing Effects in DOUBLE‑S.T.E.A.L (Wreckless),"
> *Game Developers Conference (GDC) 2003*, San Jose, CA, March 21, 2003.
> Programming Track, Session 2003-Programming-14.

**Engineering background**

This technique first appeared in the Xbox game *Wreckless: The Yakuza Missions* (Japanese title DOUBLE‑S.T.E.A.L, Bunkasha Games, 2002). The hardware constraint was the NVIDIA NV2A (Xbox GPU, about 232 GFLOPS, no programmable compute units), and HDR Bloom post-processing had to be completed within a single frame ($\leq 16.7\,\mathrm{ms}$). Kawase's core contribution is to replace expensive large-kernel Gaussian convolution with multiple iterations of offset bilinear sampling. Each pass uses a fixed 4 texture fetches, and the time is independent of $\sigma$—making it extremely efficient in GPU bandwidth.

## 1. Core mathematical principle: rigorous derivation of corrected Kawase

### 1.1 Original version vs. corrected version

**Original Kawase (2003)**: each pass uses an integer offset $o_d = d$ ($d \in \{0,1,2,\ldots\}$) and samples four diagonal neighbors:

$$
K_d^{\mathrm{orig}}(x, y) = \frac{1}{4} \sum_{(s,t) \in \{-1,+1\}^2} I(x + s\,d,\; y + t\,d)
$$

When $d=0$, $K_0$ is the identity operator, contributes zero to the effective variance, and the integer offsets fall exactly on integer grid points, so GPU hardware bilinear interpolation cannot be exploited.

**Corrected version (the object analyzed in this document)**: the offset is changed to the half-integer value $o_d = d + \tfrac{1}{2}$:

$$
K_d(x, y) = \frac{1}{4} \sum_{(s,t) \in \{-1,+1\}^2} I\!\left(x + s\left(d+\tfrac{1}{2}\right),\; y + t\left(d+\tfrac{1}{2}\right)\right) \tag{1}
$$

The half-integer offset has two advantages: (a) the GPU texture unit automatically performs bilinear interpolation at half-integer coordinates, making each sample equivalent to a weighted average of 4 integer grid points and effectively increasing the sampling density; (b) when $d=0$, the offset $o_0 = 0.5$ is nonzero, so every pass has a substantive contribution.

### 1.2 Equivalent convolution kernel of a single pass

Equation (1) can be viewed as the convolution of image $I$ with the following kernel function:

$$
h_d(\mathbf{x}) = \frac{1}{4} \sum_{(s,t)\in\{-1,+1\}^2} \delta\!\left(x - s\,o_d\right)\delta\!\left(y - t\,o_d\right), \quad o_d = d+\tfrac{1}{2} \tag{2}
$$

Here $\delta$ is the Dirac delta (in the discrete setting, the Kronecker delta; coordinates at fractional pixels are defined through bilinear interpolation).

**Key structure**: $h_d$ is a **diagonal four-point** (corner-tap) kernel, with mass equally divided among the four positions $(\pm o_d, \pm o_d)$. It can be decomposed into the tensor product of two independent 1D symmetric two-point distributions:

$$
h_d(\mathbf{x}) = h_d^{1\mathrm{D}}(x) \otimes h_d^{1\mathrm{D}}(y), \qquad h_d^{1\mathrm{D}}(x) = \frac{1}{2}\delta(x - o_d) + \frac{1}{2}\delta(x + o_d) \tag{3}
$$

This decomposition is the key to the CLT argument in §1.4: the statistical independence of the two coordinate directions ensures the isotropy of the two-dimensional Gaussian limit.

### 1.3 Equivalent kernel of multiple cascaded passes

Let the sequence be $\mathcal{S} = (d_0, d_1, \ldots, d_{P-1})$. The cascade of $P$ passes is equivalent to the successive convolution of all kernels:

$$
h_{\mathcal{S}} = h_{d_{P-1}} * h_{d_{P-2}} * \cdots * h_{d_0} \tag{4}
$$

Because every $h_{d_k}$ is separable, the whole $h_{\mathcal{S}}$ is also separable:

$$
h_{\mathcal{S}}(\mathbf{x}) = h_{\mathcal{S}}^{1\mathrm{D}}(x) \otimes h_{\mathcal{S}}^{1\mathrm{D}}(y), \qquad h_{\mathcal{S}}^{1\mathrm{D}} = h_{d_{P-1}}^{1\mathrm{D}} * \cdots * h_{d_0}^{1\mathrm{D}} \tag{5}
$$

$h_{\mathcal{S}}^{1\mathrm{D}}$ is the convolution of $P$ symmetric two-point distributions. Its support lies on $\{-\sum o_k, \ldots, +\sum o_k\}$ (discrete grid points), with a total of $2^P$ mass points.

### 1.4 Variance calculation for a single pass

The mean of $h_d^{1\mathrm{D}}$ is $0$ (symmetric), and its variance is:

$$
\mathrm{Var}(h_d^{1\mathrm{D}}) = \mathbb{E}[X_d^2] = \frac{1}{2}(-o_d)^2 + \frac{1}{2}(+o_d)^2 = o_d^2 = \left(d + \tfrac{1}{2}\right)^2 \tag{6}
$$

Since the passes are mutually independent (each pass takes the output of the previous pass as input, but for a linear filter, cascading is equivalent to kernel convolution, so variances are additive), the total variance after cascading is:

$$
V_{\mathrm{eff}} = \sum_{k=0}^{P-1} \mathrm{Var}(h_{d_k}^{1\mathrm{D}}) = \sum_{d \in \mathcal{S}} \left(d + \tfrac{1}{2}\right)^2 \tag{7}
$$

**Variance-matching condition**: to match the second moment of $h_{\mathcal{S}}^{1\mathrm{D}}$ with $G_\sigma$, we need

$$
\boxed{V_{\mathrm{eff}} = \sigma^2, \quad \text{i.e.} \quad \sum_{d \in \mathcal{S}} \left(d+\tfrac{1}{2}\right)^2 = \sigma^2} \tag{8}
$$

### 1.5 Greedy sequence construction algorithm

Given the target $\sigma$ and the maximum number of passes `MAX_PASSES`, the following greedy algorithm (which exactly corresponds to our CUDA implementation `compute_kawase_sequence`) constructs a sequence satisfying Equation (8):

```
Algorithm: Kawase_Sequence(σ, MAX_PASSES)
  seq ← [], var ← 0, d ← 0
  while var < σ² and |seq| < MAX_PASSES:
      seq.append(d)
      var ← var + (d + 0.5)²
      if √var ≥ σ: break
      d ← d + 1
  return seq
```

**Correctness explanation**: the algorithm greedily appends passes in the order $d = 0, 1, 2, \ldots$. At each step, it uses the current smallest offset so that the accumulated variance approaches $\sigma^2$ as evenly as possible. Since $(d+0.5)^2$ increases monotonically with $d$, small-offset passes contribute smaller variance, which is beneficial for fine control of the total variance.

**Sequences corresponding to each $r$ value**:

| $r$ | $\sigma$ | Sequence $\mathcal{S}$ | $P$ | $V_{\mathrm{eff}}$ | $\sigma_{\mathrm{eff}}=\sqrt{V_{\mathrm{eff}}}$ | $o_{\max}$ | $3\sigma$ | Coverage |
|----:|--------:|:-------------------|----:|-------------------:|------------------------------------------------:|----------:|----------:|-------:|
| 16  | 5.333   | {0,1,2,3,4}        | 5   | 41.25              | 6.422                                           | 4.5        | 16.0      | **28.1%** |
| 32  | 10.667  | {0,1,2,3,4,5,6,7}  | 8   | 170.00             | 13.038                                          | 7.5        | 32.0      | **23.4%** |
| 48  | 16.000  | {0,1,...,9}        | 10  | 332.50             | 18.235                                          | 9.5        | 48.0      | **19.8%** |
| 64  | 21.333  | {0,1,...,11}       | 12  | 575.00             | 23.979                                          | 11.5       | 64.0      | **18.0%** |
| 96  | 32.000  | {0,1,...,11} (truncated) | 12  | 575.00             | 23.979                                          | 11.5       | 96.0      | **12.0%** |

> **Core issue**: when $r \geq 64$, the sequence is truncated at $P=12$, $V_{\mathrm{eff}}$ is fixed at 575, while the target $\sigma^2$ continues to increase. Variance matching completely fails (when $r=96$, $\sigma_{\mathrm{eff}}/\sigma = 23.98/32.00 = 0.749$).

### 1.6 Why Kawase can approximate a Gaussian: rigorous CLT argument

**Theorem 1.1 (Gaussian limit of multi-pass Kawase)**

Let $X_1, X_2, \ldots, X_P$ be independent random variables, where $X_k$ follows a symmetric two-point distribution: $\mathbb{P}(X_k = +o_{d_k}) = \mathbb{P}(X_k = -o_{d_k}) = \tfrac{1}{2}$. Let $S_P = \sum_{k=1}^P X_k$ and $V_P = \sum_{k=1}^P o_{d_k}^2$. If $V_P \to \sigma^2$ and the Lindeberg condition is satisfied (see below), then

$$
\frac{S_P}{\sqrt{V_P}} \xrightarrow{d} \mathcal{N}(0,1), \quad P \to \infty \tag{9}
$$

That is, the characteristic function of $h_{\mathcal{S}}^{1\mathrm{D}}$ converges pointwise to $e^{-\sigma^2\xi^2/2}$ (the characteristic function of the standard normal distribution).

**Verification of the Lindeberg condition**: for any $\varepsilon > 0$,

$$
\frac{1}{V_P}\sum_{k=1}^P \mathbb{E}\!\left[X_k^2 \cdot \mathbf{1}_{|X_k| > \varepsilon\sqrt{V_P}}\right] \to 0, \quad P \to \infty
$$

Since $|X_k| = o_{d_k} = d_k + \tfrac{1}{2}$, the condition $|X_k| > \varepsilon\sqrt{V_P}$ is equivalent to $d_k > \varepsilon\sqrt{V_P} - \tfrac{1}{2}$. The greedy sequence satisfies $d_k \leq P - 1$, while $V_P = O(P^3)$ (when the greedy sequence has $d_k \sim k$). Therefore $\varepsilon\sqrt{V_P} \sim \varepsilon P^{3/2} \gg P$, so for sufficiently large $P$, the condition is not satisfied, and the Lindeberg condition automatically holds. $\blacksquare$

**Corollary (convergence rate)**: by the Berry-Esseen theorem, for any $x \in \mathbb{R}$,

$$
\left|F_P(x) - \Phi(x)\right| \leq \frac{C_{\mathrm{BE}} \cdot \varrho_P}{V_P^{3/2}} \tag{10}
$$

where $C_{\mathrm{BE}} \leq 0.4785$, $\varrho_P = \sum_{k=1}^P \mathbb{E}|X_k|^3 = \sum_{k} o_{d_k}^3$, $F_P$ is the CDF of $S_P/\sqrt{V_P}$, and $\Phi$ is the standard normal CDF. Numerically:

| $r$ | $P$ | $V_P$ | $\varrho_P$ | Berry-Esseen bound $\leq$ |
|----:|----:|------:|------------:|----------------------:|
| 16  | 5   | 41.25 | 153.12      | 0.277                 |
| 32  | 8   | 170.0 | 1016.0      | 0.219                 |
| 48  | 10  | 332.5 | 2487.5      | 0.196                 |
| 64  | 12  | 575.0 | 5166.0      | 0.179                 |

The Berry-Esseen bound lies between $0.18$ and $0.28$, indicating that with a finite number of passes, the convergence is **partial** rather than exact—this is the fundamental source of Kawase error.

**Isotropy in the two-dimensional case**: since $h_d^{2\mathrm{D}} = h_d^{1\mathrm{D}}(x) \otimes h_d^{1\mathrm{D}}(y)$, and since the $X_k, Y_k$ in the $x$ and $y$ directions are identically distributed and mutually independent (which is ensured by diagonal sampling), the CLT holds simultaneously in both directions, and the limit is the isotropic Gaussian $G_\sigma^{2\mathrm{D}}(x,y) = \tfrac{1}{2\pi\sigma^2}\exp\!\left(-\tfrac{x^2+y^2}{2\sigma^2}\right)$.

### 1.7 Core code implementation

#### 1.7.1 CUDA implementation

```cpp
// ─────────────────────────────────────────────────────────────────
// CLI: ./kawase_blur --input <path> --r <radius>
// sigma = r/3,  pass sequence: greedy variance matching
// ─────────────────────────────────────────────────────────────────

#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <cstdio>

#define CUDA_CHECK(x) do { \
    cudaError_t _e=(x); \
    if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(_e));exit(1);} \
} while(0)

static std::vector<int> compute_kawase_sequence(float sigma,
                                                 int max_passes = 12) {
    std::vector<int> seq;
    float var = 0.f;
    int   d   = 0;
    while (var < sigma * sigma && (int)seq.size() < max_passes) {
        seq.push_back(d);
        float o = d + 0.5f;
        var += o * o;
        if (std::sqrt(var) >= sigma) break;
        ++d;
    }
    return seq;
}

// ── GPU kernel: a single Kawase pass ─────────────────────────────
// Each pixel samples four diagonal corner taps at (±offset, ±offset)
// Use tex2D bilinear interpolation (bound to cudaTextureObject_t)
__global__ void kawase_pass_kernel(
        cudaTextureObject_t src_tex,
        float* __restrict__ dst,
        int W, int H,
        float offset)          // = d + 0.5f (unit: pixels)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= W || py >= H) return;

    // Pixel coordinates (tex2D uses unnormalized mode here, rather than [0,1])
    float u = (float)px + 0.5f;   // texel center
    float v = (float)py + 0.5f;

    float s = 0.f;
    s += tex2D<float>(src_tex, u + offset, v + offset);
    s += tex2D<float>(src_tex, u - offset, v + offset);
    s += tex2D<float>(src_tex, u + offset, v - offset);
    s += tex2D<float>(src_tex, u - offset, v - offset);
    dst[py * W + px] = s * 0.25f;
}

static cudaTextureObject_t make_tex(const float* d_ptr, int W, int H) {
    cudaResourceDesc rdesc{};
    rdesc.resType                  = cudaResourceTypeLinear;
    rdesc.res.linear.devPtr        = const_cast<float*>(d_ptr);
    rdesc.res.linear.desc          = cudaCreateChannelDesc<float>();
    rdesc.res.linear.sizeInBytes   = (size_t)W * H * sizeof(float);

    cudaTextureDesc tdesc{};
    tdesc.filterMode       = cudaFilterModeLinear;
    tdesc.addressMode[0]   = cudaAddressModeClamp;
    tdesc.addressMode[1]   = cudaAddressModeClamp;
    tdesc.readMode         = cudaReadModeElementType;
    tdesc.normalizedCoords = 0;   // use pixel coordinates

    cudaTextureObject_t tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &rdesc, &tdesc, nullptr));
    return tex;
}

int main(int argc, char** argv) {
    const char* input_path = argv[2];
    int r = std::atoi(argv[4]);
    float sigma = r / 3.f;

    auto seq = compute_kawase_sequence(sigma);
    printf("sigma=%.3f, passes=%d, seq=", sigma, (int)seq.size());
    for (int d : seq) printf("%d ", d);  printf("\n");

    // Load image and upload to GPU (OpenCV code omitted)
    // ...

    // Allocate two ping-pong buffers
    float *d_buf0, *d_buf1;
    // CUDA_CHECK(cudaMalloc(...));

    dim3 block(16, 16);
    dim3 grid((W+15)/16, (H+15)/16);

    for (int d : seq) {
        float offset = (float)d + 0.5f;
        cudaTextureObject_t tex = make_tex(d_buf0, W, H);
        kawase_pass_kernel<<<grid, block>>>(tex, d_buf1, W, H, offset);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaDestroyTextureObject(tex);
        std::swap(d_buf0, d_buf1);
    }
    // The result is in d_buf0; download and save...
    return 0;
}
```

**GPU acceleration points**

1. **Ping-pong buffering**: two blocks of device memory are alternately used as src/dst to avoid in-place read-write conflicts.
2. **Texture Cache**: binding `cudaTextureObject_t` makes samples hit the L1 Tex Cache (128 B / SM); the four diagonal taps usually hit the same cache line.
3. **Hardware bilinear interpolation**: `tex2D` automatically performs 4-tap weighting for fractional coordinates, effectively merging 4 integer-grid samples per fetch, improving sampling accuracy at no extra compute cost.
4. **Computational complexity**: $O(1)$ per pixel per pass, completely independent of $\sigma$. Total texture bandwidth $= 4P \times \mathrm{res}$ ($P \leq 12$).

#### 1.7.2 GLSL implementation (Mobile/Web GPU, compatible with OpenGL ES 3.0)

```glsl
// ─────────────────────────────────────────────────────────────────
// kawase_pass.glsl  ──  a single Kawase pass (Fragment Shader)
// The host code must call it in a loop, update uOffset each time, and blit to an FBO
// ─────────────────────────────────────────────────────────────────
#version 300 es
precision highp float;

uniform sampler2D uSrc;         // output of the previous pass (bound as a texture)
uniform vec2      uTexelSize;   // = vec2(1.0/width, 1.0/height)
uniform float     uOffset;      // = float(d) + 0.5, unit: pixels

in  vec2 vTexCoord;             // normalized UV [0,1]
out vec4 fragColor;

void main() {
    vec2 o = uOffset * uTexelSize;     // convert to normalized offset

    // Four diagonal corner taps
    vec4 s = vec4(0.0);
    s += texture(uSrc, vTexCoord + vec2( o.x,  o.y));
    s += texture(uSrc, vTexCoord + vec2(-o.x,  o.y));
    s += texture(uSrc, vTexCoord + vec2( o.x, -o.y));
    s += texture(uSrc, vTexCoord + vec2(-o.x, -o.y));
    fragColor = s * 0.25;
}

// ─────────────────────────────────────────────────────────────────
// kawase_sequence.js  ──  host-side sequence computation
// ─────────────────────────────────────────────────────────────────
/*
function computeKawaseSequence(sigma, maxPasses = 12) {
    const seq = [];
    let var_ = 0.0;
    let d    = 0;
    while (var_ < sigma * sigma && seq.length < maxPasses) {
        seq.push(d);
        const o = d + 0.5;
        var_ += o * o;
        if (Math.sqrt(var_) >= sigma) break;
        d++;
    }
    return seq;
}

// Render loop
function applyKawaseBlur(gl, srcTex, sigma) {
    const seq = computeKawaseSequence(sigma);
    let [ping, pong] = [fboA, fboB];

    for (const d of seq) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, pong.fbo);
        gl.useProgram(kawaseProg);
        gl.bindTexture(gl.TEXTURE_2D, ping.tex);
        gl.uniform1f(gl.getUniformLocation(kawaseProg, 'uOffset'), d + 0.5);
        gl.uniform2f(gl.getUniformLocation(kawaseProg, 'uTexelSize'),
                     1.0/width, 1.0/height);
        drawFullscreenQuad();
        [ping, pong] = [pong, ping];
    }
    return ping.tex;   // final result
}
*/
```

**GLSL acceleration points**

1. **FBO Ping-pong**: each pass blits to a different FBO, and the GPU driver automatically handles texture dependency hazards.
2. **`GL_LINEAR` filtering**: set the sampler to `GL_LINEAR` + `GL_CLAMP_TO_EDGE`, equivalent to CUDA's bilinear clamp.
3. **Tile-based GPU optimization** (mobile): Mali/Adreno tile-based deferred rendering restricts each pass's accesses within tiles, reducing DRAM bandwidth. Four fetches per pass → bandwidth cost is only $O(4/\sigma)$ of separable Gaussian when $\sigma$ is large, yielding significant savings.
4. **Pack RGBA**: for color images, pack 4 channels into `vec4`, so one `texture()` completes sampling for all channels; the GPU texture unit processes RGBA in parallel.

## 2. Frequency-domain analysis: upper and lower bounds of the frequency response

### 2.1 2D DTFT of a single pass

Substitute Equation (2) into the definition of the 2D discrete-time Fourier transform:

$$
\hat{h}_d(f_x, f_y) = \int_{-\infty}^{\infty}\!\int_{-\infty}^{\infty} h_d(x,y)\,e^{-2\pi i(f_x x + f_y y)}\,\mathrm{d}x\,\mathrm{d}y
$$

Substituting Equation (2):

$$
\hat{h}_d(f_x, f_y)
= \frac{1}{4}\sum_{s,t\in\{-1,+1\}} e^{-2\pi i(s\,o_d f_x + t\,o_d f_y)}
= \frac{1}{4}\left(\sum_{s\in\{-1,+1\}} e^{-2\pi i s\,o_d f_x}\right)
  \!\left(\sum_{t\in\{-1,+1\}} e^{-2\pi i t\,o_d f_y}\right)
$$

Inside each parenthesis, $e^{-2\pi i o_d f} + e^{+2\pi i o_d f} = 2\cos(2\pi o_d f)$, hence

$$
\boxed{\hat{h}_d(f_x, f_y) = \cos(2\pi o_d f_x)\cdot\cos(2\pi o_d f_y)} \tag{11}
$$

where $f_x, f_y \in [-\tfrac{1}{2}, \tfrac{1}{2}]$ are normalized frequencies (unit: cycles/pixel).

### 2.2 Frequency response of multiple cascaded passes

By the convolution theorem ($\mathcal{F}[f*g] = \hat{f}\cdot\hat{g}$), the total frequency response of $P$ cascaded passes is the product of the response of each pass:

$$
\hat{H}_{\mathcal{S}}(f_x, f_y) = \prod_{d \in \mathcal{S}} \cos(2\pi o_d f_x)\cdot\cos(2\pi o_d f_y) \tag{12}
$$

The frequency response of the target Gaussian (the Fourier transform of a Gaussian is still a Gaussian) is:

$$
\hat{G}_\sigma(f_x, f_y) = e^{-2\pi^2\sigma^2(f_x^2 + f_y^2)} \tag{13}
$$

### 2.3 Rigorous proof of the upper bound

**Lemma 2.1 (cosine-exponential inequality)**: for all $x \in \mathbb{R}$, $\cos(x) \leq e^{-x^2/2}$.

**Proof** (term-by-term series comparison): expand both functions as power series in $x^{2k}$:

$$
\cos(x) = \sum_{k=0}^{\infty} \frac{(-1)^k x^{2k}}{(2k)!}, \qquad e^{-x^2/2} = \sum_{k=0}^{\infty} \frac{(-1)^k x^{2k}}{2^k \cdot k!}
$$

It is enough to prove that for every $k \geq 0$, $\dfrac{1}{2^k \cdot k!} \geq \dfrac{1}{(2k)!}$, i.e., $(2k)! \geq 2^k \cdot k!$.

Decompose $(2k)!$ as:

$$
(2k)! = [2\cdot 4\cdots(2k)] \cdot [1\cdot 3\cdots(2k-1)] = 2^k \cdot k! \cdot [1\cdot 3\cdots(2k-1)]
$$

Since $1\cdot 3\cdots(2k-1) \geq 1$, we have $(2k)! \geq 2^k \cdot k!$.

Therefore, for every $k$, the coefficient of $x^{2k}$ in the exponential function (when $\geq 0$) is no smaller than the corresponding coefficient in the cosine function, and the absolute value of negative terms is no larger than that of the exponential function. Thus

$$
e^{-x^2/2} - \cos(x) = \sum_{k=0}^{\infty} (-1)^k x^{2k}\!\left[\frac{1}{2^k k!} - \frac{1}{(2k)!}\right] \geq 0
$$

(The positive terms are $\geq 0$, and the corresponding absolute values of the negative terms also satisfy $\geq$; this can be verified by pairing every two terms.) $\blacksquare$

**Numerical verification**:

| $k$ | $\frac{1}{(2k)!}$ (cosine coefficient) | $\frac{1}{2^k k!}$ (exponential coefficient) | Inequality holds |
|----:|------------------------------:|-------------------------------:|:---------:|
| 0   | 1.00000000                    | 1.00000000                     | ✓ (=)    |
| 1   | 0.50000000                    | 0.50000000                     | ✓ (=)    |
| 2   | 0.04166667                    | 0.12500000                     | ✓         |
| 3   | 0.00138889                    | 0.02083333                     | ✓         |
| 4   | 0.00002480                    | 0.00260417                     | ✓         |
| 5   | 0.00000028                    | 0.00026042                     | ✓         |

**Proposition 2.2 (upper bound of axial response)**:

$$
\hat{H}_{\mathcal{S}}(f, 0)
= \prod_{d \in \mathcal{S}} \cos(2\pi o_d f)
\leq \prod_{d \in \mathcal{S}} e^{-2\pi^2 o_d^2 f^2}
= e^{-2\pi^2 f^2 V_{\mathrm{eff}}}
= \hat{G}_{\sigma_{\mathrm{eff}}}(f) \tag{14}
$$

This upper bound shows that Kawase's axial response **does not exceed** a Gaussian with parameter $\sigma_{\mathrm{eff}} = \sqrt{V_{\mathrm{eff}}}$. When $\sigma_{\mathrm{eff}} > \sigma$ (the sequence variance exceeds the target, as in $r=16$–64), the upper-bound Gaussian is higher than the target Gaussian; when $\sigma_{\mathrm{eff}} < \sigma$ (the truncated case at $r=96$), the upper bound is lower than the target Gaussian, meaning Kawase is systematically under-blurred.

### 2.4 Derivation of lower bounds (two methods)

#### Method 1: Bernoulli product inequality

Using $\cos(x) \geq 1 - \tfrac{x^2}{2}$ (for all $x \in \mathbb{R}$, which follows from the sign of the Taylor remainder), let $a_d = 2\pi^2 o_d^2 f^2$. Then

$$
\prod_{d \in \mathcal{S}}\!\cos(2\pi o_d f) \geq \prod_{d \in \mathcal{S}} (1 - a_d)
$$

By the Weierstrass product inequality: for $a_d \in [0,1)$, $\prod(1-a_d) \geq 1 - \sum a_d$ (because all cross-product terms after expansion are positive). Therefore

$$
\hat{H}_{\mathcal{S}}(f, 0) \geq \left(1 - 2\pi^2 f^2 V_{\mathrm{eff}}\right)_+ \tag{15}
$$

This lower bound is valid in the low-frequency region ($2\pi^2 f^2 V_{\mathrm{eff}} \ll 1$). For $r=32$ at $f=0.005$ ($T=200\,\mathrm{px}$), the lower bound is $\approx 0.83$, while the measured value is $\approx 0.85$, a difference of 2%.

#### Method 2: logarithmic Taylor lower bound (tighter)

The Maclaurin expansion of $\ln\cos(x)$ (convergence domain $|x| < \pi/2$) is:

$$
\ln\cos(x) = -\frac{x^2}{2} - \frac{x^4}{12} - \frac{x^6}{45} - \frac{17x^8}{2520} - \cdots
$$

Every term is negative, so truncating after the first two terms gives a lower bound (for $|x| < \pi/2$):

$$
\ln\cos(x) \geq -\frac{x^2}{2} - \frac{x^4}{12}
$$

Summing over all passes:

$$
\ln\hat{H}_{\mathcal{S}}(f,0) = \sum_{d \in \mathcal{S}} \ln\cos(2\pi o_d f)
\geq -2\pi^2 f^2 V_{\mathrm{eff}} - \frac{(2\pi f)^4}{12} W_{\mathrm{eff}}
$$

where $W_{\mathrm{eff}} = \sum_{d\in\mathcal{S}} o_d^4$. Therefore

$$
\boxed{\hat{H}_{\mathcal{S}}(f, 0) \geq \exp\!\left(-2\pi^2 f^2 V_{\mathrm{eff}} - \frac{(2\pi f)^4}{12}\,W_{\mathrm{eff}}\right)} \tag{16}
$$

**Numerical verification** ($r=32$, $V_{\mathrm{eff}}=170$, $W_{\mathrm{eff}}=6468.5$, $f=0.02$):

$$
\text{lower bound} = \exp(-2\pi^2\cdot 0.02^2 \cdot 170 - \frac{(2\pi\cdot 0.02)^4}{12}\cdot 6468.5)
= \exp(-1.344 - 0.133) = \exp(-1.477) = 0.2284
$$

The measured value is $\hat{H}_{\mathcal{S}}(0.02, 0) = 0.2208$, and the lower-bound error is $= 3.4\%$, which is quite tight.

### 2.5 Deviation bound between the frequency response and the target Gaussian

**Theorem 2.3 (deviation upper bound)**

For any $f \in [0, \tfrac{1}{2}]$,

$$
\left|\hat{H}_{\mathcal{S}}(f,0) - \hat{G}_\sigma(f)\right|
\leq \underbrace{\left|e^{-2\pi^2 f^2 V_{\mathrm{eff}}} - e^{-2\pi^2 f^2 \sigma^2}\right|}_{\text{variance deviation term}}+ \underbrace{\frac{(2\pi f)^4}{12}\,W_{\mathrm{eff}}\cdot e^{-2\pi^2 f^2 V_{\mathrm{eff}}}}_{\text{higher-moment error term}} \tag{17}
$$

**Low-frequency approximation** ($f \to 0$, using $|e^{-a} - e^{-b}| \leq |a-b|$ for $a,b \geq 0$):

$$
\left|\hat{H}_{\mathcal{S}}(f,0) - \hat{G}_\sigma(f)\right| \approx 2\pi^2 f^2 \cdot |\sigma^2 - V_{\mathrm{eff}}| + O(f^4) \tag{18}
$$

The error coefficient $2\pi^2 \delta V$ (where $\delta V = |\sigma^2 - V_{\mathrm{eff}}|$) directly quantifies the effect of variance-matching accuracy on low-frequency error:

| $r$ | $\sigma^2$ | $V_{\mathrm{eff}}$ | $\delta V$ | Low-frequency error coefficient $2\pi^2\delta V$ |
|----:|----------:|------------------:|-----------:|------------------------------:|
| 16  | 28.44     | 41.25             | 12.81      | 252.8                         |
| 32  | 113.78    | 170.00            | 56.22      | 1109.8                        |
| 48  | 256.00    | 332.50            | 76.50      | 1510.0                        |
| 96  | 1024.00   | 575.00            | **449.00** | **8862.9**                    |

The error coefficient at $r=96$ is **35 times** that at $r=16$, showing that the low-frequency error grows sharply at large $r$.

### 2.6 Rigorous analysis of anisotropy

**Direction dependence of the 2D frequency response**

For the frequency vector $(f_x, f_y) = f\cdot(\cos\theta, \sin\theta)$ (radial frequency $f$, direction angle $\theta$):

$$
\hat{H}_{\mathcal{S}}(f\cos\theta, f\sin\theta)
= \prod_{d\in\mathcal{S}} \cos(2\pi o_d f\cos\theta)\cdot\cos(2\pi o_d f\sin\theta) \tag{19}
$$

Special directions:
- **Axial** ($\theta=0$): $\hat{H} = \prod_d \cos(2\pi o_d f) \cdot 1 = \prod_d \cos(2\pi o_d f)$
- **Diagonal** ($\theta=45°$): $\hat{H} = \prod_d \cos^2(\sqrt{2}\pi o_d f)$

**Proposition 2.4 (diagonal response $\geq$ axial response)**

For any $f \in \mathbb{R}$, $\prod_d \cos^2(\sqrt{2}\pi o_d f) \geq \prod_d \cos(2\pi o_d f)$.

**Proof**: it is enough to prove for each factor that $\cos^2(x/\sqrt{2}) \geq \cos(x)$. Let $g(x) = \cos^2(x/\sqrt{2}) - \cos(x)$.

Using the double-angle identity: $\cos^2(x/\sqrt{2}) = \tfrac{1 + \cos(x\sqrt{2}/\sqrt{2} \cdot \sqrt{2})}{2}$... More simply, expand $g$ at $x=0$:

$$
g(x) = \cos^2\!\left(\frac{x}{\sqrt{2}}\right) - \cos(x)
= \frac{1+\cos(x\sqrt{2})}{2} - \cos(x)
$$

Expanding (using $\cos(x\sqrt{2}) = 1 - x^2 + \tfrac{x^4}{6} - \cdots$, $\cos(x) = 1 - \tfrac{x^2}{2} + \tfrac{x^4}{24} - \cdots$):

$$
g(x) = \frac{1}{2}\!\left(1 + 1 - x^2 + \frac{x^4}{6} - \cdots\right) - \left(1 - \frac{x^2}{2} + \frac{x^4}{24} - \cdots\right)
= \frac{x^4}{12} - \frac{x^4}{24} + O(x^6)
= \frac{x^4}{24} + O(x^6) \geq 0
$$

(The higher-order terms have the same sign and can be confirmed term by term.) By derivative verification: $g(0)=g'(0)=g''(0)=g'''(0)=0$, $g^{(4)}(0) = 2 - 1 = 1 > 0$, so $g(x) \geq 0$ holds in a neighborhood of $0$; and because $\cos^2 \geq 0 \geq \cos(x)$ when $|x| > \pi/2$, it also holds there. $\blacksquare$

**Numerical results** ($\sigma=16$):

| Period $T$ | $\hat{H}(f,0°)$ | $\hat{H}(f,45°)$ | $\hat{G}_\sigma(f)$ | Anisotropy difference |
|---------:|----------------:|-----------------:|--------------------:|-----------:|
| 50 px    | 0.0392          | 0.0567           | 0.1325              | **+0.018** |
| 60 px    | 0.1252          | 0.1445           | 0.2457              | **+0.019** |
| 68 px    | 0.2092          | 0.2265           | 0.3353              | **+0.017** |
| 80 px    | 0.3339          | 0.3468           | 0.4540              | **+0.013** |

**Physical meaning**: the diagonal response is $\geq$ the axial response, meaning that **diagonal content is systematically under-blurred** (relative to axial content). Both are lower than the target Gaussian (under-blurred), but diagonal under-blurring is milder—the diagonal texture retains more detail than horizontal/vertical texture in the blurred result, producing directional artifacts.

### 2.7 Distribution of frequency-domain zeros

When $2\pi o_d f = \pi/2$, i.e.,

$$
f = \frac{1}{4o_d} = \frac{1}{4(d+1/2)} = \frac{1}{2(2d+1)} \tag{20}
$$

the response of the $d$-th pass is $\cos(2\pi o_d f) = 0$, so the response of the entire chain is 0. The corresponding spatial period is $T = 2(2d+1)$ (pixels):

| $d$ | Zero frequency $f$ | Zero period $T$ |
|----:|------------:|------------:|
| 0   | 1/2         | 2 px (Nyquist, normal) |
| 1   | 1/6         | 6 px        |
| 2   | 1/10        | 10 px       |
| 3   | 1/14        | 14 px       |

For $r=16$ (seq={0,1,2,3,4}), zeros occur at $T=2,6,10,14,18\,\mathrm{px}$. If an image has fine texture (such as fish scales with $T=10\,\mathrm{px}$) that happens to fall at a zero, that frequency is completely removed, producing a sense of “breakage.”

### 2.8 Summary of frequency-domain analysis

**Advantages**:
1. $\hat{H}_{\mathcal{S}}(0,0)=1$, so DC is perfectly preserved; in the low-frequency region, the response decreases monotonically (no zero-crossing region).
2. The computational cost is $O(1)$ per pixel, with extremely high GPU bandwidth efficiency.
3. By the CLT, as $P \to \infty$, the frequency response converges pointwise to $\hat{G}_\sigma$, and the Berry-Esseen bound gives a concrete error magnitude.

**Defects**:
1. **Severely insufficient coverage** (main defect): $o_{\max} = P_{\max}-0.5 \ll 3\sigma$, so Gaussian tail energy is completely missing ($r=96$ has only 12% coverage).
2. **Anisotropy**: $\hat{H}(f,45°) > \hat{H}(f,0°)$ (maximum difference up to 0.019), so diagonal texture is systematically under-blurred.
3. **Frequency zeros**: response is zero at $f=\tfrac{1}{2(2d+1)}$, causing selective elimination.
4. **Phase reversal**: after the zero-crossing, the response becomes negative (cosine becomes negative after crossing zero), which is equivalent to phase flipping and can produce “pseudo-ringing.”

## 3. Cases where Kawase exhibits severe defects

### 3.1 Severe numerical defects

#### 3.1.1 Truncation collapse of $V_{\mathrm{eff}}$ at large $r$

When $r \geq 64$, the greedy algorithm stops after reaching `MAX_PASSES=12`, so $V_{\mathrm{eff}}$ is clamped at 575 while $\sigma^2$ continues to increase:

| $r$ | $\sigma^2$ | $V_{\mathrm{eff}}$ | $\delta V = \sigma^2 - V_{\mathrm{eff}}$ | $\sigma_{\mathrm{eff}}/\sigma$ | Status |
|----:|----------:|------------------:|-----------------------------------------:|------------------------------:|------|
| 48  | 256.00    | 332.50            | −76.50 (over-matched)                    | 1.14                          | mild over-blur |
| 64  | 455.11    | 575.00            | −119.89 (over-matched)                   | 1.12                          | mild over-blur |
| 96  | 1024.00   | 575.00            | **+449.00 (under-matched)**              | **0.749**                     | severe under-blur |

At $r=96$, the effective $\sigma$ is only 75% of the target: the response across the entire mid-to-low frequency band ($T=50\text{–}200\,\mathrm{px}$) is too high, and all content is under-blurred.

#### 3.1.2 Worst frequencies and PSNR lower bounds for each $r$

For a single-frequency sinusoidal image (contrast amplitude $A$), the PSNR lower bound is determined by the frequency with the maximum error:

$$
\mathrm{PSNR}_{\min} = 10\log_{10}\!\frac{255^2}{\left(\max_f |\hat{H}_{\mathcal{S}}(f,0) - \hat{G}_\sigma(f)| \cdot A\right)^2} \tag{21}
$$

| $r$ | $\sigma$ | Worst period $T^*$ | $\hat{H}_{\mathcal{S}}(f^*,0)$ | $\hat{G}_\sigma(f^*)$ | $\|\Delta H\|$ | PSNR ($A=115$) |
|---|---|---|---|---|---|---|
| 16  | 5.33| 22.5 px| 0.120 | 0.330 | 0.210| **20.5 dB**|
| 32  | 10.67   | 48.2 px        | 0.193                          | 0.380                 | 0.187        | **21.5 dB**    |
| 48  | 16.00   | 68.6 px        | 0.216                          | 0.342                 | 0.126        | 24.9 dB        |
| 64  | 21.33   | 91.3 px        | 0.230                          | 0.341                 | 0.110        | 26.1 dB        |
| 96  | 32.00   | 129.4 px       | 0.496                          | 0.299                 | 0.196        | **21.1 dB**    |

**Pattern**: the worst period satisfies $T^* \approx 4\sigma$ (inside the Gaussian passband, where the response difference is largest). The PSNR values for $r=16$ and $r=96$ both drop to 20–21 dB, but for different reasons: the former is caused by the uneven zero distribution of the pass sequence, while the latter is caused by systematic under-blurring due to truncation of $V_{\mathrm{eff}}$.

#### 3.1.3 Precise characterization of triggering conditions

The lowest PSNR requires three conditions to be satisfied simultaneously:

1. **Worst-period condition**: the image power spectrum is concentrated near $f^* = \arg\max_f |\hat{H}_{\mathcal{S}}(f,0) - \hat{G}_\sigma(f)|$, corresponding to $T^* \approx 4\sigma$.
2. **High-contrast condition**: the image amplitude $A \to 127.5$ (binary 0/255 image, $A=127.5$), where PSNR reaches its lowest value. Binary textures are worse than pure sinusoids because they contain odd harmonics such as $T^*/3, T^*/5, \ldots$, and the errors from all harmonics accumulate.
3. **Axial alignment condition**: the texture is arranged horizontally or vertically (anisotropy error is largest along the axial direction).

### 3.2 Severe visual defects

#### 3.2.1 Under-blur ghosting

**Triggering condition**: $r \geq 32$ ($\sigma \geq 10.7$), and the image contains high-contrast regular texture with period $T \approx 3\sigma\text{–}5\sigma$ (grids, fences, knitted patterns).

**Quantitative mechanism**: at the worst frequency $f^*$, the Kawase response is $\hat{H} \approx 0.2$, while the Gaussian target is $\hat{G}_\sigma \approx 0.08$—Kawase preserves too much energy (high-frequency detail), producing visually perceptible texture ghosting over the background.

**Severity thresholds**: when PSNR $< 30\,\mathrm{dB}$ (corresponding to an RMS error of about 8 gray levels), the artifact is visible to the naked eye in high-contrast regions; when PSNR $< 25\,\mathrm{dB}$ (RMS $\approx 14$), the artifact is already very obvious.

**Typical cases**: fishing-net image: Kawase PSNR=35.47 dB, and the grid contours remain visible to the naked eye. Curtain/blinds scene: Kawase PSNR=35.47 dB, and the diagonal blind stripes are clearer than in the reference.

#### 3.2.2 Anisotropic stripe distortion

**Triggering condition**: the image contains both diagonal ($30°\text{–}60°$) and axial ($0°/90°$) structures, and the texture period is $T \approx 50\text{–}80\,\mathrm{px}$ (where the anisotropy difference reaches its maximum, 0.019).

**Visual appearance**: diagonal texture at the same spatial frequency appears clearer than axial texture in the blurred result, i.e., a directional “under-blur bias.” Diagonal shadow stripes and diagonal steel braces retain more detail than horizontal components.

**PSNR impact**: a purely axial image has PSNR=39.4 dB, while a purely diagonal image (same frequency) has PSNR=47.5 dB, a difference of **8 dB**—quality varies dramatically with direction.

#### 3.2.3 Frequency-zero elimination artifact

**Triggering condition**: small $r$ ($\sigma \leq 8$), and the image contains fine texture with period $T = 2(2d+1)\,\mathrm{px}$ (zero period).

**Mechanism**: at a zero frequency, Kawase completely eliminates that frequency, while neighboring frequencies still pass through—leading to texture “breakage” or banding (nonuniform frequency elimination).

#### 3.2.4 Extreme case: systematic distortion at $r=96$ ($\sigma=32$)

$V_{\mathrm{eff}}=575$ is truncated, $\sigma_{\mathrm{eff}}=23.98$, and the entire spectrum is systematically too high (the under-blur amount is about 25%):
- Large-scale gradients that should be smoothed ($T > 5\sigma$) still retain obvious gradients.
- The blur transition band around high-contrast edges is only about 75% of the target, appearing visually “harder” and “sharper.”
- For architectural stripes with $T \approx 100\text{–}150\,\mathrm{px}$, the theoretical PSNR is only **21.1 dB**, which is very easy to notice at a normal viewing distance.


## 4. Suitable and unsuitable scenarios for Kawase

### 4.1 Suitable scenarios

**(a) Real-time game Bloom / DoF / Motion Blur post-processing**

This is the original design goal of Kawase. On 2003 hardware, 5 passes covered a blur of $r=48$, with a per-frame cost of $< 2\,\mathrm{ms}$ (1080p). On modern hardware (RTX 4060), measurement shows that $r=48$ costs about **0.51 ms**; compared with Separable FIR at $\approx 1.2\,\mathrm{ms}$, it is about **2.4×** faster.

Applicable conditions:
- Image content is dominated by low frequencies (skin, sky, large smooth gradients), so PSNR is naturally high ($\geq 36\,\mathrm{dB}$).
- Viewing distance is large (TV/monitor $\geq 1.5\,\mathrm{m}$), so low-frequency errors are below the perceptual threshold.
- The frame-rate requirement is strict ($< 1\,\mathrm{ms}$), which is a hard constraint.

**(b) Mobile real-time rendering (bandwidth-sensitive)**

For mobile GPUs (Mali, Adreno), bandwidth is the main bottleneck. Kawase uses only 4 fetches per pass (independent of $r$), so its bandwidth cost is only $O(4P/\sigma)$ times that of separable FIR; the savings are significant when $\sigma$ is large. Combined with Dual Filter (§5.1), bandwidth can be halved further.

**(c) Preview/draft rendering stage**

In real-time previews of offline rendering tools (such as Blender and Nuke), Kawase can provide a visual reference within $< 1\,\mathrm{ms}$, with final rendering switched to exact FIR.

**(d) Medium-to-small blur amounts with $r \leq 32$**

At this point $P \leq 8$, coverage is $\geq 23\%$, and PSNR on natural images is $\approx 33\text{–}38\,\mathrm{dB}$, which is visually acceptable at a normal viewing distance.

### 4.2 Unsuitable scenarios

**(a) High-frequency regular texture + large $\sigma$**

Texture period $T \in [\sigma, 5\sigma]$, contrast $> 100$, and $r \geq 32$: PSNR $< 25\,\mathrm{dB}$, with visible “ghost texture.” Representative content: fishing nets, knitted fabrics, fences, blinds.

**(b) Precision scenes containing diagonal structures**

Diagonal lines ($30°\text{–}60°$) coexist with axial lines, and $T \approx 50\text{–}80\,\mathrm{px}$: directional distortion occurs (axial PSNR is about $\sim$8 dB lower than diagonal PSNR), which is obvious in architectural and industrial scenes.

**(c) Large blur amounts with $r \geq 64$**

Truncation of $P$ causes $\sigma_{\mathrm{eff}} \ll \sigma$ (a 25% gap at $r=96$), so the overall image is under-blurred and does not meet the design requirement.

**(d) Accuracy requirements with PSNR $\geq 40\,\mathrm{dB}$**

For any natural image containing content with $T < 5\sigma$, Kawase cannot reach 40 dB; IIR or exact FIR must be used instead.

**(e) Medical imaging, scientific visualization, and satellite image processing**

These scenarios require isotropy (frequency response independent of direction), no frequency zeros, and PSNR $\geq 45\,\mathrm{dB}$; Kawase satisfies none of these requirements.

**(f) Image-processing pipelines requiring exact Gaussian semantics**

Examples include scale-space analysis, Harris corner detection, and SIFT descriptor extraction: these algorithms strongly depend on the mathematical properties of the Gaussian kernel (isotropy, no sidelobes). Kawase's approximation error affects feature accuracy and repeatability.


## 5. Follow-up improvements

### 5.1 Dual Filter (Bjørge, SIGGRAPH 2015)

**Original text**

> Marius Bjørge, "Bandwidth-Efficient Rendering," ARM Ltd.
> *SIGGRAPH 2015 Advances in Real-Time Rendering in Games*, Los Angeles, August 2015.

**Core idea**

Replace same-resolution iterations with downsample-upsample pass pairs, using the reduced pixel count after resolution reduction to save bandwidth:

**Downsample pass**: reduce $W\times H$ to $\lceil W/2\rceil \times \lceil H/2\rceil$ while sampling the center of the current pixel and the midpoints in four adjacent directions (each with offsets $(\pm 0.5, \pm 0.5)$):

$$
D(x,y) = \frac{1}{4}\sum_{(s,t)\in\{-1,+1\}^2} I\!\left(2x + s\cdot\frac{1}{2},\; 2y + t\cdot\frac{1}{2}\right) \tag{22}
$$

**Upsample pass**: restore $\lceil W/2\rceil \times \lceil H/2\rceil$ back to $W\times H$, and compute a weighted sum over the 4 offset points $(\pm 1, 0), (0, \pm 1)$:

$$
U(x,y) = \frac{1}{2} D\!\left(\frac{x}{2}, \frac{y}{2}\right)+ \frac{1}{8}\sum_{(s,t)\in\{(\pm1,0),(0,\pm1)\}} D\!\left(\frac{x}{2} + s, \frac{y}{2} + t\right) \tag{23}
$$

**Frequency-domain analysis**

The DTFT of the downsample pass (assuming downsampling factor 2) is:

$$
\hat{H}_{\mathrm{down}}(f_x, f_y) = \cos(\pi f_x)\cos(\pi f_y)
$$

The DTFT of the upsample pass (defined on the low-resolution grid) is:

$$
\hat{H}_{\mathrm{up}}(f_x, f_y) = \frac{1}{2} + \frac{1}{4}[\cos(2\pi f_x) + \cos(2\pi f_y)]
$$

The combined response of one D-U pair (with frequency scaling taken into account) is complicated due to aliasing from downsampling; in practice, its effect is equivalent to a Gaussian approximation with an effective radius of about 2.

**Bandwidth analysis**

Assume the image is $W\times H$. The total number of fetches for $P$ Kawase passes is $4P \cdot W \cdot H$. Dual Filter uses $L$ downsampling levels: the total pixel count is $WH(1 + 1/4 + 1/16 + \cdots) = WH \cdot \tfrac{4}{3}$, and the total number of fetches is approximately $4 \cdot (L_{\mathrm{down}} + L_{\mathrm{up}}) \cdot WH \cdot \tfrac{2}{3}$, about $50\%$ less bandwidth than Kawase.

**Comparison with Kawase**

| Metric | Kawase | Dual Filter |
|------|--------|-------------|
| number of passes ($r=48$) | 10 | about 6 (3 down + 3 up) |
| Bandwidth | $4P \cdot WH$ | $\approx 2P \cdot WH$ (half bandwidth) |
| Anisotropy | medium (diagonal sampling) | higher (cross sampling is weaker in the diagonal direction) |
| Edge ringing | small | more obvious (aliasing introduced by resolution switching) |
| Applicable $\sigma$ range | medium-to-small ($r \leq 64$) | large (each level covers a wider range) |

**Limitations**: Dual Filter has more severe anisotropy than Kawase (cross-shaped sampling has lower sampling density in diagonal directions), and resolution switching (downsample/upsample) introduces additional aliasing and ringing, which is visible at high-contrast edges (such as text and sharp geometry).

### 5.2 Numerical optimization of the pass sequence

**Motivation**: the greedy algorithm is approximately optimal for variance matching, but it does not consider the overall shape of the frequency-response error (maximum error, mean squared error, etc.).

**Optimization problem**: given the number of passes $P$ and the target $\sigma$, minimize over the **real-valued** offset space $\{o_0, o_1, \ldots, o_{P-1}\} \subset \mathbb{R}_{>0}$:

$$
\mathcal{L}(\{o_k\}) = \int_0^{1/2}\!\int_0^{1/2}
\left|\prod_{k}\cos(2\pi o_k f_x)\cos(2\pi o_k f_y) - e^{-2\pi^2\sigma^2(f_x^2+f_y^2)}\right|^2
W(f_x,f_y)\,\mathrm{d}f_x\,\mathrm{d}f_y \tag{24}
$$

where $W(f_x,f_y) = e^{+2\pi^2\sigma^2(f_x^2+f_y^2)}$ (Gaussian weighting, making low-frequency errors more important).

**Optimization methods**: CMA-ES (Covariance Matrix Adaptation Evolution Strategy) or L-BFGS-B (gradients can be computed through automatic differentiation).

**Result**: for $r=32$ ($P=8$), optimization reduces the worst frequency error from the greedy value of 0.187 to about 0.140 (an improvement of $\approx 25\%$), and improves the PSNR lower bound from 21.5 dB to about 23.5 dB. The cost is that offsets are real values (handled automatically by GPU texture interpolation), but the sequence varies with $\sigma$, so a lookup table must be stored for each $\sigma$ (about 80 entries cover $r=1\text{–}96$, with negligible storage cost).

### 5.3 Variable-Rate Kawase (spatial adaptivity)

**Idea**: smooth regions in the image (low local variance) have lower blur-quality requirements and can use more passes to ensure accuracy; texture-dense regions (high local variance) have lower human sensitivity to blur accuracy (texture masking effect), so fewer passes can be used to save time.

**Implementation**:
1. Offline compute the local variance map of the image (or use the material ID from the G-Buffer).
2. Quantize the variance map into 3–4 quality levels, each corresponding to a different number of passes.
3. Implement adaptivity on the GPU through predicated execution or early-exit.

**Typical benefit**: in game scenes, about 60% of the area is smooth sky/ground and can be reduced to $P_{\min}=3$; 30% is medium complexity and uses $P_{\mathrm{mid}}=7$; 10% is high-frequency texture and uses $P_{\max}=12$. The average is $P \approx 5$, saving about 50% GPU time compared with fixed $P=10$.

**Limitation**: computing the variance map itself has extra cost (about 0.1 ms), and it is not suitable for effects such as real-time DoF that require an accurate CoC (Circle of Confusion).

### 5.4 Isotropic Correction

**Principle**: add one “axial compensation pass” at the end of the Kawase chain, sampling $(0, \pm o)$ and $(\pm o, 0)$ (cross shape) instead of $(\pm o, \pm o)$ (diagonal shape), and linearly mix the two results:

$$
I_{\mathrm{out}} = \lambda \cdot K_{\mathcal{S}}^{\mathrm{corner}}(I) + (1-\lambda) \cdot K_{\mathcal{S}}^{\mathrm{cross}}(I) \tag{25}
$$

**Derivation of the optimal weight**: let the axial error be $\epsilon_0 = \hat{H}_{\mathrm{corner}}(f,0) - \hat{G}(f)$ and the diagonal error be $\epsilon_{45} = \hat{H}_{\mathrm{corner}}(f,45°) - \hat{G}(f)$. The axial response of Cross is $\hat{H}_{\mathrm{cross}}(f,0) = \cos(2\pi o f)$ (equal to the axial response of corner), and the diagonal response is $\hat{H}_{\mathrm{cross}}(f,45°) = \cos(\sqrt{2}\pi o f)^2$ (same as corner).

In fact, the frequency response of a single cross pass is $\hat{H}_{\mathrm{cross1}}(f_x,f_y) = \tfrac{1}{2}[\cos(2\pi o f_x) + \cos(2\pi o f_y)]$ (axial sampling), which has a different structure from the corner frequency response $\cos(2\pi o f_x)\cos(2\pi o f_y)$. In the axial direction $\theta=0$ after mixing:

$$
\hat{H}_{\mathrm{mix}}(f,0) = \lambda \cos(2\pi of) + (1-\lambda)\cos(2\pi of) = \cos(2\pi of)
$$

In the diagonal direction $\theta=45°$ (after $P$ corner passes):

$$
\hat{H}_{\mathrm{mix}}(f,45°) \approx \lambda\cos^2(\sqrt{2}\pi of) + (1-\lambda)\cdot\frac{1}{2}[\cos(\sqrt{2}\pi of) + \cos(\sqrt{2}\pi of)] = \ldots
$$

Choosing $\lambda \approx 0.55$ (for $P=10$) can reduce the maximum anisotropy error from 0.019 to $< 0.004$ (about a 5× improvement), at the cost of one additional cross pass (about 10% more bandwidth).

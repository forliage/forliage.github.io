---
title: "Case 6 README"
---

![dissolve](images/bunny_dissolve.gif)

The source cuda code can be found [here](https://github.com/forliage/forliage.github.io/blob/main/src/pages/showcase/_data/codes/bunny_dissolve.cu),  which you should noted is that the stb_image.h, stb_image_write.h and the stanford-bunny.obj must be placed at the same file level.

## 1. Project Objectives

This project implements a particle-based dissolve animation from `stanford-bunny.obj`. The core visual sequence is:

1. Build a static bunny silhouette  
2. Trigger fracture from a single ear-tip seed (propagation-based)  
3. Let particles disperse under normal-direction blast + Curl Noise field  
4. Finish with either reconstruction (`regroup`) or disappearance (`vanish`)  
5. Run a two-orbit camera path with near/far shot transitions  
6. Export a PNG sequence and encode it into a high-quality GIF

## 2. Overall Technical Architecture

The system uses an offline rendering pipeline:

1. CPU side: OBJ parsing, triangle construction, area-weighted particle sampling  
2. GPU side: per-frame particle dynamics update (CUDA kernel)  
3. GPU side: particle projection + multi-segment trail splat accumulation (CUDA kernel)  
4. GPU side: background composition, bloom, vignette, gamma (CUDA kernel)  
5. CPU side: write per-frame PNG files  
6. FFmpeg: generate GIF with `palettegen + paletteuse`

Key idea: instead of triangle rasterization, reconstruct the mesh surface as a dense particle cloud, then apply stage-based dynamics and stylized shading.

## 3. Geometry Data and Particle Initialization

### 3.1 OBJ Parsing and Triangulation

- Parse `v` and `f` lines, including first-field index parsing for `f v/vt/vn` format.
- Triangulate polygonal faces via fan triangulation.
- Skip degenerate triangles (very small normal length).

### 3.2 Normalization

- Compute the bounding-box center and translate geometry near the origin.
- Uniformly scale by the maximum extent to stabilize framing and motion scale.

### 3.3 Area-Uniform Surface Sampling

- Compute triangle areas and build a CDF.
- For each particle:
  - Pick a triangle with area-weighted probability
  - Use barycentric sampling
    $$
    b_0 = 1-\sqrt{u},\quad
    b_1 = \sqrt{u}(1-v),\quad
    b_2 = \sqrt{u}v
    $$
  - Position: $\mathbf{p}=b_0\mathbf{a}+b_1\mathbf{b}+b_2\mathbf{c}$
  - Store surface normal, random phase `phase`, and random seed `seed`

This yields a particle distribution that visually behaves like a true mesh surface instead of random volumetric sampling.

## 4. Procedural Noise and Curl Field

### 4.1 Hash-Noise Foundation

- Use `pcg_hash` + integer lattice hashing to construct parallel-evaluable pseudo-random value noise.
- `value_noise_3d` produces continuous noise via trilinear interpolation and Hermite smoothing.

### 4.2 Curl Noise

- Build a 3D vector potential field $\mathbf{F}(x)$, then compute curl:
  $$
  \mathbf{v}_{curl} = \nabla \times \mathbf{F}
  $$
- Estimate derivatives with finite differences for stable and parallel-friendly evaluation.

Meaning: the curl field is approximately divergence-free, so particle motion looks like coherent advection/vortical flow instead of unstructured jitter.

## 5. Fracture Dynamics

Particles are updated each frame with explicit integration:
$$
\mathbf{v}_{t+\Delta t}=d\cdot\mathbf{v}_t + \mathbf{F}\Delta t,\quad
\mathbf{p}_{t+\Delta t}=\mathbf{p}_t+\mathbf{v}_{t+\Delta t}\Delta t
$$

where `d` is drag and `F` is total force.

### 5.1 Ear-Tip Single-Seed Propagation Mask

Given the ear-tip source point $\mathbf{s}_{ear}$:
$$
r = \|\mathbf{p}_0-\mathbf{s}_{ear}\|
$$
$$
delay = 0.34\cdot smoothstep(0,1.85,r) + 0.07\cdot phase
$$
$$
local\_progress = clamp((progress-delay)\cdot 1.60, 0,1)
$$

Interpretation: particles farther from the ear tip fracture later, producing a clear propagating wavefront.

### 5.2 Blast Pulse and Wavefront Enhancement

- A two-peak `pulse` controls primary and secondary bursts.
- `source_gain` strengthens the near-ear region.
- `wave_front` boosts the currently advancing fracture front.

Approximate blast term:
$$
F_{blast}\propto normal \cdot blast(local\_progress,r)\cdot pulse
$$

### 5.3 Force Composition

The total force is the sum of:

1. Normal-direction blast force (along surface normals)  
2. Shock radial force (outward from ear tip)  
3. Curl Noise flow force (smoke/fluid feel)  
4. Global outward expansion force (detaching from body)  
5. Tangential swirl force (adds rolling structure)  
6. End-state force:
   - `regroup`: pull back to initial surface position  
   - `vanish`: gravity-driven sinking + alpha fade

### 5.4 Stage-Based Opacity and Regroup Stability

- `regroup`: raise opacity after dispersion; final `snap` interpolation enforces stable silhouette closure.
- `vanish`: delay fade-out to preserve longer trail readability before disappearance.


## 6. Camera System (Two Orbits + Near/Far Shots)

The camera is a parameterized orbital camera:

- Total orbit angle: `4π` (two full revolutions)
- Radius switching between far `~3.3` and near `~1.72`
- Near-shot windows driven by two Gaussian pulses in mid/late timeline
- Mild vertical oscillation + target offset coupling

Effect: combines global shape readability (far shots) with high-impact fracture details (close shots).

## 7. Particle Rendering Model (Screen Space)

### 7.1 Projection

- Use camera basis vectors `right/up/forward` for view-space projection.
- Particle depth controls screen radius and base opacity.

### 7.2 Velocity Trails (Key Visual Enhancer)

- Project world-space velocity into screen space to get motion direction.
- Compute dynamic `tail_len` from speed magnitude.
- Use up to 4 `trail_taps` for reverse-direction multi-segment splats.
- Control each segment’s radius, alpha, and color independently (toward violet-pink at the tail).

This is the core implementation behind “longer pink-purple trails”, without relying on post-process motion blur.

### 7.3 Accumulation Strategy

- Accumulate all particles into an `accum` buffer with `atomicAdd` (RGB + alpha).
- Normalize color by density during composition.

## 8. Composition and Stylization

Implemented in `compose_frame`:

1. Deep-to-light purple gradient background  
2. Dual-layer halo (central + offset halo)  
3. Foreground-over-background blending (density-based)  
4. Light bloom enhancement  
5. Vignette for subject focus  
6. Gamma correction (approximate sRGB)

Overall color strategy:

- Static pink-white `c_static`
- Mid-stage magenta `c_mid`
- High-energy deep purple `c_deep`
- Trail accent color `c_tail`

## 9. CUDA Parallelization and Memory Layout

### 9.1 Kernel Breakdown

- `init_particles`
- `step_particles`
- `clear_accum`
- `raster_particles`
- `compose_frame`

### 9.2 Data Layout

- Particle state: `float4` arrays (initial, position, velocity)
- Accumulation buffer: `float4` (RGB sum + alpha sum)
- Output frame: `uchar3` (stored as linear byte array)

### 9.3 Complexity

- Simulation: $O(N)$
- Rasterization: approximately $O(N \cdot r^2 \cdot taps)$, affected by particle radius and trail segments
- Main bottleneck: `atomicAdd` contention in high-density regions

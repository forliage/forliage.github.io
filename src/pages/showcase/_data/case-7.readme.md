---
title: "Case 7 README"
---

![dissolve](images/bunny_flow_field.gif)

The source cuda code can be found [here](https://github.com/forliage/forliage.github.io/blob/main/src/pages/showcase/_data/codes/bunny_flow_field.cu),  which you should noted is that the **stb_image.h**, **stb_image_write.h** and the **stanford-bunny.obj** must be placed at the same file level.

## 1. Goal and Visual Outcome

This project implements a particle flow-field engine. The core visual progression is:

- Particles start from a random chaos cloud;
- They evolve under a continuous dynamical system `dx/dt = F(x)`;
- They gradually converge from disorder and attach to the Stanford Bunny surface;
- In the final segment, the camera performs a full orbit around the bunny to present a stable structure.

The target effect is a strong "chaos -> structure emergence" transition.

## 2. Overall Architecture

The engine pipeline has five stages:

1. Mesh loading  
Parse vertices and triangle faces from `stanford-bunny.obj`, then normalize scale and center the mesh.

2. Target distribution sampling  
Build a triangle-area CDF and uniformly sample points on the bunny surface as per-particle targets `target[i]`.

3. Chaos particle initialization  
Initialize particle positions in a random spherical cloud with Gaussian noise; initial velocities are small random values.

4. Particle dynamics evolution  
Integrate particle position/velocity each frame. Early stage keeps chaotic perturbations; late stage switches to strong attraction and settling.

5. Projection rendering and output  
Project particles to 2D, accumulate energy, apply tone mapping, export PNG frames, then encode a GIF with `ffmpeg`.

## 3. Dynamical System Design

Each particle state is `(p, v)`, with target point `t`.  
The system uses two stage weights:

- `alpha = morph_progress(t)`: structure formation progress;
- `settle = settle_progress(t)`: late-stage strong convergence progress.

The force field is a blend of two components:

- `F_chaos`: sinusoidal noise + vortex + random jitter, creating early unordered motion;
- `F_form`: attraction toward target + mild spiral guidance + damping, producing stable geometric contours.

Overall form:

`F = (1 - blend) * F_chaos + blend * F_form`

`blend` increases with `alpha/settle`.  
In the final stage, an additional `snap` term (position interpolation toward target) plus stronger damping is applied to make convergence more complete and sharper.

## 4. Convergence and Clarity Strategy

To avoid a blurry final frame, three key strategies are used:

- Dynamic trail decay  
`decay` decreases as `settle` increases, so historical trails are cleared faster in the final stage.

- Dynamic exposure  
Exposure is reduced near the end to suppress overexposed, washed-out contours.

- Outlier particle suppression  
Rendering intensity is weighted by convergence `convergence`; particles far from targets contribute less in the final stage.

## 5. Camera and Composition

The camera uses a segmented trajectory:

- First 70%: slow approach and observation to preserve continuity from chaos to structure;
- Last 30%: a full 360-degree orbit around the bunny to show the final form.

Field-of-view and orbit radius are tuned to avoid extreme close-up only in the last stage and improve overall readability.

## 6. Color System (Adjustable)

The engine supports three color parameter groups (`0..1`):

- `--chaos-color r,g,b`: main color in the chaos stage;
- `--form-color r,g,b`: main color in the converged stage;
- `--bg-color r,g,b`: background base color.

During rendering, color blending follows `alpha + convergence + settle`, creating a smooth transition from chaos color to structure color.

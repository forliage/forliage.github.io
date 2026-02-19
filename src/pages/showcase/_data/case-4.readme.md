---
title: "Case 4 README"
---

**Disclaimer**: This method for "melting ice cream and chocolate bunnies" is *not* my creative idea. I was inspired by the article [***Melting and Flowing***](https://faculty.cc.gatech.edu/~turk/my_papers/melt.pdf) by Mark Carlson, Peter J. Mucha, R. Brooks Van Horn III, Greg Turk, and others. And all the contents below comes from my understanding of this paper. The ice cream bunny is also shown in the original article, while the chocolate bunny is my own creation.

**Melting Ice cream Bunny**:

|FRONT|![front](images/melt_front.gif)|![side](images/melt_side.gif)|SIDE|
|---|---|---|---|
|**TOP**|![top](images/melt_top.gif)|![back](images/melt_back.gif)|**BACK**|

**Melting chocolate Bunny**:

|FRONT|![front](images/chocolate_front.gif)|![side](images/chocolate_side.gif)|SIDE|
|---|---|---|---|
|**TOP**|![top](images/chocolate_top.gif)|![back](images/chocolate_back.gif)|**BACK**|

The source cuda code can be found here: [melting ice cream bunny](https://github.com/forliage/forliage.github.io/blob/main/src/pages/showcase/_data/codes/melt_icecream_bunny.cu) and [melting chocolate bunny](https://github.com/forliage/forliage.github.io/blob/main/src/pages/showcase/_data/codes/melt_chocolate_bunny.cu),  which you should noted is that the stb_image.h, stb_image_write.h and the stanford-bunny.obj must be placed at the same file level.

## 1. Overview
 
The goal is to use one unified fluid framework to cover a continuous transition:
“near-solid -> semi-fluid -> low-viscosity liquid”.

Core strategy:

1. Use incompressible viscous flow equations for motion.  
2. Use a temperature field to drive viscosity changes.  
3. Use implicit diffusion for high-viscosity stability.  
4. Use particles to mark free surfaces and reconstruct renderable geometry.

## 2. Governing Equations

### 2.1 Incompressibility Constraint

$$
\nabla\cdot\mathbf{u}=0
$$

### 2.2 Momentum Equation (Variable Viscosity)

$$
\frac{\partial \mathbf{u}}{\partial t}
=-(\mathbf{u}\cdot\nabla)\mathbf{u}
+\nabla(\nu\nabla\mathbf{u})
-\frac{1}{\rho}\nabla p
+\mathbf{f}
$$

Where:

- $\mathbf{u}$: velocity field
- $\nu$: kinematic viscosity (space/time varying)
- $p$: pressure
- $\mathbf{f}$: body force (usually gravity)

### 2.3 Temperature Transport Equation

$$
\frac{\partial T}{\partial t}=k\nabla^2T-(\mathbf{u}\cdot\nabla)T+Q
$$

- $T$: temperature
- $k$: thermal diffusivity
- $Q$: heat source/cooling term

## 3. State Representation (Eulerian + Lagrangian)

The method uses a hybrid representation:

1. Eulerian grid: stores velocity, pressure, temperature, viscosity, fluid flags.  
2. Lagrangian particles: track the free surface, avoiding coarse-grid-only artifacts.

At the end of each step, particles write back which cells are fluid.

## 4. Time Integration (Operator Splitting)

Each time step is split into:

1. Advection + body force (explicit)  
2. Viscous diffusion (implicit)  
3. Pressure projection (enforce incompressibility)  
4. Particle advection + fluid marker rebuild  
5. Temperature advection + temperature diffusion + heating/cooling  
6. Viscosity update from temperature

Equation form:

### 4.1 Advection + Body Force

$$
\mathbf{u}^*=\mathbf{u}^n+\Delta t\left[-(\mathbf{u}\cdot\nabla)\mathbf{u}+\mathbf{f}\right]
$$

### 4.2 Implicit Viscous Diffusion

$$
\mathbf{u}^{**}=\mathbf{u}^*+\Delta t\,\nabla(\nu\nabla\mathbf{u}^{**})
$$

Discrete linear system:

$$
(I-\Delta t\,L_\nu)\mathbf{u}^{**}=\mathbf{u}^*
$$

### 4.3 Pressure Projection

Solve Poisson first:

$$
\nabla^2 p=\frac{\rho}{\Delta t}\nabla\cdot\mathbf{u}^{**}
$$

Then correct velocity:

$$
\mathbf{u}^{n+1}=\mathbf{u}^{**}-\frac{\Delta t}{\rho}\nabla p
$$

## 5. Variable-Viscosity Discretization and Face Viscosity

With variable viscosity, diffusion is no longer a constant-coefficient Laplacian.  
Viscosity should be defined at cell faces, e.g.:

$$
\nu_{i+1/2,j,k}
$$

A practical face viscosity is the geometric mean:

$$
\nu_{face}=\sqrt{\nu_L\nu_R}
$$

Compared to arithmetic averaging, this often reduces over-damping when melt drips flow along high-viscosity regions.

## 6. High-Viscosity Stability and Implicit Solving

Explicit diffusion has a strict stability condition:

$$
\nu\Delta t/h^2<\mathcal{O}(1)
$$

For high viscosity, $\Delta t$ becomes prohibitively small.  
So diffusion should be implicit, solved iteratively (CG/Jacobi/PCG, etc.).

Desirable system properties:

1. Symmetric  
2. Positive definite  
3. Sparse

These properties improve solver robustness and efficiency.

## 7. Free-Flight Momentum Reintroduction

At high viscosity, implicit diffusion can artificially kill velocity of detached flying blobs.  
A practical correction:

1. Identify connected fluid components not touching solid boundaries.  
2. Compute pre-/post-diffusion bulk velocity:

$$
\mathbf{v}_{bulk}^{pre},\quad \mathbf{v}_{bulk}^{post}
$$

3. Add the difference back to all cells in that component:

$$
\Delta\mathbf{v}=\mathbf{v}_{bulk}^{pre}-\mathbf{v}_{bulk}^{post}
$$

$$
\mathbf{u}\leftarrow\mathbf{u}+\Delta\mathbf{v}
$$

This mostly restores **translational momentum**, which is usually enough to prevent the “frozen-in-air” artifact.

## 8. Temperature Solve and Heat Source Model

### 8.1 Temperature Advection

$$
T^*=T^n-\Delta t(\mathbf{u}\cdot\nabla T)
$$

### 8.2 Implicit Temperature Diffusion

$$
(I-\Delta t\,k\nabla^2)T^{n+1}=T^*+\Delta t\,Q
$$

### 8.3 Practical Heating/Cooling Terms

Common engineering choices:

1. Ambient cooling: $Q_{cool}\propto (T_{env}-T)$  
2. Local heater: distance-weighted heat injection

$$
Q_{heater}\propto w(r)\,(T_{heater}-T),\quad w(r)=\max(0,1-r/R)
$$

## 9. Temperature-to-Viscosity Constitutive Mapping

Melting requires viscosity to span multiple orders of magnitude.  
A practical mapping:

1. Normalize temperature into a melt transition band: $s\in[0,1]$  
2. Use a smooth function (e.g., smoothstep) to avoid hard jumps  
3. Interpolate viscosity in log-space

$$
s=\text{smoothstep}(T_{m}-\Delta T,\;T_{m}+\Delta T,\;T)
$$

$$
\nu(T)=\exp\left((1-s)\ln\nu_{cold}+s\ln\nu_{hot}\right)
$$

This gives “near-solid at low temperature, rapid liquefaction near melt”.

## 10. Free-Surface Reconstruction (Particle Splatting)

Renderable surfaces are usually not extracted from the coarse simulation grid directly:

1. Build a higher-resolution volume $V$  
2. Splat surface particles into volume (tent kernel)

$$
V(\mathbf{x}) \mathrel{+}= \prod_{a\in\{x,y,z\}}\max\left(0,1-\frac{|a-a_p|}{r}\right)
$$

3. Clamp + low-pass smoothing  
4. Iso-surface extraction (or direct volume ray marching)

A typical practical kernel width is around 2.5 voxels.

## 11. Minimal End-to-End Pipeline

1. Initialize: OBJ voxelization + particle seeding.  
2. Main loop: coupled velocity/pressure/temperature/viscosity updates.  
3. Advect particles and rebuild fluid markers.  
4. Splat particles into high-res volume.  
5. Render and export frame sequence.  

## 12. Parameter Tuning Guide (Bunny Melting)

Tune three groups first:

1. Viscosity range: $\nu_{cold},\nu_{hot}$  
2. Melt center and transition width: $T_m,\Delta T$  
3. Heat input strength/range: $T_{heater},R,\text{gain}$

Practical effects:

1. More “solid feel”: increase $\nu_{cold}$, decrease $\Delta T$.  
2. More “runny melt”: decrease $\nu_{hot}$, increase heater radius or gain.  
3. Less numerical jitter: increase implicit iterations or reduce substep size.  

## 13. Validation Criteria

A good melting simulation should satisfy:

1. Local melting starts first (not globally uniform softening).  
2. High-viscosity and low-viscosity regions coexist in the same frame.  
3. Detached droplets keep inertia and do not instantly stall from diffusion.

If these three are met, the melting-method chain is usually correct.


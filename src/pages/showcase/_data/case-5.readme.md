---
title: "Case 5 README"
---

**Disclaimer**: This method for "The bunny and the spheres fell into the water" is *not* my creative idea. I was inspired by the article [***Solving General Shallow Wave Equations on Surfaces***](https://faculty.cc.gatech.edu/~turk/paper_pages/2007_shallow_waves/shallow_water.pdf) by Huamin Wang,  Gavin Miller and Greg Turk. And all the contents below comes from my understanding of this paper. The sphere is also shown in the original article, while I use the method for stanford bunny.

|Bunny|Sphere|Both|
|---|---|---|
|![bunny](images/bunny_top.gif)|![sphere](images/spheres_top.gif)|![both](images/both_top.gif)|

The source cuda code can be found [here](https://github.com/forliage/forliage.github.io/blob/main/src/pages/showcase/_data/codes/shallow_spheres_bunny.cu),  which you should noted is that the stb_image.h, stb_image_write.h and the stanford-bunny.obj must be placed at the same file level.

## 1. Top-View Shallow Water Method Overview

Primary target:

1. Reproduce SPHERES-style coupling behavior (waves/splashes from floating objects).  
2. Keep **identical mass but different volumes** for spheres.  
3. Replace sphere geometry with a **Stanford Bunny** option.  
4. Render/export only **top-view** frames.

## 2. Connection to the Original GSWE Formulation

The paper writes (in continuous form):

$$
\mathbf{u}_t = -(\mathbf{u}\cdot\nabla)\mathbf{u} - \frac{1}{\rho}\nabla P_{ext} + \mathbf{a}_{ext}
$$

$$
h_t + \nabla\cdot\big((h-b)\mathbf{u}\big)=0
$$

and derives a second-order height update with implicit force terms, finally solved in matrix form:

$$
(A_g + A_s + I - I_c)\,h^t = b - b_c
$$

In this implementation:

1. Surface is planar regular grid (not curved mesh GSWE).  
2. Surface tension is disabled ($\gamma=0$), matching SPHERES table style.  
3. Coupling with rigid bodies is kept through per-cell constraints (an $I_c,b_c$-like term).  


## 3. State Variables (Planar Grid + Rigid Bodies)

### 3.1 Fluid grid

On a 2D grid $(i,j)$:

- $h_{i,j}$: water height
- $u_{i,j}, v_{i,j}$: horizontal velocity

### 3.2 Rigid bodies

Each body stores:

- position $(x,y,z)$
- velocity $(v_x,v_y,v_z)$
- mass $m$
- equivalent radius $R_{eq}$
- shape type: sphere or bunny

## 4. Time Integration

Each simulation step uses operator splitting:

1. Update rigid bodies and build splash source.
2. Build coupling maps from body-water contacts.
3. Advect height (semi-Lagrangian).
4. Build right-hand side with artificial viscosity term.
5. Solve implicit height system by Jacobi iterations.
6. Update velocity from height gradient + friction damping.
7. Apply boundary damping.

## 5. Height Advection and RHS

### 5.1 Semi-Lagrangian advection

For cell center $\mathbf{x}_{ij}$:

$$
h_{adv}(\mathbf{x}_{ij}) = h^n\!\left(\mathbf{x}_{ij} - \Delta t\,\mathbf{u}^n(\mathbf{x}_{ij})\right)
$$

### 5.2 Prospective RHS (paper-style $\tau$ term + external source)

Code form:

$$
b_{ij}=h_{adv,ij} + (1-\tau)(h_{adv,ij} - h^{n-1}_{ij}) + \Delta t\,S_{ij}
$$

where $S$ is splash/body forcing written into the height equation.

In current settings: $\tau=0$.

## 6. Implicit Height Solve with Coupling

The solver uses a local nonlinear Jacobi update:

$$
\sigma_{ij} = \frac{\Delta t^2 g}{\Delta x^2}\max(h_{ij}, h_{min})
$$

$$
h_{ij}^{k+1} =
\frac{
b_{ij} + c^w_{ij}c^t_{ij} + \sigma_{ij}(h_{L}+h_{R}+h_{D}+h_{U})
}{
1 + 4\sigma_{ij} + c^w_{ij}
}
$$

- $c^w$: coupling weight map (constraint strength)
- $c^t$: coupling target (object bottom height proxy)

Interpretation:

1. $1 + 4\sigma$ is the implicit gravity stiffness.  
2. $c^w,c^t$ is the rigid/fluid coupling term (similar role to $I_c,b_c$).  

## 7. Velocity Update and Friction

After solving $h^{n+1}$:

$$
u \leftarrow u - \Delta t\,g\,\partial_x h,\quad
v \leftarrow v - \Delta t\,g\,\partial_y h
$$

Then apply paper-style depth-dependent friction damping:

$$
d_{fric}(h)=\frac{h}{h+d_0},\quad
(u,v)\leftarrow d_{fric}(h)\,(u,v)
$$

with constant $d_0$ in code.

## 8. Rigid-Body Dynamics Used for SPHERES/BUNNY

### 8.1 Buoyancy proxy

A sphere-cap proxy is used for submerged volume:

$$
V_{cap}(R,h)=\pi h^2\left(R-\frac{h}{3}\right),\quad h\in[0,2R]
$$

For bunny, the cap estimate is scaled by volume ratio:

$$
V_{sub}^{bunny}
\approx
V_{cap}(R_{proxy},\cdot)\cdot\frac{V_{bunny}}{V_{sphere}(R_{proxy})}
$$

Buoyancy force:

$$
F_b=\rho g V_{sub}
$$

### 8.2 Motion update

Vertical:

$$
m\ddot z = F_b - mg - c_z\dot z
$$

Horizontal (wave-slope-driven drift):

$$
\ddot x = -k_s g\,\partial_x h - c_{xy}\dot x,\quad
\ddot y = -k_s g\,\partial_y h - c_{xy}\dot y
$$

plus floor and wall collision response with damping/restitution.

## 9. Coupling Map Construction

For each body, for each covered cell:

1. Compute an approximate object bottom elevation $z_{bot}(x,y)$.
2. Write:

$$
c^w_{ij}=\max(c^w_{ij},k_{couple}),\quad
c^t_{ij}=\min(c^t_{ij}, z_{bot})
$$

For spheres:

$$
z_{bot}=z_c-\sqrt{R^2-r^2}
$$

For bunny:

1. Transform world $(x,y)$ into bunny local coordinates.
2. Sample precomputed profile $z_{bot}^{local}(u,v)$ from OBJ-derived map.
3. Convert back to world height.

## 10. Splash Source Term

A local Gaussian source is added to height RHS:

$$
S(\mathbf{x}) \mathrel{+}= A\exp\!\left(-\frac{\|\mathbf{x}-\mathbf{x}_b\|^2}{2\sigma^2}\right)
$$

Amplitude $A$ is driven by:

1. downward impact speed
2. lateral speed
3. body size

This mimics the SPHERES visual effect (waves + splash rings) without introducing particles.

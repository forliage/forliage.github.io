---
title: "Particle Systems"
lecture: 2
course: "anim"
date: 2026-02-10
---

# I. What is a Particle System?

## 1.1 Core Idea

The core idea of ​​a particle system can be summarized in one sentence: **"Very simple procedural rules can create very deep visual effects."**

Traditional 3D modeling techniques, such as polygonal meshes or parametric surfaces, are well-suited for constructing static objects with clear boundaries (e.g., tables, cars). However, these techniques fall short when attempting to simulate natural phenomena that are fuzzy, dynamically changing, and without fixed shapes—what we call "fuzzy" phenomena. These phenomena include:

* **Natural Phenomena**: Flames, smoke, clouds, flowing water (waterfalls, waves), rain, snow, falling leaves

* **Special Effects**: Explosions, sparks, fountains, fireworks

* **Aggregates**: Grass, hair, nebulae, swarms of organisms

Particle systems were developed to solve this problem. It doesn't attempt to establish a unified, complex geometric model for the entire fuzzy object, but rather views it as a collection of **numerous, simple, independent particles**. The overall behavior and appearance of the system are programmatically defined by rules controlling the generation, movement, evolution, and demise of each particle.

## 1.2 History and Motivation

The concept of particle systems was first formally proposed by **William T. Reeves** in his landmark 1983 SIGGRAPH paper, "Particle Systems - a Technique for Modeling a Class of Fuzzy Objects".

The direct driving force behind this technology came from the needs of film special effects. In the 1982 film **Star Trek II: The Wrath of Khan**, there is a famous scene known as the "Genesis Effect," which needed to show a barren planet rapidly transforming into a vibrant world after being engulfed in flames. Drawing thousands of flying sparks using traditional animation techniques was extremely difficult and unrealistic. Reeves and his team developed a particle system for this purpose, successfully simulating the grand spectacle of walls of fire surrounding and terraforming a planet—a breakthrough considered significant in the history of computer graphics.

Since then, particle systems have become a core method for generating visual effects in film special effects, video games, and scientific visualization. Reeves himself has become an industry legend for his outstanding contributions to Pixar, including his involvement in the production of the Oscar-winning animated short film *Tin Toy*.

# II. Basic Architecture of a Particle System

A typical particle system simulation follows a fixed loop that executes every frame. We can break it down into four core phases:

1.  **Generation (Emission/Generation)**: In the current time step, create new particles based on the emitter's rules.
2.  **Update (Update/Animation)**: Update the state of all active particles. This is the core of the simulation, mainly involving physical dynamics calculations.
3.  **Culling (Culling/Reaping)**: Remove particles that have reached their lifespan or other death conditions.
4.  **Rendering (Rendering)**: Draw all active particles in some way to the screen.

## 2.1 Basic Assumptions

To achieve efficient computation, classic particle systems typically rely on several simplifying assumptions:

*   **Particles do not interact with each other**: Particles usually do not collide with or apply forces to other particles (advanced systems like SPH break this assumption). This significantly reduces computational complexity, from $O(n^2)$ to $O(n)$.
*   **No self-shadowing**: Particles usually do not cast shadows on other particles.
*   **One-way environment interaction**: Particles can be affected by the environment (e.g., collisions with the ground), but usually do not change the environment.
*   **No reflection**: Particles are usually considered self-emitting or only affected by simple lighting, without complex global illumination calculations.
*   **Finite lifespan**: Each particle has a lifespan, which decreases over time and the particle dies when it reaches zero.

These assumptions allow us to simulate thousands or even millions of particles in real-time.

# III. Math and Physics Models

The core of a particle system lies in the "update" stage, which involves calculating the motion of each particle based on physical laws.

## 3.1 Particle Attributes

Each particle essentially represents a point mass, and its state is described by a set of attributes. These attributes form the basis of the particle system state vector.

*   **Position (Position)**: $\vec{x}(t)$, a three-dimensional vector.
*   **Velocity (Velocity)**: $\vec{v}(t) = \frac{d\vec{x}}{dt}$, a three-dimensional vector.
*   **Acceleration (Acceleration)**: $\vec{a}(t) = \frac{d\vec{v}}{dt}$, a three-dimensional vector.
*   **Mass (Mass)**: $m$, a scalar.
*   **Lifespan (Lifespan)**: $l$, a scalar, representing how long the particle can live.
*   **Rendering Attributes**: Color(Color), Alpha(Alpha), Size(Size), etc.
*   **Force Accumulator (Force Accumulator)**: $\vec{F}_{total}$, used to accumulate all forces acting on the particle.

## 3.2 Motion Equations

Particles follow Newton's second law: 
$$ 
\vec{F}_{total} = m \vec{a}(t) 
$$ 
From this, we can obtain the acceleration:
$$ 
\vec{a}(t) = \frac{\vec{F}_{total}}{m} 
$$ 
This is a second-order ordinary differential equation (ODE):
$$ 
\frac{d^2\vec{x}}{dt^2} = \frac{\vec{F}_{total}(\vec{x}, \vec{v}, t)}{m} 
$$ 
To facilitate numerical integration, we usually convert it into a system of two first-order ODEs. We define a system's **state vector** $S(t)$: 
$$ 
S(t) = \begin{pmatrix} \vec{x}(t) \\ \vec{v}(t) \end{pmatrix} 
$$ 
its derivative is: 
$$ 
\frac{dS(t)}{dt} = \begin{pmatrix} \frac{d\vec{x}}{dt} \\ \frac{d\vec{v}}{dt} \end{pmatrix} = \begin{pmatrix} \vec{v}(t) \\ \vec{a}(t) \end{pmatrix} = \begin{pmatrix} \vec{v}(t) \\ \frac{\vec{F}_{total}(\vec{x}, \vec{v}, t)}{m} \end{pmatrix} 
$$ 
This form $dS/dt = f(S, t)$ is the standard form we need to solve numerically.

## 3.3 Force Modeling

The total force $\vec{F}_{total}$ is the vector sum of all individual forces. Common force models include:

1. **Unary Forces**: Forces that relate only to the state of a single particle.

   * **Gravity**: $\vec{F}_{g} = m\vec{g}$, where $\vec{g}$ is the gravitational acceleration constant (e.g., `(0, -9.8, 0)`).

   * **Viscous Drag**: $\vec{F}_{d} = -k_{d} \vec{v}$, where $k_{d}$ is the drag coefficient. This force simulates the resistance a particle experiences in a medium such as air or liquid, and its direction is opposite to the velocity direction.

2. **N-ary Forces**: Forces that relate to the interactions of multiple particles.

   * **Spring-Damper:** This is the most common model connecting two particles $a$ and $b$, used to simulate cloth, soft bodies, etc.

      * Let the positions of particles $a$ and $b$ be $\vec{x}_{a}, \vec{x}_{b}$, and their velocities be $\vec{v}_{a}, \vec{v}_{b}$.

      * The vector $\vec{d} = \vec{x}_{a} - \vec{x}_{b}$, the current length $l = ||\vec{d}||$, and the direction unit vector $\hat{d} = \vec{d}/l$.

      * The resting length of the spring is $l_0$.

   * **Hooke's Law**: $\vec{F}_{spring} = -k_s (l - l_0) \hat{d}$, where $k_s$ is the spring constant.

   * **Damping Force**: $\vec{F}_{damper} = -k_d (\vec{v}_{a} - \vec{v}_{b}) \cdot \hat{d}$, where $k_d$ is the damping coefficient. This force dissipates energy, stabilizing the system.

      * The total force acting on particle $a$ is $\vec{F}_{a} = \vec{F}_{spring} + \vec{F}_{damper}$. The force acting on particle $b$ is $-\vec{F}_{a}$.

3. **Environmental Forces**: Forces exerted by the external environment.

   * **Wind Fields**: Can be a constant vector or a complex vector field function $\vec{F}_{wind} = f(\vec{x}, t)$.

   * **Vortex**: Simulates a rotating force field.

## 3.4 Numerical Integration

We have established the equation of motion `dS/dt = f(S, t)`, but except for a very few cases, this equation cannot be solved analytically. Therefore, we must use numerical methods to approximate the solution on discrete time steps $h$ (or $\Delta t$).

### 3.4.1 Euler's Method

This is the simplest numerical integration method, based on the first-order Taylor expansion of the function: 
$$ 
S(t+h) \approx S(t) + h \frac{dS(t)}{dt} 
$$ 
Substituting our state vector, we obtain the update rules: 
$$
 \vec{v}(t+h) = \vec{v}(t) + h \cdot \vec{a}(t) 
$$
$$ 
\vec{x}(t+h) = \vec{x}(t) + h \cdot \vec{v}(t) 
$$ 
This method is called the **explicit Euler method**. It is very simple, but has a fatal flaw: **numerical instability**. It systematically overestimates velocity, causing energy to continuously increase during the simulation, eventually leading to the system "exploding". In practice, it is rarely used alone unless extremely small time steps are used.

### 3.4.2 Verlet Integration

To address the instability problem of the Euler method, the Verlet integration and its variants are widely used in the field of physical simulation.

**Derivation**: We perform a forward and backward third-order Taylor expansion on position $\vec{x}(t)$: 
$$ \vec{x}(t+h) = \vec{x}(t) + h\vec{v}(t) + \frac{h^2}{2}\vec{a}(t) + \frac{h^3}{6}\vec{b}(t) + O(h^4) 
$$
$$ \vec{x}(t-h) = \vec{x}(t) - h\vec{v}(t) + \frac{h^2}{2}\vec{a}(t) - \frac{h^3}{6}\vec{b}(t) + O(h^4) 
$$ 
(where $\vec{b}(t)$ is the jerk). Adding the two equations, the odd-order terms are eliminated: 
$$ 
\vec{x}(t+h) + \vec{x}(t-h) = 2\vec{x}(t) + h^2\vec{a}(t) + O(h^4) 
$$ 
After simplification, we obtain the updated formula for the **basic Welley integral**: 
$$ 
\vec{x}(t+h) = 2\vec{x}(t) - \vec{x}(t-h) + h^2\vec{a}(t) 
$$ 
**Advantages**:

* **Good numerical stability**: It is a symplectic integrator, which better preserves the system's energy and is less prone to divergence even at large time steps.

* **Time reversibility**.

**Disadvantages**:

* The velocity $\vec{v}(t)$ is not directly calculated. While it's possible to approximate $\vec{v}(t) = (\vec{x}(t+h) - \vec{x}(t-h)) / (2h)$, this presents inconveniences in terms of accuracy and starting the simulation (requiring two prior positions).

### 3.4.3 Velocity Verlet

This is one of the most commonly used and robust integration methods in practice. It updates position and velocity simultaneously and synchronously while maintaining the excellent stability of the Verlet integral.

The update process consists of two steps:

1. First, update the position and half-step velocity: 
$$ 
\vec{x}(t+h) = \vec{x}(t) + h\vec{v}(t) + \frac{h^2}{2}\vec{a}(t) 
$$

2. Then, calculate the new acceleration $\vec{a}(t+h) = \vec{F}(\vec{x}(t+h)) / m$ based on the new position $\vec{x}(t+h)$.

3. Finally, update the final velocity using the average of the old and new accelerations: 
$$ 
\vec{v}(t+h) = \vec{v}(t) + \frac{h}{2}(\vec{a}(t) + \vec{a}(t+h)) 
$$

**The Velocimetry method** offers a perfect balance of stability and ease of use and is highly recommended for use in particle systems. 
## 3.5 Collision Detection and Response

The interaction between particles and the environment (such as the ground and walls) is key to the realism of special effects.

1. **Collision Detection**: The simplest case is the collision of a particle with an infinitely large plane.

* The plane is defined by the normal vector $\vec{N}$ and a point $\vec{P_0}$ on the plane.

* For any particle position $\vec{P}$, its signed distance to the plane is $d = (\vec{P} - \vec{P_0}) \cdot \vec{N}$.

* When $d < 0$, the particle has crossed the plane, resulting in a collision.

2. **Collision Response**: After a collision is detected, the particle's position and velocity need to be corrected.

* **Position Correction:** Reflects the particle back onto the plane: $\vec{P}_{new} = \vec{P} - d \cdot \vec{N}$.

* **Velocity Correction:** Decomposes the velocity into a **normal component** $\vec{v}_n$ perpendicular to the normal and a **tangential component** $\vec{v}_t$ parallel to the normal. 
$$
 \vec{v}_n = (\vec{v} \cdot \vec{N})\vec{N} 
$$
$$ 
\vec{v}_t = \vec{v} - \vec{v}_n 
$$ 
Collisions primarily affect the normal velocity. We invert the normal velocity and multiply it by a **coefficient of restitution** $\varepsilon$ ($0 <= \varepsilon <= 1$, where 1 represents a perfectly elastic collision and 0 represents a perfectly inelastic collision). The tangential velocity can remain constant or be multiplied by a coefficient of friction. 
$$ 
\vec{v}'_n = -\varepsilon \vec{v}_n 
$$ 
$$ 
\vec{v}'_t = (1 - \mu) \vec{v}_t \quad (\text{where } \mu \text{ is the coefficient of friction}) 
$$ 
The new velocity is: 
$$ 
\vec{v}_{new} = \vec{v}'_n + \vec{v}'_t 
$$

# IV. Rendering

The final step is to convert the calculated particle data into a visual image. The rendering method directly determines the final appearance of the particle system.

1. **Point Sprites**

   * **Method**: Render each particle as a pixel or a small glowing point.

   * **Implementation**: In modern graphics APIs (such as OpenGL, Vulkan), the Point Sprites feature can be used to directly render a point as a 2D texture map.

   * **Effect**: Suitable for simulating sparks, stars, magical dust, etc.

   * **Blending Mode**: Typically uses **Additive Blending**, `FinalColor = SrcColor + DstColor`. This simulates the overlapping effect of glowing objects; overlapping areas of particles become brighter.

2. **Textured Billboards**

   * **Method**: Render each particle as a rectangular quad that always faces the camera and apply a texture with an alpha channel.

   * **Implementation:** In the vertex shader, rotation is eliminated through the inverse transformation of the model-view matrix, ensuring that the facets always face the viewpoint.

   * **Effect:** This is the most commonly used technique for simulating smoke, flames, explosions, and clouds. Textures provide rich detail, while the alpha channel creates soft edges.

   * **Blending Mode:** Typically, **Alpha Blending** is used: `FinalColor = SrcColor.alpha * SrcColor + (1 - SrcColor.alpha) * DstColor`.

3. **Metaballs/Implicit Surfaces**

   * **Method:** Each particle is not rendered directly but is treated as the center of a **field function**, which has a maximum value at the particle center and decays outwards. The surface of the entire particle system is defined as an **isosurface** where the sum of the field function values ​​of all particles equals a certain threshold.

   * **Mathematics:** The field function of a particle centered at $\vec{c}_i$ can be $f_i(\vec{p}) = \frac{R^2}{||\vec{p} - \vec{c}_i||^2}$. The field function of the entire system is $F(\vec{p}) = \sum_i f_i(\vec{p})$. The rendered surface is the set of all points $\vec{p}$ that satisfy $F(\vec{p}) = T$ (threshold).

   * **Implementation:** Ray Marching or Moving Cubes algorithms are typically used to generate the final geometry.

   * **Effects:** Ideal for simulating fluid effects with merging and separating behaviors, such as liquids, lava, and slime, like the liquid metal robot T-1000 in *Terminator 2*.

# V. Implementing a Complete Particle System in C++

Below, we will implement a basic but complete particle system in C++, containing all the core components we have discussed, and employing the **velocity Welley integral method**.

## V.1 Data Structure

First, define the basic 3D vector class and particle structure.

```cpp
// Vec3.h
struct Vec3 {
    float x = 0, y = 0, z = 0;
    // ... (overload +, -, *, / operators)
};

// Particle.h
struct Particle {
    Vec3 position;
    Vec3 velocity;
    Vec3 acceleration;
    Vec3 forceAccumulator; // force accumulator

    float mass = 1.0f;
    float lifespan = 0.0f;
    float age = 0.0f;

    // rendering attributes
    Vec4 color;
    float size = 1.0f;

    bool isAlive() const { return age < lifespan; }
};
```

## V.2 Particle System Class

This class is the manager of the entire system.

```cpp
// ParticleSystem.h
#include <vector>
#include "Particle.h"

class ParticleSystem {
public:
    ParticleSystem(int maxParticles);
    ~ParticleSystem();

    void update(float dt);
    void render();

    void emit(int count);

private:
    void applyForces(Particle& p);
    void integrate(Particle& p, float dt);
    void handleCollisions(Particle& p);
    void initializeParticle(Particle& p);

    std::vector<Particle> m_particles;
    int m_maxParticles;
};
```

## V.3 Implementation Details

```cpp
// ParticleSystem.cpp
#include "ParticleSystem.h"

ParticleSystem::ParticleSystem(int maxParticles) : m_maxParticles(maxParticles) {
    m_particles.resize(m_maxParticles);
}

ParticleSystem::~ParticleSystem() {}

// Core update loop
void ParticleSystem::update(float dt) {
    for (auto& p : m_particles) {
        if (!p.isAlive()) continue;

        p.age += dt;
        if (!p.isAlive()) continue;

        // 1. Clear force accumulator
        p.forceAccumulator = {0, 0, 0};

        // 2. Apply forces
        applyForces(p);

        // 3. Integration (core physics simulation)
        integrate(p, dt);
        
        // 4. Collision handling
        handleCollisions(p);
    }
}

void ParticleSystem::render() {
    // Rendering logic: traverse all alive particles and draw based on selected rendering method
    // For example, use OpenGL to draw point sprites or billboards
    for (const auto& p : m_particles) {
        if (p.isAlive()) {
            // ... OpenGL/Vulkan/DirectX rendering code ...
            // drawParticle(p.position, p.size, p.color);
        }
    }
}

void ParticleSystem::emit(int count) {
    int emitted = 0;
    for (auto& p : m_particles) {
        if (emitted >= count) break;
        if (!p.isAlive()) {
            initializeParticle(p);
            emitted++;
        }
    }
}

// Initialize a new particle
void ParticleSystem::initializeParticle(Particle& p) {
    p.age = 0.0f;
    p.lifespan = 2.0f + static_cast<float>(rand()) / RAND_MAX * 3.0f; // 2-5s
    p.position = {0, 0, 0}; // emit from origin
    p.velocity = {
        (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 5.0f,
        static_cast<float>(rand()) / RAND_MAX * 10.0f,
        (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 5.0f
    }; // emit upward randomly
    p.mass = 1.0f;
}

// Apply various forces
void ParticleSystem::applyForces(Particle& p) {
    // Gravity
    Vec3 gravity = {0.0f, -9.8f * p.mass, 0.0f};
    p.forceAccumulator += gravity;

    // Air resistance
    Vec3 drag = p.velocity * -0.5f; // k_d = 0.5
    p.forceAccumulator += drag;
}

// Velocity Verlet integration
void ParticleSystem::integrate(Particle& p, float dt) {
    // Remember old acceleration
    Vec3 old_acceleration = p.acceleration;

    // 1. Update position
    p.position = p.position + p.velocity * dt + old_acceleration * (0.5f * dt * dt);

    // 2. Calculate new acceleration (force from new position)
    // Here we assume force does not depend on position, so new acceleration equals F/m
    Vec3 new_acceleration = p.forceAccumulator / p.mass;

    // 3. Update velocity
    p.velocity = p.velocity + (old_acceleration + new_acceleration) * (0.5f * dt);

    // Update acceleration state
    p.acceleration = new_acceleration;
}


// Collision handling
void ParticleSystem::handleCollisions(Particle& p) {
    // Collision with ground (plane y=0, normal N=(0,1,0))
    if (p.position.y < 0.0f) {
        // 1. Position correction
        p.position.y = 0.0f;
        
        // 2. Velocity response
        float epsilon = 0.75f; // restitution coefficient
        p.velocity.y = -p.velocity.y * epsilon;
    }
}
```

# VI. Advanced Topic: Material Point Method (MPM)

Traditional particle systems (also known as mass-spring systems) are well-suited for simulating sparse, separated phenomena. However, when simulating continuous media (such as snow, sand, water, elastic bodies) with large deformations, fractures, and phase changes, they encounter difficulties.

**Material Point Method (MPM)** is an advanced particle-based method that combines the advantages of the Lagrangian method (particles carry material properties) and the Eulerian method (fixed background grid for calculations).

*   **Particles**: Store all state variables, such as mass, velocity, position, deformation gradient, etc. They move with the material.
*   **Grid**: A fixed background grid. At each time step:
    1.  Particle information (mass, momentum) is **mapped** to grid nodes.
    2.  Compute forces (e.g., pressure, stress) on the grid and solve momentum equations to update grid node velocities.
    3.  Interpolate grid node velocities back to particles to update particle velocities and positions.
    4.  Reset the grid.

MPM can simulate various materials in a unified framework and naturally handle topological changes, making it a hot topic in computer graphics research and widely used in movie special effects for snow, sand, and fluid simulation.

# Conclusion

Particle systems are a powerful and flexible procedural animation technique. Their core advantages lie in:

*   **Widely applicable**: Can simulate various "fuzzy" phenomena such as fire, smoke, water, and star clouds.
*   **High computational efficiency**: Based on simplified physical assumptions, it can simulate a large number of particles in real-time.
*   **Artistic controllability**: Through adjusting emitter parameters, force fields, and rendering styles, artists can create diverse visual effects.

From basic Newtonian dynamics to robust numerical integration methods, and then to various rendering techniques, a deep understanding of particle systems is a must for students and researchers in computer graphics and animation.
---
title: "3D Image Morphing"
lecture: 5
course: "anim"
date: 2026-02-13
---


# I.Introduction: What is 3D Morphing?

First, let's experience the charm of this technology intuitively through a few vivid examples.

Imagine a game character, before our very eyes, smoothly and seamlessly transforming from an ordinary human form into a powerful demonic form. Or, in a car design showcase, a concept car's design evolves fluidly between different aerodynamic shapes. This is the magic of morphing.

So, what exactly is 3D morphing?

**Definition:** 3D morphing (3D shape morphing) refers to the process of transforming the shape of one 3D object (the **source object**) into the shape of another 3D object (the **target object**) through a smooth, continuous transformation.

This process is far more than a simple fade-in/fade-out between two objects; it involves the "growth" and "evolution" of the geometric structure itself.

Let's look at a classic example: a 3D dinosaur model. Every vertex, every edge, and every face undergoes continuous change, ultimately transforming into an electric iron, a common object in our daily lives. At any given moment during this transformation process, what we see is not a blurry double image, but a complete, independent, and entirely new three-dimensional model.

# II. Why Use 3D Morphing? The Essential Difference from 2D Morphing

You might wonder, we've often seen 2D image morphing in many software applications, such as turning one person's face into another's. What is the fundamental difference between those techniques and the technology we're discussing today?

This is a very crucial question, and it is the key to understanding the huge advantages of 3D Morphing.

*   **Limitations of 2D Morphing**:
    
    *   2D morphing technology deals with **images**. The essence of an image is a collection of pixels, which itself **contains no three-dimensional geometric information**.
    *   This means that when you perform morphing on two images, you are merely performing pixel-level distortion and color blending.
    *   Its biggest limitation is that the resulting intermediate result is still a 2D image. You **cannot change the viewing angle**, nor can you recalculate the lighting for the scene. If the camera is moving in the source and target animations, 2D morphing is helpless.
*   **Advantages of 3D Morphing**:
    
    *   3D morphing deals with **3D models**. We operate on real geometric data such as vertex coordinates and topological structures.
    *   The intermediate frames it generates are **complete, brand-new 3D models**, not just pictures.
    *   This means that once we obtain the intermediate model sequence from the source model to the target model, we can:
        *   Observe this transformation process from **any camera angle**.
        *   Recalculate the lighting and shadow effects on the intermediate model under **any lighting conditions**.
        *   Seamlessly **integrate this transformation process** into other complex 3D scenes.

Although the algorithm for 3D morphing is much more complex, it is precisely because it can generate such realistic and vivid special effects that it has attracted the investment of countless computer graphics researchers.

In the film industry, this technology has long been a star. For example, in "Scooby-Doo 2," the slime monster's transformation from a hand to a monster face is a typical application of 3D Morphing. In the "Transformers" series, every transformation process has been carefully designed by artists and engineers, making it a model of 3D Morphing technology.

# III.3D Morphing Technology Classification

Having understood what 3D Morphing is and why it is used, let's look at how it is done. Academically, 3D Morphing technology is generally divided into two categories:

1.  **Surface-based Representation**:
    
    *   This method directly operates on the surface mesh of the object, such as the common polyhedron models we see.
    *   Its core challenge lies in two key steps:
        *   **Establishing Vertex Correspondences:** Finding a one-to-one correspondence between points on the surface of the source model and points on the surface of the target model. This is the most difficult step, which we call the "Correspondence Problem."
        *   **Vertex Interpolation:** After establishing the correspondence, interpolating the attributes of the corresponding vertices, such as position, color, and normal vector. This is relatively simple and is called the "Interpolation Problem."
    *   A common limitation of this method is that it usually requires the two models to have **topological consistency** (e.g., both are closed spheres without holes), or it requires complex algorithms to unify their topological structures.
2.  **Volume-based Representation**:
    
    *   This method does not care about the surface of the object but regards the object as a "body" in three-dimensional space, usually represented by a three-dimensional voxel grid.
    *   It first needs to **voxelize** the surface model (such as a polygon mesh) and convert it into a volumetric representation. This process may lead to information loss or distortion.
    *   Its core idea is to **warp the entire space**, not just the object's surface. When the space is warped, the object embedded in the space also deforms accordingly.
    *   This method is usually computationally expensive, but its advantage is that it can naturally handle objects with completely different topological structures.

Next, we will delve into these two categories of methods to discuss their principles and mathematical details in detail.

# IV. Deep Dive: Surface-based Morphing Methods

## 4.1 A Very Simple Ideal Case

Let's start from the most ideal situation. Suppose we have two three-dimensional models, such as a sphere and a soda can. If they satisfy two extremely harsh conditions:

1.  **The number of vertices is exactly the same**.
2.  **The topological structure is exactly the same** (i.e., the connectivity of the vertices, which is the composition of edges and faces, is exactly the same).

Then, the Morphing process becomes extremely simple. We only need to pair the $i$-th vertex $V_{S,i}$ of the source model $M_S$ with the $i$-th vertex $V_{T,i}$ of the target model $M_T$.

In time $t \in [0, 1]$ (where $t=0$ corresponds to the source model and $t=1$ corresponds to the target model), any vertex $V_i(t)$ of an intermediate model $M(t)$ can be obtained through linear interpolation (Lerp):

$$
V_i(t) = (1-t) \cdot V_{S,i} + t \cdot V_{T,i}
$$

This is a very simple 3D Morphing. But in reality, almost no two meaningful models satisfy such strict conditions. Therefore, we must find more general methods.

## 4.2 Polyhedral Morphing Methods Based on Star-shaped Objects

Now, let's look at a milestone work. In 1992, Kent, Carlson, and Parent proposed a morphing method for a special type of polyhedron—**star-shaped objects**. This is a very clever solution to the "Correspondence Problem" in the early days.

Before delving into the algorithm, we must first unify some **basic concepts**.

*   **Object**: A solid with three-dimensional surface geometry.
*   **Shape**: The set of points that make up the surface of an object.
*   **Model**: A complete description of the shape of an object, such as a polygon mesh model.
*   **Topology**: The connectivity structure of the model (how vertices, edges, and faces are connected). It does not care about the specific coordinates of the vertices.
*   **Geometry**: A specific instance of a topological structure, obtained by assigning coordinates to each vertex.
*   **Homeomorphic**: If there exists a continuous and invertible one-to-one mapping between the surfaces of two objects, we say they are homeomorphic or topologically equivalent. A classic example is that a coffee cup and a donut are topologically equivalent because they both have a "hole."
*   **Euler's Formula**: For a closed, well-behaved polyhedron, the number of vertices (V), edges (E), and faces (F) satisfy a beautiful relationship: 
$$
V - E + F = 2 - 2G
$$ 
Here, $G$ is called **Genus**, which intuitively is the number of "holes" on the surface of the object. A sphere has a genus of $G=0$, and a donut (torus) has a genus of $G=1$. Models that satisfy this formula are called **Euler valid**.

Kent et al.'s method is mainly aimed at polyhedra with **genus 0**, that is, objects that are topologically equivalent to spheres.

### **Algorithm Core Idea**

The basic idea of this algorithm, as we mentioned earlier, is divided into two steps:

1.  **Correspondence**: This is the difficult part.
2.  **Interpolation**: This is relatively simple.

For two polyhedra with genus 0 (such as a cube and a sphere made of many patches), they are both homeomorphic to a sphere. This gives us a wonderful inspiration: we can use the **sphere** as a "common reference space."

Imagine these two polyhedra are like two differently shaped balloons. We can "blow" them both into a standard spherical shape. In this process, any point on the surface of the source object will be uniquely mapped to the sphere.

*   If the point $V_1$ on the source object and the point $V_2$ on the target object are mapped to the **same point** on the sphere, we consider $V_1$ and $V_2$ to be a pair of **corresponding points**.

This is the basic idea of merging topological structures.

### **Algorithm Steps**

This process can be broken down into three elegant steps:

1.  **Projection**: Project the source model $M_a$ and the target model $M_b$ onto a unit sphere, respectively, obtaining two topological networks on the sphere, $(M_a)_p$ and $(M_b)_p$.
2.  **Merge**: Merge these two spherical topological networks $(M_a)_p$ and $(M_b)_p$ into a new, more refined topological network $M_c$. This new network contains all the vertices and edges of the previous two networks.
3.  **Map Back**: Map the merged topological network $M_c$ back to the original two object surfaces, respectively, obtaining two **new models**, $M_a^\_$ and $M_b^\_$.

At this point, something magical happens: $M_a^\_$ and $M_b^\_$ have **exactly the same number of vertices and topological structure**, while their shapes remain consistent with the original models $M_a$ and $M_b$, respectively.

Now, the problem is reduced to the simplest case from Section 4.1! We can directly perform linear interpolation on the corresponding vertices of $M_a^-$ and $M_b^-$.

Let's break down the details of these steps one by one.

**Step 1: Projection**

The projection method must satisfy two conditions: one-to-one correspondence and continuity. For **star-shaped objects**, there is a very intuitive projection method.

**Definition: Star-shaped Object** An object is star-shaped if there exists at least one point inside it (called the **kernel point**) from which a ray can see all internal vertices of the object without occlusion. Convex bodies are a special case of star-shaped objects.

The projection process is as follows:

1.  Select a kernel point $O$ inside the star-shaped object (usually the centroid of the object).
2.  For any vertex $V$ on the surface of the object, construct a ray $\\vec{OV}$ starting from $O$ and passing through $V$.
3.  The intersection of this ray with the unit sphere is the projection point $(V)\_p$ of vertex $V$ on the sphere.

Mathematically, the coordinates of the projection point $(V)\_p$ can be expressed as:

$$
(V)_p = \frac{V - O}{|V - O|}
$$

**Step 2: Topology Merging**

When the two topological networks $(M_a)_p$ and $(M_b)_p$ are projected onto the sphere, we need to merge them. This is essentially a **Map Overlay** problem in computational geometry.

*   **New vertex set**: Contains all vertices of $(M_a)_p$, all vertices of $(M_b)_p$, and all **intersection points** of the edges of $(M_a)_p$ and $(M_b)_p$.
*   **New edge set**: All **small edge segments** formed by splitting the original edges by the above vertex set.
*   **New face set**: The **minimum region** enclosed by the new edge set.

This process is like drawing the maps of two countries on a globe and then generating a new map that contains all the national borders and provincial boundaries.

**Step 3: Interpolation**

After obtaining $M_a^-$ and $M_b^-$ with the same topology, we interpolate the vertices. Besides geometric position, other attributes such as color, texture coordinates, and opacity can also be interpolated. For these non-geometric attributes, **Barycentric Coordinates** are typically used for interpolation.

*   Assume a point $P(t)$ on the intermediate frame model $M(t)$ is located on a certain triangular face. We can calculate its barycentric coordinates $(\alpha, \beta, \gamma)$ on this face, where $\alpha+\beta+\gamma=1$。
*   Using the same barycentric coordinates, we can find the corresponding points $P_a^-$ and $P_b^-$ on the corresponding faces of $M_a^-$ and $M_b^-$.
*   The attributes of $P_a^-$ and $P_b^-$ (such as color $C_a$ and $C_b$) can then be interpolated to obtain the color of $P(t)$: 
$$
C(t) = (1-t)C_a + t C_b
$$

**Potential problems in the interpolation process:**

*   **Non-coplanarity**: For faces with more than three vertices, the vertices may no longer lie on the same plane during the interpolation process. A simple solution is to **triangulate** all faces after merging the topology.
*   **Self-intersection**: During the interpolation process, some parts of the model may pass through each other, resulting in invalid geometry. This is a very difficult problem that usually requires more complex algorithms, such as energy minimization methods or volume constraints.

This classic work demonstrates the solution to the Morphing problem, from the smooth transition from a wine glass to a human face, from the letter S to a portrait, all proving its effectiveness.

# V. Further Exploration: Morphing and Parametricization of General Mesh Models

Kent et al.'s method is ingenious, but it relies on the premises of "star-shaped objects" and "genus 0". How can it be generalized to arbitrarily complex mesh models (such as those with holes or highly convex and concave surfaces)?

The answer is to introduce a more powerful and general mathematical tool——**Parameterization**.

Parameterization is essentially the process of assigning a two-dimensional coordinate to every point on the surface of a three-dimensional model. This is like unfolding an irregular globe (3D) into a flat world map (2D). This 2D "map" is our **parameter domain**. Once two different 3D models are mapped to the same parameter domain, we can establish a correspondence on this parameter domain.

Common parameterization methods include:

*   **Spherical Parameterization**: Maps models with genus 0 to a sphere.
*   **Planar Parameterization**: Maps models that are topologically like a disk (an open surface with boundaries) to a 2D plane.
*   **Patch Parameterization**: For complex models with genus greater than 0 (such as those with holes), they are first cut into several topological disk patches, and then each patch is parameterized in a plane.
*   **Polycube Parameterization**: Maps the model to a shape composed of cubes, often used to generate high-quality quadrilateral meshes.

## 5.1 Spherical Parameterization

For genus 0 models, we can use more robust algorithms than the previous star-shaped projection to parameterize them onto a sphere. A famous strategy is **Progressive Mesh Parameterization**.

*   **Step 1: Simplification**. First, through a series of "edge collapse" operations, the high-precision input model is continuously simplified until a very simple base mesh (such as an octahedron) is obtained.
*   **Step 2: Parameterize the base mesh**. The vertices of this simplest base mesh are directly mapped to the sphere.
*   **Step 3: Progressive Recovery and Optimization**. Then, the simplification process is executed in reverse (called "vertex splitting"), and the model is restored to its original precision step by step. At each step of recovery (i.e., adding a vertex), we find an optimal position on the sphere for the new vertex, aiming to **minimize the distortion** of the parameterization mapping. Distortion is usually measured by the change in angle or area.

This process guarantees that the final spherical mapping is **globally non-overlapping** (bijective) and **minimally distorted**.

Once we have parameterized the horse model and the cow model (or the gargoyle and the rabbit model) onto the sphere, we can merge their topological structures on the sphere as before, and then perform interpolation to achieve high-quality Morphing.

## 5.2 Planar Parameterization and Harmonic Map

When our model has boundaries (such as a human face model, excluding the back of the head), we can parameterize it onto a 2D plane, usually a unit disk. **Harmonic Map** is a very popular and effective method.

Its physical model can be imagined as a fishing net connected by countless small springs. We fix the boundary of the fishing net to a circle and let it naturally relax to an **energy-minimizing** equilibrium state. The shape of the fishing net in this equilibrium state is the planar parameterization result we want.

**Mathematical Derivation**

1.  **Boundary Fixed**: First, we fix the $n$ vertices that make up the boundary of the model to the boundary of the unit disk in order, usually proportional to their edge lengths along the boundary in 3D space.
    
2.  **Energy Function:** For the interior vertices, their positions are unknown. We define a total energy function $E_{\text{harm}}$, which represents the sum of the elastic energies of all the "springs":
$$
E_{\text{harm}}(\mathbf{v}) = \frac{1}{2} \sum_{{i,j} \in \text{Edges}(H)} k_{i,j} |v_i - v_j|^2
$$

   *   $\mathbf{v}$ is the set of all vertex positions in the 2D plane.
   *   $v_i, v_j$ are the coordinates of vertices $i$ and $j$ in the 2D plane.
   *   $k_{i,j}$ is the "spring coefficient" of the edge connecting vertices $i,j$. This coefficient is crucial; it should reflect the original geometric information in the 3D model. A commonly used choice is **cotangent weight**, which preserves angles well and reduces mapping distortion.

3.  **Energy Minimization**: Our goal is to find a set of internal vertices that minimizes $E_{\text{harm}}$. According to the variational principle, at the optimal point, the gradient (partial derivative) of the energy function with respect to each movable vertex must be zero. $\nabla E_{\text{harm}} = \mathbf{0}$
    
4.  **Solving the Linear System**: Since $E_{\text{harm}}$ is a quadratic function of the vertex coordinates $\mathbf{v}$, its gradient is a linear function. Therefore, $\nabla E_{\text{harm}} = \mathbf{0}$ forms a large **sparse linear system**. We can arrange the unknown coordinates $(v_x, v_y)$ of all vertices into a long vector $\mathbf{V}$. The system of equations can be written in matrix form. More precisely, we can divide the vertices into **free parts** $V_\alpha$ (internal vertices) and **fixed parts** $V_\beta$ (boundary vertices). The energy function can be expanded as: 
$$
E_{\text{harm}} = \frac{1}{2} \begin{bmatrix} V_\alpha^T & V_\
\beta^T \end{bmatrix} \begin{bmatrix} H_{\alpha\alpha} & H_{\alpha\beta} \\ H_{\beta\alpha} & H_{\beta\beta} \end{bmatrix} \begin{bmatrix} V_\alpha \\ V_\beta\end{bmatrix}
$$
We take the partial derivative of the free part $V_\alpha$ with respect to $V_\alpha$ and set it to zero: 
$$\frac{\partial E_{\text{harm}}}{\partial V_\alpha} = H_{\alpha\alpha}V_\alpha + H_{\alpha\beta}V_\beta = \mathbf{0}
$$
This is a standard linear system: 
$$
H_{\alpha\alpha}V_\alpha = -H_{\alpha\beta}V_\beta
$$
Since $V_\beta$ is known (we fix it to the boundary of the disk), we can find the positions of all internal vertices $V_\alpha$ by solving this system of equations.

This method allows us to obtain a smooth and non-overlapping planar parameterization result. This provides a solid foundation for feature correspondence between different models.

**C++ Data Structure Example** To make it more concrete, a simple mesh data structure can be designed as follows:

```cpp
#include <vector>
#include <Eigen/Sparse> // A popular library for linear algebra

// Assuming we have a 3D vector class Vector3
struct Vector3 { float x, y, z; };
// And a 2D vector class Vector2
struct Vector2 { float u, v; };

struct Vertex {
    Vector3 position;    // 3D geometric position
    Vector2 uv;          // 2D parameterization coordinates
    bool is_boundary;    // Flag to identify boundary vertices
};

struct Face {
    // Stores indices of vertices that form this face
    std::vector<int> vertex_indices; 
};

class TriangleMesh {
public:
    std::vector<Vertex> vertices;
    std::vector<Face> faces;

    // Method to compute the harmonic map (planar parameterization)
    void compute_harmonic_map() {
        // 1. Identify boundary vertices and map them to a circle.
        // 2. Build the sparse matrix H_alpha_alpha and the vector -H_alpha_beta * V_beta.
        //    The weights k_ij (e.g., cotangent weights) are computed here.
        // 3. Solve the linear system H_alpha_alpha * V_alpha = b using a sparse solver.
        // 4. Store the resulting 2D coordinates in vertices[i].uv.
    }
};
```

# VI. Taking a different approach: Morphing based on volume representation

Now let's change our thinking and stop dwelling on the complex problem of surface correspondence. Let's enter the world of **Volume-based Morphing**.

This method is inspired by **Field Warping** in 2D image processing. In 1995, Lerios et al. successfully extended it to three-dimensional space.

**Core Idea**:

1.  **Geometric Alignment (Warping)**: First, the animator specifies pairs of **feature elements** in the source body S and the target body T (for example, the tip of the nose on the source object corresponds to the tip of the nose on the monster).
2.  **Blending**: Then, the algorithm generates a spatial transformation (Warping function) that can "twist" the source body S and the target body T into two geometrically aligned intermediate bodies S' and T'.
3.  **Blending**: The voxel attributes (such as density, color) of the two warped bodies S' and T' are interpolated to obtain the final intermediate body.

## 6.1 Geometric Alignment

*   **Feature Elements**: The features specified by the animator can be **pairs of points, lines, rectangles, or cuboids**.
*   **Local Coordinate System**: Each feature defines a local coordinate system. For example, a feature line segment can define an origin, a main axis direction, and two perpendicular directions.
*   **Spatial Transformation**: The process of Morphing can be seen as the process of smoothly **translating, rotating, and scaling** the local coordinate system of the source feature to the local coordinate system of the target feature.
*   **Warping Function Calculation (Inverse Mapping)**: For each pixel in the final rendered image, we trace back to the point $P'$ in 3D space. We want to know which point $P_S$ in the source body S this point $P'$ should sample color from.
    1.  For each feature pair, there is an interpolated feature $e'$ at time $t$.
    2.  We calculate the coordinates of $P'$ in the local coordinate system of $e'$, $(p_x, p_y, p_z)$.
    3.  We assume that this **local coordinate is invariant** (this is the core assumption of local parameterization).
    4.  Using the same local coordinates $(p_x, p_y, p_z)$, we can calculate the position of the corresponding source space point $P_S$ in the coordinate system of the source feature $e_S$. 
    $$
    P_S = \mathbf{c}_S + p_x s_{Sx} \mathbf{x}_S + p_y s_{Sy} \mathbf{y}_S + p_z s_{Sz} \mathbf{z}_S
    $$ 
    The same method can be used to find the corresponding point $P_T$ of $P'$ in the target body T. In this way, we have found the "past and present" of any point $P'$ in space in the source and target bodies.

## 6.2 Multi-feature Processing

When there are multiple feature pairs, each feature pair will calculate a corresponding source point $P_{S,i}$ for point $P'$. The final corresponding point $P_S$ is the **weighted average** of these $P_{S,i}$: 
$$
P_S = \frac{\sum\limits_{i=1}^{n} w_i P_{S,i}}{\sum\limits_{i=1}^{n} w_i}
$$

*   **Weight $w_i$**: The magnitude of the weight depends on the **distance** $d_i$ between point $P'$ and the $i$-th interpolation feature $e'_i$. The closer the distance, the greater the influence. Usually, the weight is inversely proportional to the square of the distance: 
$$
w_i = \frac{1}{(d_i + \epsilon)^2}
$$
where $\epsilon$ is a small positive number to prevent the denominator from being zero. The calculation method of the distance $d_i$ depends on the type of feature (point-point distance, point-line segment distance, point-cuboid distance, etc.).

## 6.3 Blending

After calculating the warped bodies $S'$ and $T'$ (in fact, we obtain them implicitly through inverse mapping), we perform blending.

*   **Incorrect Method**: Render $S'$ and $T'$ separately (Volume Rendering) to get two images, and then fade between the two images. This approach loses the correct lighting and occlusion relationships.
*   **Correct Method**: Interpolate the **color and opacity** of the corresponding voxels in $S'$ and $T'$ to obtain a new blended body. Then, perform volume rendering on this new body. This can produce a three-dimensional Morphing that is physically more realistic and visually better.

Volume Morphing shows powerful capabilities in handling objects with different shapes and topologies (such as darts to airplanes, lions to horses). However, its disadvantages are also obvious: voxelization may introduce aliasing (jaggies), the accuracy is not as good as surface methods; for high precision, a very large volume of data is required, and the computational overhead is huge.

# VII. Advanced Progress: Variational Implicit Surface-Based Morphing

Finally, let's explore a more advanced and mathematically elegant method that can even handle **topological changes** in Morphing.

*   **Implicit Surface**: A method that defines a surface not through vertices and faces, but through a functional equation $f(x, y, z) = 0$. The surface is the set of points where the function value is zero.
*   **Implicit Surface Morphing**: If the source object and the target object are defined by implicit functions $f_S=0$ and $f_T=0$ respectively, then a simple linear interpolation function $f(t) = (1-t)f_S + t f_T = 0$ can describe a Morphing process.

But the question is, how to find a "good" implicit function for any given mesh model?

In 1999, Turk and O'Brien proposed an amazing method that transforms the **N-dimensional Morphing problem into an (N+1)-dimensional scattered data interpolation problem**.

**Core Idea (Taking 2D Morphing as an example)**

1.  **Dimension Lifting**: We want to perform Morphing between two 2D shapes. We construct a 3D space, placing all constraints of the source shape on the $z=0$ plane and the constraints of the target shape on the $z=1$ plane.
2.  **Setting Constraints**:
    *   On the **boundary** of the shape, we specify the function value as 0.
    *   **Inside** the shape (a small distance along the boundary normal), we specify the function value as 1.
3.  **Variational Interpolation**: We look for a function $f(x, y, z)$ in 3D space that must satisfy the values of all the constraint points we set and be as **smooth** as possible itself. This "smoothest" is achieved by minimizing an energy function (usually the integral of the function gradient, simulating the bending energy of a thin plate). This process is called **variational interpolation**.
4.  **Slicing to Obtain the Intermediate Frame:** After obtaining the 3D implicit function $f(x,y,z)$, we slice it using a plane $z=t$ (where $t \in [0, 1]$). The intersection of the slice and $f=0$ is the intermediate shape at time $t$!

**Mathematical Tool: Radial Basis Functions (RBFs)** Solving the above variational interpolation problem can be represented as a linear combination of a special function—the **Radial Basis Function (RBF)**—plus a low-order polynomial. 
$$
f(\mathbf{x}) = \sum_{j=1}^{n} d_j \phi(|\mathbf{x} - \mathbf{c}_j|) + P(\mathbf{x})
$$

*   $\mathbf{c}_j$ is the position of the constraint point.
*   $d_j$ is the unknown weight coefficient.
*   $P(\mathbf{x})$ is a low-order polynomial (e.g., linear $p_0 + p_1x + p_2y + p_3z$).
*   $\phi(r)$ is the RBF, a common choice is $\phi(r) = r^2 \log(r)$.

Substituting all constraints $f(\mathbf{c}_i) = h_i$ ($h_i$ is a specified value at the constraint point, 0 or 1) into the above equation, we obtain a large system of linear equations. Solving this system of equations yields all the unknown coefficients, thus determining the unique implicit function $f$.

Compared to simple distance-field-based implicit surfaces, this method generates smoother intermediate results, effectively avoiding unnatural "squeezing" or "sharp" sections during transformation.

The essence of this method lies in unifying shape representation and shape interpolation within a single framework. For 3D Morphing, we construct a 4D implicit function $f(x,y,z,t)=0$, obtained by solving a 4D RBF interpolation problem. The intermediate frame is a 3D "slice" when $t \in [0,1]$. This method can naturally handle topological changes, such as a complete object smoothly splitting into two.

# Conclusion

This lecture systematically reviews the development and technological framework of 3D Morphing, focusing on the problem of 3D shape deformation, from theoretical foundations to classic algorithms and advanced methods. First, by comparing the essential differences between 2D and 3D deformation, it emphasizes the crucial role of 3D geometric information in viewpoint, lighting, and spatial consistency. Then, it focuses on surface-based deformation methods, including ideal correspondence, star-shaped object mapping, and parametric techniques, providing a unified framework for solving vertex correspondence and topological consistency problems.

Building on this foundation, the course further explores volumetric deformation methods based on voxel representation. Through spatial warping and volumetric data fusion, it achieves a natural transition to complex topological structures, while analyzing its limitations in computational cost and accuracy. Finally, it introduces advanced methods based on variational implicit surfaces, transforming the deformation problem into a high-dimensional function interpolation problem, allowing shape changes and topological evolution to be uniformly characterized within the same mathematical framework.

Overall, the development of 3D Morphing reflects an evolutionary trend from explicit geometric modeling to parametric mapping and then to implicit representation. Surface methods emphasize precise control and high accuracy, voxel methods emphasize topological flexibility, while implicit methods balance smoothness and structural changeability. Future research is expected to further integrate geometric modeling, physical constraints, and data-driven methods to achieve more realistic and efficient 3D morphological evolution while ensuring controllability, providing stronger technical support for fields such as animation production, industrial design, and virtual reality.
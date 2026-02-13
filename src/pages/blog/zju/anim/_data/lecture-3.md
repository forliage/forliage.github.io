---
title: "2D Image Morphing"
lecture: 3
course: "anim"
date: 2026-02-13
---

# I.The Concept and Intuition of Morphing

## **1.1 What is Image Morphing?**

Everyone may have seen such special effects in movies, advertisements, or other media: a person's face smoothly transforms into another person's face, or a cat seamlessly transforms into a tiger. This process, we call **Morphing**, which means "deformation".

Technically speaking, **Image Morphing** is the process of transforming one digital image (Source Image) into another digital image (Destination Image) in a natural, smooth, and dramatic way. This process generates a series of intermediate frames, which, when played in sequence, create a smooth transition animation.

The most famous example is undoubtedly the facial morphing effect used at the end of Michael Jackson's music video "Black or White" in 1991, which caused a huge sensation and popularized this technology.

## **1.2 The Core Idea of Morphing: Decomposition and Synthesis**

A successful Morphing effect is actually the perfect combination of two basic operations:

1.  **Image Warping**: This is a geometric transformation that changes the position of pixels in an image but does not change their color. It is responsible for gradually "pinching" the shape of the source image into the shape of the destination image, and also "pinches" the shape of the destination image into the shape of the source image.
2.  **Cross-dissolving**: This is a color transformation that achieves a gradual fade-out of one image and a gradual fade-in of another image by averaging the color values of the two images.

We can describe the intermediate frame image $I_t$ at time point $t$ (where $t$ ranges from 0 to 1) using a simple formula:

$$
I_t = (1 - t) * \text{Warped\_Source} + t * \text{Warped\_Destination}
$$

Here, $\text{Warped\_Source}$ is the result of warping the source image, and $\text{Warped\_Destination}$ is the result of warping the destination image. When $t=0$, the image is entirely the source image; when $t=1$, the image is entirely the destination image.

## **1.3 2D Morphing vs. 3D Morphing**

Before we dive into the 2D technique, it is important to distinguish it from 3D Morphing:

*   **3D Morphing**:
    *   Operates on 3D models. Requires separate modeling of source and target objects.
    *   Strict topological requirements for models (e.g., the number of vertices, faces, and connectivity relationships need to be consistent).
    *   Interpolates 3D vertex coordinates, normals, texture coordinates, etc.
    *   Advantages: More physically accurate, can change perspectives and lighting arbitrarily.
    *   Disadvantages: Modeling is complex, establishing correspondences between models is extremely difficult.
*   **2D Morphing**:
    *   Operates on 2D images. No need to model source and target objects.
    *   No strict topological requirements for images (e.g., the number of pixels, connectivity relationships need to be consistent).
    *   Interpolates pixel values.
    *   Advantages: Avoids complex 3D modeling process. When the virtual camera position is fixed, the effect is outstanding, able to produce the illusion of "three-dimensional shape change".
    *   Disadvantages: Lacks true 3D geometric information, cannot handle changes in perspective and lighting, may produce distortions that do not follow physical laws.

Today, we will focus on 2D Morphing, as it achieves impressive visual effects at a lower cost in many applications.


# II. Classic Morphing Algorithms

Now, let's explore two mainstream classic methods for implementing Morphing.

## **2.1 Method One: Grid-Based Morphing**

This is one of the earliest Morphing methods, with a very intuitive idea: cover a control grid on the source image and the target image, and drive the entire image's deformation by manipulating the grid.

**2.1.1 Algorithm Flow**

1.  **Define Control Grid**: On the source image $I_S$, define a control grid $M_S$, and on the target image $I_D$, define a corresponding control grid $M_D$. $M_S$ and $M_D$ have the same topological structure (i.e., the same number of rows and columns), but the positions of the control vertices are different, matching the features of the two images.
2.  **Interpolate Intermediate Grid**: For any time $t$ ($0\leq t\leq 1$), interpolate the control vertex coordinates of $M_S$ and $M_D$ linearly to obtain an intermediate control grid $M_t$. 
$$
M_t(i, j) = (1 - t) * M_S(i, j) + t * M_D(i, j)
$$
3.  **Image Warping**: This is the most critical step. We need to generate two warped images:
    *   $I_S'$: Warp the source image $I_S$ from its original grid $M_S$ to the intermediate grid $M_t$.
    *   $I_D'$: Warp the target image $I_D$ from its original grid $M_D$ to the intermediate grid $M_t$.
4.  **Cross Dissolve**: Cross dissolve the two warped images $I_S'$ and $I_D'$ to obtain the final intermediate frame $I_t$. 
$$
I_t(p) = (1 - t) * I_S'(p) + t * I_D'(p)
$$

**2.1.2 Mathematically Implementing Warping: Inverse Mapping and Bilinear/Bicubic Interpolation**

Directly mapping pixels from the source grid to the target grid (forward mapping) may cause issues in the target image, such as empty areas (multiple source pixels mapped to the same location) or overlaps (some target pixels without corresponding source pixels). Therefore, we usually use **inverse mapping (Inverse Mapping)**:

For each pixel $p$ in the intermediate frame $I_t$, we ask: where are the original positions $p_s$ and $p_d$ in the source image $I_S$ and target image $I_D$?

*   **Steps**:
    1.  Determine which quadrilateral cell in the intermediate grid $M_t$ the pixel $p$ is located in.
    2.  Calculate the relative position of $p$ within the cell (usually using $(u,v)$ coordinates, where 0 ≤ u,v ≤ 1), to find the corresponding positions $p_s$ and $p_d$ in the source grid $M_S$ and target grid $M_D$ respectively. This calculation process is **bilinear interpolation** or smoother **bicubic spline interpolation**.
    3.  Since $p_s$ and $p_d$ are usually not integers, we need to use the surrounding pixels in $I_S$ and $I_D$ to perform bilinear interpolation again to obtain the exact color values.

**Bicubic Spline Surface**

To achieve smoother deformation, the grid is usually considered as control points of a bicubic spline surface. A spline surface defined by $(n+1)\times(n+1)$ control points $P_{ij}$ can be represented as:

$$
p(u,v) = \sum_{i=0}^{n} \sum_{j=0}^{n} B_{i,n}(u) B_{j,n}(v) P_{ij}
$$

where $B_{i,n}$ is the basis function (e.g., Bernstein basis functions for Bezier surfaces), $(u,v)$ is the parameterized coordinate $(u,v) \in [0,1]\times[0,1]$.

The core challenge of inverse mapping lies in solving the equation $p = p(u,v)$ to reverse and obtain $(u,v)$. This is a nonlinear equation system, usually without a closed-form solution. In practice, we can use numerical methods, such as **gradient descent**, to solve it.

*   **Objective Function**: $F(u,v) = || p(u,v) - p_{target} ||^2$
*   **Iteration Update**: $(u_{k+1}, v_{k+1}) = (u_k, v_k) - \eta * \nabla F(u_k, v_k)$ where $\eta$ is the learning rate, $\nabla F$ is the gradient.

**2.1.3 Advantages and Disadvantages of Grid-Based Morphing**

*   **Advantages**: Simple concept, smooth deformation.
*   **Disadvantages**: Control is not intuitive. In complex feature regions of the image, a very dense grid is needed, manually placing and adjusting these grid points is a tedious and arduous task.


## **2.2 Method Two: Feature-Based Morphing**

Thaddeus Beier and Shawn Neely proposed a revolutionary method in 1992, which allows animators to control deformation by specifying **line pairs**. This technology has become the standard in the industry and has been used in countless movie special effects.

**2.2.1 Core Idea**

Animators draw a series of corresponding feature line segments on the source image and target image. For example, in face Morphing, lines can be drawn on eyebrows, eyes, nose, mouth, and facial contours. These line pairs define the mapping relationship between points in the image space.

**2.2.2 Single Line Pair Definition**

Let's first consider a reverse mapping from target image $I_D$ to source image $I_S$, defined by a pair of line segments $P'Q'$ (in $I_D$) and $PQ$ (in $I_S$).

For any pixel point $X'$ in $I_D$, how do we find its corresponding point $X$ in $I_S$?

1.  **Calculate the coordinates $(u, v)$ of $X'$ relative to the line segment $P'Q'$**:
    
    *   $u$：$X'$ in the projection point of the line segment $P'Q'$ along the direction of the line segment. $u=0$ at $P'$, $u=1$ at $Q'$. 
    $$
    u = \frac{(X'-P')\cdot(Q'-P')}{\|Q'-P'\|^2}
    $$
    *   $v$：$X'$ to the signed vertical distance of the line segment $P'Q'$. The distance is in pixels. 
    $$
    v = \frac{(X'-P')\cdot\text{Perpendicular}(Q'-P')}{\|Q'-P'\|}
    $$ 
    where $\text{Perpendicular}((dx, dy))$ returns $(-dy, dx)$.
2.  **Reconstruct $X$ in the source image using $(u, v)$**： Using the same $(u, v)$ coordinates, we can calculate the position of $X$ relative to the line segment $PQ$: 
$$
X = P + u * (Q - P) + v * \frac{\text{Perpendicular}(Q - P) }{ \|Q - P\|}
$$
    

This transformation essentially consists of a composite transformation made up of rotation, translation, and scaling.

**2.2.3 Multiple Line Pairs Definition**

When multiple feature lines exist, a pixel will be affected by all the lines. Beier and Neely's method is to calculate a source position $X_i$ for each line pair $i$ from $X'$, then average these $X_i$ with weights, to get the final source position $X$.

*   **Displacement**: For each line pair $i$, it generates a displacement vector $D_i = X_i - X'$ from the target point $X'$ to the source point $X_i$.
*   **Weighted Average**: The final source position $X_{source}$ is determined by adding the weighted average of all displacements to the target position $X'_{dest}$: 
$$
X_{source} = X'_{dest} + \frac{\sum_i D_i * w_i}{\sum_i w_i}
$$
*   **Weight Function (Weight Function)**: The calculation of weight $w_i$ is crucial, as it determines the influence range and strength of each line. 
$$
w_i = \left(\frac{\text{length}^p}{a + \text{dist}}\right)^b
$$
where $\text{length}$ is the length of line segment $P'Q'$. $p$ controls the influence of length ($p=0$ means length is irrelevant, $p=1$ means longer lines have greater influence). $\text{dist}$ is the shortest distance from pixel $X'$ to line segment $P'Q'$. 

This is the main source of influence, with closer distances resulting in greater influence. $a$ is a very small constant to prevent division by zero when $\text{dist}$ is zero. It also controls the decay of influence. $b$ controls the strength of influence decay with distance. The larger $b$, the faster the influence decays, making the deformation more "local". $b$ is usually in the range $[0.5, 2]$.

**2.2.4 Complete Beier-Neely Morphing Algorithm Flow**

1.  **Input**: Source image $I_S$, target image $I_D$, source feature line set ${S_i}$, target feature line set ${D_i}$.
2.  **Animation Loop**: For each frame $t$ from 0 to 1: 
   * a. Create an empty intermediate frame image $I_t$. 
   * b. **Interpolate Feature Lines**: Calculate the feature line set ${L_i}$ for the current frame. 
   $$
   L_i = (1 - t) * S_i + t * D_i
   $$ 
   * c. **Reverse Mapping Pixels**: For each pixel $p_{\text{dest}}$ in $I_t$: 
      * i. **Calculate Source Position**: Use ${L_i}$ as target lines, ${S_i}$ as source lines, and apply the above multi-line pair transformation algorithm to calculate the source pixel position $p_{\text{source}_S}$ of $p_{\text{dest}}$ in $I_S$. 
      * ii. **Calculate Target Position**: Use ${L_i}$ as target lines, ${D_i}$ as source lines, to calculate the source pixel position $p_{\text{source}_D}$ of $p_{\text{dest}}$ in $I_D$. 
      * iii. **Sample Colors**: Perform bilinear interpolation at $p_{\text{source}_S}$ and $p_{\text{source}_D}$ positions in $I_S$ and $I_D$ respectively, to obtain colors $\text{Color}_S$ and $\text{Color}_D$. 
      * iv. **Cross Dissolve**: Calculate the final color and assign it to $I_t(p_{\text{dest}})$.
      $$
      \text{Color}_t = (1 - t) * \text{Color}_S + t * \text{Color}_D
      $$ 
3.  **Output**: Generate all intermediate frames $I_t$ sequence.


# III.Code Implementation (Python/NumPy)

We provide a simplified Python implementation of the Beier-Neely Morphing algorithm based on feature line pairs to help you understand the core logic.

```python
import numpy as np
from PIL import Image
from scipy.interpolate import interp2d

# Define a line segment data structure
class Line:
    def __init__(self, p1, p2):
        self.P = np.array(p1, dtype=float)
        self.Q = np.array(p2, dtype=float)

def warp_image(image, lines_src, lines_dst):
    """
    Warps an image based on source and destination lines.
    Inverse mapping: for each pixel in the destination grid, find its source.
    """
    height, width, _ = image.shape
    warped_img = np.zeros_like(image, dtype=np.uint8)

    # Create a coordinate grid
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    dest_pixels = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)

    # Parameters
    a = 0.001
    b = 2.0
    p = 0.5
    
    displacements = np.zeros_like(dest_pixels, dtype=float)
    total_weights = np.zeros(len(dest_pixels), dtype=float)

    for line_src, line_dst in zip(lines_src, lines_dst):
        # Calculate u, v relative to the target line
        vec_PQ_dst = line_dst.Q - line_dst.P
        vec_Perp_PQ_dst = np.array([-vec_PQ_dst[1], vec_PQ_dst[0]])
        len_sq_PQ_dst = np.dot(vec_PQ_dst, vec_PQ_dst)

        # Avoid division by zero
        if len_sq_PQ_dst == 0:
            len_sq_PQ_dst = 1e-6

        vec_XP_dst = dest_pixels - line_dst.P

        u = np.dot(vec_XP_dst, vec_PQ_dst) / len_sq_PQ_dst
        v = np.dot(vec_XP_dst, vec_Perp_PQ_dst) / np.sqrt(len_sq_PQ_dst)

        # Calculate the source pixel position
        vec_PQ_src = line_src.Q - line_src.P
        vec_Perp_PQ_src = np.array([-vec_PQ_src[1], vec_PQ_src[0]])
        len_PQ_src = np.sqrt(np.dot(vec_PQ_src, vec_PQ_src))
        
        # Avoid division by zero
        if len_PQ_src == 0:
            len_PQ_src = 1e-6
        
        # u and v are vectorized, so X_src is also vectorized
        X_src = line_src.P + u[:, np.newaxis] * vec_PQ_src + (v[:, np.newaxis] / len_PQ_src) * vec_Perp_PQ_src
        
        # Calculate displacement and weight
        D = X_src - dest_pixels
        
        # Calculate dist
        dist = np.zeros_like(u)
        # u < 0
        dist[u < 0] = np.linalg.norm(dest_pixels[u < 0] - line_dst.P, axis=1)
        # u > 1
        dist[u > 1] = np.linalg.norm(dest_pixels[u > 1] - line_dst.Q, axis=1)
        # 0 <= u <= 1
        mask = (u >= 0) & (u <= 1)
        dist[mask] = np.abs(v[mask])
        
        weight = (len(vec_PQ_dst)**p / (a + dist))**b
        
        displacements += D * weight[:, np.newaxis]
        total_weights += weight

    # Avoid division by zero
    total_weights[total_weights == 0] = 1e-6
    
    final_displacements = displacements / total_weights[:, np.newaxis]
    source_pixels = dest_pixels + final_displacements
    
    # Bilinear interpolation sampling colors
    x_src = source_pixels[:, 0]
    y_src = source_pixels[:, 1]
    
    # Boundary check
    x_src = np.clip(x_src, 0, width - 1)
    y_src = np.clip(y_src, 0, height - 1)
    
    # Use floor and ceil for bilinear interpolation
    x_floor, y_floor = np.floor(x_src).astype(int), np.floor(y_src).astype(int)
    x_ceil, y_ceil = np.ceil(x_src).astype(int), np.ceil(y_src).astype(int)
    
    x_ceil = np.clip(x_ceil, 0, width - 1)
    y_ceil = np.clip(y_ceil, 0, height - 1)

    # Calculate interpolation weights
    dx = x_src - x_floor
    dy = y_src - y_floor

    # Sample the colors of the four points
    c00 = image[y_floor, x_floor]
    c01 = image[y_floor, x_ceil]
    c10 = image[y_ceil, x_floor]
    c11 = image[y_ceil, x_ceil]

    # Bilinear interpolation
    c0 = c00 * (1 - dx)[:, np.newaxis] + c01 * dx[:, np.newaxis]
    c1 = c10 * (1 - dx)[:, np.newaxis] + c11 * dx[:, np.newaxis]
    color = c0 * (1 - dy)[:, np.newaxis] + c1 * dy[:, np.newaxis]

    warped_img[y_coords.ravel(), x_coords.ravel()] = color.astype(np.uint8)
    
    return warped_img


def morph(img_src, img_dst, lines_src, lines_dst, t):
    """
    Performs morphing between two images at time t.
    """
    # 1. Interpolate lines
    lines_t = []
    for ls, ld in zip(lines_src, lines_dst):
        p1_t = (1 - t) * ls.P + t * ld.P
        p2_t = (1 - t) * ls.Q + t * ld.Q
        lines_t.append(Line(p1_t, p2_t))
    
    # 2. Distort the source and destination images to the intermediate shape
    warped_src = warp_image(img_src, lines_src, lines_t)
    warped_dst = warp_image(img_dst, lines_dst, lines_t)
    
    # 3. Cross dissolve
    morphed_img = ((1 - t) * warped_src.astype(float) + t * warped_dst.astype(float)).astype(np.uint8)
    
    return morphed_img

# --- Usage Example ---
# 1. Load images (make sure they have the same size)
# img1 = np.array(Image.open('source.jpg'))
# img2 = np.array(Image.open('destination.jpg'))

# 2. Define feature lines (manually or using a UI tool)
# lines1 = [Line((100, 120), (150, 125)), Line((200, 130), (250, 128))] # Example source lines
# lines2 = [Line((110, 115), (160, 118)), Line((210, 125), (260, 122))] # Example target lines

# 3. Generate Morphing sequence
# num_frames = 30
# for i in range(num_frames):
#     t = i / (num_frames - 1)
#     morphed_frame = morph(img1, img2, lines1, lines2, t)
#     Image.fromarray(morphed_frame).save(f'frame_{i:03d}.png')
```

**Code Interpretation**:

*   We use NumPy for vectorized calculations, which greatly improves efficiency. The code calculates the u, v, displacement, and weights of all pixels at once.
*   `warp_image` function implements the core inverse mapping and distortion process.
*   `morph` function encapsulates the entire process: interpolate lines, distort the source and destination images to the intermediate shape, and finally cross dissolve.
*   The code includes detailed bilinear interpolation implementation, which is key to maintaining image quality.


# IV. Advanced Topics and Extensions

## **4.1 Transition Control**

In our algorithm, the parameter $t$ changes linearly. This means that the deformation and color dissolution speed are constant. We can introduce a **non-uniform transition function** $f(t)$ to control the rhythm, creating more dramatic effects.

For example, using an "ease-in-ease-out" function, the deformation can be slower at the beginning and end, and faster in the middle. $t_{\text{effective}} = f(t) = t*t*(3 - 2*t)$ can be used to replace all $t$ in the algorithm with $t_{\text{effective}}$.

## **4.2 Other Morphing Methods**

*   **Radial Basis Functions (RBF)**: This method is based on point pairs (not line pairs) of corresponding relationships. Each point pair defines a radial basis function (Gaussian function or thin plate spline function), and the deformation of the entire space is a linear combination of all these functions. It is well-suited for handling dispersed, unstructured features.
*   **Moving Least Squares (MLS)**: This is a more powerful and flexible deformation technique. For each point in the image, MLS calculates a local best affine transformation (or similarity transformation, rigid transformation), instead of calculating a global weighted displacement like Beier-Neely. This makes MLS perform better in maintaining image local structure, resulting in less distortion.

## **4.3 View Morphing**

Traditional 2D Morphing has a fatal weakness that it does not consider the three-dimensional space and perspective. When the shooting viewpoints of the two images are different, directly performing 2D Morphing will produce severe geometric distortions. For example, a straight line in three-dimensional space may curve during the Morphing process.

**Seitz and Dyer proposed View Morphing** in 1996 to solve this problem. It aims to generate intermediate view points that are physically correct.

**Core idea**: Convert the complex perspective transformation problem into a simple parallel view interpolation problem through reprojection (reprojection).

**Three-step algorithm**:

1.  **Pre-warp**: Utilize the camera projection matrix of the two images (or the basic matrix calculated through feature points), transform the source image $I_S$ and the target image $I_D$ respectively, project them to a common plane parallel to the camera baseline, and obtain $I_S'$ and $I_D'$. In these "corrected" images, corresponding points are located on the same horizontal scan line.
2.  **Morphing**: Interpolate the corrected images $I_S'$ and $I_D'$ simply or perform 2D Morphing. Since the geometry has been aligned, this process will not produce distortion, but only simple pixel mixing and displacement.
3.  **Post-warp**: Project the generated intermediate frame $I_t'$ from the common plane to the final intermediate view point through the intermediate camera matrix, and obtain the final result $I_t$.

View Morphing can generate realistic three-dimensional rotation effects, but it requires additional information about camera geometry.

**Anti-aliasing**:

*   **Folds**: When a surface visible in the source view is occluded in the target view.
*   **Holes**: When a surface occluded in the source view becomes visible in the target view. These issues can be resolved by using depth information (or its approximate value — **disparity**). Through the idea of Z-Buffering, we can determine which surface should be in front, thus correctly handling occlusion relationships.


# Conclusion

## **Summary**

*   **Core Principle**: Image distortion + cross dissolve.
*   **Main Algorithms**:
    *   **Based on Grid**: Intuitive but control is complex.
    *   **Based on Feature Line Pairs (Beier-Neely)**: Control is intuitive, results are outstanding, and it is the industry standard. We delved into its mathematical derivation and code implementation.
*   **Advanced Extensions**:
    *   **Transition Control**: Achieve non-linear animation rhythms.
    *   **View Morphing**: Through the combination of camera geometry, generate physically correct intermediate views, solving the perspective distortion problem of traditional 2D Morphing.

## **Limitations**

Despite the power of 2D morphing, it has its inherent limitations:

*   **Lack of Three-Dimensional Information**: Unable to handle changes in lighting, shadow, and material.
*   **Dependent on Manual Annotation**: Whether it is grid or feature line pairs, a large amount of manual interaction is required to specify corresponding relationships.
*   **Sensitive to Large-Scale Transformations**: When the source and target are significantly different, it may produce unnatural "ghosts" or tearing effects.

## **Prospects: The Age of Deep Learning**

Recently, with the development of deep learning, especially generative adversarial networks (GANs), image-to-image conversion has entered a new era. Models like StyleGAN and CycleGAN can learn the high-dimensional feature distribution of datasets, achieving astonishing image conversion effects, such as aging faces, gender conversion, style transfer, etc.

These deep learning methods compared to the classic morphing we learn today:

*   **Advantages**: Usually do not need to manually specify features, can generate more realistic and detailed textures, and can even "imagine" the hidden parts.
*   **Disadvantages**: Controllability is poor ("black box" operation), requires a large amount of training data, and may produce results unrelated to the input and unpredictable.

Classic morphing technology, with its **precise controllability, intuitive manipulation of geometry, and independence from large-scale training data**, is still an indispensable tool in many professional fields (such as movie special effects, medical imaging, scientific visualization).

Mastering these classic algorithms not only allows you to understand the foundation of computer graphics, but also helps you understand and develop more advanced generative models in the future.
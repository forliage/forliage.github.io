---
title: "Mathematical foundations, Keyframe technology and Velocity Control"
lecture: 1
course: "anim"
date: 2026-02-08
---

# Mathematical foundations, Keyframe technology and Velocity Control

## I.Transsformation and rotation representation

In images, we can use equations to define the movement of points. But objects in animation are not just points; they have size, shape, and orientation.
We need a set of mathematical tools to describe changes in these properties, and that's where **geometric transformations** come in.

### 1.1.Basic Transformations and Homogeneous Coordinates

The most basic three-dimensional transformations include:
- Translate: 
$$
\mathbf{p'} = \mathbf{p}+\mathbf{T}
$$
- Scale:
$$
\mathbf{p'} = \mathbf{Sp}
$$
- Rotate: 
$$
\mathbf{p'} = \mathbf{Rp}
$$

Here, translation is addition, while scaling and rotation are multiplication, resulting in inconsistent forms. To unify them within the framework of matrix multiplication, we introduce **homogeneous coordinates**. We expand a three-dimensional point $(x,y,z)$ into a four-dimensional vector $[x,y,z,1]^T$.

In this way, all transformations can be represented by a $4\times 4$ matrix:
- Translation Matrix: 
$$
\begin{bmatrix}1&0&0&t_x \\ 0&1&0&t_y \\ 0&0&1&t_z \\ 0&0&0&1\end{bmatrix}
$$
- Scaling Matrix: 
$$
\begin{bmatrix}s_x&0&0&0\\0&s_y&0&0\\0&0&s_z&0\\0&0&0&1\end{bmatrix}
$$
- Rotating Matrix (Around the Z-axis): 
$$
\begin{bmatrix}\cos\theta&-\sin\theta&0&0\\\sin\theta&\cos\theta&0&0\\0&0&1&0\\0&0&0&1\end{bmatrix}
$$

The greatest advantage of using homogeneous coordinates in **concatenation of transformations** is that a series of complex transformations (e.g., scaling, rotation, and translation) can be precombined into a single composite transormation matrix:$\mathbf{M}=\mathbf{M}_{\text{translate}}\mathbf{M}_{\text{rotate}}\mathbf{M}_{\text{scale}}$. Then this $\mathbf{M}$ is used to transform all the vertices of the model, greatly improving efficiency.

Note: Matrix multiplication is not commutative; the order of transformations is crucial!

### 1.2.Challenges of Rotation Representation

Interpolation for translation and scaling is straightforward; it simply involves linear interpolation of the translation vector and the scaling factor.
However, interpolation for rotation is a very tricky problem.

**1.Rotation Matrix.**

The rotation matrix is a $3\times 3$ orthogonal matrix (the row and column vectors are all unit vectors and mutually orthogonal), with a determinant of 1.
- **Advantage**: It is mathematically rigorous and has no singularity.
- **Disadvantage**: **Redundancy**: Representing a rotation with only 3 degree of freedom using 9 numbers; **Interpolation difficulties**: Directly performing linear interpolation on the 9 elements of the two rotation matrices results in an intermediate matrix that is no longer orthogonal, which introduces unwanted scaling and skewing distortions.

Example: Rotate from $90^\circ$ to $-90^\circ$ around the Z-axis:

$$
 \mathbf{R}(90^\circ)=\begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix}\qquad \mathbf{R}(-90^\circ)=\begin{bmatrix}0&1&0\\-1&0&0\\0&0&1\end{bmatrix}
$$

Linear interpolation to half (weight 0.5):
$$
\mathbf{R}_{\text{half}}=0.5\mathbf{R}(90^\circ)+0.5\mathbf{R}(-90^\circ)=\begin{bmatrix}0&0&0\\0&0&0\\0&0&1\end{bmatrix}
$$
This is a strange matrix that flattens objects onto the Z-axis, which is not the $0^\circ$ rotation (identity matrix) we want!

**2.Euler Angles / Fixed Angles.**

This is the most intuitive representation. It decomposes any three-dimensional rotation into three consecutive rotations around three coordinate axes (e.g., X, Y, Z).
- Fixed Angle: Each rotation revolves around an axis of the world coordinate system.
- Euler angles: The first rotation is around the world coordinate axis, and subsequent rotations are around the axes of the object's own (local) coordinate system. An interesting property is that a rotation with Euler angles in the order of (X, Y, Z) is equivalent to a rotation with fixed angles in the order of (Z, Y, X). As showm below:
![alt](./images/img1.png)
- Advantages: **Compact**: Uses only 3 numbers $(\theta_x, \theta_y, \theta_z)$; **Intuitive**: Easy for humans to understand and edit.
- Disadvantages: **Non-unique/non-uniform interpolation paths**: Linear interpolation of three angles often results in rotational paths that are not the shortest and have non-uniform angular velocities; **Gimbal Lock**: This is a fatal flaw of Euler angles. When the middle rotation axis (e.g., the Y-axis) rotates $90^\circ$, the first axis (X-axis) and the third axis (Z-axis) coincide. At this point, the system loses a rotational degree of freedom. Regardless of whether we change the angle of the X-axis or Z-axis, only rotation around the same world axis will occur. This leads to sudden, unnatural, rapid flips in animation.As showm below:
![Gimbal Lock](./images/img3.gif)

**3.Axis-Angle.**

According to Euler's rotation theorem, any three-dimensional rotation can be represented as rotating about an axis $\mathbf{a}$ in any direction by an angle $\theta$.
- Advantages: **Intuitive interpolation**: Rotation axes and rotation angles can be interpolated between two attitudes; **no gimbal lock-up**.
- Disadvantages: **Difficult to chain together**: Two axis-angle representations of rotation, the axes and angles of the composite rotation do not have simple calculation formulas; **Not unique representation**: $(\mathbf{a},\theta)$ and $(-\mathbf{a},-\theta)$ represent the same rotation.

### 1.3.Ultimate Solution: Quaternions

A quaternion is a mathematical concept created by the Irish mathematician William Nouwen Hamilton in 1843. It is usually denoted by $\mathbb {H}$.

From a definite perspective, a quaternion is a non-commutative extension of complex numbers. If the set of quaternions is considered as a multidimensional real number space, then a quaternion represents a four-dimensional space, while complex numbers represent a two-dimensional space.

As a coordinate representation for describing real space, quaternions were created based on complex numbers and are expressed in the form $a + bi + cj + dk$ to indicate the location of a point in space. $i$, $j$, and $k$ participate in operations as special imaginary units, and follow these operational rules: $i^2 = j^2 = k^2 = -1$.

The geometric meaning of $i$, $j$, and $k$ can be understood as a rotation. The $i$ rotation represents a rotation from the positive X-axis to the positive Y-axis in the plane where the X and Y axes intersect; the $j$ rotation represents a rotation from the positive Z-axis to the positive X-axis in the plane where the Z and X axes intersect; the $k$ rotation represents a rotation from the positive Y-axis to the positive Z-axis in the plane where the Y and Z axes intersect; and $-i$, $-j$, and $-k$ represent the opposite rotations of $i$, $j$, and $k$, respectively.

**Definition**

Quaternions are all composed of real numbers plus three elements $i,j,k$, and they have the following relationship:
$$
i^2=j^2=k^2=ijk=-1
$$

Each quaternion is a linear combination of $1,i,j,k$, and can generally be represented as $a+bi+cj+dk$.

To add two quaternions, simply add their similar coefficients, just like with complex numbers. The multiplication rule follows the multiplication table below:

$$
\begin{array}{c|cccc}
\times & 1 & i & j & k \\
\hline
1 & 1 & i & j & k \\
i & i & -1 & k & -j \\
j & j & -k & -1 & i \\
k & k & j & -i & -1 \\
\end{array}
$$

The multiplication of the identity elements of quaternions forms an 8th-order quaternion group, $Q_8$.

**Property**

Unlike real or complex numbers, quaternions do not obey the anticommutative law of multiplication and are therefore noncommutative, for example:
$$
ij=k, ji=-k\\
jk=i, kj=-i\\
ki=j, ik=-j
$$

Quaternions are an example of division rings. Except for the lack of a commutative property for multiplication, division rings are analogous to fields. Specifically, the associative property of multiplication still applies, and each non-zero element has a unique inverse.

Quaternions form a four-dimensional associative algebra (actually a division algebra) over real numbers, including complex numbers, but not forming an associative algebra with complex numbers. Quaternions (as well as real and complex numbers) are simply finite-dimensional associative division algebras of real numbers.

The noncommutativity of quaternions often leads to unexpected results. For example, an n-order polynomial of a quaternion can have more than $n$ distinct roots. For instance, the equation $h^2+1=0$ has infinitely many solutions. If any real number satisfies $b^2+c^2+d^2=1$, then $h = bi + cj + dk$ is a solution.

The conjugate of a quaternion $h = a + bi + cj + dk$ is defined as: 
$$
h^*=a-bi-cj-dk
$$

Its absolute value is a non-negative real number, defined as: 
$$
|h|=\sqrt{h\cdot h^*}=\sqrt{a^2+b^2+c^2+d^2}
$$

Note that $(hk)^*=k^*h^*$ is generally not equal to $h^*k^*$

The multiplicative inverse of a quaternion can be calculated using $h^{-1}=\frac{h^*}{|h|^2}$

By using the distance function $d(h,k)=|h-k|$, quaternions can become a metric space homeomorphic to $\mathbb{R}^4$ and have continuous arithmetic operations. Furthermore, for all quaternions $h$ and $k$, $|hk| = |h||k|$. Modulo the absolute value, quaternions can form a real Banach space.

## II.Keyframe interpolation and Velocity Control

## Conclusion
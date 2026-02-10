---
title: "Mathematical foundations, Keyframe technology and Velocity Control"
lecture: 1
course: "anim"
date: 2026-02-08
---

# I.Transsformation and rotation representation

In images, we can use equations to define the movement of points. But objects in animation are not just points; they have size, shape, and orientation.
We need a set of mathematical tools to describe changes in these properties, and that's where **geometric transformations** come in.

## 1.1.Basic Transformations and Homogeneous Coordinates

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

## 1.2.Challenges of Rotation Representation

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

## 1.3.Ultimate Solution: Quaternions

A quaternion is a mathematical concept created by the Irish mathematician William Nouwen Hamilton in 1843. It is usually denoted by $\mathbb {H}$.

From a definite perspective, a quaternion is a non-commutative extension of complex numbers. If the set of quaternions is considered as a multidimensional real number space, then a quaternion represents a four-dimensional space, while complex numbers represent a two-dimensional space.

As a coordinate representation for describing real space, quaternions were created based on complex numbers and are expressed in the form $a + bi + cj + dk$ to indicate the location of a point in space. $i$, $j$, and $k$ participate in operations as special imaginary units, and follow these operational rules: $i^2 = j^2 = k^2 = -1$.

The geometric meaning of $i$, $j$, and $k$ can be understood as a rotation. The $i$ rotation represents a rotation from the positive X-axis to the positive Y-axis in the plane where the X and Y axes intersect; the $j$ rotation represents a rotation from the positive Z-axis to the positive X-axis in the plane where the Z and X axes intersect; the $k$ rotation represents a rotation from the positive Y-axis to the positive Z-axis in the plane where the Y and Z axes intersect; and $-i$, $-j$, and $-k$ represent the opposite rotations of $i$, $j$, and $k$, respectively.

### 1.3.1.Definition

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

### 1.3.2.Property

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

### 1.3.3.Group Rotation

The conjugate action of the multiplicative group of nonzero quaternions on the part of $\mathbf{R}^3$ where the real part is zero can realize a rotation. The conjugate action of a unit quaternion (a quaternion with an absolute value of 1) with a real part of $\cos t$ is a rotation of angle $2t$, with the axis of rotation being the direction of the imaginary part. The advantages of quaternions are:
- The expression has no singularities (compared to representations such as Euler angles)
- It is more concise (and faster) than matrices
- A pair of unit quaternions can represent a rotation in four-dimensional space.

The set of all unit quaternions forms a three-dimensional sphere $S^3$ and a group (a Lie group) under multiplication. $S^3$ is a double superposition of the group $SO(3,R)$ of real orthogonal $3\times 3$ orthogonal matrices with determinant 1, since every two unit quaternions correspond to a rotation by the above relation. The group $S^3$ is isomorphic to $SU(2)$, which is the group of complex unitary $2\times 2$ matrices with determinant 1. Let $A$ be the set of quaternions of the form $a + bi + cj + dk$, where $a, b, c, d$ are either integers or rational numbers with odd numerators and denominators of 2. The set $A$ is a ring and a lattice. There are 24 quaternions in the ring, which are the vertices of a regular 24-cell structure with the Schleifli notation {3,4,3}.

### 1.3.4.Application

**Rotation using quaternions**: To rotate a 3D vector $\mathbf{p}$ using a unit quaternion $q$, first promote $\mathbf{p}$ to a pure quaternion $p=[0,\mathbf{p}]$, then perform the following operation:
$$
\mathbf{p'}=qpq^{-1}
$$

The scalar part of the result $\mathbf{p'}$ will be 0, and its vector part is the rotated vector.

Proof Summary: This multiplication is very complex after expansion, but its core idea is that the operation preserves the magnitude of the vector, and the scalar part of the final result is 0, therefore it is a pure rotation.

**Composite Rotation**: Rotating first with $q_1$, then with $q_2$, is equivalent to performing a rotation using the composite quaternion $q_{\text{comp}} = q_2q_1$. This is much simpler than serial axis-angle!

**Quaternion Interpolation**: **SLERP** The biggest advantage of quaternions is that they provide a smooth, singularity-free interpolation method. A unit quaternion can be viewed as a point on a unit hypersphere in four-dimensional space. Two orientations correspond to two points on the hypersphere.
- **Linear Interpolation (Lerp)**: Directly interpolating the four components of the quaternion linearly, then normalizing. Geometrically, this is equivalent to interpolating on a chord connecting two points, resulting in uneven speed.
- **Spherical Linear Interpolation (SLERP)**: This is the correct interpolation method. It performs uniform interpolation on a great circle connecting two points.

Given two unit quaternions $q_1, q_2$ and interpolation parameter $u\in[0,1]$, the SLERP formula is:
$$
\text{slerp}(q_1,q_2,u)=\frac{\sin(1-u)\Omega}{\sin\Omega}q_1+\frac{\sin u\Omega}{\sin\Omega}q_2
$$
where $\Omega=\arccos(q_1\cdot q_2)$ is the angle between the two quaternions in four-dimensional space.

**Practical Tip: "Shortest Path"** Since $q$ and $-q$ represent the same rotation, the interpolation from $q_1$ to $q_2$ has two paths (the shorter arc and the longer arc). We usually want to take the shortest path. This can be determined by checking the dot product $q_1\cdot q_2$. If $q_1\cdot q_2 < 0$, it means the angle is greater than 90 degrees, and we should interpolate to $-q_2$ instead of $q_2$ to ensure that the shorter arc is taken.

```cpp
// C++ Quaternion Slerp pseudocode
Quaternion Quaternion::slerp(const Quaternion& q1, const Quaternion& q2, float u) {
    Quaternion q2_temp = q2;
    float dot = q1.dot(q2);

    // If the dot product is negative, use the opposite of q2 to take the shortest path
    if (dot < 0.0f) {
        q2_temp = -q2;
        dot = -dot;
    }

    // Prevent the dot product from being slightly greater than 1 due to floating-point error
    if (dot > 0.9995f) {
        // Angle is very small, directly linearly interpolate and normalize to avoid division by 0
        return normalize(lerp(q1, q2_temp, u));
    }

    float theta_0 = acos(dot);        // Angle between q1 and q2
    float theta = theta_0 * u;        // Interpolated angle
    float sin_theta = sin(theta);
    float sin_theta_0 = sin(theta_0);

    float s1 = cos(theta) - dot * sin_theta / sin_theta_0;
    float s2 = sin_theta / sin_theta_0;

    return (q1 * s1) + (q2_temp * s2);
}
```

Conclusion: In animation systems, the common practice is as follows: externally, expose intuitive Euler angles to animators for editing; internally, convert Euler angles into quaternions for storage, interpolation, and computation; finally, when needed for application to vertices, convert the quaternions back to rotation matrices. Quaternions are the best standard for internally representing rotations!

# II.Keyframe interpolation and Velocity Control

# Conclusion
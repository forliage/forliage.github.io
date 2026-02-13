---
title: "Joint (character) animation (I)"
lecture: 8
course: "anim"
date: 2026-02-13
---

# I.The charm and applications of joint animation

In the world of computer animation, if we want visuals to be more than just cold, moving geometric shapes, but full of life, we need to introduce **characters**—such as people, animals, or even moving robots. These characters can instantly bring the entire scene to life. **Joint animation** is the core technology that enables all of this; it's the cornerstone that gives these characters life and vitality.

## **1.1 Ubiquitous Applications**

Joint animation technology permeates almost every aspect of our digital lives, and its depth and breadth of application may far exceed most people's imagination.

**In games, it's the soul of the character...**

Whether it's Cristiano Ronaldo's iconic celebration in FIFA, the powerful swings of orc warriors in World of Warcraft, the dazzling skills unleashed by heroes in League of Legends, or the fluid dunks of basketball superstars in the NBA 2K series—all these lifelike character movements are driven by complex joint animation systems. It allows virtual characters to simulate complex movements in the real world.

**In animation, it can breathe life into everything...**

The magic of this technology lies in its applicability beyond living characters. We can even construct a virtual "skeleton" inside inanimate, soft objects (like a flour sack). By controlling this skeleton, we can make it behave like a thinking, emotional character, performing anthropomorphic movements such as squeezing, stretching, and jumping. This is a key technique for animators to give objects "personality."

**In film, it's the creator of visual spectacles...**

The film industry is the epitome of articulated animation technology. From the lifelike prehistoric beasts in early films like *Dinosaurs*, to the leisurely swimming turtles in Pixar's classic *Finding Nemo*, from the powerful and emotional action of the epic *King Kong*, to the octopus-tentacle-wielding, strangely moving Davy Jones in *Pirates of the Caribbean*, and the incredibly complex, fluidly transforming robots in *Transformers*—every movement, every posture, relies on powerful articulated animation technology.

In particular, with the development of motion capture and performance capture technologies, such as in "Polar Express" and later "Alita: Battle Angel," we have been able to map the detailed facial expressions and movement data of live actors onto the joints and skeletons of virtual characters in real time, thereby creating an unprecedented sense of realism and immersion.

# II.Core Models and Concepts of Joint Animation

Now that we've experienced the power of joint animation, how is it technically achieved? We need to grasp a few crucial core concepts.

## 2.1 Model and Skeleton

A movable character typically consists of two parts:

1. **Model (Skin/Mesh)**: This is what we ultimately see on the screen, the character's "skin" or "shell," usually a polygonal mesh.

2. **Skeleton (Rig)**: This is a hidden driving structure within the model, composed of a series of joints and links (often called bones).

Animators don't directly drag each vertex on the model; instead, they indirectly move the entire model by controlling the movement of the bones. This is similar to how the skeleton drives the muscles and skin in human motion.

## 2.2 Model Hierarchy vs. Motion Hierarchy

To effectively organize and control the skeleton, we introduce the concept of **Hierarchy**.

* **Usually described using relative motion**: Imagine your body. Your forearm moves relative to your upper arm, your upper arm moves relative to your shoulder, and your shoulder moves relative to your torso. We don't calculate the absolute position of each part in the world, but rather describe this "child relative to parent" motion relationship.

* **The Composition of Motion Levels**:

* **Object Level + Relative Motion = Motion Level**.

* **Links**: Represent different parts of the body, such as the upper arm, forearm, and hand.

* **Joints**: Connect these links and define how they move relative to each other.

* **Motion is usually restricted**: For example, your knee can only bend in one direction, while your shoulder can rotate more freely. These restrictions are inherent properties of joints, which we call **Degrees of Freedom (DOF)**.

## 2.3 Kinematics vs. Dynamics

In the field of animation, we often hear two terms: kinematics and dynamics. Their distinction is crucial:

* **Kinematics**: Studies the **geometric properties of motion**, i.e., "how to move". It only cares about position, velocity, acceleration, etc., without considering the forces that cause these movements. The core question of this course is: "How do we animate links by setting position parameters over time?" This is the kinematics problem.

* **Dynamics:** This studies the **forces that cause motion**, i.e., "why the object moves." It considers physical quantities such as mass, inertia, force, and torque, and is commonly used in physics simulations, such as cloth movement and collision effects.

**Today, our focus will be entirely on kinematics.**

## 2.4 Articulated Model

A joint can be abstracted as a **tree structure** composed of links and joints.

* **Represent a joint as a series of links connected by joints.**

In this tree structure:

* **Node:** Represents a part of an object (e.g., upper arm, head).

* **Edge:** Represents the joint connecting two parts and the relative transformations between them.

* **Root Node:** Typically the core part of a character (e.g., the pelvis), its position and orientation are defined in world coordinates.

## 2.5 Degrees of Freedom (DOF)

**Degrees of freedom (DOF)** refer to the minimum number of independent coordinates required to fully define the motion of an object.

* A rigid body moving freely in three-dimensional space has **6 degrees of freedom (6 DOF)**:

* 3 Translational DOF: Translation along the X, Y, and Z axes.

* 3 Rotational DOF: Rotation about the X, Y, and Z axes (often called Pitch, Yaw, Roll).

For joint models:

* **Single-DOF Joint:** Allows movement in only one direction.

* **Revolute Joint:** Such as an elbow joint, allowing only rotation.

* **Prismatic Joint:** Such as a piston, allowing only translation.

* **Complex Joints**: Possess multiple degrees of freedom.

* **2-DOF Joints**: Such as the wrist, which can bend up and down and left and right.

* **3-DOF Joints**:

* **Ball and Socket Joints**: Such as the shoulder joint, which can rotate freely in three axes.

* **Gimbals**: Common 3-DOF rotary joints in mechanics.

An important concept is that a complex joint with n degrees of freedom can be viewed as n joints with 1 degree of freedom connected by n-1 links of length 0. This provides significant convenience in mathematical and programming implementation.

**Degrees of Freedom in Human Models**

A typical human model has the following degree of freedom distribution:

* **Root Node**: Located in the pelvis, possessing all 6 degrees of freedom (3 translations + 3 rotations), determining the character's overall position and orientation in the world.

* **Other Joints**: Typically only rotational joints are used.

* **Shoulder Joint:** 3 DOF

* **Wrist Joint:** 2 DOF

* **Knee Joint:** 1 DOF

## 2.6 Data Structures

How can we represent this hierarchical structure in code? We typically use a tree-like data structure.

* **Hierarchical Links can be represented by a tree structure:**

* **Root Node:** Corresponds to the root of the object; its transformations (position and orientation) are given in the **world coordinate system**.

* **Child Nodes:** Their transformations are represented **relative to their parent node**.

* **Leaf Nodes:** The terminal nodes of the tree, such as fingertips or toes.

**Code Implementation Ideas**

We can define a `Joint` or `Node` struct/class:

```cpp
#include <vector>
                #include <string>
                #include "glm/glm.hpp" // Use the GLM library to process vectors and matrices.

                struct Joint {
                    std::string name;

                    // Relative transformation information from the parent joint
                    glm::vec3 offset;       // Translation from the parent joint to this joint (length and direction of the link)
                    glm::vec3 rotation;     // Current rotation angle of this joint (e.g., Euler angles)

                    // Hierarchical structure
                    Joint* parent;
                    std::vector<Joint*> children;

                    // Calculated transformation matrix
                    glm::mat4 localTransform; // Local transformation matrix
                    glm::mat4 worldTransform; // World transformation matrix

                    // Constructor and methods
                    Joint(const std::string& name, const glm::vec3& offset)
                        : name(name), offset(offset), rotation(0.0f), parent(nullptr) {}

                    void AddChild(Joint* child) {
                        children.push_back(child);
                        child->parent = this;
                    }
                    
                    // Methods for updating the transformation matrix...
                };
                
```

# III.Forward Kinematics

Now, let's move on to our first core technical point today: **Forward Kinematics**.

**Definition:** Animators directly specify the motion parameters (usually rotation angles) for each joint, and the system calculates the final position and orientation of each link and end effector (such as a hand or foot) in world space.

This is a "root-to-leaf" calculation process. You set the cause (joint angles) and then solve for the effect (final position).

## 3.1 Mathematical Principles of Hierarchical Transformations

The core of Forward Kinematics (FK) is matrix concatenation. The world coordinates of a child node are the result of multiplying the world coordinate transformation of its parent node by the cumulative local coordinate transformation of itself.

Suppose we have a vertex $V$ defined in the local coordinate system of a link $\text{Link\_k}$. To obtain its position $V'$ in the world coordinate system, we need to left-multiply all the transformation matrices from the root node to $\text{Link\_k}$.

$$
V' = M_{\text{world}} \cdot V = (M_0 \cdot M_1 \cdot ... \cdot M_k) \cdot V
$$

Where $M_i$ is the local transformation matrix of the $i$th node relative to its parent node. It typically consists of a translation matrix $T_i$ and a rotation matrix $R_i(\theta_i)$.

$$
M_i = T_i \cdot R_i(\theta_i)
$$

**Let's understand this process through an example:**

Consider a two-segment arm $\text{Link}_0 \to \text{Link}_1 \to \text{Link}_{1,1}$.

* $V_0$: A vertex on $\text{Link}_0$.
* $V_1$: A vertex on $\text{Link}_1$.
* $V_{1,1}$: A vertex on $\text{Link}_{1,1}$.
* $T_0$: The transformation of $\text{Link}_0$ (the root) in world coordinates.
* $T_1$: The translation of $\text{Link}_1$ relative to $\text{Link}_0$ (i.e., the length of $\text{Link}_0$).
* $R_1(\theta_1)$: The rotation of $\text{Link}_1$ about its joints.
* $T_{1,1}$: The translation of $\text{Link}_{1,1}$ relative to $\text{Link}_1$.
* $R_{1,1}(\theta_{1,1})$: The rotation of $\text{Link}_{1,1}$ about its joints.

Therefore, the positions of these vertices in the world coordinate system are:

* $V_0' = T_0 \cdot V_0$
* $V_1' = T_0 \cdot (T_1 \cdot R_1(\theta_1) \cdot V_1) = (T_0 T_1 R_1(\theta_1)) \cdot V_1$
* $V_{1,1}' = T_0 \cdot T_1 R_1(\theta_1) \cdot (T_{1,1} R_{1,1}(\theta_{1,1}) \cdot V_{1,1}) = (T_0 T_1 R_1(\theta_1) T_{1,1} R_{1,1}(\theta_{1,1})) \cdot V_{1,1}$

**Note the order of the matrices!** The transformations are applied sequentially from parent to child.

## 3.2 Implementation of FK: Tree Traversal

The process of calculating the FK of the entire skeletal tree is essentially performing a **Depth-First Search (DFS)** on the tree.

**Algorithm Pseudocode:**

```cpp
// Use a matrix stack to store the cumulative transformations of the parent node

MatrixStack matrix_stack;

matrix_stack.push(IdentityMatrix); // Initialized as an identity matrix

function TraverseAndCalculateFK(joint, parentWorldTransform):

// 1. Calculate the local transformation matrix of the current joint

// localTransform = TranslationMatrix(joint.offset) * RotationMatrix(joint.rotation)

joint.calculateLocalTransform();

// 2. Calculate the world transformation matrix of the current joint

// worldTransform = parentWorldTransform * localTransform

joint.worldTransform = parentWorldTransform * joint.localTransform;

// 3. Recursively traverse all child joints

for each child in joint.children:

TraverseAndCalculateFK(child, joint.worldTransform);

// Called starting from the root node

TraverseAndCalculateFK(root_joint, IdentityMatrix);

```

## **3.3 FK Example: A Three-Link Planar Arm**

Let's calculate a concrete example. Assume a three-link planar robotic arm, with the base at the origin.

* Link lengths: $l_1, l_2, l_3$

* Joint angles: $\theta_1, \theta_2, \theta_3$ (relative to the horizontal axis or the previous link)

What are the coordinates $(x, y)$ of the end effector?

We can obtain the following coordinates by summing the projections of each link onto the X and Y axes:

* **Coordinates of the end of link 1:** $(l_1 \cos(\theta_1), l_1 \sin(\theta_1))$

* **Relative orientation of link 2:** $\theta_1 + \theta_2'$ (Here, we assume $\theta_2$ is a relative angle. For simplicity, we define all angles relative to a fixed axis, such as the horizontal axis, as shown in the diagram above).

* According to the diagram, $\theta_2$ is the angle with a downward-extending dashed line. If the horizontal line is 0 degrees and the angle of link 1 is $\theta_1$, then the angle between link 2 and the horizontal line is $\theta_1 - (180 - \theta_2')$, which is quite complex.

Let's redefine the angles to make the problem clearer: Let $\theta_1, \theta_2, \theta_3$ be the angles between the link and the positive horizontal direction.

* The final coordinates $(x, y)$ are the sum of the components of all links in the x and y directions: 
$$
x = l_1 \cos(\theta_1) + l_2 \cos(\theta_2) + l_3 \cos(\theta_3)
$$ 
$$
y = l_1 \sin(\theta_1) + l_2 \sin(\theta_2) + l_3 \sin(\theta_3)
$$

**If the angles are relative** (i.e., $\theta_i$ is the angle of $\text{link}_i$ relative to $\text{link}_{i-1}$), then:

* The world angle of $\text{link}_1$ is $\theta_1$
* The world angle of $\text{link}_2$ is $\theta_1 + \theta_2$
* The world angle of $\text{link}_3$ is $\theta_1 + \theta_2 + \theta_3$

* The final coordinates are: 
$$
x = l_1 \cos(\theta_1) + l_2 \cos(\theta_1 + \theta_2) + l_3 \cos(\theta_1 + \theta_2 + \theta_3)
$$ 
$$
y = l_1 \sin(\theta_1) + l_2 \sin(\theta_1 + \theta_2) + l_3 \cos(\theta_1 + \theta_2 + \theta_3)
$$

This is forward kinematics—**given joint parameters, solving for the end effector position**

# IV.Inverse Kinematics

Now, let's look at the other side of the problem, which is also the more powerful but also more challenging part of joint animation: **Inverse Kinematics**.

**Definition:** The animator specifies the target position and/or orientation of the end effector, and the system works backward to calculate the angles that all joints in the joint chain need to be set to achieve that target.

This is a "leaf-to-root" way of thinking. You set the effect (target position) and then work backward to find the cause (joint angles). This aligns perfectly with our intuition: when you want to pick up a cup from a table, you don't think, "My shoulder needs to turn 30 degrees, my elbow needs to bend 70 degrees..."; you only think, "Let my hand move to the position of the cup," and your brain automatically performs the IK calculations for you.

## 4.1 Challenges of IK

The IK problem is far more complex than FK because it faces several inherent challenges:

1. **Existence of Solutions**:

* **No Solution**: If the target point is beyond the reach of the robotic arm, then the IK problem has no solution.
* **Unique Solution:** This situation is extremely rare, typically occurring only when the number of joints and constraints are perfectly matched.
* **Multiple Solutions/Infinitely Many Solutions:** This is the most common case. For a simple 2D two-link arm, there are usually two possible poses to reach a point ("elbow up" and "elbow down"). For a human arm with more joints (shoulder, elbow, wrist), there are infinitely many possible poses to reach the same target point. We call this **redundancy**.

2. **Singularities**: Under certain joint configurations, the joint chain may lose one or more degrees of freedom, resulting in an inability to move in certain directions. The most common example is when your arm is fully extended, you cannot move your hand "forward" further in the extended direction. At these points, the mathematical solution to the IK problem becomes unstable or undefined.

3. **Solution Complexity**: The equations in the FK problem are direct trigonometric functions, while the IK problem requires solving a highly **nonlinear** system of trigonometric equations, which typically does not have a simple algebraic solution.

## 4.2 Methods for Solving IK Problems: Analytical Methods

For very simple joint chains (such as a 2D two-bar linkage), we can find an **analytical solution** using geometric or algebraic methods.

**Example: Analytical Solution of a 2D Two-Bar Linkage**

Let's derive the solution to this classic problem.

* Given: Link lengths $L_1, L_2$, target point coordinates $(X, Y)$.
* Solve for: Joint angles $\theta_1, \theta_2$.

**Derivation Steps:**

1. **Establish Geometric Relationships** We have a triangle formed by the origin (0,0), the elbow joint, and the target point (X,Y). The lengths of the three sides are $L_1$, $L_2$, and $D = \sqrt{X^2 + Y^2}$.


2. **Solve for $\theta_2$ using the Law of Cosines** In $\triangle ABC$, according to the Law of Cosines: 
$$
D^2 = L_1^2 + L_2^2 - 2L_1L_2 \cos(\alpha)
$$ 
where $\alpha$ is the interior angle at the elbow joint.

$$
\cos(\alpha) = \frac{L_1^2 + L_2^2 - D^2}{2L_1L_2} = \frac{L_1^2 + L_2^2 - (X^2 + Y^2)}{2L_1L_2}
$$

As can be seen from the diagram, the relationship between the joint angle $\theta_2$ and the interior angle $\alpha$ is $\alpha = 180^\circ - \theta_2$ (assuming $\theta_2$ is positive to represent outward bending). Because $\cos(180^\circ - \theta_2) = -\cos(\theta_2)$, therefore: 
$$
-\cos(\theta_2) = \frac{L_1^2 + L_2^2 - X^2 - Y^2}{2L_1L_2}
$$

Therefore, the solution to $\theta_2$ is: 
$$
\theta_2 = \pm \arccos\left(\frac{X^2 + Y^2 - L_1^2 - L_2^2}{2L_1L_2}\right)
$$

The $+$ and $-$ here correspond to the two cases: "elbow down" and "elbow up".

3. **Solving $\theta_1$** Now that we know $\theta_2$, how do we find $\theta_1$? $\theta_1$ can be seen as the sum or difference of two angles: $\theta_1 = \beta \pm \gamma$
* $\beta$ is the angle between the line connecting Base to Goal and the X-axis. $\beta = \text{atan2}(Y, X)$ (The $\text{atan2}$ function can correctly handle all quadrants).
* $gamma$ is the interior angle of triangle ABC at Base.

Using the Law of Cosines again to find $\gamma$: 
$$
L_2^2 = L_1^2 + D^2 - 2L_1D \cos(\gamma)
$$
$$
\cos(\gamma) = \frac{L_1^2 + D^2 - L_2^2}{2L_1D} = \frac{L_1^2 + X^2 + Y^2 - L_2^2}{2L_1\sqrt{X^2+Y^2}}
$$
$$
\gamma = \arccos\left(\frac{L_1^2 + X^2 + Y^2 - L_2^2}{2L_1\sqrt{X^2+Y^2}}\right)
$$

Finally, the solution to $\theta_1$ is: 
$$
\theta_1 = \text{atan2}(Y, X) \mp \gamma
$$

Note that the $\mp$ here corresponds to the $\pm$ of $\theta_2$. If $\theta_2$ is positive (elbow down), then $\theta_1$ needs to be subtracted from $\gamma$; if $\theta_2$ is negative (elbow up), then $\theta_1$ needs to be added to $\gamma$.

**Limitations of Analytical Methods**: Although this method is accurate, it is only suitable for very simple structures. For chains with more than 3 joints, or in three-dimensional space, the algebraic equations become extremely complex or even unsolvable. Therefore, in practical applications, we rely more on **numerical methods**.

## 4.3 Methods for Solving IK: Numerical Methods (Introduction)

When analytical methods fail, we use iterative numerical methods to **approximate** the solution. The basic idea of ​​these methods is:

1. Start from the current joint pose.
2. Calculate the difference (error) between the current position of the end effector and the target position.
3. Based on this error, fine-tune the angles of all joints so that the end effector moves a small step closer to the target.
4. Repeat steps 2 and 3 until the error is less than an acceptable threshold, or the maximum number of iterations is reached.

Common numerical methods for solving IK include:

* **Inverse-Jacobian method**: This is one of the most classic and popular methods. It establishes a linear relationship between joint velocities and end effector velocities using a tool called the "Jacobian matrix."
* **Cyclic Coordinate Descent (CCD)**: A simpler and more intuitive method that starts with the end effector joints and rotates each joint sequentially until it points towards the target.
* **FABRIK (Forward And Backward Reaching Inverse Kinematics)**: A geometry-based iterative method that approximates the target by stretching the joint chain forward and backward.

**Optimization-based methods:** These methods treat the IK problem as an optimization problem, aiming to minimize the distance between the end effector and the target, while potentially considering other constraints (such as joint limitations, maintaining balance, etc.).

We will explore these powerful numerical methods in detail in the next lecture, particularly the inverse Jacobian method.

# Conclusion

This lecture focuses on the core issues of character joint animation, systematically outlining the basic theoretical framework for character motion generation, starting with skeletal modeling and hierarchical structures. By introducing the concept of model-skeleton separation and hierarchical joint structures, the lecture clarifies the transmission relationship between local transformations and global posture in character motion control, providing a clear data structure foundation for the organization and management of complex character animations.

At the kinematic level, the course emphasizes the matrix cascading mechanism of forward kinematics, enabling the mapping process from joint parameters to end effector positions to have clear geometric meaning and an efficient implementation method. Simultaneously, examples such as planar robotic arms demonstrate the intuitiveness and controllability advantages of FK in character pose generation.

In the inverse kinematics section, the course delves into the multiple solutions, singularities, and nonlinear nature of the IK problem, and systematically introduces two main solution approaches: analytical methods and numerical iterative methods. Analytical methods are suitable for simple joint chains, while numerical methods provide a general and flexible solution for high-degree-of-freedom characters, laying the foundation for realistic interaction and intelligent control.

Overall, this lecture established a complete theoretical framework from skeletal structure modeling to FK and IK solutions, enabling character animation to achieve a balance between precise control and natural expression. Future developments will further integrate physical constraints, learning methods, and perception-driven mechanisms to continuously improve joint animation in terms of realism, adaptability, and automation.
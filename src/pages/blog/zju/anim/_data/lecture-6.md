---
title: "Group Animation (I)"
lecture: 6
course: "anim"
date: 2026-02-13
---

# I. The Symphony of Nature: The Ubiquitous Collective Behavior

* **Ballet of the Ocean:** A school of tropical fish swims in unison through coral reefs, forming neat formations as if guided by an invisible commander. How do they manage to avoid collisions while swimming at high speeds and coordinate their movements to evade predators?

* **Dance of Ink in the Sky:** At dusk, thousands of starlings gather in the sky, forming an ever-changing "black cloud." This phenomenon, known as "Murmuration," is a breathtaking aerial dance. There is no leader, no pre-arranged choreography; each individual simply responds to its companions, yet together they create a grand, flowing visual spectacle.

* **Vortex of Life:** In the ocean, when sardines face predator threats, they quickly gather, forming a giant, rotating "ball of fish." This is a defense mechanism, using collective strength to confuse the enemy and increase the individual's chances of survival. The presence of the diver in the picture allows us a glimpse into the spectacular scene within this vortex of life.

* **Order on the Grasslands:** On vast pastures, flocks of sheep graze leisurely or migrate. They maintain a delicate balance—gathering together for safety while leaving enough space for each other.

These examples demonstrate a powerful, bottom-up organizational force—**Collective Behavior**.

This force is not limited to the animal world. It is equally ubiquitous in our human societies:

* **The Flow of History:** Imagine the charge on an ancient battlefield or the bustling traffic hubs of a modern city. At the famous Shibuya intersection in Tokyo, when the green light turns on, crowds from all directions converge and weave, each person independently heading towards their own destination, yet collectively forming an efficient and orderly flow.

* **The Pulse of the City:** The endless stream of vehicles on highway overpasses, the resonance and transmission of emotions among crowds at large gatherings. These are macroscopic phenomena resulting from the convergence of countless individual decisions.

All of this presents us with a profound challenge: **How ​​do we use computers to simulate these complex dynamic systems composed of a large number of autonomous individuals?** This is precisely the core question that group animation seeks to answer.

# II. The Magic on Screen: Examples of Group Animation Applications

Let's shift our focus from the real world to the art of light and shadow, and see how group animation technology works its magic in movies and games.

* **A Milestone in The Lion King** Everyone surely remembers the breathtaking scene from Disney's classic animated film *The Lion King*—the wildebeest stampede. Young Simba is trapped in a canyon, and thousands of wildebeest thunder past him. In those days, making each wildebeest look unique and move naturally was a huge technological challenge. This is an early application of group animation technology, using algorithms to give each individual basic behavioral logic, thus generating this classic scene in animation history.

* **Virtual Legions in the Game World** In modern video games, whether it's the massive battles of the *Total War* series or the raid dungeons in *World of Warcraft*, large-scale NPC (non-player character) interactions have become commonplace. To make these virtual soldiers appear well-trained, capable of forming ranks, charging, flanking, and dodging, rather than a chaotic mob, a highly efficient swarm AI and animation system is needed. All of this must be calculated in real-time on the player's device, placing extremely high demands on the algorithm's efficiency.
    

# III. Core Theory: Deconstructing Collective Behavior

We've already appreciated many phenomena of collective behavior; now, let's delve into its scientific core and give it a more precise definition.

**What is Collective Behavior?**

"Swarms of animals—such as schools of fish, flocks of birds, and swarms of insects—often exhibit complex and coordinated behaviors resulting from social interactions among individuals."

This definition contains two key terms: **interaction** and **emergence**.

* **Interaction**: The root of behavior is not a command from a central brain, but rather simple interactions between an individual and its local neighbors.

* **Emergence**: This is the most fascinating and important concept in the entire field. Emergence refers to the phenomenon where **many microscopic, simple units, through interaction, spontaneously and unpredictably generate entirely new, complex properties and patterns at a macroscopic level**.

* **A single neuron has no intelligence, but billions of neurons connected together create "consciousness."** **A single water molecule has no "fluidity," but when countless water molecules gather, "waves" and "whirlpools" emerge.**

Similarly, in swarm animation, a single Boid (a virtual creature we'll discuss later) follows only three simple rules, but when thousands of Boids gather, realistic flocking bird behavior emerges.**

This embodies the philosophical idea that "the whole is greater than the sum of its parts" in science and engineering. Our goal is to design simple **local rules** to "emerge" the grand effects we desire at the global level.

This field is highly interdisciplinary, attracting leading scholars from biology, physics, computer science, and other fields. Professor Iain Couzin of the Max Planck Institute for Collective Behavior in Germany is a leading figure in this field. His team, through sophisticated experiments and mathematical modeling, has revealed the decision-making mechanisms of biological groups such as schools of fish and swarms of locusts. Their research frequently graces the covers of top journals like *Nature* and *Science*, greatly inspiring swarm simulation algorithms in computer graphics.

# IV. 群组动画的基本框架

Now, let's return to the context of computer graphics and establish a clear engineering framework for "groups."

**Definition of a Group**

In the same physical environment, a group of individuals with consistent or similar goals will exhibit significantly different behavioral patterns compared to their individual behavior when alone, due to the presence of the group.

**Hierarchical Structure of Behavior**

To manage complexity, we break down group behavior into three levels:

*   **Crowd Level**: Focuses on the highest-level macroscopic performance, such as the overall flow direction of the crowd, density changes, and the formation of formations.
*   **Group Level**: Focuses on the behavior of small groups composed of several individuals, such as a family walking hand-in-hand in the park, or a squad of soldiers fighting together on the battlefield.
*   **Individual Level**: Focuses on the behavior of the most basic units, i.e., the animation, physics, and basic decisions of individual characters, such as walking, running, and avoiding obstacles.

**Core Difference: Particle Systems vs. Agent-based Systems**

Please remember this key difference:

* **Particle System**: Each "particle" in it is **not intelligent**. They are like dust, passively following global physical forces (such as gravity, wind). Suitable for simulating sparks, rain and snow, waterfalls, etc.
* **Agent-based System**: Each "individual" in it is an **Agent**. It has its own "Perceive-Think-Act" cycle. It can "see" the surrounding environment and neighbors, and **make decisions autonomously** based on them. This is the correct way to simulate life groups.

**All the techniques we will discuss today belong to the category of agent-based systems.**

# V. The Driving Force: Why Pursue Group Animation?

The immense effort invested in researching this technology is driven by powerful industrial demands:

1.  **Creating Visual Spectacles**: In movies and games, grand crowd scenes are key to creating a sense of epic scale and immersion. Without group animation, we would never have seen the orc armies in *The Lord of the Rings*, nor could we experience the suffocating zombie hordes in *World War Z*.
2.  **Liberating Productivity**: Having animators manually adjust the movements of thousands of characters is an impossible task. Group animation, through **Procedural Generation**, elevates the animator's role from "operator" to "director." They only need to set the macro-level rules and behavioral logic, and the computer handles the rest automatically.
3.  **Reducing Production Costs and Risks**: Filming a war scene with tens of thousands of people in reality would require astronomical amounts of manpower, materials, and capital, and would be extremely dangerous. Through CG technology, we can achieve more spectacular and imaginative effects than live-action filming, at a lower cost, higher efficiency, and absolute safety.

From *Avatar* to *Rise of the Planet of the Apes*, from *The Great Wall* to *The Wandering Earth*, almost all visual blockbusters rely on the silent contribution of group animation technology.

# VI.Technical Challenges: The "Hard Nuts" of Group Animation

To achieve realistic group animation, we need to tackle several "hard nuts to crack":

* **Massive Data and Computation:** Thousands upon thousands of intelligent agents, each containing models, skeletons, animations, and AI data, pose a huge challenge to memory and computing resources.

* **The Unity of Order and Chaos:** Group movement must possess both overall regularity (e.g., consistent direction) and local randomness (subtle differences between individuals). Finding the perfect balance between these two is a dual challenge of art and technology.

These macro-level characteristics, in terms of specific technical implementation, break down into the following core issues:

* **Motion Control:** How to design algorithms that allow each intelligent agent to navigate autonomously, avoid collisions, and exhibit behavior consistent with its character setting? This is the focus of this lecture.

* **Efficient Rendering:** How to render thousands upon thousands of high-quality characters within a single frame (typically 1/60th of a second)? * **Artistic Creation (Authoring):** How can we provide artists with intuitive and efficient tools to easily "direct" a large-scale group performance?

* **Level of Detail (LOD):** Distant characters don't need to be as detailed. How can we dynamically switch between models and animations of different complexities based on distance to save computational resources?

* **Collision Detection & Response:** How can we quickly and stably handle collisions between individuals and between individuals and their environment?

# VII. The "Trinity" Model of Motion Control

To solve the most core problem of motion control, **Craig Reynolds** proposed a classic hierarchical architecture that decomposes the complex movement behavior of an intelligent agent into three clear levels. We can call it the "Trinity" model of motion control.

1.  **Action Selection - Strategic Layer**
    
    *   This is the "**brain**" of the intelligent agent. It makes decisions based on high-level AI logic (such as state machines, behavior trees). It answers the question: "**What should I do?**" For example, a soldier on the battlefield may be in a state of "patrolling," "detecting enemies," "attacking," "seeking cover," or "retreating." This layer determines the current goals and intentions.
2.  **Steering - Tactical Layer**
    
    *   This is the "**driver**" of the intelligent agent. Once the brain determines the target (such as "attack that enemy"), the navigation layer is responsible for calculating the **force (Steering Force)** or acceleration that should be applied in the current frame to achieve this target. It handles abstract movement behaviors such as "Seek," "Flee," "Path Following," etc. It answers the question: "**How should I get there?**" **The Boids model and various navigation behaviors we will discuss next belong to this layer.**
3.  **Locomotion - Execution Layer**
    
    *   This is the "**body**" of the intelligent agent. It receives instructions (that force) from the navigation layer and applies them to the character's physical model to update its position and orientation in the world. At the same time, it selects and plays the most appropriate animation segment (such as walking, running, attacking animation) based on the current speed and state. It answers the question: "**How should I move my limbs specifically?**"

The advantage of this layering is **decoupling**. We can design and replace the algorithms of each layer independently. For example, we can use the same navigation algorithm to drive humanoid characters, cars, or spaceships, just by replacing their different locomotion layer implementations.

**Two Core Control Philosophies**

When implementing group movement, there are mainly two philosophies:

*   **Bottom-up**: The macro-level group behavior **emerges naturally** by setting simple local rules for each individual. This is like a free-developing society, full of vitality and unpredictability. **The Boids model is its typical representative.**
*   **Top-down**: The designer directly defines a global "field" (such as velocity field or density field) to **forcefully guide** the movement of individuals. This is like a well-planned urban traffic system, which is efficient and controllable. **Flow Field Following is its typical representative.**

In modern industrial applications, **the two are often combined**, using the "top-down" method for macro-level guidance and the "bottom-up" rules for local details, to balance controllability and realism.

# VIII.The "secret weapon" of industry

Before diving into the ocean of algorithms, let's first visit the "arsenal" of the industrial world to see what powerful commercial software artists and engineers are using.

*   **Golaem Crowd**、**Atoms Crowd**、**Anima**：These are the mainstream choices in the film and television and architectural visualization fields today. They are usually integrated as plug-ins into software such as Maya and Houdini, providing a friendly artist workflow and a powerful feature set.
    
*   **Massive: The Legend of Crowd Simulation** This software holds a position in the field of crowd animation comparable to that of Photoshop in image processing; it is groundbreaking. It was created specifically for the epic battle scenes in the\*\*\*The Lord of the Rings\*\*\* trilogy.
    
    *   **Its creator** is the brilliant engineer **Stephen Regelous** from New Zealand. He spent two years developing this "divine weapon" for director Peter Jackson, earning him an Academy Award for Science and Engineering.
    *   **The core philosophy of Massive** is to endow each digital character with a "brain" based on **fuzzy logic**. This brain can process complex perceptual information ("What do I see?" "What do I hear?") and make decisions that are close to human, non-binary judgments. It provides artists with a set of visual AI creation tools, enabling even those without a programming background to "design souls."
    *   可以说，Massive is not just a piece of software; it is a complete philosophy on how to create large-scale, believable digital life.

From the dinosaur stampede in *King Kong* to the ecosystem of Pandora in *Avatar*, and the zombie hordes in *World War Z*, these impressive scenes are backed by the power of Massive or similar technologies.

# IX. The Origin of All Things: A Deep Dive into the Boids Model

Now, after all this preparation, let's formally enter the core part of this lecture. We will derive and understand the most important and fundamental model in the history of crowd animation from scratch—the **Boids model**.

This model was published by the Craig W. Reynolds we mentioned repeatedly in 1987 at SIGGRAPH. The paper title was "Flocks, Herds, and Schools: A Distributed Behavioral Model." As of today, the Google Scholar citation count for this paper has exceeded 15,000, making it one of the most cited papers in the field of computer graphics.

**The Core Idea of Boids**

Reynolds' insight was revolutionary: **We don't need to simulate the entire flock; we just need to simulate one bird, and then let thousands of such "birds" interact with each other, and the realistic flocking behavior will miraculously emerge.**

The "bird" he referred to is called **Boid** (short for bird-oid object, meaning "bird-like object").

**How to simulate a Boid?**

Reynolds proposed that the behavior of each Boid is driven by three simple yet incredible rules based on its local neighbors. These three rules have decreasing priority:

1.  **Collision Avoidance / Separation**: Avoid colliding with nearby companions.
2.  **Velocity Matching / Alignment**: Try to match the average velocity and direction of nearby companions.
3.  **Flock Centering / Cohesion**: Try to move towards the average position of nearby companions to maintain group cohesion.

These three simple rules form the **digital DNA** of all complex group behaviors. Now, let's express them precisely in the language of mathematics and code.

First, let's define our basic data structure.

**Boid Data Structure (C++)**

```cpp
#include <vector>

struct Vector3 {
    float x, y, z;
    // ... Various vector operations, such as addition, subtraction, scalar multiplication, dot product, cross product, normalization, length calculation, etc.
};

class Boid {
public:
    Vector3 position;      // Position
    Vector3 velocity;      // Velocity (direction and magnitude)
    Vector3 acceleration;  // Acceleration
    
    float max_speed;       // Maximum speed limit
    float max_force;       // Maximum steering force limit
    float perception_radius; // Perception radius
    
    void update() {
        // Update physical state: velocity += acceleration, position += velocity
        velocity += acceleration;
        // Limit speed
        velocity.limit(max_speed); 
        position += velocity;
        // Reset acceleration every frame
        acceleration *= 0; 
    }
    
    void applyForce(const Vector3& force) {
        acceleration += force;
    }
    
    // ... Other methods
};
```

Our main loop will be: For each Boid in the scene, calculate the resultant force it receives based on the three rules, and then update its physical state.

## **Rule 1: Separation**

**Goal**: Avoid crowding neighbors.

**Intuition**: Check all neighbors within the perception range. If a neighbor is too close, generate a force to move away from it. The closer the neighbor, the stronger the repulsive force.

**Mathematical Derivation**: Let the current Boid be $b_i$ with position $\vec{P}_i$. For any neighbor $b_j$ (position $\vec{P}_j$) within the perception range, the distance vector between them is $\vec{D} = \vec{P}_i - \vec{P}_j$. We want a repulsive force that is in the same direction as $\vec{D}$ and whose magnitude is inversely proportional to the distance. A common approach is to make the force magnitude inversely proportional to the square of the distance, i.e., $1/|\vec{D}|^2$. Therefore, the repulsive force from neighbor $b_j$ can be expressed as $\frac{\hat{D}}{|\vec{D}|}$ or more simply $\frac{\vec{D}}{|\vec{D}|^2}$. Summing the repulsive forces from all neighbors gives the total separation force.

**Pseudocode**：

```cpp
function separation(boids):
  steering_force = Vector3(0, 0, 0)
  total = 0
  
  for each other_boid in boids:
    distance = distance(self.position, other_boid.position)
    
    // Check if the neighbor is within the perception range and the distance is greater than 0
    if distance > 0 and distance < self.perception_radius:
      // Calculate a repulsive force inversely proportional to the distance
      diff = self.position - other_boid.position
      diff.normalize()
      diff = diff / distance // The closer the distance, the greater the force
      steering_force += diff
      total += 1
      
  if total > 0:
    steering_force /= total // Find the average force
    
  // If the average force is not zero, adjust it to the desired magnitude.
  if steering_force.length() > 0:
    steering_force.normalize()
    steering_force *= self.max_speed
    steering_force -= self.velocity // Steering force = Desired velocity - Current velocity
    steering_force.limit(self.max_force)
    
  return steering_force
```

## **Rule 2: Alignment**

**Goal**: Align with the flight direction of neighbors.

**Intuition**: Observe all neighbors within my perception range, calculate their average flight velocity (direction and speed), and then adjust my own velocity to approach this average velocity.

**Mathematical Derivation**: Assume there are $N$ neighbors within Boid $b_i$'s perception range, with velocities $\vec{V}_1, \vec{V}_2, ..., \vec{V}_N$. Their average velocity is $\vec{V}_{avg} = \frac{1}{N} \sum\limits_{j=1}^{N} \vec{V}_j$. The goal of alignment behavior is to make Boid $b_i$ generate a steering force that gradually brings its own velocity $\vec{V}_i$ closer to $\vec{V}_{avg}$. This desired steering force is $\vec{F}_{align} = \vec{V}_{avg} - \vec{V}_i$.

**Pseudocode**:

```cpp
function alignment(boids):
  steering_velocity = Vector3(0, 0, 0)
  total = 0
  
  for each other_boid in boids:
    distance = distance(self.position, other_boid.position)
    
    if distance > 0 and distance < self.perception_radius:
      steering_velocity += other_boid.velocity
      total += 1
      
  if total > 0:
    steering_velocity /= total // Find the average velocity
    steering_velocity.normalize()
    steering_velocity *= self.max_speed // Expect to move forward at maximum speed
    
    steering_force = steering_velocity - self.velocity
    steering_force.limit(self.max_force)
    return steering_force
    
  return Vector3(0, 0, 0)
```

## **Rule 3: Cohesion**

**Goal**: Keep the flock compact and move towards the center of the flock.

**Intuition**: Find all neighbors within my perception range, calculate their geometric center, and then generate a force pointing towards that center.

**Mathematical Derivation**: Assume there are $N$ neighbors within Boid $b_i$'s perception range, with positions $\vec{P}_1, \vec{P}_2, ..., \vec{P}_N$. Their geometric center (centroid) is $\vec{P}_{center} = \frac{1}{N} \sum\limits_{j=1}^{N} \vec{P}_j$. Boid $b_i$ wants to move towards this center, so the target position is $\vec{P}_{center}$. The desired velocity vector from the current position $\vec{P}_i$ to the target position is $\vec{V}_{desired} = \vec{P}_{center} - \vec{P}_i$. The corresponding steering force is $\vec{F}_{cohesion} = \vec{V}_{desired} - \vec{V}_i$.

**Pseudocode**:

```cpp
function cohesion(boids):
  center_of_mass = Vector3(0, 0, 0)
  total = 0
  
  for each other_boid in boids:
    distance = distance(self.position, other_boid.position)
    
    if distance > 0 and distance < self.perception_radius:
      center_of_mass += other_boid.position
      total += 1
      
  if total > 0:
    center_of_mass /= total // Calculate the geometric center
    
    // Calculate the desired velocity towards the center
    desired_velocity = center_of_mass - self.position
    desired_velocity.normalize()
    desired_velocity *= self.max_speed
    
    // Calculate the steering force
    steering_force = desired_velocity - self.velocity
    steering_force.limit(self.max_force)
    return steering_force
    
  return Vector3(0, 0, 0)
```

## **组合规则 (Arbitration)**

Now that we have three independent forces, how do we combine them and apply them to the Boid?

The simplest method is **weighted summation**: 
$$
\vec{F}_{total} = w_{sep}\vec{F}_{sep} + w_{align}\vec{F}_{align} + w_{coh}\vec{F}_{coh}
$$ 
where $w$ is the weight coefficient, which can be adjusted by the artist to achieve different group effects (for example, increasing the cohesion weight will make the group more compact, and increasing the separation weight will make the group more loose).

**Main Loop Pseudocode**:

```cpp
// In the main program
std::vector<Boid> flock;
// ... Initialize flock ...

// Update loop for each frame
void update_simulation() {
  for (Boid& b : flock) {
    // Calculate the forces for the three rules
    Vector3 separation_force = b.separation(flock);
    Vector3 alignment_force = b.alignment(flock);
    Vector3 cohesion_force = b.cohesion(flock);
    
    // Apply weights
    separation_force *= 1.5; // Separation usually has the highest priority
    alignment_force *= 1.0;
    cohesion_force *= 1.0;
    
    // Apply the combined force
    b.applyForce(separation_force);
    b.applyForce(alignment_force);
    b.applyForce(cohesion_force);
    
    // Update the boid's physical state
    b.update();
  }
}
```

A more robust method is **priority arbitration**. Because "avoiding collisions" is usually more urgent than "maintaining formation". We can set a force limit `max_force`. Priority is given to the separation force, and if there is still room, it is allocated to the alignment force, and so on.

# Conclusion

Although the Boids model is simple, it laid the foundation for the entire field. Countless subsequent studies have been based on it for expansion and improvement:

*   **Obstacle Avoidance**: In addition to avoiding companions, Boids also need to avoid obstacles such as pillars and walls in the environment. This is usually achieved by projecting a "perception whisker" in front of it. If the whisker touches an obstacle, a normal repulsive force is generated.
*   **Leader Following**: One or more Boids can be designated as leaders, and another rule "follow the leader" can be added to the behavior rules of other Boids.
*   **Path Following**: Let the entire group move along a preset path.
*   **Flow Field**: As mentioned above, the top-down control method can be combined with the local rules of Boids to achieve macro-controllable and micro-vivid effects.
*   **More complex behaviors**: such as predator and prey behavior. The predator will have a "chasing" force, while the prey will have a strong "escape" force, and when the prey sees the predator, the weights of its separation, alignment, and cohesion rules will change dynamically.
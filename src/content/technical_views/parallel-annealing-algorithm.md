---
title: "Parallel Annealing Algorithm"
description: "An in-depth analysis of Parallel Annealing Algorithm."
pubDate: 2026-02-14
tags: ["Optimization", "Parallel Computing", "Simulated Annealing"]
heroImage: "/images/PSA.gif"
---

# I.Introduction - From Physical Annealing to Simulated Annealing (SA)

Combinatorial optimization is a well-known and challenging problem in computer science, including problems like the Transformation of Spirals (TSP) and VLSI Layout. The solution space for these problems typically grows exponentially with the problem size, making brute-force enumeration impractical.

Many local search algorithms, such as Hill Climbing, while efficient, are prone to getting trapped in local optimization and failing to find the global optimum.

The inspiration for simulated annealing comes from the physical annealing process in metallurgy.

*   **Physical Annealing**：A solid (such as a metal) is heated to a sufficiently high temperature, causing its internal particles to be in a disordered but high-energy state. Then, the temperature is slowly and controlledly lowered. During this process, the particles have enough time and "opportunity" to find the lowest-energy, most structurally stable lattice state. If the cooling is too rapid (quenching), the particles will be "frozen" in a higher-energy, metastable, amorphous state.
*   **Simulated Annealing,SA**：Inspired by this physical process, S. Kirkpatrick, C. D. Gelatt Jr., M. P. Vecchi abstracted this idea into a general random optimization algorithm.
    *   **State**：This corresponds to a solution $S$ to the optimization problem.
    *   **Energy**：This corresponds to the cost function $E(S)$ of the optimization problem. Our goal is to find the solution $S$ that minimizes $E(S)$.
    *   **Temperature**：This corresponds to the control parameter $T$ and decreases gradually with each iteration of the algorithm.
    *   **Finding New States**：This corresponds to generating a new solution $S'$ from the current solution $S$ through a neighborhood function.

**The Core Idea of SA**:

Algorithm not only accepts better new solutions ($\Delta E=E(S')-E(S)<0$), but also accepts worse solutions with a certain probability ($\Delta E>0$). This acceptance probability is given by the Metropolis criterion: 
$$
P(\text{accept } S')=\begin{cases}1,\quad\text{ if } \Delta E<0\\ e^{-\frac{\Delta E}{T}},\text{if }\Delta E\geq 0 \end{cases}
$$ 
This probability of "going downhill" is the soul of the SA algorithm.

*   **High Temperature**($T$ large)：$e^{-\frac{\Delta E}{T}}$ approaches 1. The algorithm almost accepts any new solution, showing strong random exploration behavior, allowing it to "climb over" mountains and avoid getting stuck in local optima.
*   **Low Temperature**($T$ small)：$e^{-\frac{\Delta E}{T}}$ approaches 0. The algorithm becomes very greedy, almost only accepting better solutions, allowing it to perform fine-grained search in the already found better region (Exploitation).

**Algorithm Pseudo Code**

```cpp
1. Initialization: 
   - Initial solution S_current = generate_initial_solution()
   - Initial temperature T = T_initial
   - Final temperature T_final
   - Cooling rate alpha (0 < alpha < 1)
   - S_best = S_current
2. while T > T_final:
3.    for i = 1 to L: // Iterate L times at each temperature
4.       S_new = generate_neighbor(S_current)
5.       delta_E = E(S_new) - E(S_current)
6.       if delta_E < 0:
7.          S_current = S_new
8.          if E(S_current) < E(S_best):
9.             S_best = S_current
10.      else:
11.         if random(0, 1) < exp(-delta_E / T):
12.            S_current = S_new
13.   T = T * alpha // Cooling
14. return S_best
```

# II.The mathematical foundation of simulated annealing algorithm

The convergence of SA can be rigorously proven through Markov Chain theory.

We consider the execution process of the algorithm as a sequence of states $S_0,S_1,S_2,...$，where each state is a solution in the solution space.

## 2.1. Fixed temperature homogeneous Markov chain

First, we consider that the algorithm's iteration process at a fixed temperature $T$ forms a homogeneous Markov chain (Homogeneous Markov Chain).

*   **State Space**：The set of all possible solutions $\Omega$.
*   **Transition Probability**：The probability of transitioning from state $i$ to state $j$ at temperature $T$，denoted as $P_{ij}(T)$.

$$
P_{ij}(T) = G_{ij} \cdot A_{ij}(T)
$$ 
where $G_{ij}$ is the probability of generating neighbor $j$ from state $i$ (determined by the neighborhood function), and $A_{ij}(T)$ is the acceptance probability of $j$ (determined by the Metropolis criterion).

This Markov chain has an important property: it has a **stationary distribution**. When $t \to \infty$, the probability $\pi_i (T)$ that the system is in state $i$ converges to a value that does not change with time. This steady-state distribution is the **Gibbs/Boltzmann distribution**: 
$$
\pi_i (T)=\frac{1}{Z(T)} e^{-\frac{E(i)}{T}}
$$ 
where $Z(T)=\sum\limits_j e^{-\frac{E(j)}{T}}$ is the normalization factor, called the partition function.

**Proof (Based on the Detailed Balance Condition):**

A Markov chain has a steady-state distribution $\pi$ if and only if the Detailed Balance Condition holds: 
$$
\pi_i P_{ij} = \pi_j P_{ji}\quad \forall i,j \in \Omega
$$
We verify this. Assuming $E(j)>E(i)$, then $\Delta E=E(j)-E(i)>0$. 
$$
A_{ij}(T) = e^{-\frac{E(j)-E(i)}{T}}\text{ and } A_{ji}(T)=1
$$ 
Substitute into the detailed balance condition: 
$$
\text{LHS}=\pi_i P_{ij} = \left(\frac{1}{Z(T)}e^{-\frac{E(i)}{T}}\right)\cdot G_{ij}\cdot e^{-\frac{E(i)-E(j)}{T}}=\frac{G_{ij}}{Z(T)}e^{-\frac{E(j)}{T}}
$$ 
$$
\text{RHS}=\pi_j P_{ji} = \left(\frac{1}{Z(T)}e^{-\frac{E(j)}{T}}\right)\cdot G_{ji}\cdot 1 = \frac{G_{ji}}{Z(T)} e^{-\frac{E(j)}{T}}
$$

If our neighborhood generation function is symmetric, i.e., $G_{ij}=G_{ji}$, for example, in TSP, randomly swapping two cities, this operation is reversible with the same probability, then the left and right sides are equal, and the detailed balance condition is satisfied.

**The Significance of the Steady-State Distribution:**

At temperature $T$, after sufficient iterations, the probability that the algorithm visits state $i$ is proportional to $e^{-\frac{E(i)}{T}}$. This means that: states with lower energy are visited with exponentially higher probability.

## 2.2. Cooling Process and Inhomogeneous Markov Chain

The temperature in SA algorithm is changing, so it is an inhomogeneous Markov Chain (Inhomogeneous Markov Chain). Its convergence proof is much more complex, but the core idea is:

**Theorem**: If the following two conditions are satisfied, the SA algorithm will converge to the global optimal solution with probability 1:

1.  **Ergodicity**：For any temperature $T > 0$, the corresponding Markov chain is irreducible. This means that from any solution $i$, it is possible to reach any other solution $j$ in a finite number of steps. This requires our neighborhood function to be designed appropriately, so that the entire solution space is connected.
2.  **Sufficiently Slow Cooling**：Temperature $T(k)$（$k$ is iteration number） must decrease slowly enough to ensure that the system has enough time to approach its steady-state distribution at each temperature. Geman and Geman proved in 1984 that if the cooling schedule satisfies: 
$$
T(k) \ge \frac{C}{\log(k+k_0)}
$$ 
where $C$ is a sufficiently large constant (at least the maximum height of the energy barrier in the solution space), the algorithm can guarantee convergence to the global optimum.

**Intuitive Understanding**：

When $T\to 0$，the Gibbs distribution $\pi_i (T)$ has the following properties: 
$$
\lim\limits_{T\to 0} \pi_i (T) = \begin{cases}\frac{1}{|S_{opt}|}\text{ if } i \in S_{opt}\\ 0\quad \text{otherwise}\end{cases}
$$

where $S_{opt}$ is the set of global optimal solutions. This means that as temperature approaches 0, the probability concentrates entirely on the global optimal solutions. Slow cooling ensures that the non-homogeneous Markov chain can "follow" the changing steady-state distribution at each temperature, eventually settling at the global optimum.

However, logarithmic cooling is too slow in practice. We typically use exponential cooling $T_{k+1} =\alpha T_k$. While it cannot theoretically guarantee 100% convergence to the global optimum, it can obtain very high-quality approximate solutions within a finite time, representing a trade-off between theory and practice.

# III. Parallelization Motivation and Challenges

SA algorithm has two fatal weaknesses, making it powerless against large-scale problems:

1.  **Inherent Sequentiality**：The state at iteration $k+1$ depends on the result of iteration $k$, which is a strict Markov chain, making it difficult to parallelize the internal chain.
2.  **Slow Convergence**：To ensure solution quality, the cooling process must be very slow, leading to a large number of iterations.

Parallel computing has provided us with powerful weapons to overcome these weaknesses. The core goal of parallelization is: to shorten the real time (Wall Clock Time) needed to find the optimal solution without significantly sacrificing (sometimes even improving) the solution quality.

**Challenges of Parallelization**:

*   **Breakdown of Markov Property**：Parallel execution may destroy the Markov property of the algorithm, leading to its theoretical convergence no longer holding.
*   **Communication Overhead**：Information exchange between parallel processes/threads incurs additional time overhead. If communication is too frequent or the data volume is too large, it may offset the benefits of parallel computing.
*   **Load Balancing**：How to evenly distribute computing tasks among all processing units, avoiding some units being idle while others are overloaded.
*   **Synchronization**：How multiple processes coordinate their work, especially when sharing information (e.g., the current optimal solution).

# IV. Major Flavors of Parallel Simulated Annealing

Based on the granularity and strategy of parallelization, PSA algorithms can be divided into several major categories.

## Flavor One: Independent Searches (Multi-start SA)

This is the simplest and most direct way to parallelize.

*   **Idea**：On $N$ processors, run $N$ independent SA algorithms that are completely unrelated. Each algorithm has its own initial solution, random seed, and complete cooling process. Finally, select the best one from these $N$ results as the final solution.
*   **Advantages**:
    *   No communication overhead: processes exchange zero information, perfect parallelization, and linear speedup.
    *   Easy to implement: minimal code changes.
    *   Enhanced exploration: starting from different initial points increases the probability of finding the global optimal solution.
*   **Disadvantages**: No synergy effect: one process's "good discovery" cannot help other processes, wasting valuable computing information. Multiple processes may repeatedly search in the same suboptimal region.

## Flavor Two: Parallel Moves

This strategy attempts to parallelize the internal loop of a single SA chain.

*   **Idea**：At each temperature, the master process holds the current solution $S_\text{current}$, which it broadcasts to $N$ slave processes. Each slave process independently generates a neighbor solution $S'_i$ of $S_\text{current}$ and calculates its energy $E(S'_i)$. Then, all $S'_i$ are sent back to the master process, which selects one as the next $S_\text{current}$ based on some rule.
    *   **Rule 1** (Most Greedy)：Select the lowest energy solution among all $S'_i$ and $S_\text{current}$.
    *   **Rule 2** (Metropolis Variant)：Randomly select one from all accepted moves (including those probabilistically accepted bad moves).
*   **Advantages**：Explore more neighbors in one iteration, potentially accelerating convergence.
*   **Disadvantages**：
    *   High communication/synchronization overhead: each iteration requires broadcasting and collecting, frequent synchronization between master and slave processes.
    *   Acceptance rate reduction: when multiple neighbors are generated in parallel, as long as one is a good move, it may be selected, reducing the acceptance probability of bad moves, making the algorithm behavior greedy and easily trapped in local optima.
    *   Weak theoretical foundation: This severely disrupts the original SA's Markov chain structure, making convergence difficult to guarantee.

## Flavor Three: Interactive Searches (Cooperative SA)

This is a compromise between independent searches and parallel moves, and it is currently the most widely studied and applied flavor. It allows multiple SA chains (called Walkers or Agents) to run in parallel, but they periodically or asynchronously exchange information.

*   **Idea**：$N$ SA chains run in parallel. They can exchange information including:
    *   Current solution
    *   Best solution found so far
    *   Current temperature
*   **Common Interaction Strategies**:
    *   Migration Model：Similar to parallel genetic algorithms. Each SA chain runs independently for a period (one epoch) and then performs a "migration". For example, each chain sends its best solution to its neighbor and replaces its current solution with the received better solution. This helps spread good genes (solution structure) throughout the population.
    *   Central Blackboard Model：All chains share a global "blackboard" that records the global best solution $S_\text{global\_best}$. Each chain runs its own SA process locally, but periodically:
        *   Updates the best solution found so far to the blackboard.
        *   Reads $S_\text{global\_best}$ from the blackboard and resets (re-seeds) its current solution with a certain probability, thus escaping the local optimum it is in.
*   **Advantages**:
    *   Cooperative search: Combines exploration (multiple independent chains) and exploitation (information sharing), good solutions can guide other chains' search direction.
    *   Robust: One chain getting stuck in a local optimum can be "pulled out" by other chains.
    *   Controllable communication overhead: Communication frequency can be adjusted based on the problem, far below the parallel move model.
*   **Disadvantages**:
    *   Increased parameters: Need to design migration topology, communication frequency, information exchange strategy, etc., increasing algorithm complexity.
    *   Complex theoretical analysis: Coupled behavior of multiple Markov chains is very difficult to analyze.

# V.Extensions and Frontiers

*   **Adaptive Parallel Annealing**：Algorithm parameters (e.g., cooling rate, migration frequency) are no longer fixed but dynamically adjusted based on search feedback (e.g., solution diversity, acceptance rate), making the algorithm more "intelligent".
*   **Heterogeneous Parallel Annealing**：On CPU+GPU etc. heterogeneous platforms, different computing units perform different tasks. For example, let the GPU execute many independent short chains for extensive exploration, while the CPU executes a few long chains for deep mining and responsible coordination.
*   **Machine Learning Integration**：
    *   Use reinforcement learning to dynamically adjust SA parameters.
    *   SA can be embedded into the training of neural networks for weight optimization, especially for network structures where gradients are not obvious.
*   **Quantum Annealing**：This is a physical implementation, not a simulation. It utilizes the quantum tunneling effect to "pass through" the energy barrier, rather than "flipping" it as in classical scalars. D-Wave's quantum computer is based on this principle. It is the ultimate simulation of scalars in the field of quantum computing.

# VI. Important Application Areas

PSA's powerful ability makes it shine in many NP-hard problems:

1.  Electronic Design Automation (EDA):
    *   VLSI Layout (Placement): Place millions of logic gates on a chip, with the goal of minimizing bus length and routing congestion. The solution space is extremely large.
    *   VLSI routing connects pre-laid logic gates, aiming for $100\%$ connectivity while satisfying various physical constraints.
2.  Traveling Salesman Problem (TSP) and its variants:
    *   Vehicle Routing Problem (VRP), logistics distribution, drone path planning, etc.
3.  Bioinformatics:
    *   Protein folding: Predict the three-dimensional structure of proteins, which is an energy minimization problem, with its conformation space being astronomical.
    *   Gene sequence alignment.
4.  Image Processing:
    *   Image recovery/denoising: Treat noisy images as high-energy states, finding the original clear image with the lowest energy.
    *   Image segmentation.
5.  Machine Learning:
    *   Hyperparameter optimization: Find the best configuration in the vast hyperparameter space for models.
    *   Training Boltzmann Machines: This is a random neural network, and its training process is closely related to SA.

# VII. Practical Implementation - High Performance Implementation

Theory must be put into practice. We take the classic TSP problem as an example to show two implementations of PSA: C++ multi-threading (a simplified version of interactive search - independent search) and CUDA (large-scale independent search).

**Problem Definition**: Given $N$ city coordinates, find the shortest path that visits each city once and returns to the starting point.

*   **Definition of a solution**: A permutation of cities, such as $[0, 4, 1, 3, 2]$.
*   **Cost function**: The total Euclidean distance of the path.
*   **Neighborhood function**: 2-opt, which randomly selects two segments of the path, disconnects them, and reconnects them in another way (equivalent to inverting a segment of the path between two cities).

1.  C++ Multithreaded Implementation (Independent Search Model)

We will use std::thread to start multiple independent SA instances.

```cpp
// parallel_sa_tsp.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <thread>
#include <mutex>

// City structure
struct City {
    double x, y;
};

// Calculate the distance between two cities
double distance(const City& a, const City& b) {
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}

// Calculate the total path length
double total_distance(const std::vector<int>& path, const std::vector<City>& cities) {
    double dist = 0.0;
    for (size_t i = 0; i < path.size() - 1; ++i) {
        dist += distance(cities[path[i]], cities[path[i + 1]]);
    }
    dist += distance(cities[path.back()], cities[path.front()]); // Return to the starting point
    return dist;
}

// Single simulated annealing thread function
void simulated_annealing_worker(
    int thread_id,
    const std::vector<City>& cities,
    std::vector<int>& best_path,
    double& min_distance,
    std::mutex& mtx) 
{
    // Thread-safe random number generator
    std::mt19937 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count() + thread_id);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Initialize path
    std::vector<int> current_path(cities.size());
    std::iota(current_path.begin(), current_path.end(), 0);
    std::shuffle(current_path.begin() + 1, current_path.end(), rng); // Randomly shuffle (fixed starting point)

    double current_energy = total_distance(current_path, cities);
    std::vector<int> local_best_path = current_path;
    double local_min_energy = current_energy;

    double T = 10000.0;
    double T_final = 1e-8;
    double alpha = 0.999;

    while (T > T_final) {
        for (int i = 0; i < 100; ++i) { // Each temperature iteration 100 times
            // Generate new neighbor (2-opt)
            std::vector<int> new_path = current_path;
            int a = std::uniform_int_distribution<int>(1, cities.size() - 2)(rng);
            int b = std::uniform_int_distribution<int>(a + 1, cities.size() - 1)(rng);
            std::reverse(new_path.begin() + a, new_path.begin() + b + 1);

            double new_energy = total_distance(new_path, cities);
            double delta_E = new_energy - current_energy;

            if (delta_E < 0 || dist(rng) < std::exp(-delta_E / T)) {
                current_path = new_path;
                current_energy = new_energy;
                if (current_energy < local_min_energy) {
                    local_best_path = current_path;
                    local_min_energy = current_energy;
                }
            }
        }
        T *= alpha;
    }

    // Update global optimal solution (need to lock)
    std::lock_guard<std::mutex> lock(mtx);
    if (local_min_energy < min_distance) {
        min_distance = local_min_energy;
        best_path = local_best_path;
        std::cout << "Thread " << thread_id << " found new best distance: " << min_distance << std::endl;
    }
}

int main() {
    // Create TSP problem instance
    const int num_cities = 50;
    std::vector<City> cities(num_cities);
    std::mt19937 city_rng(123); // Fixed seed to reproduce
    std::uniform_real_distribution<double> coord_dist(0.0, 100.0);
    for (int i = 0; i < num_cities; ++i) {
        cities[i] = {coord_dist(city_rng), coord_dist(city_rng)};
    }

    const int num_threads = std::thread::hardware_concurrency(); // Get CPU core count
    std::cout << "Using " << num_threads << " threads." << std::endl;

    std::vector<int> global_best_path;
    double global_min_distance = std::numeric_limits<double>::max();
    std::mutex mtx;

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(simulated_annealing_worker, i, std::ref(cities), 
                             std::ref(global_best_path), std::ref(global_min_distance), std::ref(mtx));
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "\nFinal best distance: " << global_min_distance << std::endl;
    std::cout << "Path: ";
    for (int city_idx : global_best_path) {
        std::cout << city_idx << " -> ";
    }
    std::cout << global_best_path[0] << std::endl;

    return 0;
}
```

This implementation clearly shows the parallel strategy of the independent search model: each thread is an independent solver, and they update the global optimal solution through a mutex lock.

2.  CUDA Implementation (Large-scale Independent Search)

GPU has thousands of computing cores, making it ideal for performing large-scale independent searches. Each CUDA thread will be responsible for a complete SA annealing process.

Key points:

*   Device-side random numbers: When performing random algorithms on the GPU, each thread must initialize an independent random number generator state. We will use the cuRAND library.
*   Data structure: City coordinates, paths, etc. need to be copied from CPU (Host) to GPU (Device).
*   Kernel function: This is the core code executed on the GPU, where each thread runs its own SA loop.

```cpp
// parallel_sa_tsp.cu
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <curand_kernel.h>

#define NUM_CITIES 50
#define NUM_WALKERS 10240 // Launch a large number of independent SA instances (walkers)

// City structure
struct City {
    float x, y;
};

// GPU device function: Calculate the distance between two points
__device__ float distance_gpu(const City& a, const City& b) {
    return sqrtf(powf(a.x - b.x, 2) + powf(a.y - b.y, 2));
}

// GPU device function: Calculate the total path length
__device__ float total_distance_gpu(int* path, City* cities) {
    float dist = 0.0f;
    for (int i = 0; i < NUM_CITIES - 1; ++i) {
        dist += distance_gpu(cities[path[i]], cities[path[i + 1]]);
    }
    dist += distance_gpu(cities[path[NUM_CITIES - 1]], cities[path[0]]);
    return dist;
}

// CUDA Kernel: Each thread executes a complete SA process
__global__ void parallel_sa_kernel(City* d_cities, int* d_best_paths, float* d_min_distances, curandState* states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= NUM_WALKERS) return;

    // 1. Initialize each thread's random number generator
    curandState local_state = states[tid];

    // 2. Initialize path (stored in each thread's local memory)
    int current_path[NUM_CITIES];
    for (int i = 0; i < NUM_CITIES; ++i) current_path[i] = i;
    // Fisher-Yates shuffle
    for (int i = NUM_CITIES - 1; i > 0; --i) {
        int j = curand_uniform(&local_state) * (i + 1);
        int temp = current_path[i];
        current_path[i] = current_path[j];
        current_path[j] = temp;
    }

    float current_energy = total_distance_gpu(current_path, d_cities);
    
    int local_best_path[NUM_CITIES];
    for(int i=0; i T_final) {
        for (int i = 0; i < 50; ++i) {
            // Generate neighbor (2-opt)
            int a = 1 + (int)(curand_uniform(&local_state) * (NUM_CITIES - 2));
            int b = 1 + (int)(curand_uniform(&local_state) * (NUM_CITIES - 2));
            if (a == b) continue;
            if (a > b) { int temp = a; a = b; b = temp; }
            
            // Reverse the sub-path in a temporary array
            int new_path[NUM_CITIES];
            for(int k=0; k>>(d_states, NUM_WALKERS, time(0));
    
    
    // 5. Launch SA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_WALKERS + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "Launching " << NUM_WALKERS << " walkers on GPU..." << std::endl;
    parallel_sa_kernel<<>>(d_cities, d_best_paths, d_min_distances, d_states);
    cudaDeviceSynchronize(); // Waiting for the kernel to finish executing
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }


    // 6. Copy results back to CPU
    std::vector<int> h_best_paths(NUM_WALKERS * NUM_CITIES);
    std::vector<float> h_min_distances(NUM_WALKERS);
    cudaMemcpy(h_best_paths.data(), d_best_paths, NUM_WALKERS * NUM_CITIES * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_min_distances.data(), d_min_distances, NUM_WALKERS * sizeof(float), cudaMemcpyDeviceToHost);

    // 7. Find the global optimal solution on the CPU
    float global_min_dist = h_min_distances[0];
    int best_walker_idx = 0;
    for (int i = 1; i < NUM_WALKERS; ++i) {
        if (h_min_distances[i] < global_min_dist) {
            global_min_dist = h_min_distances[i];
            best_walker_idx = i;
        }
    }

    std::cout << "\nFinal best distance (from GPU): " << global_min_dist << std::endl;
    std::cout << "Path: ";
    for (int i = 0; i < NUM_CITIES; ++i) {
        std::cout << h_best_paths[best_walker_idx * NUM_CITIES + i] << " -> ";
    }
    std::cout << h_best_paths[best_walker_idx * NUM_CITIES] << std::endl;

    // 8. Free GPU memory
    cudaFree(d_cities);
    cudaFree(d_best_paths);
    cudaFree(d_min_distances);
    cudaFree(d_states);

    return 0;
}
```

This CUDA implementation leverages the large-scale parallelism of the GPU, completing thousands of independent annealing processes in an instant, greatly increasing the probability of finding high-quality solutions, and saving a lot of time compared to serial execution on the CPU.

## Conclusion and Prospects

In the future, as computing power continues to improve, and with the integration of AI, quantum computing, and other frontier fields, parallel annealing and its derivative algorithms will play an increasingly important role in solving the more complex optimization problems humans face.
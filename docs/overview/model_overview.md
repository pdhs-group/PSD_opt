# Model Overview

## 1. Population Balance Equation (PBE)

The **Population Balance Equation (PBE)** is a partial differential equation that describes the time evolution of particle populations in terms of their number density distribution.  
Let $ n(x, t) $ denote the particle number density, where $ x $ represents an internal variable (e.g., particle size or volume), and $ t $ denotes time.

A general time-dependent 1d-PBE can be written as:

$$
\frac{\partial n(x,t)}{\partial t} = \text{(contributions from various mechanisms)}
$$

Typical contributions on the right-hand side include:

- **Agglomeration (Coalescence):** particles combine to form larger ones.  
- **Breakage (Fragmentation):** particles split into smaller ones.  
- **Growth:** particles increase in size due to processes such as deposition or crystallization.  
- **Nucleation:** new particles are generated.  

⚠️ In this project, the PBE is formulated **for a closed system**, meaning **no convection or diffusion terms** (external exchange terms) are included.

PBEs are widely used in chemical engineering, materials science, and particle technology to model systems where particle size distributions (PSD) evolve over time.  
Examples include:

- Crystallization and precipitation processes  
- Polymerization  
- Granulation and agglomeration in powder technology  
- Aerosol and colloid dynamics  

By solving the PBE, one can predict the evolution of particle size distributions, which are crucial for product quality, process control, and optimization.

In this project, the implemented models focus primarily on:

- **Agglomeration (coalescence) modeling**  
- **Breakage (fragmentation) modeling**  

Other phenomena (e.g., growth, nucleation) are not the current focus but can be incorporated within the same PBE framework if needed.

---

## 2. DPBE Solver

The **DPBESolver** is a grid-based method for solving the Population Balance Equation (PBE).  
It discretizes the internal variable $x$.  
When $x$ represents **particle volume**, each grid node corresponds to a collection of particles of a specific size.

### Grid Types

The solver currently supports two types of grids:

#### 1. Uniform Grid (uni grid)

- Defined directly on evenly spaced grid points.  
- The spacing is equal to the smallest particle volume $x_\text{min}$.  
- Due to this definition, the **range of particle sizes** strongly affects the number of required grid points.  
- If the system contains particles spanning a wide size range, the uniform grid may require a very large number of nodes, especially in higher dimensions, which leads to high computational cost.

#### 2. Geometric Grid (geo grid)

- Based on the **cell average method** ([Kumar & Ramkrishna, 2007](https://www.sciencedirect.com/science/article/pii/S0032591007002896)).  
- Instead of uniform spacing, nodes can be distributed in a **geometric sequence**, covering a wide particle size range with fewer grid points.  
- This approach redistributes newly formed particles through weighted averaging.  

⚠️ **Accuracy note:**  
The cell average method guarantees correct **zeroth** and **first-order moments**, but higher-order moments may suffer from approximation errors.

### Implemented Mechanisms

The DPBE solver currently includes models for:

- **Agglomeration**  
- **Breakage**

Multiple models are available for both mechanisms.  
They can be found in:

- `optframework/utils/func/jit_kernel_agg`  
- `optframework/utils/func/jit_kernel_break`

Users can easily add custom models by extending these files.  
All models are **shared across solvers**, not limited to DPBE.

### Dimensionality

- **1D:** particle volume distribution  
- **2D:** distribution over two material volumes (representing two particle components)  

⚠️ **Historical note:** In 2D mode, variables for the second material are internally labeled with the index **3** instead of 2.  

### Numerical Solver

The time evolution of the PBE is solved using `scipy.integrate.solve_ivp`  
(located in `optframework.dpbe_core.solve_PBE()`).

This solver works for most scenarios.  
If numerical convergence issues occur (after excluding non-physical parameter choices), users may try adjusting the solver tolerances:

- **`atol`**: absolute tolerance  
- **`rtol`**: relative tolerance

---

## 3. PBMSolver

The **PBMSolver** is based on the **Quadrature Method of Moments (QMOM)**.  
Unlike grid-based approaches such as DPBE, QMOM does not solve directly for the number density $ n(x,t) $.  
Instead, it evolves the **moments** of the particle size distribution.

### Moment Definition

For an internal variable $x$ (e.g., particle volume), the $k$-th moment of the number density function is defined as:

$$
M_k(t) = \int_{0}^{\infty} x^k \, n(x,t) \, dx
$$

where  

- $M_0$: total number of particles,  
- $M_1$: total particle volume (or mass, depending on definition),  
- higher-order moments: capture more detailed distribution properties.  

The original PBE can be reformulated as a system of ordinary differential equations (ODEs) governing the evolution of these moments.

---

### Quadrature Method of Moments (QMOM)

QMOM uses **weighted orthogonal polynomials** to represent the moments, effectively discretizing the continuous equation.  

- The number of quadrature nodes depends on the **order** of the orthogonal polynomial.  
- In practice, **2nd- or 3rd-order quadratures** are usually sufficient for accurate results.  
- Compared to DPBE, the **matrix size is much smaller**, which generally results in faster computations.

### Implemented Algorithms

The PBMSolver includes multiple variants of moment methods:

- **QMOM (basic)**: based on the **Wheeler algorithm**.  
- **GQMOM** (Generalized QMOM): available for **1D** problems.  
- **CQMOM** (Conditional QMOM): applied for **2D** problems.  

### Stability Considerations

While efficient, QMOM has specific stability issues:

- When the particle distribution becomes **overly concentrated** (e.g., nearly all particles at the same volume, a **delta distribution**), the solver may fail to converge.  
- For **agglomeration** and **breakage** processes:  
  - In DPBE, the grid boundaries provide natural “limits” to particle growth/shrinkage.  
  - In QMOM, no such explicit boundaries exist.  
    - This can sometimes lead to results that better reflect real physical behavior.  
    - However, in extreme cases, it may also trigger numerical instabilities.

## 4. MCPBESolver

The **MCPBESolver** is based on **Monte Carlo (MC) random algorithms and statistical methods**.  
Unlike DPBE or PBMSolver, this solver does **not** attempt to solve the PBE as a partial differential equation.  
Instead, it **directly simulates thousands of individual particles** and models their interactions through stochastic rules.

### Principle

- Each particle is explicitly represented in the simulation.  
- Particle dynamics (agglomeration, breakage, etc.) are modeled probabilistically.  
- By performing repeated random experiments and averaging results, the solver estimates the statistical behavior of the particle system.  

The larger the number of simulated particles and repetitions, the **more stable and realistic** the resulting distribution becomes.

### Computational Cost

- MC algorithms explicitly **track interactions between all particles**.  
- This leads to computational demands that are **significantly higher** than DPBE or PBMSolver.  
- Consequently, MCPBE is usually less efficient for large-scale simulations, but it is extremely powerful for obtaining reference or benchmark results.

### Accuracy and Stability

- Since MCPBE does not rely on discretizing the PBE, it avoids errors introduced by grid resolution (DPBE) or moment truncation (PBM).  
- There are **no inherent stability issues** associated with numerical schemes.  
- The method can, in principle, achieve **very high accuracy**, limited only by the number of particles and repetitions that can be simulated given computational resources.

## 5. ExtruderPBESolver

The **ExtruderPBESolver** is an extension of the `DPBESolver`.  
In addition to solving the PBE within each region using the DPBE approach, it introduces **convection terms** between adjacent regions.  

This allows the solver to handle **multi-region problems**, such as processes where particles are transported through different spatial zones (e.g., extrusion or segmented reactors).

However, similar to the original DPBE, the internal variables do not include spatial coordinates. This means the PBE does not account for spatial variations of the particle number density. In other words, both DPBE and ExtruderPBESolver assume that particles are **uniformly distributed** within each region.

## 6. Summary

All solvers in this project — `DPBESolver`, `PBMSolver`, and `MCPBESolver` — share the **same kernel models** for agglomeration and breakage, located in:

- `optframework/utils/func/jit_kernel_agg`  
- `optframework/utils/func/jit_kernel_break`

Users can easily extend these files to add new physical models, which will then be available to all solvers.

### Comparison of Solvers

| Solver         | Computational Complexity | Accuracy & Stability Characteristics |
|----------------|--------------------------|--------------------------------------|
| **DPBESolver** | High (grid-based, scales poorly with large particle ranges or higher dimensions) | Accurate and stable within grid resolution; stability ensured by grid boundaries; potential discretization error if grid resolution is low. |
| **PBMSolver**  | Low to Moderate (moment-based, complexity depends on polynomial order, usually 2–3 nodes) | Efficient and compact; accurate for smooth distributions; may become unstable for highly concentrated (delta-like) distributions; no natural particle size boundaries. |
| **MCPBESolver**| Very High (explicitly simulates thousands of particles; cost grows with particle count and repetitions) | Benchmark-level accuracy; no discretization or stability issues; results improve with more particles and repetitions but at a high computational cost. |

In practice:  

- **DPBE** is suitable when detailed particle size distribution (PSD) resolution is required and computational resources are sufficient.  
- **PBM** is often preferred for efficient simulations in engineering applications where approximate distributions are acceptable.  
- **MCPBE** serves as a high-fidelity reference solver, best suited for validation or small-scale studies where accuracy is prioritized over speed.

```{toctree}
:maxdepth: 1

```

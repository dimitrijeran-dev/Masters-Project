# Physics-Based Crack Growth Framework  
## J-Integral Validation → Deterministic → Stochastic Fracture Modeling

**Author:** Dimitrije Randjelovic  
**Program:** M.Eng. Aerospace Engineering — Rensselaer Polytechnic Institute (RPI)  
**Lab:** Intelligent Structural Systems Laboratory (ISSL)

---

## Overview

This repository contains a custom finite element fracture mechanics framework developed to compute, validate, and propagate cracks using energy-based fracture metrics.

The long-term objective is to build a physics-based, uncertainty-aware crack growth prediction pipeline capable of producing probabilistic life predictions.

The workflow progresses in three stages:

1. Deterministic FEM fracture validation  
2. Deterministic crack propagation  
3. Stochastic / probabilistic crack growth modeling  

---

## Motivation

Fatigue crack growth and remaining life prediction are critical in aerospace structures for:

- Damage tolerance analysis  
- Inspection interval planning  
- Reliability and risk assessment  

Reliable extraction of fracture driving forces (ΔK, K, J) is essential, yet many finite element workflows:

- Do not directly output fracture metrics  
- Produce mesh-dependent crack-tip stresses  
- Require manual post-processing  

This project develops a repeatable, validated computational pipeline grounded in LEFM and energy methods.

---

## Core Capabilities

### Deterministic FEM Solver

- 2D Linear Elastic FEM (Q4 elements)
- Adaptive crack-tip mesh refinement
- Traction loading with physical boundary conditions
- Field extraction:
  - Displacement `u`
  - Stress `σ`
  - Strain `ε`
  - von Mises stress
- VTK export for ParaView visualization

---

### Fracture Driving Force Extraction

Implements the domain form J-integral:

\[
J = \int_A \left( \sigma_{ij} \frac{\partial u_i}{\partial x_1} - W \delta_{1j} \right) \frac{\partial q}{\partial x_j} \, dA
\]

Features:

- Smooth weighting (q-field)
- Crack-face exclusion
- Domain/path independence validation
- Automatic computation of:

\[
K_I = \sqrt{J E'}
\]

---

## Verification

The solver is validated against LEFM asymptotic theory:

- Near-tip stress field scaling  
- Domain independence of J  
- Analytical vs FEM \(K_I\) comparison  
- Stress and von Mises crack-tip behavior  

---

## Current Results

- Stable J-integral plateau across domain sweep  
- Correct near-tip singular stress field behavior  
- Consistent \(K_I\) extraction from J  
- Verified displacement and stress fields  
- FEM crack-tip field structure consistent with LEFM  

---

## Deterministic Crack Propagation (Next Step)

Planned features:

- Incremental crack growth using ΔK or J  
- Paris-law fatigue modeling  
- Automatic remeshing around advancing crack  
- Crack length vs load cycles tracking  
- Crack trajectory prediction  

---

## Stochastic Crack Growth (Long-Term Goal)

Introduce uncertainty in:

- Material properties \(E, \sigma_y\)  
- Load magnitude / spectrum  
- Initial crack size and geometry  
- Numerical / mesh variability  

Methods:

- Monte Carlo simulation  
- Latin Hypercube sampling (future)  
- Probabilistic fracture metrics  

Outputs:

- Distribution of crack length vs cycles  
- Failure probability / reliability metrics  
- Sensitivity to material and load uncertainty  
- Foundation for surrogate / reduced-order models  

---

## Repository Structure
FEM/
│── solver_q4.py              # FEM solver
│── J_Integral.py             # Domain J-integral implementation
│── validate_fields.py        # LEFM validation & plots
│── crack_propagation.py      # (future)
│── stochastic_pipeline.py    # (future)

Data/
│── solution_q4.vtu
│── results/
│── plots/

docs/
│── figures/
│── references/

## Running the code
FEM/fem_solver.py in terminal
Outputs will be displacements, stresses and a .vtk file for Paraview

## Compute J-Integral
from J_Integral import compute_J_domain_q4

J, KI = compute_J_domain_q4(
    pts, conn, U,
    tip,
    E=70e9,
    nu=0.33,
    plane_stress=True,
    r_in=0.01,
    r_out=0.03
)

## Key References
Rice, J. R. (1968). A Path Independent Integral and the Approximate Analysis of Strain Concentration by Notches and Cracks. Journal of Applied Mechanics.

Anderson, T. L. Fracture Mechanics: Fundamentals and Applications.

Domain integral methods for energy release rate evaluation.

Stochastic fracture and XFEM literature.

## Research Contribution
This work aims to provide:

 - A validated fracture-driving-force extraction framework

 - eterministic → stochastic crack growth pipeline

 - Uncertainty-aware fracture prediction

 - A foundation for probabilistic life modeling in aerospace structures

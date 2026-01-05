# Masters-Project

Automated / Machine Learning Crack Growth and Lifing for Aerospace Materials and Structures

---

## Overview

This repository captures early-stage tooling for a machine-learning-enabled
workflow to predict crack growth and remaining life in aerospace
materials and structures.

The initial focus is a **2D plate with a central hole and a Mode I crack
emanating from the hole edge**, analyzed within the framework of **linear elastic
fracture mechanics (LEFM)**. Siemens NX finite element simulations provide
near-tip displacement fields, which are post-processed using the
**Displacement Correlation Method (DCM)** to extract stress intensity factors
(SIFs) for downstream crack-growth and lifing models.

---

## Repository Structure

- `src/dcm.py` – Lightweight Python utilities and CLI for computing Mode I stress
  intensity factors using the displacement correlation method.
- `README.md` – Project overview, modeling assumptions, and quickstart
  instructions.

---

## Requirements

- Python 3.10+
- pandas, numpy, matplotlib (for Siemens NX nodal post-processing and plotting)

---

## Modeling Assumptions

- 2D linear elastic fracture mechanics (LEFM)
- Homogeneous, isotropic material behavior
- Small-scale yielding
- **Plane stress** conditions (thin plate assumption)
- Fixed-crack analyses (no automatic crack propagation)

---

## Quickstart: Computing Mode I SIF with DCM

### Governing Expression (Plane Stress)

For a 2D isotropic, linear elastic body under **plane stress**, the Mode I stress
intensity factor is computed using the Displacement Correlation Method (DCM)
based on the crack opening displacement:

\[
K_I = \frac{E}{8}\,\delta_y(r)\sqrt{\frac{2\pi}{r}}
\]

where:

- \(E\) is Young’s modulus  
- \(\delta_y(r) = u_y^{+} - u_y^{-}\) is the crack opening displacement between
  the upper and lower crack faces  
- \(r\) is the distance behind the crack tip measured along the crack line  

This expression is derived from the near-tip asymptotic displacement field of
linear elastic fracture mechanics and is valid within the **K-dominant zone**.

> **Note:** A more general DCM formulation exists using Lamé parameters and the
> Kolosov constant \(\kappa\). Because this project explicitly assumes
> **plane stress**, the simplified expression above is used directly without
> loss of generality.

---

### Practical DCM Procedure

1. Identify the crack tip location.
2. Select multiple distances \(r\) behind the crack tip.
3. At each \(r\), extract nodal displacements \(u_y^{+}\) and \(u_y^{-}\) on the
   upper and lower crack faces.
4. Compute the crack opening displacement:
   \[
   \delta_y(r) = u_y^{+} - u_y^{-}
   \]
5. Compute \(K_I(r)\) using the DCM expression above.
6. Identify a **plateau region** where \(K_I(r)\) is approximately constant.
7. Report the final \(K_I\) as the mean or robust statistic over the plateau
   region.

---

### Direct Entry (Single Evaluation)

```bash
python src/dcm.py \
  --material-E 7.31e10 \    # Young's modulus (Pa)
  --material-nu 0.33 \      # Poisson's ratio (retained for completeness)
  --r 1.0e-3 \              # distance behind crack tip (m)
  --uy-upper 2.0e-5 \       # upper crack-face displacement (m)
  --uy-lower -2.0e-5        # lower crack-face displacement (m)



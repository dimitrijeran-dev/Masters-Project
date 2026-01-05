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

### Integrating with Siemens NX output
The `--nx-csv` mode pairs upper/lower crack-face nodes and sweeps the near-tip
region to identify a K_I plateau.

1. Export crack-face nodal displacements from Siemens NX into a CSV containing
   at minimum the columns `x`, `y`, and `uy`. Optional columns `ux` and
   `node_id` are preserved in the output.
2. Specify the crack tip coordinates and pairing tolerances:
   ```bash
   python src/dcm.py \
     --material-E 7.3e10 \
     --material-nu 0.33 \
     --nx-csv nx_nodes.csv \
     --x-tip 0.0 --y-tip 0.0 \
     --y-band 0.02 --x-match-tol 0.02 \
     --r-min 0.04 --r-max 2.0 \
     --plot
   ```
3. The script pairs nodes, filters them in the requested `r` window, computes
   K_I for each pair using the E'/8 COD relation, and writes
   `dcm_pairs_and_KI.csv` next to the input file. The log reports a robust
   plateau estimate (median of the middle 50% of r samples). Set `--plot` to
   visualize the K_I vs. r curve.

## Next Steps
- Add parsing helpers for Siemens NX native output formats.
- Extend the calculator to Mode II/III and mixed-mode interaction integrals.
- Integrate the SIF pipeline with the machine learning crack-growth models.

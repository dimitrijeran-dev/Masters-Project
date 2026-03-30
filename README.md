# Masters Project

Physics-based fracture mechanics, stochastic FEM, displacement-correlation post-processing, and fatigue lifing tools for cracked plates.

## What This Repository Does

This repository now covers five connected workflows:

1. Deterministic 2D linear-elastic FEM for cracked plates with Q4 elements.
2. Fracture post-processing with domain-form J-integrals and crack-tip field validation.
3. Stochastic FEM with random Young's modulus fields, including spatially correlated KL/lognormal fields.
4. Non-intrusive generalized polynomial chaos (GPC) / uncertainty quantification studies built on the stochastic solver and validator.
5. Fatigue and experimental post-processing utilities for building `K_I(a)`, `Delta K(a)`, geometry-factor curves, and Monte Carlo life estimates.

The codebase is organized around simulation outputs written into `Data/`, especially `Data/New Data/`.

## Repository Layout

```text
FEM/
  mesh.py                         Deterministic Gmsh quad mesh generator
  fem_solver.py                   Deterministic FEM solve + J / K_I extraction
  validate_fields.py              Deterministic LEFM / path-independence validation
  mesh_convergence_validation.py  Mesh-convergence aggregation and plots
  Convergence_Plot.py             Convergence-fit helper for mesh studies

Stochastic_FEM/
  stochastic_mesh.py                  Stochastic mesh generator
  stochastic_fem_solver.py            Stochastic FEM solver for multiple realizations
  stochastic_validate_fields.py       Stochastic validation and aggregate statistics
  J_Integral_stochastic_modified.py   Corrected J* and interaction-integral utilities

Lifing/
  build_ki_csv_from_runs.py       Collect K_I from validated FEM run folders
  make_delta_k_curve.py           Build Delta K(a)
  make_dadn_curve.py              Paris-law and crack-growth plots
  stochastic_life_mc.py           Monte Carlo life simulation
  fatigue_lifing_utils.py         Shared fatigue helper functions

UQ/
  distributions.py                Random-variable mapping helpers
  quadrature.py                   Tensor collocation rules
  qoi_extractors.py               Extract K_I QoIs from validation outputs
  surrogate.py                    Polynomial-chaos fitting utilities
  study_runner.py                 GPC study driver used by pipelines/run_gpc_study.py

pipelines/
  run_deterministic_study.py      Manifest-driven deterministic orchestration
  run_stochastic_study.py         Manifest-driven stochastic orchestration
  run_gpc_study.py                Manifest-driven GPC/UQ orchestration
  manifests/*.example.json        Example manifests for the pipeline drivers

src/
  J_Integral.py                   Deterministic J-integral library
  dcm.py                          DCM / COD-based K_I extraction utility
  extract_KI_vs_a.py              Plateau K_I(a) extraction from DCM CSVs
  plot_F_vs_ab.py                 Geometry-factor plotting utility
  configs/run_io.py               Shared runtime-config reader/writer
  run_manifest.py                 Hashed run-manifest helper
  save_runs.py                    JSON save helper
  visualize.py                    Small hard-coded K_I vs r_out plotting helper

Data/
  New Data/                       Default FEM and stochastic output root
  Fatigue Outputs/                Fatigue post-processing outputs
  Old Wrong Data/                 Archived legacy data / older DCM inputs

tests/
  test_*.py                       Unit and regression checks for DCM, J/J*, UQ, and summaries
```

## Environment Setup

Python `3.12.1` is available in this workspace and is a good target for this repo.

Create or activate an environment, then install the packages used by the scripts:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib pandas meshio gmsh pytest
```

Optional:

- ParaView, if you want to inspect `.vtk` results visually.
- `export MPLCONFIGDIR=/tmp/matplotlib` if Matplotlib warns about a non-writable config directory.

## Important Usage Pattern

Not every script is CLI-driven, and not every stage is configured in the same place.

- The main FEM and stochastic workflows are configured by editing the dataclasses near the top of each script.
- The fatigue, DCM, extraction, and pipeline helpers are mostly normal command-line tools.
- The mesh stages seed a `runtime_config.json` file in each run folder, and later solve/validate stages reuse those geometry/material settings if present.
- Most run-producing stages also write a hashed `run_manifest.json` so downstream summaries can carry a reproducibility fingerprint.

For the config-driven scripts, keep `run_name`, geometry, and load/material settings consistent across the mesh, solve, and validation files. The scripts communicate through shared folders under `Data/New Data/`.

## Run Folder Metadata

Two small JSON files now tie the workflows together:

- `runtime_config.json`: evolving stage-to-stage state for a run folder, including geometry, material, and artifact paths.
- `run_manifest.json`: hashed description of the workflow inputs used to generate a run.

In practice, that means:

- `FEM/mesh.py` and `Stochastic_FEM/stochastic_mesh.py` write the initial geometry/material context.
- The solver and validator stages read that context back instead of relying only on hard-coded defaults.
- Lifing and validation summaries may include a `manifest_hash_sha256` so results can be traced back to the originating setup.

## Deterministic FEM Workflow

This workflow generates an edge-crack mesh, solves the deterministic plate problem, computes `J` and `K_I`, and validates the crack-tip fields against LEFM behavior.

### 1. Build the deterministic mesh

Edit the `Config` dataclass in `FEM/mesh.py` to set values such as:

- `run_name`
- `W`, `H`, `a`, `crack_gap`
- `lc_global`, `lc_tip`, `tip_refine_r`
- `E`, `nu`, `thickness`, `traction`

Then run:

```bash
python FEM/mesh.py
```

Output:

- `Data/New Data/<run_name>/plate_edge_crack_q4.msh`
- `Data/New Data/<run_name>/runtime_config.json`
- `Data/New Data/<run_name>/run_manifest.json`

### 2. Solve the deterministic FEM model

Edit `SolverConfig` in `FEM/fem_solver.py` so it points to the same run folder and geometry.

Then run:

```bash
python FEM/fem_solver.py
```

Outputs written into the run folder include:

- `solution_q4.vtk`
- `fields.npz`
- `metadata.json`
- `<run_name>.json`

The solver also imports [`src/J_Integral.py`](src/J_Integral.py) to compute `J` and `K_I`.

### 3. Validate the deterministic fields

Edit `ValConfig` in `FEM/validate_fields.py` so it uses the same `run_name`, crack length, material, and load.

Then run:

```bash
python FEM/validate_fields.py
```

Typical outputs:

- `validation_summary.json`
- `validation_line.csv`
- `KI_vs_rout.png`
- `J_path_independence.png`
- `sigma_sqrt_2pir_vs_r.png`
- `sigma_yy_and_vonmises_vs_r.png`

### 4. Aggregate a mesh-convergence study

If you have multiple validated run folders inside the same study directory, run:

```bash
python FEM/mesh_convergence_validation.py --study-dir "Data/New Data" --metric lc_tip
```

Outputs:

- `mesh_convergence_summary.csv`
- `mesh_convergence_report.txt`
- `KI_vs_mesh_refinement.png`
- `J_vs_mesh_refinement.png`
- contour-spread and J-K residual plots

### 5. Optional convergence-fit helper

`FEM/Convergence_Plot.py` reads `Data/New Data/mesh_convergence_summary.csv` and fits an asymptotic convergence curve:

```bash
python FEM/Convergence_Plot.py
```

It writes:

- `Data/New Data/mesh_convergence_plot.png`
- `Data/New Data/mesh_convergence_error_plot.png`

## Stochastic FEM Workflow

This workflow extends the FEM model to heterogeneous materials and repeated realizations.

Supported stochastic modulus modes in [`Stochastic_FEM/stochastic_fem_solver.py`](Stochastic_FEM/stochastic_fem_solver.py):

- `uniform_bounded`
- `gaussian_clipped`
- `kl_lognormal`

Supported geometries in [`Stochastic_FEM/stochastic_mesh.py`](Stochastic_FEM/stochastic_mesh.py):

- `plate_edge_crack`
- `plate_hole_edge_crack`

### 1. Build the stochastic mesh

Edit `MeshConfig` in `Stochastic_FEM/stochastic_mesh.py`, especially:

- `run_name`
- `geometry_type`
- `W`, `H`, `a`
- `hole_radius`, `hole_center` if using `plate_hole_edge_crack`
- `lc_global`, `lc_tip`, `tip_refine_r`

Then run:

```bash
python Stochastic_FEM/stochastic_mesh.py
```

Outputs:

- `Data/New Data/<run_name>/mesh_q4.msh`
- `Data/New Data/<run_name>/geometry_metadata.json`
- `Data/New Data/<run_name>/runtime_config.json`
- `Data/New Data/<run_name>/run_manifest.json`

### 2. Run the stochastic FEM solver

Edit `SolverConfig` in `Stochastic_FEM/stochastic_fem_solver.py`, especially:

- `run_name`
- `geometry_type`
- `stochastic_E`
- `E_mode`
- `E_mean`, `E_rel_std`, `E_rel_clip`
- `n_realizations`, `random_seed`
- KL settings if using `kl_lognormal`

Then run:

```bash
python Stochastic_FEM/stochastic_fem_solver.py
```

Outputs per realization:

- `solution_q4_mcXXXX.vtk`
- `fields_mcXXXX.npz`
- `metadata_mcXXXX.json`

### 3. Validate stochastic realizations

Edit `ValConfig` in `Stochastic_FEM/stochastic_validate_fields.py` so `run_dir` points at the stochastic run folder.

Then run:

```bash
python Stochastic_FEM/stochastic_validate_fields.py
```

This script uses corrected heterogeneous-material post-processing from [`Stochastic_FEM/J_Integral_stochastic_modified.py`](Stochastic_FEM/J_Integral_stochastic_modified.py) and writes:

- `validation_summary_mcXXXX.json`
- `validation_summary_all_realizations.json`
- `validation_summary_all_realizations.csv`
- `validation_line_mcXXXX.csv`
- `J_star_path_independence_mcXXXX.png`
- `KI_from_Jstar_vs_rout_mcXXXX.png`
- interaction-integral plots when enabled
- optional DCM-based `K_I` estimates when `enable_dcm_from_fields` is enabled

### 4. Optional crack-length sweep + lifing comparison

`Stochastic_FEM/stochastic_validate_fields.py` can also compare deterministic and stochastic crack-length sweeps if you enable `run_crack_length_sweep = True` in `ValConfig`.

Point these settings at already validated run folders:

- `sweep_root`
- `deterministic_glob`
- `stochastic_glob`
- `lifing_out_dir`

When enabled, the validator collects `K_I(a)` from deterministic `validation_summary.json` files and stochastic mean `validation_summary_all_realizations.json` files, then writes:

- `ki_vs_a_deterministic.csv`
- `ki_vs_a_stochastic.csv`
- `delta_k_curve_deterministic.csv`
- `delta_k_curve_stochastic.csv`
- `a_vs_N_deterministic.png`
- `a_vs_N_stochastic.png`
- `life_histogram_deterministic.png`
- `life_histogram_stochastic.png`
- `lifing_stats_deterministic.json`
- `lifing_stats_stochastic.json`

## GPC / UQ Workflow

The repository also supports non-intrusive generalized polynomial chaos studies through [`pipelines/run_gpc_study.py`](pipelines/run_gpc_study.py) and the `UQ/` package.

### 1. Prepare a base mesh folder

`UQ/study_runner.py` expects `solver_run_dir` to contain a deterministic mesh file named:

- `plate_edge_crack_q4.msh`

That usually means you first run the deterministic mesh workflow once and point the GPC manifest at that run directory.

### 2. Edit the GPC manifest

Start from:

- `pipelines/manifests/gpc_manifest.example.json`

Key fields:

- `run_name`
- `base_output_dir` or `run_dir`
- `solver_run_dir`
- `quadrature_order`
- `polynomial_order`
- `compute_jstar`
- `compute_dcm`
- `random_inputs`

Currently supported random-input distributions in `UQ/distributions.py` are:

- `normal`
- `uniform`

Each collocation case maps the random inputs to physical parameters, runs one stochastic realization with those overrides, validates it, and extracts the QoIs `KI_Jstar` and `KI_DCM`.

### 3. Run the study

```bash
python pipelines/run_gpc_study.py pipelines/manifests/gpc_manifest.example.json
```

Typical outputs in the GPC run directory:

- `case_0000/`, `case_0001/`, ... with copied mesh, solve outputs, and validation outputs
- `gpc_samples.csv`
- `gpc_coefficients_KI_Jstar.json`
- `gpc_coefficients_KI_DCM.json` if DCM extraction is enabled and available
- `gpc_qoi_summary.json`
- `gpc_config_resolved.json`

## Fatigue Lifing Workflow

The fatigue tools build a crack-length history from validated `K_I` data and then apply Paris-law style post-processing.

### 1. Collect `K_I(a)` from validated deterministic runs

`Lifing/build_ki_csv_from_runs.py` reads `validation_summary.json` from each run folder and extracts `KI_ref`.
`--a-values` is required and must contain one entry per `--run-dirs` folder.
For stochastic studies, the same script can also read `validation_summary_all_realizations.json` or a specific `validation_summary_mcXXXX.json` via `--summary-mode`.

Example using the run folders already present in `Data/New Data/`:

```bash
python Lifing/build_ki_csv_from_runs.py \
  --run-dirs \
  "Data/New Data/meshrun_10mm" \
  "Data/New Data/meshrun_15mm" \
  "Data/New Data/meshrun_20mm" \
  "Data/New Data/meshrun_25mm" \
  "Data/New Data/meshrun_30mm" \
  "Data/New Data/meshrun_35mm" \
  "Data/New Data/meshrun_40mm" \
  "Data/New Data/meshrun_45mm" \
  "Data/New Data/meshrun_50mm" \
  "Data/New Data/meshrun_55mm" \
  --a-values 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05 0.055 \
  --out-csv "Data/Fatigue Outputs/ki_vs_a.csv"
```

Stochastic-mean example:

```bash
python Lifing/build_ki_csv_from_runs.py \
  --run-dirs "Data/New Data/stochastic_20mm" "Data/New Data/stochastic_25mm" \
  --a-values 0.02 0.025 \
  --summary-mode stochastic_mean \
  --out-csv "Data/Fatigue Outputs/ki_vs_a_stochastic_mean.csv"
```

### 2. Convert `K_I(a)` to `Delta K(a)`

Using load ratio `R`:

```bash
python Lifing/make_delta_k_curve.py \
  --ki-max-csv "Data/Fatigue Outputs/ki_vs_a.csv" \
  --R 0.1 \
  --outdir "Data/Fatigue Outputs"
```

Or using separate max/min files:

```bash
python Lifing/make_delta_k_curve.py \
  --ki-max-csv path/to/ki_max.csv \
  --ki-min-csv path/to/ki_min.csv \
  --outdir "Data/Fatigue Outputs"
```

Output:

- `Data/Fatigue Outputs/delta_k_curve.csv`

### 3. Generate Paris-law crack-growth plots

`make_dadn_curve.py` also supports an optional threshold line through `--delta-k-th`.

```bash
python Lifing/make_dadn_curve.py \
  --delta-k-csv "Data/Fatigue Outputs/delta_k_curve.csv" \
  --C 1e-10 \
  --m 3.0 \
  --outdir "Data/Fatigue Outputs"
```

Outputs:

- `dadn_vs_delta_k.png`
- `dadn_vs_a.png`
- `a_vs_N.png`
- `dadn_summary.json`

### 4. Run the Monte Carlo lifing study

```bash
python Lifing/stochastic_life_mc.py \
  --delta-k-csv "Data/Fatigue Outputs/delta_k_curve.csv" \
  --C-mean 1e-10 \
  --C-cov 0.1 \
  --m-mean 3.0 \
  --m-std 0.1 \
  --sigma-scale-mean 1.0 \
  --sigma-scale-cov 0.05 \
  --nsamples 1000 \
  --outdir "Data/Fatigue Outputs"
```

Output:

- `life_histogram.png`
- `life_samples.csv`
- `life_summary.json`


## Pipeline Drivers (Manifest-based Orchestration)

Three orchestration drivers are available in `pipelines/`:

- `pipelines/run_deterministic_study.py`
- `pipelines/run_stochastic_study.py`
- `pipelines/run_gpc_study.py`

The deterministic and stochastic drivers:

- accept a manifest path,
- execute steps in order by importing and calling module functions,
- stop if metadata compatibility checks fail,
- and write a top-level summary report with artifact paths and key metrics.

The GPC driver reads a `GPCConfig` JSON manifest and then delegates to `UQ/study_runner.py`.

Example manifests are included at:

- `pipelines/manifests/deterministic_manifest.example.json`
- `pipelines/manifests/stochastic_manifest.example.json`
- `pipelines/manifests/gpc_manifest.example.json`

Run examples:

```bash
python pipelines/run_deterministic_study.py pipelines/manifests/deterministic_manifest.example.json
python pipelines/run_stochastic_study.py pipelines/manifests/stochastic_manifest.example.json
python pipelines/run_gpc_study.py pipelines/manifests/gpc_manifest.example.json
```

## DCM and Geometry-Factor Utilities

The `src/` utilities support displacement-correlation measurements, experimental/NX post-processing, and geometry-factor extraction.

### 1. Run DCM / COD-based `K_I` extraction

[`src/dcm.py`](src/dcm.py) supports three modes.

Direct single-point entry:

```bash
python src/dcm.py \
  --material-E 7.31e10 \
  --material-nu 0.33 \
  --r 0.002 \
  --uy-upper 2.0e-5 \
  --uy-lower -2.0e-5
```

Simple CSV mode:

```bash
python src/dcm.py \
  --material-E 7.31e10 \
  --material-nu 0.33 \
  --csv path/to/displacements.csv
```

NX node-pairing mode:

```bash
python src/dcm.py \
  --material-E 7.31e10 \
  --material-nu 0.33 \
  --nx-csv path/to/nx_nodes.csv \
  --nx-units mm \
  --auto-tip \
  --auto-r-window \
  --plot
```

NX mode writes files next to the input CSV:

- `dcm_<stem>_pairs_and_KI_all.csv`
- `dcm_<stem>_pairs_and_KI_window.csv`
- `dcm_<stem>_KI_vs_r.png`

### 2. Extract a single plateau `K_I` per crack length

```bash
python src/extract_KI_vs_a.py \
  --inputs path/to/dcm_case1_pairs_and_KI_window.csv path/to/dcm_case2_pairs_and_KI_window.csv \
  --a-mm 4 6 \
  --out "Data/Fatigue Outputs/KI_summary.csv"
```

Outputs:

- `KI_summary.csv`
- `KI_summary.png`

### 3. Build a geometry-factor curve

```bash
python src/plot_F_vs_ab.py \
  --ki-summary "Data/Fatigue Outputs/KI_summary.csv" \
  --b-mm 100 \
  --R-mm 15 \
  --sigma-mpa 25 \
  --out-csv "Data/Fatigue Outputs/F_vs_ab.csv" \
  --out-png "Data/Fatigue Outputs/F_vs_ab.png"
```

Outputs:

- `F_vs_ab.csv`
- `F_vs_ab.png`

## Other Utilities

- [`src/J_Integral.py`](src/J_Integral.py): reusable deterministic J-integral library imported by the FEM solver and validator.
- [`Stochastic_FEM/J_Integral_stochastic_modified.py`](Stochastic_FEM/J_Integral_stochastic_modified.py): reusable corrected J* and interaction-integral library for heterogeneous materials.
- [`src/save_runs.py`](src/save_runs.py): helper for saving FEM run summaries to JSON.
- [`Lifing/fatigue_lifing_utils.py`](Lifing/fatigue_lifing_utils.py): shared helpers for `Delta K`, Paris law, and cycle integration.
- [`src/visualize.py`](src/visualize.py): small hard-coded plotting helper for one stored `K_I` versus `r_out` dataset.

## Typical Output Structure

Deterministic run folder:

```text
Data/New Data/<run_name>/
  run_manifest.json
  runtime_config.json
  plate_edge_crack_q4.msh
  solution_q4.vtk
  fields.npz
  metadata.json
  validation_summary.json
  validation_line.csv
  KI_vs_rout.png
  J_path_independence.png
  sigma_sqrt_2pir_vs_r.png
  sigma_yy_and_vonmises_vs_r.png
```

Stochastic run folder:

```text
Data/New Data/<run_name>/
  run_manifest.json
  runtime_config.json
  mesh_q4.msh
  geometry_metadata.json
  solution_q4_mc0000.vtk
  fields_mc0000.npz
  metadata_mc0000.json
  validation_summary_mc0000.json
  validation_summary_all_realizations.json
  validation_summary_all_realizations.csv
  ...
```

GPC study folder:

```text
Data/New Data/<gpc_run_name>/
  case_0000/
  case_0001/
  gpc_samples.csv
  gpc_coefficients_KI_Jstar.json
  gpc_coefficients_KI_DCM.json
  gpc_qoi_summary.json
  gpc_config_resolved.json
```

## Testing

The repository includes unit and regression tests for DCM, J/J* helpers, quadrature, surrogates, QoI extraction, and golden validation summaries.

Run the full suite with:

```bash
pytest tests
```

## Summary

At this point the repository is no longer just a single fracture-validation script. It is a broader crack-mechanics toolkit that can:

- generate deterministic and stochastic cracked-plate FEM models,
- compute and validate `J`, `K_I`, and crack-tip fields,
- run manifest-driven deterministic, stochastic, and GPC/UQ studies,
- post-process displacement data with DCM,
- build `K_I(a)` and `Delta K(a)` curves,
- and estimate fatigue life distributions.

## Git hygiene

If pull requests were opened in the wrong order, use the recovery guide in `docs/pr-order-recovery.md`.

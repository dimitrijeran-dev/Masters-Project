#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class StudyRow:
    run_dir: str
    num_nodes: float
    num_elements: float
    lc_tip: float
    lc_global: float
    J_selected: float
    KI_selected: float
    KI_from_J_selected: float
    JK_relative_residual: float
    J_relative_span: float
    KI_relative_span: float
    r_in: float
    r_out_selected: float
    Y_estimate: float


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _to_float(val, default=np.nan) -> float:
    try:
        if val is None or val == "":
            return float(default)
        return float(val)
    except Exception:
        return float(default)


def load_one_summary(run_dir: Path) -> Optional[Dict[str, float]]:
    summary_path = run_dir / "validation_summary.json"
    if not summary_path.exists():
        return None

    with open(summary_path, "r", encoding="utf-8") as f:
        row = json.load(f)

    meta_path = run_dir / "mesh_info.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            row.update(json.load(f))

    row["run_dir"] = str(run_dir)
    return row


def collect_rows(study_dir: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for child in sorted(study_dir.iterdir()):
        if not child.is_dir():
            continue
        row = load_one_summary(child)
        if row is not None:
            rows.append(row)

    if len(rows) < 2:
        raise RuntimeError(
            f"Need at least 2 validated run folders inside {study_dir}. "
            f"Each run folder should contain validation_summary.json."
        )
    return rows


def sorted_xy(rows: Sequence[Dict[str, float]], x_key: str, y_key: str):
    xs = np.array([_to_float(r.get(x_key)) for r in rows], dtype=float)
    ys = np.array([_to_float(r.get(y_key)) for r in rows], dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size == 0:
        raise RuntimeError(f"No finite data found for x='{x_key}', y='{y_key}'")
    order = np.argsort(xs)
    return xs[order], ys[order]


def save_csv(rows: Sequence[Dict[str, float]], out_csv: Path) -> None:
    keys = sorted({k for row in rows for k in row.keys()})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logging.info("Wrote %s", out_csv)


def write_text_report(rows: Sequence[Dict[str, float]], metric: str, out_txt: Path) -> None:
    xs, kis = sorted_xy(rows, metric, "KI_ref")
    _, js = sorted_xy(rows, metric, "J_ref")
    _, jspan = sorted_xy(rows, metric, "J_relative_span")
    _, kspan = sorted_xy(rows, metric, "KI_relative_span")
    _, jkres = sorted_xy(rows, metric, "JK_relative_residual")

    def pct_change_last_two(arr: np.ndarray) -> float:
        if arr.size < 2:
            return np.nan
        denom = max(abs(arr[-1]), 1e-30)
        return float(abs(arr[-1] - arr[-2]) / denom)

    lines = []
    lines.append("Mesh Convergence Study Summary")
    lines.append("=" * 32)
    lines.append(f"Metric used for x-axis: {metric}")
    lines.append(f"Number of runs: {len(rows)}")
    lines.append("")
    lines.append("What to look for:")
    lines.append("- K_I and J should approach a stable value as the mesh is refined.")
    lines.append("- J_relative_span and KI_relative_span should get smaller or stay small.")
    lines.append("- JK_relative_residual should stay near zero.")
    lines.append("")
    lines.append("Quick numerical checks:")
    lines.append(f"- Last-two-mesh relative change in K_I : {pct_change_last_two(kis):.4%}")
    lines.append(f"- Last-two-mesh relative change in J   : {pct_change_last_two(js):.4%}")
    lines.append(f"- Finest-mesh J contour spread         : {jspan[-1]:.4%}")
    lines.append(f"- Finest-mesh K_I contour spread       : {kspan[-1]:.4%}")
    lines.append(f"- Finest-mesh J-K residual             : {jkres[-1]:.4%}")
    lines.append("")
    lines.append("Rule-of-thumb interpretation:")
    lines.append("- A flattening K_I curve is a good sign.")
    lines.append("- A flattening J curve is a good sign.")
    lines.append("- Roughly < 2-5% contour spread in the plateau region is encouraging.")
    lines.append("- Large drift with refinement usually means the tip mesh or contour choice still needs work.")

    out_txt.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Wrote %s", out_txt)


def _finish_plot(
    xlabel: str,
    ylabel: str,
    title: str,
    out_png: Path,
    invert_x: bool = False,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if invert_x:
        plt.gca().invert_xaxis()
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
    logging.info("Wrote %s", out_png)


def plot_single_metric(
    rows: Sequence[Dict[str, float]],
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_png: Path,
    invert_x: bool = False,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    x, y = sorted_xy(rows, x_key, y_key)
    plt.figure()
    plt.plot(x, y, marker="o")
    _finish_plot(xlabel, ylabel, title, out_png, invert_x=invert_x, log_x=log_x, log_y=log_y)


def plot_overlay_J_KI(
    rows: Sequence[Dict[str, float]],
    x_key: str,
    xlabel: str,
    out_png: Path,
    invert_x: bool = False,
) -> None:
    x1, yj = sorted_xy(rows, x_key, "J_ref")
    x2, yk = sorted_xy(rows, x_key, "KI_ref")

    plt.figure()
    ax1 = plt.gca()
    ax1.plot(x1, yj, marker="o")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Selected J (N/m)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(x2, yk, marker="s")
    ax2.set_ylabel(r"Selected $K_I$ (Pa$\sqrt{m}$)")

    if invert_x:
        ax1.invert_xaxis()

    plt.title("J and K_I versus mesh refinement")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
    logging.info("Wrote %s", out_png)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aggregate validated runs into a mesh-convergence study with plots and summary files."
    )
    p.add_argument("--study-dir", required=True, help="Folder containing one subfolder per solved/validated mesh run.")
    p.add_argument(
        "--metric",
        default="lc_tip",
        help="Field to use on the x-axis. Common choices: lc_tip, num_nodes, num_elements.",
    )
    p.add_argument(
        "--label",
        default=None,
        help="Optional x-axis label override.",
    )
    return p


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()

    study_dir = Path(args.study_dir)
    rows = collect_rows(study_dir)

    metric = args.metric
    xlabel = args.label if args.label else metric
    invert_x = metric in {"lc_tip", "element_size", "h_tip"}

    save_csv(rows, study_dir / "mesh_convergence_summary.csv")
    write_text_report(rows, metric, study_dir / "mesh_convergence_report.txt")

    plot_single_metric(
        rows, metric, "KI_ref",
        xlabel=xlabel,
        ylabel=r"Selected $K_I$ (Pa$\sqrt{m}$)",
        title=r"$K_I$ vs. mesh refinement",
        out_png=study_dir / "KI_vs_mesh_refinement.png",
        invert_x=invert_x,
    )

    plot_single_metric(
        rows, metric, "J_ref",
        xlabel=xlabel,
        ylabel="Selected J (N/m)",
        title="J vs. mesh refinement",
        out_png=study_dir / "J_vs_mesh_refinement.png",
        invert_x=invert_x,
    )

    plot_single_metric(
        rows, metric, "J_relative_span",
        xlabel=xlabel,
        ylabel="Relative contour spread in J",
        title="J contour-spread vs. mesh refinement",
        out_png=study_dir / "J_contour_spread_vs_mesh_refinement.png",
        invert_x=invert_x,
    )

    plot_single_metric(
        rows, metric, "KI_relative_span",
        xlabel=xlabel,
        ylabel="Relative contour spread in K_I",
        title=r"$K_I$ contour-spread vs. mesh refinement",
        out_png=study_dir / "KI_contour_spread_vs_mesh_refinement.png",
        invert_x=invert_x,
    )

    plot_single_metric(
        rows, metric, "JK_relative_residual",
        xlabel=xlabel,
        ylabel="Relative J-K consistency residual",
        title="J-K consistency residual vs. mesh refinement",
        out_png=study_dir / "JK_residual_vs_mesh_refinement.png",
        invert_x=invert_x,
    )

    plot_overlay_J_KI(
        rows, metric, xlabel=xlabel,
        out_png=study_dir / "J_and_KI_vs_mesh_refinement.png",
        invert_x=invert_x,
    )

    logging.info("Done. Review the plots in %s", study_dir)


if __name__ == "__main__":
    main()

"""Compact, reproducible PDF plots for analysis artifacts."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/slri-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/slri-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _finish(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_distance_influence(table: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(5.5, 3.5))
    if not table.empty and "normalized_total_l1" in table:
        scopes = (
            table.groupby("metric_scope", sort=False)
            if "metric_scope" in table
            else [("influence", table)]
        )
        for scope, frame in scopes:
            plt.semilogy(
                frame["distance"],
                frame["normalized_total_l1"].clip(lower=1e-30),
                marker="o",
                label=f"{scope}: shell total",
            )
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No influence rows", ha="center")
    plt.xlabel("Graph distance")
    plt.ylabel("Normalized influence")
    _finish(path)


def plot_pathwise(table: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(4.5, 4.0))
    if (
        not table.empty
        and {"full_influence_fro", "geodesic_influence_fro"} <= set(table)
    ):
        plt.scatter(
            table["full_influence_fro"],
            table["geodesic_influence_fro"],
            alpha=0.7,
        )
        maximum = max(
            float(table["full_influence_fro"].max()),
            float(table["geodesic_influence_fro"].max()),
            1e-12,
        )
        plt.plot([0, maximum], [0, maximum], linestyle="--", color="black")
        plt.xscale("symlog", linthresh=1e-12)
        plt.yscale("symlog", linthresh=1e-12)
    else:
        plt.text(0.5, 0.5, "No pathwise rows", ha="center")
    plt.xlabel("Full Jacobian Frobenius norm")
    plt.ylabel("Geodesic Jacobian Frobenius norm")
    _finish(path)


def plot_curvature(table: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(5.5, 3.5))
    if not table.empty and "curvature" in table:
        frame = table
        if "curvature_alpha" in frame:
            frame = frame[frame["curvature_alpha"] == 0.5]
        if "length_scheme" in frame:
            frame = frame[frame["length_scheme"] == "omega_inverse"]
        for layer, layer_frame in frame.groupby("layer"):
            plt.scatter(
                layer_frame["unit_curvature"],
                layer_frame["curvature"],
                s=8,
                alpha=0.45,
                label=f"layer {layer}",
            )
        if frame["layer"].nunique() <= 8:
            plt.legend(fontsize=7)
    else:
        plt.text(0.5, 0.5, "Curvature sidecar not run", ha="center")
    plt.xlabel("Unit-length Ollivier--Ricci curvature")
    plt.ylabel("Primary learned-metric curvature")
    _finish(path)


def plot_bottleneck(table: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(5.5, 3.5))
    if not table.empty and "omega" in table:
        frame = table[~table["is_self_loop"]]
        summary = frame.groupby("layer")["omega"].agg(["min", "mean", "max"])
        plt.plot(summary.index, summary["mean"], marker="o", label="mean")
        plt.fill_between(
            summary.index, summary["min"], summary["max"], alpha=0.2
        )
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No sheaf geometry", ha="center")
    plt.xlabel("Diffusion layer")
    plt.ylabel(r"Effective strength $\omega_e$")
    _finish(path)


def plot_anisotropy(table: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(5.5, 3.5))
    if not table.empty and "normalized_transport_condition_number" in table:
        frame = table[~table["is_self_loop"]]
        grouped = frame.groupby("layer")[
            "normalized_transport_condition_number"
        ]
        plt.plot(grouped.median().index, grouped.median().values, marker="o")
        plt.yscale("log")
    else:
        plt.text(0.5, 0.5, "No sheaf singular spectra", ha="center")
    plt.xlabel("Diffusion layer")
    plt.ylabel("Median transport condition number")
    _finish(path)


def plot_orthogonal_restriction_rotations(table: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(6.5, 3.8))
    if not table.empty and "rotation_angle" in table:
        frame = table[table["map_kind"] == "transport"]
        if frame.empty:
            frame = table
        for key, group in frame.groupby(
            ["path_index", "path_position", "source", "target"],
            sort=False,
        ):
            path_index, path_position, source, target = key
            sign = int(group["determinant_sign"].iloc[0])
            marker = "o" if sign >= 0 else "x"
            label = f"p{path_index}:{path_position} {source}->{target}"
            plt.plot(
                group["layer"],
                group["rotation_angle"],
                marker=marker,
                label=label,
            )
        if frame.groupby(["path_index", "path_position"]).ngroups <= 8:
            plt.legend(fontsize=7)
    else:
        plt.text(0.5, 0.5, "No orthogonal rotations", ha="center")
    plt.xlabel("Diffusion layer")
    plt.ylabel("Rotation angle (radians)")
    _finish(path)

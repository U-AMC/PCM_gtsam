#!/usr/bin/env python3
"""
covariance_propagation_demo.py — Standalone covariance propagation demonstration.

Shows how adding inter-robot loop-closure factors to a pose graph
incrementally reduces the marginal covariance (uncertainty) at each node.

Produces a side-by-side figure:
  Left:  Iteration 1  — large covariance circles  (high uncertainty)
  Right: Final iteration — tiny covariance circles (low uncertainty)

This matches the style of figure/gt_propagation.png.

Usage:
    python3 covariance_propagation_demo.py
    MPLBACKEND=Agg MPLCONFIGDIR=/tmp python3 covariance_propagation_demo.py

Requires: gtsam, numpy, matplotlib
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

import gtsam
from gtsam import (
    BetweenFactorPose3,
    LevenbergMarquardtOptimizer,
    LevenbergMarquardtParams,
    NonlinearFactorGraph,
    Point3,
    Pose3,
    PriorFactorPose3,
    Rot3,
    Values,
    noiseModel,
)

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

# Node layout (matches figure/gt_propagation.png):
#
#   x_i_a (0,1) ---- intra-robot A ---- x_j_a (1,1)
#       \                                   /
#        \  z_ik^ab                z_jl^ab  /
#         \                               /
#   x_l_b (0,0) ---- intra-robot B ---- x_k_b (1,0)

NODE_POSES = {
    0: Pose3(Rot3(), Point3(0.0, 1.0, 0.0)),  # x_i^a
    1: Pose3(Rot3(), Point3(1.0, 1.0, 0.0)),  # x_j^a
    2: Pose3(Rot3(), Point3(0.0, 0.0, 0.0)),  # x_l^b
    3: Pose3(Rot3(), Point3(1.0, 0.0, 0.0)),  # x_k^b
}

NODE_LABELS = {
    0: r"$x\_i\_a$",
    1: r"$x\_j\_a$",
    2: r"$x\_l\_b$",
    3: r"$x\_k\_b$",
}

NODE_COLORS = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:red",
    3: "tab:green",
}

# Intra-robot odometry edges (same robot)
ODOM_EDGES = [(0, 1), (2, 3)]

# All possible inter-robot loop-closure pairs
INTER_ROBOT_PAIRS = [(0, 3), (1, 2), (0, 2), (1, 3)]


def build_base_graph():
    """Build a 4-node pose graph with anchor priors and intra-robot odometry.

    Returns the graph, initial estimate, and the true poses dict.
    """
    graph = NonlinearFactorGraph()
    initial = Values()

    for key, pose in NODE_POSES.items():
        initial.insert(key, pose)

    # Anchor priors — tight on Robot A origin, moderate on Robot B origin.
    anchor_noise = noiseModel.Diagonal.Sigmas(np.full(6, 1e-3))
    moderate_noise = noiseModel.Diagonal.Sigmas(np.full(6, 0.3))
    graph.add(PriorFactorPose3(0, NODE_POSES[0], anchor_noise))
    graph.add(PriorFactorPose3(2, NODE_POSES[2], moderate_noise))

    # Intra-robot odometry factors.
    odom_noise = noiseModel.Diagonal.Sigmas(np.full(6, 0.1))
    for k1, k2 in ODOM_EDGES:
        rel = NODE_POSES[k1].between(NODE_POSES[k2])
        graph.add(BetweenFactorPose3(k1, k2, rel, odom_noise))

    return graph, initial


def generate_loop_closure_factors(num_factors=26, seed=42):
    """Generate noisy inter-robot loop-closure factors.

    Each factor is a BetweenFactorPose3 between a Robot-A key and a Robot-B key,
    with small additive noise on the true relative transformation.
    """
    rng = np.random.default_rng(seed)
    loop_noise = noiseModel.Diagonal.Sigmas(np.full(6, 0.1))

    factors = []
    for i in range(num_factors):
        key_a, key_b = INTER_ROBOT_PAIRS[i % len(INTER_ROBOT_PAIRS)]
        true_rel = NODE_POSES[key_a].between(NODE_POSES[key_b])

        # Small perturbation to simulate a noisy measurement.
        d_rot = Rot3.RzRyRx(*rng.normal(0, 0.02, 3))
        d_trans = Point3(*rng.normal(0, 0.02, 3))
        noisy_rel = true_rel.compose(Pose3(d_rot, d_trans))

        factors.append((key_a, key_b, noisy_rel, loop_noise))

    return factors


# ---------------------------------------------------------------------------
# Covariance helpers
# ---------------------------------------------------------------------------

def optimize(graph, initial):
    params = LevenbergMarquardtParams()
    return LevenbergMarquardtOptimizer(graph, initial, params).optimize()


def covariance_ellipse(graph, values, key, n_std=2.0):
    """Return a matplotlib Ellipse representing the 2-sigma position covariance."""
    marginals = gtsam.Marginals(graph, values)
    cov6 = np.asarray(marginals.marginalCovariance(key), dtype=float)
    cov_xy = cov6[3:5, 3:5]  # position xy block

    eigvals, eigvecs = np.linalg.eigh(cov_xy)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2 * n_std * np.sqrt(max(eigvals[0], 0))
    height = 2 * n_std * np.sqrt(max(eigvals[1], 0))

    center = (NODE_POSES[key].x(), NODE_POSES[key].y())
    return Ellipse(xy=center, width=width, height=height, angle=angle)


def covariance_traces(graph, values):
    """Return the trace of the 6x6 marginal covariance for each of the 4 keys."""
    marginals = gtsam.Marginals(graph, values)
    return {
        key: float(np.trace(np.asarray(marginals.marginalCovariance(key), dtype=float)))
        for key in NODE_POSES
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def draw_frame(ax, graph, initial, factors_used, iteration):
    """Draw one snapshot: nodes, covariance ellipses, and factor edges."""
    values = optimize(graph, initial)

    ax.set_title(f"Covariance Propagation - Iteration {iteration}", fontsize=12)

    # Intra-robot odometry edges.
    for k1, k2 in ODOM_EDGES:
        p1 = (NODE_POSES[k1].x(), NODE_POSES[k1].y())
        p2 = (NODE_POSES[k2].x(), NODE_POSES[k2].y())
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--", linewidth=0.8, alpha=0.6)

    # Inter-robot loop-closure edges (all possible pairs shown).
    for ka, kb in INTER_ROBOT_PAIRS:
        p1 = (NODE_POSES[ka].x(), NODE_POSES[ka].y())
        p2 = (NODE_POSES[kb].x(), NODE_POSES[kb].y())
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--", linewidth=0.8, alpha=0.6)

    # Covariance ellipses and node markers.
    for key in sorted(NODE_POSES):
        ellipse = covariance_ellipse(graph, values, key, n_std=2.0)
        ellipse.set_facecolor("none")
        ellipse.set_edgecolor("blue")
        ellipse.set_linewidth(2.0)
        ax.add_patch(ellipse)

        cx, cy = NODE_POSES[key].x(), NODE_POSES[key].y()
        ax.plot(cx, cy, "o", color=NODE_COLORS[key], markersize=8, zorder=5)
        ax.annotate(
            NODE_LABELS[key],
            (cx, cy),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=10,
        )

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")


def main():
    base_graph, initial = build_base_graph()
    factors = generate_loop_closure_factors(num_factors=26, seed=42)

    # --- Build graph at iteration 1 (one inter-robot factor) ---
    graph_iter1 = NonlinearFactorGraph(base_graph)
    ka, kb, meas, noise = factors[0]
    graph_iter1.add(BetweenFactorPose3(ka, kb, meas, noise))

    # --- Build graph at final iteration (all factors) ---
    graph_final = NonlinearFactorGraph(base_graph)
    for ka, kb, meas, noise in factors:
        graph_final.add(BetweenFactorPose3(ka, kb, meas, noise))

    # --- Side-by-side figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_edgecolor("green")
    fig.patch.set_linewidth(3)

    draw_frame(ax1, graph_iter1, initial, factors[:1], iteration=1)
    draw_frame(ax2, graph_final, initial, factors, iteration=len(factors))

    # Green arrow between the two subplots.
    ax1.annotate(
        "",
        xy=(0.525, 0.5),
        xytext=(0.475, 0.5),
        xycoords="figure fraction",
        arrowprops=dict(arrowstyle="->", color="green", lw=3),
    )

    plt.tight_layout(rect=[0, 0, 1, 1])

    # Save when running headless.
    if matplotlib.get_backend().lower() == "agg":
        out = Path("figure") / "covariance_propagation_demo.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"Saved figure: {out}")

    plt.show()

    # --- Print covariance trace table ---
    print("\nCovariance trace progression (6x6 marginal trace per node):\n")
    header = f"{'Iter':>4}  {'x_i_a':>10}  {'x_j_a':>10}  {'x_l_b':>10}  {'x_k_b':>10}"
    print(header)
    print("-" * len(header))

    g = NonlinearFactorGraph(base_graph)
    for step in range(len(factors) + 1):
        if step > 0:
            ka, kb, meas, noise = factors[step - 1]
            g.add(BetweenFactorPose3(ka, kb, meas, noise))

        traces = covariance_traces(g, optimize(g, initial))
        print(
            f"{step:>4}  {traces[0]:>10.6f}  {traces[1]:>10.6f}"
            f"  {traces[2]:>10.6f}  {traces[3]:>10.6f}"
        )

    # Summary.
    first = covariance_traces(graph_iter1, optimize(graph_iter1, initial))
    final = covariance_traces(graph_final, optimize(graph_final, initial))
    print("\nShrink ratios (final / iteration-1):")
    for key in sorted(NODE_POSES):
        ratio = final[key] / first[key] if first[key] > 0 else float("nan")
        print(f"  {NODE_LABELS[key]:>10}: {ratio:.4f}")


if __name__ == "__main__":
    main()

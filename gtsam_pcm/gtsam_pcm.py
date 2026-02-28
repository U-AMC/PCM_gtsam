import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import sin, sqrt
from pathlib import Path
from typing import NamedTuple

# Ensure the project root is on sys.path so pcm_common is importable
# when this file is executed directly (python3 gtsam_pcm/gtsam_pcm.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gtsam
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gtsam import Pose3, Rot3, Point3, noiseModel, BetweenFactorPose3, NonlinearFactorGraph, Values, LevenbergMarquardtParams, LevenbergMarquardtOptimizer, PriorFactorPose3
from matplotlib.animation import FuncAnimation, PillowWriter

from pcm_common import (
    apply_maximum_clique,
    compute_metrics,
    generate_consistency_graph,
    parse_clique_indices,
    save_run_artifacts,
)

# Create a Pose3 given position (x, y, z) and yaw rotation (in rad)
def create_pose(x, y, z, yaw):
    rotation = Rot3.Yaw(yaw)
    translation = Point3(x, y, z)
    return Pose3(rotation, translation)


@dataclass
class GTSAMPCMConfig:
    seed: int = 0
    num_poses: int = 20
    yaw_increment: float = 0.01
    max_distance: float = 5.5
    pcm_threshold: float = 5.0
    intensity: float = 1.0
    num_loop_attempts: int = 40
    true_positive_attempts: int = 30
    use_sliding_window_pcm: bool = True
    pcm_window_size: int = 8
    pcm_window_step: int = 1
    visualize: bool = True
    save_results: bool = True
    output_dir: str = "results"
    run_tag: str = "gtsam_pcm_demo"
    track_covariance_progression: bool = True
    covariance_probe_indices: tuple = (10,)
    realtime_covariance_plot: bool = False
    realtime_pause_sec: float = 0.2
    animate_covariance_progression: bool = False
    animation_interval_ms: int = 300
    save_covariance_animation: bool = False
    covariance_animation_path: str = ""
    animate_pose_covariance_nodes: bool = True
    save_pose_covariance_animation: bool = False
    pose_covariance_animation_path: str = ""


class GTSAMLoopClosure(NamedTuple):
    idx_a: int
    idx_b: int
    pose_a: Pose3
    pose_b: Pose3
    relative_pose: Pose3
    is_true_positive: bool


def build_anchored_base_graph(poses_robot1, poses_robot2):
    """Build a solvable pose graph with priors + odometry for covariance queries."""
    num_poses = len(poses_robot1)
    graph = NonlinearFactorGraph()
    initial_estimate = Values()

    for i, pose in enumerate(poses_robot1):
        initial_estimate.insert(i, pose)
    for i, pose in enumerate(poses_robot2):
        initial_estimate.insert(i + num_poses, pose)

    prior_noise = noiseModel.Diagonal.Sigmas([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    odom_noise = noiseModel.Diagonal.Sigmas([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

    graph.add(PriorFactorPose3(0, poses_robot1[0], prior_noise))
    graph.add(PriorFactorPose3(num_poses, poses_robot2[0], prior_noise))

    for i in range(num_poses - 1):
        rel_odom_robot1 = poses_robot1[i].between(poses_robot1[i + 1])
        rel_odom_robot2 = poses_robot2[i].between(poses_robot2[i + 1])
        graph.add(BetweenFactorPose3(i, i + 1, rel_odom_robot1, odom_noise))
        graph.add(BetweenFactorPose3(i + num_poses, i + 1 + num_poses, rel_odom_robot2, odom_noise))

    return graph, initial_estimate


def optimize_graph(graph, initial_estimate):
    params = LevenbergMarquardtParams()
    optimizer = LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    return optimizer.optimize()


def get_covariance_snapshot(graph, values, probe_keys):
    """Return covariance diagnostics for selected keys."""
    marginals = gtsam.Marginals(graph, values)
    snapshot = []
    for key in probe_keys:
        try:
            cov = marginals.marginalCovariance(key)
            cov = np.asarray(cov, dtype=float)
            sign, logdet = np.linalg.slogdet(cov)
            snapshot.append(
                {
                    "key": int(key),
                    "trace": float(np.trace(cov)),
                    "logdet": float(logdet) if sign > 0 else float("nan"),
                }
            )
        except RuntimeError:
            snapshot.append({"key": int(key), "trace": float("nan"), "logdet": float("nan")})
    return snapshot


def compute_covariance_progression(poses_robot1, poses_robot2, inlier_loop_factors, probe_keys, on_step_callback=None):
    """
    Incrementally add inlier loop factors and track marginal covariance per step.
    Each step corresponds to one additional inlier factor.
    """
    graph, current_estimate = build_anchored_base_graph(poses_robot1, poses_robot2)
    progression = []

    # Baseline before adding any inter-robot inlier loops.
    optimized = optimize_graph(graph, current_estimate)
    progression.append(
        {
            "step": 0,
            "used_inlier_factors": 0,
            "covariance": get_covariance_snapshot(graph, optimized, probe_keys),
        }
    )
    if on_step_callback is not None:
        on_step_callback(progression[-1])
    current_estimate = optimized

    loop_noise = noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    for step, (idx1, idx2, _p1, _p2, rel_pose, _is_true_positive) in enumerate(inlier_loop_factors, start=1):
        robot1_key = int(idx1)
        robot2_key = int(idx2 + len(poses_robot1))
        graph.add(BetweenFactorPose3(robot1_key, robot2_key, rel_pose, loop_noise))

        optimized = optimize_graph(graph, current_estimate)
        progression.append(
            {
                "step": int(step),
                "used_inlier_factors": int(step),
                "covariance": get_covariance_snapshot(graph, optimized, probe_keys),
            }
        )
        if on_step_callback is not None:
            on_step_callback(progression[-1])
        current_estimate = optimized

    return progression


def plot_covariance_progression(covariance_progression):
    if not covariance_progression:
        return

    keys = [entry["key"] for entry in covariance_progression[0]["covariance"]]
    steps = [entry["step"] for entry in covariance_progression]

    plt.figure(figsize=(10, 5))
    for key in keys:
        traces = []
        for step in covariance_progression:
            key_entries = [entry for entry in step["covariance"] if entry["key"] == key]
            traces.append(key_entries[0]["trace"] if key_entries else float("nan"))
        plt.plot(steps, traces, marker="o", label=f"Key {key}")

    plt.title("Marginal Covariance Trace Over Inlier-Factor Iterations")
    plt.xlabel("Iteration (inlier factors added)")
    plt.ylabel("Covariance trace")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


def create_realtime_covariance_plot(probe_keys):
    fig, ax = plt.subplots(figsize=(10, 5))
    lines = {}
    trace_data = {}
    for key in probe_keys:
        line, = ax.plot([], [], marker="o", label=f"Key {key}")
        lines[key] = line
        trace_data[key] = []

    ax.set_title("Realtime Marginal Covariance Trace")
    ax.set_xlabel("Iteration (inlier factors added)")
    ax.set_ylabel("Covariance trace")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show(block=False)

    return {
        "fig": fig,
        "ax": ax,
        "lines": lines,
        "steps": [],
        "trace_data": trace_data,
    }


def update_realtime_covariance_plot(plot_state, progression_step, pause_sec=0.2):
    if plot_state is None:
        return

    step = progression_step["step"]
    plot_state["steps"].append(step)

    trace_map = {entry["key"]: entry["trace"] for entry in progression_step["covariance"]}
    for key, line in plot_state["lines"].items():
        plot_state["trace_data"][key].append(trace_map.get(key, float("nan")))
        line.set_data(plot_state["steps"], plot_state["trace_data"][key])

    plot_state["ax"].relim()
    plot_state["ax"].autoscale_view()
    plot_state["fig"].canvas.draw_idle()
    plt.pause(max(0.0, float(pause_sec)))


def animate_covariance_progression(covariance_progression, interval_ms=300, save_path=""):
    if not covariance_progression:
        return None

    keys = [entry["key"] for entry in covariance_progression[0]["covariance"]]
    steps = [entry["step"] for entry in covariance_progression]
    traces_by_key = {}
    for key in keys:
        traces_by_key[key] = []
        for step_entry in covariance_progression:
            key_entries = [entry for entry in step_entry["covariance"] if entry["key"] == key]
            traces_by_key[key].append(key_entries[0]["trace"] if key_entries else float("nan"))

    fig, ax = plt.subplots(figsize=(10, 5))
    lines = {}
    for key in keys:
        line, = ax.plot([], [], marker="o", label=f"Key {key}")
        lines[key] = line

    ax.set_title("Animated Marginal Covariance Trace")
    ax.set_xlabel("Iteration (inlier factors added)")
    ax.set_ylabel("Covariance trace")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()

    def _init():
        for line in lines.values():
            line.set_data([], [])
        return tuple(lines.values())

    def _update(frame_idx):
        for key, line in lines.items():
            x_vals = steps[: frame_idx + 1]
            y_vals = traces_by_key[key][: frame_idx + 1]
            line.set_data(x_vals, y_vals)
        ax.relim()
        ax.autoscale_view()
        return tuple(lines.values())

    animation = FuncAnimation(
        fig,
        _update,
        frames=len(steps),
        init_func=_init,
        interval=max(1, int(interval_ms)),
        blit=False,
        repeat=False,
    )

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = PillowWriter(fps=max(1, int(round(1000.0 / max(1.0, float(interval_ms))))))
        animation.save(str(out_path), writer=writer)

    plt.show()
    return animation

def compute_pcm_matrix(loop_queue, pcm_threshold=5.0, intensity=1.0):
    loop_count = len(loop_queue)
    if loop_count == 0:
        return np.zeros((0, 0), dtype=int)

    pcm_matrix = np.zeros((loop_count, loop_count), dtype=int)
    
    for i in range(loop_count):
        idx1, idx2, t_aj, t_bk, z_aj_bk, loop_positive = loop_queue[i]
        for j in range(i + 1, loop_count):
            idx3, idx4, t_ai, t_bl, z_ai_bl, _ = loop_queue[j]

            z_ai_aj = t_ai.between(t_aj)
            z_bk_bl = t_bk.between(t_bl)
            
            resi = residual_pcm(z_aj_bk, z_ai_bl, z_ai_aj, z_bk_bl, intensity)
            pcm_matrix[i, j] = 1 if resi < pcm_threshold else 0
    
    return pcm_matrix

def residual_pcm(inter_jk, inter_il, inner_ij, inner_kl, intensity):
    inter_il_inv = inter_il.inverse()
    res_pose = inner_ij.compose(inter_jk).compose(inner_kl).compose(inter_il_inv)
    res_vec = Pose3.Logmap(res_pose)
    
    v = np.full((6, 1), intensity)
    m_cov = np.diag(v.flatten())
    
    return sqrt(res_vec.transpose().dot(m_cov).dot(res_vec))

def visualize_inlier_loop_pairs(graph, max_clique, loop_queue):
    # Visualize the graph with the maximum clique highlighted as inliers and display loop pair information.
    if graph.number_of_nodes() == 0:
        print("Consistency graph is empty. No inlier loop pairs to visualize.")
        return

    pos = nx.spring_layout(graph, seed=84)  # Use fixed seed for consistent layout
    max_clique_subgraph = graph.subgraph(max_clique)

    plt.figure(figsize=(10, 8))
    nx.draw(graph, pos=pos, with_labels=True, node_size=700, node_color='lightgray', edge_color='gray', font_size=10)
    nx.draw(max_clique_subgraph, pos=pos, with_labels=True, node_size=700, node_color='lightgreen', edge_color='red', font_size=10)
    plt.title("Maximum Clique in Consistency Graph for Loop Closures")
    plt.show()

    # Display loop pair information for the maximum clique
    if not max_clique:
        print("No maximum clique found. No inlier loop pairs.")
        return

    print("Loop Pairs in Maximum Clique:")
    for idx in max_clique:
        loop_id = int(idx.split()[1])
        id_0, id_1, pose_1, pose_2, relative_pos, _ = loop_queue[loop_id]
        print(f"Loop Pair: ({id_0}, {id_1}), Relative Pose: {relative_pos}")
        

def generate_corrected_inlier_loop_pairs(max_clique, loop_queue):
    #Generate the loop pair information for the corrected inlier loop closures.
    if not max_clique or not loop_queue:
        return []

    corrected_inliers = []
    for idx in max_clique:
        loop_id = int(idx.split()[1])
        if loop_id >= len(loop_queue):
            continue
        id_0, id_1, pose_1, pose_2, relative_pos, _ = loop_queue[loop_id]
        corrected_inliers.append(loop_queue[loop_id])
    return corrected_inliers


def run_pcm_sliding_window(loop_queue, num_poses, window_size, window_step=1, pcm_threshold=5.0, intensity=1.0):
    """
    Run PCM in a sliding window over pose indices.
    For each pose-index window [start, end], collect loop pairs where both
    robot-local indices lie in [start, end], then run PCM on that subset.
    Returns per-step history and cumulative accepted inlier indices.
    """
    if not loop_queue or num_poses <= 0:
        return []

    history = []
    accepted_indices = set()
    safe_window_size = max(1, int(window_size))
    safe_window_step = max(1, int(window_step))
    max_start = max(0, int(num_poses) - safe_window_size)

    step_counter = 0
    for window_start in range(0, max_start + 1, safe_window_step):
        step_counter += 1
        window_end = min(int(num_poses) - 1, window_start + safe_window_size - 1)
        window_indices = []
        for global_loop_idx, (idx1, idx2, _p1, _p2, _rel_pose, _is_true_positive) in enumerate(loop_queue):
            if window_start <= int(idx1) <= window_end and window_start <= int(idx2) <= window_end:
                window_indices.append(global_loop_idx)

        window_queue = [loop_queue[idx] for idx in window_indices]

        if window_queue:
            adjacency_matrix = compute_pcm_matrix(window_queue, pcm_threshold=pcm_threshold, intensity=intensity)
            consistency_graph = generate_consistency_graph(adjacency_matrix)
            max_clique = apply_maximum_clique(consistency_graph)
            local_inliers = parse_clique_indices(max_clique)
            window_inliers = [
                window_indices[local_idx]
                for local_idx in local_inliers
                if 0 <= local_idx < len(window_indices)
            ]
        else:
            local_inliers = []
            window_inliers = []

        accepted_indices.update(window_inliers)

        history.append(
            {
                "step": int(step_counter),
                "window_pose_start": int(window_start),
                "window_pose_end": int(window_end),
                "window_indices": window_indices,
                "window_inlier_indices": sorted(window_inliers),
                "accepted_inlier_indices": sorted(accepted_indices),
                "window_max_clique_size": int(len(local_inliers)),
            }
        )

    return history


def build_graph_from_loop_indices(poses_robot1, poses_robot2, loop_queue, loop_indices):
    graph, initial_estimate = build_anchored_base_graph(poses_robot1, poses_robot2)
    loop_noise = noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    num_poses = len(poses_robot1)

    for idx in sorted(set(loop_indices)):
        if idx < 0 or idx >= len(loop_queue):
            continue
        idx1, idx2, _p1, _p2, rel_pose, _is_true_positive = loop_queue[idx]
        graph.add(BetweenFactorPose3(int(idx1), int(idx2 + num_poses), rel_pose, loop_noise))

    return graph, initial_estimate


def get_all_pose_covariance_traces(graph, values, total_keys):
    marginals = gtsam.Marginals(graph, values)
    traces = []
    for key in range(total_keys):
        try:
            cov = np.asarray(marginals.marginalCovariance(key), dtype=float)
            traces.append(float(np.trace(cov)))
        except RuntimeError:
            traces.append(float("nan"))
    return traces


def compute_sliding_window_covariance_states(poses_robot1, poses_robot2, loop_queue, sliding_history, probe_keys):
    """
    Compute covariance traces for each sliding-window step using cumulative accepted inliers.
    """
    states = []
    total_keys = len(poses_robot1) + len(poses_robot2)

    for step in sliding_history:
        graph, initial_estimate = build_graph_from_loop_indices(
            poses_robot1,
            poses_robot2,
            loop_queue,
            step["accepted_inlier_indices"],
        )
        optimized = optimize_graph(graph, initial_estimate)
        states.append(
            {
                "step": int(step["step"]),
                "window_pose_start": int(step["window_pose_start"]),
                "window_pose_end": int(step["window_pose_end"]),
                "window_indices": list(step["window_indices"]),
                "window_inlier_indices": list(step["window_inlier_indices"]),
                "accepted_inlier_indices": list(step["accepted_inlier_indices"]),
                "node_covariance_traces": get_all_pose_covariance_traces(graph, optimized, total_keys),
                "probe_covariance": get_covariance_snapshot(graph, optimized, probe_keys),
            }
        )

    return states


def animate_pose_covariance_nodes(
    poses_robot1,
    poses_robot2,
    loop_queue,
    covariance_states,
    interval_ms=300,
    save_path="",
):
    """
    Animate pose-node covariance over sliding-window PCM steps.
    - Node size/color reflect covariance trace.
    - Window true/false loop pairs are shown each frame.
    - Cumulative accepted inliers are emphasized with thicker edges.
    """
    if not covariance_states:
        return None

    num_poses = len(poses_robot1)
    total_keys = num_poses * 2
    node_positions = {}
    for i, pose in enumerate(poses_robot1):
        node_positions[i] = (pose.x(), pose.y())
    for i, pose in enumerate(poses_robot2):
        node_positions[i + num_poses] = (pose.x(), pose.y())

    odom_edges = [(i, i + 1) for i in range(num_poses - 1)] + [
        (i + num_poses, i + 1 + num_poses) for i in range(num_poses - 1)
    ]

    all_traces = []
    for state in covariance_states:
        for trace in state["node_covariance_traces"]:
            if np.isfinite(trace):
                all_traces.append(trace)
    if all_traces:
        vmin = float(min(all_traces))
        vmax = float(max(all_traces))
    else:
        vmin, vmax = 0.0, 1.0
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(11, 6))
    scalar_mappable = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    scalar_mappable.set_array([])
    fig.colorbar(scalar_mappable, ax=ax, label="Marginal covariance trace")

    def _draw_frame(frame_idx):
        state = covariance_states[frame_idx]
        ax.clear()

        # Odometry chains.
        for u, v in odom_edges:
            x1, y1 = node_positions[u]
            x2, y2 = node_positions[v]
            ax.plot([x1, x2], [y1, y2], color="lightgray", linewidth=1.8, zorder=1)

        accepted_set = set(state["accepted_inlier_indices"])
        window_set = set(state["window_indices"])
        window_inliers = set(state["window_inlier_indices"])

        # Current window loop pairs with true/false color and inlier emphasis.
        for idx in window_set:
            idx1, idx2, _p1, _p2, _rel_pose, is_true_positive = loop_queue[idx]
            u = int(idx1)
            v = int(idx2 + num_poses)
            x1, y1 = node_positions[u]
            x2, y2 = node_positions[v]
            base_color = "green" if is_true_positive else "red"
            lw = 2.6 if idx in window_inliers else 1.0
            alpha = 0.95 if idx in window_inliers else 0.4
            ax.plot([x1, x2], [y1, y2], linestyle="--", color=base_color, linewidth=lw, alpha=alpha, zorder=2)

        # Also overlay accepted historical inliers (blue) for context.
        for idx in accepted_set:
            if idx in window_set:
                continue
            idx1, idx2, _p1, _p2, _rel_pose, _is_true_positive = loop_queue[idx]
            u = int(idx1)
            v = int(idx2 + num_poses)
            x1, y1 = node_positions[u]
            x2, y2 = node_positions[v]
            ax.plot([x1, x2], [y1, y2], linestyle="--", color="royalblue", linewidth=1.2, alpha=0.35, zorder=1)

        traces = np.array(state["node_covariance_traces"], dtype=float)
        trace_safe = np.where(np.isfinite(traces), traces, vmax)
        scale = (trace_safe - vmin) / (vmax - vmin)
        sizes = 80.0 + 420.0 * scale

        x_robot1 = [node_positions[i][0] for i in range(num_poses)]
        y_robot1 = [node_positions[i][1] for i in range(num_poses)]
        x_robot2 = [node_positions[i + num_poses][0] for i in range(num_poses)]
        y_robot2 = [node_positions[i + num_poses][1] for i in range(num_poses)]

        ax.scatter(
            x_robot1,
            y_robot1,
            c=trace_safe[:num_poses],
            s=sizes[:num_poses],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            edgecolors="black",
            marker="o",
            label="Robot 1",
            zorder=3,
        )
        ax.scatter(
            x_robot2,
            y_robot2,
            c=trace_safe[num_poses:total_keys],
            s=sizes[num_poses:total_keys],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            edgecolors="black",
            marker="s",
            label="Robot 2",
            zorder=3,
        )

        ax.set_title(
            "Sliding-Window PCM Covariance Propagation | "
            f"Step {state['step']}/{len(covariance_states)} | "
            f"PoseWindow=[{state['window_pose_start']},{state['window_pose_end']}] | "
            f"WindowInliers={len(window_inliers)} | AcceptedInliers={len(accepted_set)}"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.axis("equal")
        ax.legend(loc="upper right")

    animation = FuncAnimation(
        fig,
        _draw_frame,
        frames=len(covariance_states),
        interval=max(1, int(interval_ms)),
        repeat=False,
        blit=False,
    )

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = PillowWriter(fps=max(1, int(round(1000.0 / max(1.0, float(interval_ms))))))
        animation.save(str(out_path), writer=writer)

    plt.tight_layout()
    plt.show()
    return animation

def visualize_initial_pose_graph(poses_robot1, poses_robot2, loop_queue):
    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.grid(False)
    ax.set_axis_off()
    # Plot Robot 1 poses
    old_pose = None
    for i, pose in enumerate(poses_robot1):
        x, y, z = pose.x(), pose.y(), pose.z()
        ax.plot([x], [y], [z], 'bo', label='Robot 1' if i == 0 else "")
        if old_pose is not None:
            x_o,y_o,z_o = old_pose.x(), old_pose.y(), old_pose.z()
            ax.plot([x, x_o], [y, y_o], [z,z_o], color='grey', linestyle='-')
        old_pose = pose

    # Plot Robot 2 poses
    old_pose = None
    for i, pose in enumerate(poses_robot2):
        x, y, z = pose.x(), pose.y(), pose.z()
        ax.plot([x], [y], [z], 'ro', label='Robot 2' if i == 0 else "")
        if old_pose is not None:
            x_o,y_o,z_o = old_pose.x(), old_pose.y(), old_pose.z()
            ax.plot([x, x_o], [y, y_o], [z,z_o], color='grey', linestyle='-')
        old_pose = pose

    # Adding loop closures
    for idx1, idx2, p1, p2, rel_pose, is_true_positive in loop_queue:
        # p1 = poses_robot1[idx1].translation()
        # p2 = poses_robot2[idx2].translation()
        x1, y1, z1 = p1.x(), p1.y(), p1.z()
        x2, y2, z2 = p2.x(), p2.y(), p2.z()
        color = 'g' if is_true_positive else 'r'
        ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, linestyle='--')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()

def visualize_inlier_only_pose_graph(poses_robot1, poses_robot2, corrected_inliers):
    # Visualize the pose graph with only inlier loop closures and odometry edges.
    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.grid(False)
    ax.set_axis_off()
    # Plot Robot 1 poses
    old_pose = None
    for i, pose in enumerate(poses_robot1):
        x, y, z = pose.x(), pose.y(), pose.z()
        ax.plot([x], [y], [z], 'bo', label='Robot 1' if i == 0 else "")
        if old_pose is not None:
            x_o,y_o,z_o = old_pose.x(), old_pose.y(), old_pose.z()
            ax.plot([x, x_o], [y, y_o], [z,z_o], color='grey', linestyle='-')
        old_pose = pose

    # Plot Robot 2 poses
    old_pose = None
    for i, pose in enumerate(poses_robot2):
        x, y, z = pose.x(), pose.y(), pose.z()
        ax.plot([x], [y], [z], 'ro', label='Robot 2' if i == 0 else "")
        if old_pose is not None:
            x_o,y_o,z_o = old_pose.x(), old_pose.y(), old_pose.z()
            ax.plot([x, x_o], [y, y_o], [z,z_o], color='grey', linestyle='-')
        old_pose = pose

    # Adding loop closures
    for idx1, idx2, p1, p2, rel_pose, is_true_positive in corrected_inliers:
        # p1 = poses_robot1[idx1].translation()
        # p2 = poses_robot2[idx2].translation()
        x1, y1, z1 = p1.x(), p1.y(), p1.z()
        x2, y2, z2 = p2.x(), p2.y(), p2.z()
        color = 'g' if is_true_positive else 'r'
        ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, linestyle='--')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()

def run_demo(config=None):
    if config is None:
        config = GTSAMPCMConfig()

    # Generate poses with rotation and position for Robot 1 and Robot 2
    poses_robot1 = []
    poses_robot2 = []
    for i in range(config.num_poses):
        x1, y1, z1 = i * 1.0, sin(i * 0.1) * 10, 0.0
        yaw1 = i * config.yaw_increment
        poses_robot1.append(create_pose(x1, y1, z1, yaw1))

        x2, y2, z2 = i * 1.0 + 2.0, (i * 0.1) * 10, 0.0
        yaw2 = i * config.yaw_increment
        poses_robot2.append(create_pose(x2, y2, z2, yaw2))

    # Create a factor graph and add initial poses
    graph = NonlinearFactorGraph()
    initial_estimate = Values()

    # Add poses to initial estimates
    for i, pose in enumerate(poses_robot1):
        initial_estimate.insert(i, pose)
    for i, pose in enumerate(poses_robot2):
        initial_estimate.insert(i + config.num_poses, pose)

    # Noise model for loop closures (small standard deviation for translation and rotation)
    loop_closure_noise = noiseModel.Diagonal.Sigmas([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])  # XYZRPY
    faulty_loop_closure_noise = noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    # Generate loop closures with noise, checking Euclidean distance
    np.random.seed(config.seed)
    loop_queue = []
    positive_sort = None

    for i in range(config.num_loop_attempts):
        idx1 = np.random.randint(0, config.num_poses - 1)
        if i < config.true_positive_attempts:
            step = np.random.choice([1, 2])
            if np.random.rand() > 0.5:
                idx2 = (idx1 + step) % config.num_poses
            else:
                idx2 = (idx1 - step) % config.num_poses
            positive_sort = True
            # Distance calculation, only 2D info to filter far euclidean distance in true pair.
            p1 = poses_robot1[idx1]
            p2 = poses_robot2[idx2]
            distance = sqrt((p1.x() - p2.x()) ** 2 + (p1.y() - p2.y()) ** 2)
            if abs(idx1 - idx2) > 2.0 or (distance > config.max_distance):
                positive_sort = False
        else:
            step = np.random.randint(2, 5)
            idx2 = (idx1 + step) % config.num_poses if np.random.rand() > 0.5 else (idx1 - step) % config.num_poses
            positive_sort = False
            p1 = poses_robot1[idx1]
            p2 = poses_robot2[idx2]
            distance = sqrt((p1.x() - p2.x()) ** 2 + (p1.y() - p2.y()) ** 2)
            # Remove false-positive pair that are too far.
            if distance > 15.0:
                continue
            if abs(idx1 - idx2) > 4.0 or (distance < 3.0):
                continue

        dx, dy, dz = np.random.uniform(-0.1, 0.1, 3)
        droll, dpitch, dyaw = np.random.uniform(0, 0.1, 3)
        delta_rotation = Rot3.RzRyRx(droll, dpitch, dyaw)
        delta_translation = Point3(dx, dy, dz)
        noisy_transform = Pose3(delta_rotation, delta_translation)
        # Add to loop queue with true/false positive flag.
        loop_queue.append(GTSAMLoopClosure(idx1, idx2, p1, p2, noisy_transform, positive_sort))
        # Add between factor to graph.
        robot1_key = idx1
        robot2_key = idx2 + config.num_poses

        # TODO: add distributed pose-graph optimization scheme.
        # Add noise for false positive and true positive in graph.
        if positive_sort is False:
            graph.add(BetweenFactorPose3(robot1_key, robot2_key, noisy_transform, faulty_loop_closure_noise))
        else:
            graph.add(BetweenFactorPose3(robot1_key, robot2_key, noisy_transform, loop_closure_noise))

    # Start processing the pose graph.
    max_clique = []
    selected_indices = []
    sliding_window_history = []
    pose_covariance_animation_file = ""

    if config.use_sliding_window_pcm:
        sliding_window_history = run_pcm_sliding_window(
            loop_queue,
            num_poses=config.num_poses,
            window_size=config.pcm_window_size,
            window_step=config.pcm_window_step,
            pcm_threshold=config.pcm_threshold,
            intensity=config.intensity,
        )
        if sliding_window_history:
            selected_indices = sliding_window_history[-1]["accepted_inlier_indices"]
        corrected_inlier_loop_pairs = [
            loop_queue[idx] for idx in selected_indices if 0 <= idx < len(loop_queue)
        ]
        max_clique = [f"Loop {idx}" for idx in selected_indices]
    else:
        if config.visualize:
            visualize_initial_pose_graph(poses_robot1, poses_robot2, loop_queue)

        adjacency_matrix = compute_pcm_matrix(loop_queue, pcm_threshold=config.pcm_threshold, intensity=config.intensity)
        consistency_graph = generate_consistency_graph(adjacency_matrix)
        max_clique = apply_maximum_clique(consistency_graph)
        selected_indices = parse_clique_indices(max_clique)

        if config.visualize:
            visualize_inlier_loop_pairs(consistency_graph, max_clique, loop_queue)

        corrected_inlier_loop_pairs = generate_corrected_inlier_loop_pairs(max_clique, loop_queue)
        if config.visualize:
            visualize_inlier_only_pose_graph(poses_robot1, poses_robot2, corrected_inlier_loop_pairs)

    print("\nCorrected Inlier Loop Pairs:")
    for pair in corrected_inlier_loop_pairs:
        print(f"Loop Pair: ({pair.idx_a}, {pair.idx_b}), Relative Pose: {pair.relative_pose}")

    covariance_progression = []
    covariance_shrink_summary = []
    covariance_animation_file = ""
    if config.track_covariance_progression:
        probe_local_indices = sorted(
            set(
                int(i)
                for i in config.covariance_probe_indices
                if 0 <= int(i) < config.num_poses
            )
        )
        if not probe_local_indices:
            probe_local_indices = [max(0, config.num_poses // 2)]
        probe_keys = probe_local_indices + [i + config.num_poses for i in probe_local_indices]

        if config.use_sliding_window_pcm:
            covariance_states = compute_sliding_window_covariance_states(
                poses_robot1,
                poses_robot2,
                loop_queue,
                sliding_window_history,
                probe_keys,
            )
            covariance_progression = [
                {
                    "step": int(state["step"]),
                    "used_inlier_factors": int(len(state["accepted_inlier_indices"])),
                    "covariance": state["probe_covariance"],
                }
                for state in covariance_states
            ]
        else:
            realtime_plot_state = None
            if config.visualize and config.realtime_covariance_plot:
                realtime_plot_state = create_realtime_covariance_plot(probe_keys)

            def _on_cov_step(step_entry):
                if realtime_plot_state is not None:
                    update_realtime_covariance_plot(
                        realtime_plot_state,
                        step_entry,
                        pause_sec=config.realtime_pause_sec,
                    )

            covariance_progression = compute_covariance_progression(
                poses_robot1,
                poses_robot2,
                corrected_inlier_loop_pairs,
                probe_keys,
                on_step_callback=_on_cov_step,
            )

        # Summarize start/end shrinkage for each probed key.
        if covariance_progression:
            start_cov = covariance_progression[0]["covariance"]
            end_cov = covariance_progression[-1]["covariance"]
            for start_entry in start_cov:
                key = start_entry["key"]
                end_entry = next((item for item in end_cov if item["key"] == key), None)
                if end_entry is None:
                    continue
                start_trace = start_entry["trace"]
                end_trace = end_entry["trace"]
                shrink_ratio = (
                    float(end_trace / start_trace)
                    if np.isfinite(start_trace) and start_trace > 0 and np.isfinite(end_trace)
                    else float("nan")
                )
                covariance_shrink_summary.append(
                    {
                        "key": int(key),
                        "start_trace": float(start_trace),
                        "end_trace": float(end_trace),
                        "shrink_ratio_end_over_start": float(shrink_ratio),
                    }
                )

        if config.visualize:
            plot_covariance_progression(covariance_progression)
            if config.use_sliding_window_pcm and config.animate_pose_covariance_nodes:
                if config.save_pose_covariance_animation:
                    if config.pose_covariance_animation_path:
                        pose_covariance_animation_file = config.pose_covariance_animation_path
                    else:
                        pose_covariance_animation_file = str(
                            Path(config.output_dir) / f"{config.run_tag}_pose_covariance_animation.gif"
                        )
                animate_pose_covariance_nodes(
                    poses_robot1,
                    poses_robot2,
                    loop_queue,
                    covariance_states,
                    interval_ms=config.animation_interval_ms,
                    save_path=pose_covariance_animation_file,
                )
            if config.animate_covariance_progression:
                if config.save_covariance_animation:
                    if config.covariance_animation_path:
                        covariance_animation_file = config.covariance_animation_path
                    else:
                        covariance_animation_file = str(
                            Path(config.output_dir) / f"{config.run_tag}_covariance_animation.gif"
                        )
                animate_covariance_progression(
                    covariance_progression,
                    interval_ms=config.animation_interval_ms,
                    save_path=covariance_animation_file,
                )

    truth_labels = [bool(entry.is_true_positive) for entry in loop_queue]
    metrics = compute_metrics(selected_indices, truth_labels, len(loop_queue))
    run_results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "generated_loops": int(len(loop_queue)),
        "max_clique": list(max_clique),
        "max_clique_size": int(len(max_clique)),
        "selected_indices": selected_indices,
        "truth_labels": truth_labels,
        "metrics": metrics,
        "sliding_window_history": sliding_window_history,
        "covariance_progression": covariance_progression,
        "covariance_shrink_summary": covariance_shrink_summary,
        "covariance_animation_file": covariance_animation_file,
        "pose_covariance_animation_file": pose_covariance_animation_file,
    }

    print("\nRun Metrics:")
    print(
        "Precision={:.3f}, Recall={:.3f}, F1={:.3f}, RejectionRatio={:.3f}, MaxCliqueSize={}".format(
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["rejection_ratio"],
            len(max_clique),
        )
    )
    if covariance_shrink_summary:
        print("Covariance trace shrink summary (end/start):")
        for entry in covariance_shrink_summary:
            print(
                "  key {}: start={:.6f}, end={:.6f}, ratio={:.6f}".format(
                    entry["key"],
                    entry["start_trace"],
                    entry["end_trace"],
                    entry["shrink_ratio_end_over_start"],
                )
            )

    if config.save_results:
        run_id, json_path, csv_path = save_run_artifacts(run_results, config.output_dir, config.run_tag)
        print(f"Saved run artifacts: run_id={run_id}")
        print(f"- JSON: {json_path}")
        print(f"- CSV:  {csv_path}")

    return run_results


# Example
if __name__ == "__main__":
    run_demo()

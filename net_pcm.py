import itertools
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import NamedTuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation as R

from pcm_common import (
    apply_maximum_clique,
    compute_metrics,
    generate_consistency_graph,
    parse_clique_indices,
    save_run_artifacts,
)


@dataclass
class NetPCMConfig:
    seed: int = 42
    pcm_threshold: float = 5.0
    intensity: float = 1.0
    visualize: bool = True
    save_results: bool = True
    output_dir: str = "results"
    run_tag: str = "net_pcm_demo"


class NetLoopClosure(NamedTuple):
    idx_a: int
    idx_b: int
    relative_pose: list


# Step 1: Import Loop Pair Information
def import_loop_pairs(loop_queue):
    """
    Import loop pair information. Each loop pair should be represented as a tuple (id_0, id_1, relative_pose).
    """
    return loop_queue

# Step 2: Generate Adjacency Matrix
def generate_adjacency_matrix(loop_queue, pcm_threshold=5.0, intensity=1.0):
    """
    Generate the adjacency matrix from loop pair information based on consistency checks.
    """
    loop_count = len(loop_queue)
    if loop_count == 0:
        return np.zeros((0, 0))
    adjacency_matrix = np.zeros((loop_count, loop_count))

    def pose3_between(pose1, pose2):
        r1 = R.from_euler('xyz', pose1[:3])
        r2 = R.from_euler('xyz', pose2[:3])
        relative_rotation = r1.inv() * r2
        relative_translation = np.array(pose2[3:]) - np.array(pose1[3:])
        return list(relative_rotation.as_euler('xyz')) + list(relative_translation)

    def residualPCM(inter_jk, inter_il, inner_ij, inner_kl, intensity):
        v = np.array([intensity] * 6)
        m_cov = np.diag(v)
        res_pose = [ij + jk + kl - il for ij, jk, kl, il in zip(inner_ij, inter_jk, inner_kl, inter_il)]
        res_vec = np.array(res_pose, dtype=np.float64)
        return np.sqrt(res_vec.T @ m_cov @ res_vec)

    for i, j in itertools.combinations(range(loop_count), 2):
        id_0_i, id_1_i, z_aj_bk = loop_queue[i]
        id_0_j, id_1_j, z_ai_bl = loop_queue[j]
        t_aj = [0, 0, 0, id_0_i, id_0_i, id_0_i]
        t_ai = [0, 0, 0, id_0_j, id_0_j, id_0_j]
        t_bk = [0, 0, 0, id_1_i, id_1_i, id_1_i]
        t_bl = [0, 0, 0, id_1_j, id_1_j, id_1_j]

        z_ai_aj = pose3_between(t_ai, t_aj)
        z_bk_bl = pose3_between(t_bk, t_bl)

        resi = residualPCM(z_aj_bk, z_ai_bl, z_ai_aj, z_bk_bl, intensity)
        if resi < pcm_threshold:
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1

    return adjacency_matrix


# Step 5: Visualize Initial Pose Graph with Odometry and Loop Closures
def visualize_initial_pose_graph(loop_queue):
    """
    Visualize the initial pose graph with odometry and loop closures.
    """
    graph = nx.Graph()

    # Add nodes for robots A and B
    nodes_A = [f"A{i + 1}" for i in range(5)]
    nodes_B = [f"B{i + 1}" for i in range(5)]
    graph.add_nodes_from(nodes_A)
    graph.add_nodes_from(nodes_B)

    # Add odometry edges (horizontal edges)
    odometry_edges = [(nodes_A[i], nodes_A[i + 1]) for i in range(4)] + [(nodes_B[i], nodes_B[i + 1]) for i in range(4)]
    graph.add_edges_from(odometry_edges)

    # Add loop closure edges (colored dotted lines)
    loop_colors = ['cyan', 'blue', 'red', 'green', 'magenta', 'orange', 'brown', 'pink', 'purple', 'olive']
    for idx, (id_0, id_1, _) in enumerate(loop_queue):
        node_1 = f"A{id_0 + 1}" if id_0 < 5 else f"B{id_0 - 4}"
        node_2 = f"A{id_1 + 1}" if id_1 < 5 else f"B{id_1 - 4}"
        graph.add_edge(node_1, node_2, style='dotted', color=loop_colors[idx % len(loop_colors)])

    # Define positions for nodes to be linear and perpendicular
    pos = {f"A{i + 1}": (i, 1) for i in range(5)}
    pos.update({f"B{i + 1}": (i, 0) for i in range(5)})

    # Plot the initial pose graph
    plt.figure(figsize=(10, 6))
    nx.draw_networkx_edges(graph, pos, edgelist=odometry_edges, width=2, edge_color='black')
    for idx, (id_0, id_1, _) in enumerate(loop_queue):
        node_1 = f"A{id_0 + 1}" if id_0 < 5 else f"B{id_0 - 4}"
        node_2 = f"A{id_1 + 1}" if id_1 < 5 else f"B{id_1 - 4}"
        nx.draw_networkx_edges(graph, pos, edgelist=[(node_1, node_2)], style='dotted', edge_color=loop_colors[idx % len(loop_colors)], width=1.5)
    nx.draw_networkx_nodes(graph, pos, node_size=1000, node_color='white', edgecolors='black')
    nx.draw_networkx_labels(graph, pos, font_size=12, font_family='sans-serif')
    plt.title("Corrected Inlier Loop Closures with Odometry and Consistent Loop Closures")
    plt.axis('off')
    plt.show()

# Step 6: Visualize Inlier Loop Pairs and Loop Pair Information
def visualize_inlier_loop_pairs(graph, max_clique, loop_queue):
    """
    Visualize the graph with the maximum clique highlighted as inliers and display loop pair information.
    """
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
        id_0, id_1, relative_pose = loop_queue[loop_id]
        print(f"Loop Pair: ({id_0}, {id_1}), Relative Pose: {relative_pose}")

# Step 7: Generate Loop Pair Information for Corrected Inlier Loop Closures
def generate_corrected_inlier_loop_pairs(max_clique, loop_queue):
    """
    Generate the loop pair information for the corrected inlier loop closures.
    """
    if not max_clique or not loop_queue:
        return []

    corrected_inliers = []
    for idx in max_clique:
        loop_id = int(idx.split()[1])
        if loop_id >= len(loop_queue):
            continue
        corrected_inliers.append(loop_queue[loop_id])
    return corrected_inliers

# Step 8: Visualize Inlier Only Pose Graph
def visualize_inlier_only_pose_graph(loop_queue, corrected_inliers):
    """
    Visualize the pose graph with only inlier loop closures and odometry edges.
    """
    graph = nx.Graph()

    # Add nodes for robots A and B
    nodes_A = [f"A{i + 1}" for i in range(5)]
    nodes_B = [f"B{i + 1}" for i in range(5)]
    graph.add_nodes_from(nodes_A)
    graph.add_nodes_from(nodes_B)

    # Add odometry edges (horizontal edges)
    odometry_edges = [(nodes_A[i], nodes_A[i + 1]) for i in range(4)] + [(nodes_B[i], nodes_B[i + 1]) for i in range(4)]
    graph.add_edges_from(odometry_edges)

    # Add inlier loop closure edges (colored dotted lines)
    loop_colors = ['cyan', 'blue', 'red', 'green', 'magenta', 'orange', 'brown', 'pink', 'purple', 'olive']
    for idx, (id_0, id_1, _) in enumerate(corrected_inliers):
        node_1 = f"A{id_0 + 1}" if id_0 < 5 else f"B{id_0 - 4}"
        node_2 = f"A{id_1 + 1}" if id_1 < 5 else f"B{id_1 - 4}"
        graph.add_edge(node_1, node_2, style='dotted', color=loop_colors[idx % len(loop_colors)])

    # Define positions for nodes to be linear and perpendicular
    pos = {f"A{i + 1}": (i, 1) for i in range(5)}
    pos.update({f"B{i + 1}": (i, 0) for i in range(5)})

    # Plot the inlier-only pose graph
    plt.figure(figsize=(10, 6))
    nx.draw_networkx_edges(graph, pos, edgelist=odometry_edges, width=2, edge_color='black')
    for idx, (id_0, id_1, _) in enumerate(corrected_inliers):
        node_1 = f"A{id_0 + 1}" if id_0 < 5 else f"B{id_0 - 4}"
        node_2 = f"A{id_1 + 1}" if id_1 < 5 else f"B{id_1 - 4}"
        nx.draw_networkx_edges(graph, pos, edgelist=[(node_1, node_2)], style='dotted', edge_color=loop_colors[idx % len(loop_colors)], width=1.5)
    nx.draw_networkx_nodes(graph, pos, node_size=1000, node_color='white', edgecolors='black')
    nx.draw_networkx_labels(graph, pos, font_size=12, font_family='sans-serif')
    plt.title("Pose Graph with Inlier Loop Closures and Odometry")
    plt.axis('off')
    plt.show()

def get_example_loop_data():
    loop_queue_example = [
        NetLoopClosure(0, 5, [0, 0, 0, 1, 0, 0]),
        NetLoopClosure(1, 7, [0, 0.1, 0, 1, 1, 0]),
        NetLoopClosure(2, 6, [0.2, 0, 0, 0, 1, 1]),
        NetLoopClosure(3, 8, [0, 0.1, -0.1, 2, 2, 0]),
        NetLoopClosure(4, 9, [0, 0, 0, 3, 3, 1]),
        NetLoopClosure(0, 6, [0, -0.1, 0, 1, 0, 1]),
        NetLoopClosure(1, 8, [0.1, 0, 0, 2, 1, 0]),
        NetLoopClosure(2, 9, [-0.1, 0, 0, 1, 2, 1]),
        NetLoopClosure(3, 7, [0.1, -0.1, 0, 2, 1, 1]),
        NetLoopClosure(4, 5, [0, 0.1, 0, 3, 0, 1])
    ]
    # Ground-truth labels for the synthetic set above.
    loop_truth_labels = [True, True, True, True, False, True, False, True, False, False]
    return loop_queue_example, loop_truth_labels


def run_demo(config=None):
    if config is None:
        config = NetPCMConfig()

    np.random.seed(config.seed)
    loop_queue_example, loop_truth_labels = get_example_loop_data()
    loop_queue = import_loop_pairs(loop_queue_example)

    if config.visualize:
        # Step 5: Visualize initial pose graph with odometry and loop closures
        visualize_initial_pose_graph(loop_queue)

    # Step 2: Generate adjacency matrix
    adjacency_matrix = generate_adjacency_matrix(
        loop_queue,
        pcm_threshold=config.pcm_threshold,
        intensity=config.intensity,
    )

    # Step 3: Generate consistency graph
    consistency_graph = generate_consistency_graph(adjacency_matrix)

    # Step 4: Apply maximum clique problem
    max_clique = apply_maximum_clique(consistency_graph)

    if config.visualize:
        # Step 6: Visualize inlier loop pairs and loop pair information
        visualize_inlier_loop_pairs(consistency_graph, max_clique, loop_queue)

    # Step 7: Generate loop pair information for corrected inlier loop closures
    corrected_inlier_loop_pairs = generate_corrected_inlier_loop_pairs(max_clique, loop_queue)
    print("\nCorrected Inlier Loop Pairs:")
    for pair in corrected_inlier_loop_pairs:
        print(f"Loop Pair: ({pair.idx_a}, {pair.idx_b}), Relative Pose: {pair.relative_pose}")

    if config.visualize:
        # Step 8: Visualize inlier only pose graph
        visualize_inlier_only_pose_graph(loop_queue, corrected_inlier_loop_pairs)

    selected_indices = parse_clique_indices(max_clique)
    metrics = compute_metrics(selected_indices, loop_truth_labels, len(loop_queue))
    run_results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "max_clique": list(max_clique),
        "max_clique_size": int(len(max_clique)),
        "selected_indices": selected_indices,
        "truth_labels": loop_truth_labels,
        "metrics": metrics,
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

    if config.save_results:
        run_id, json_path, csv_path = save_run_artifacts(run_results, config.output_dir, config.run_tag)
        print(f"Saved run artifacts: run_id={run_id}")
        print(f"- JSON: {json_path}")
        print(f"- CSV:  {csv_path}")

    return run_results


if __name__ == "__main__":
    run_demo()

# Import necessary libraries
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from networkx.algorithms.clique import find_cliques
import itertools

# Set random seed for consistency
np.random.seed(42)

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

# Step 3: Generate Consistency Graph
def generate_consistency_graph(adjacency_matrix):
    """
    Generate the consistency graph based on the adjacency matrix.
    """
    loop_count = len(adjacency_matrix)
    graph = nx.Graph()
    graph.add_nodes_from([f"Loop {i}" for i in range(loop_count)])
    for i in range(loop_count):
        for j in range(i + 1, loop_count):
            if adjacency_matrix[i, j] == 1:
                graph.add_edge(f"Loop {i}", f"Loop {j}")
    return graph

# Step 4: Apply Maximum Clique Problem
def apply_maximum_clique(graph):
    """
    Apply the maximum clique algorithm to find the largest set of mutually consistent loop closures.
    """
    all_cliques = list(find_cliques(graph))
    max_clique = max(all_cliques, key=len)
    return max_clique

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
    pos = nx.spring_layout(graph, seed=84)  # Use fixed seed for consistent layout
    max_clique_subgraph = graph.subgraph(max_clique)

    plt.figure(figsize=(10, 8))
    nx.draw(graph, pos=pos, with_labels=True, node_size=700, node_color='lightgray', edge_color='gray', font_size=10)
    nx.draw(max_clique_subgraph, pos=pos, with_labels=True, node_size=700, node_color='lightgreen', edge_color='red', font_size=10)
    plt.title("Maximum Clique in Consistency Graph for Loop Closures")
    plt.show()

    # Display loop pair information for the maximum clique
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
    corrected_inliers = []
    for idx in max_clique:
        loop_id = int(idx.split()[1])
        id_0, id_1, relative_pose = loop_queue[loop_id]
        corrected_inliers.append((id_0, id_1, relative_pose))
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

# Example Usage
if __name__ == "__main__":
    # Example loop pair information (id_0, id_1, relative_pose)
    loop_queue_example = [
        (0, 5, [0, 0, 0, 1, 0, 0]),
        (1, 7, [0, 0.1, 0, 1, 1, 0]),
        (2, 6, [0.2, 0, 0, 0, 1, 1]),
        (3, 8, [0, 0.1, -0.1, 2, 2, 0]),
        (4, 9, [0, 0, 0, 3, 3, 1]),
        (0, 6, [0, -0.1, 0, 1, 0, 1]),
        (1, 8, [0.1, 0, 0, 2, 1, 0]),
        (2, 9, [-0.1, 0, 0, 1, 2, 1]),
        (3, 7, [0.1, -0.1, 0, 2, 1, 1]),
        (4, 5, [0, 0.1, 0, 3, 0, 1])
    ]

    # Step 1: Import loop pair information
    loop_queue = import_loop_pairs(loop_queue_example)

    # Step 5: Visualize initial pose graph with odometry and loop closures
    visualize_initial_pose_graph(loop_queue)

    # Step 2: Generate adjacency matrix
    adjacency_matrix = generate_adjacency_matrix(loop_queue)

    # Step 3: Generate consistency graph
    consistency_graph = generate_consistency_graph(adjacency_matrix)

    # Step 4: Apply maximum clique problem
    max_clique = apply_maximum_clique(consistency_graph)

    # Step 6: Visualize inlier loop pairs and loop pair information
    visualize_inlier_loop_pairs(consistency_graph, max_clique, loop_queue)

    # Step 7: Generate loop pair information for corrected inlier loop closures
    corrected_inlier_loop_pairs = generate_corrected_inlier_loop_pairs(max_clique, loop_queue)
    print("\nCorrected Inlier Loop Pairs:")
    for pair in corrected_inlier_loop_pairs:
        print(f"Loop Pair: ({pair[0]}, {pair[1]}), Relative Pose: {pair[2]}")

    # Step 8: Visualize inlier only pose graph
    visualize_inlier_only_pose_graph(loop_queue, corrected_inlier_loop_pairs)

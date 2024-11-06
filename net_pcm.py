import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import networkx as nx

# Define a function to draw ellipses based on covariance matrices
def draw_ellipse(ax, pos, cov, scale=0.05, **kwargs):
    """Draw an ellipse representing a scaled covariance matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * scale * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=pos, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

# Initialize initial covariances for each state node
np.random.seed(42)
initial_cov_x_i_a = np.array([[0.5 + 0.1 * np.random.randn(), 0.1 * np.random.randn()],
                              [0.1 * np.random.randn(), 0.5 + 0.1 * np.random.randn()]])
initial_cov_x_j_a = np.array([[0.6 + 0.1 * np.random.randn(), 0.2 * np.random.randn()],
                              [0.2 * np.random.randn(), 0.6 + 0.1 * np.random.randn()]])
initial_cov_x_k_b = np.array([[0.4 + 0.1 * np.random.randn(), 0.15 * np.random.randn()],
                              [0.15 * np.random.randn(), 0.4 + 0.1 * np.random.randn()]])
initial_cov_x_l_b = np.array([[0.7 + 0.1 * np.random.randn(), 0.25 * np.random.randn()],
                              [0.25 * np.random.randn(), 0.7 + 0.1 * np.random.randn()]])

# Scaling factor for covariance reduction
scaling_factor = 0.8
num_iterations = 100 # Number of steps to visualize

# Define graph structure
G = nx.DiGraph()
nodes = ["x_i^a", "x_j^a", "x_k^b", "x_l^b"]
edges = [("x_i^a", "x_j^a", {"label": "intra-robot A"}),
         ("x_k^b", "x_l^b", {"label": "intra-robot B"}),
         ("x_i^a", "x_k^b", {"label": "z_ik^ab"}),
         ("x_j^a", "x_l^b", {"label": "z_jl^ab"})]
G.add_nodes_from(nodes)
G.add_edges_from(edges)
pos = {"x_i^a": (0, 1), "x_j^a": (1, 1), "x_k^b": (1, 0), "x_l^b": (0, 0)}

# Sequential plots showing covariance propagation
for iteration in range(num_iterations):
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d["label"] for u, v, d in G.edges(data=True)}, ax=ax)
    ax.set_title(f"Covariance Propagation - Iteration {iteration + 1}")

    # Scale each covariance to simulate reduction in uncertainty
    current_cov_x_i_a = initial_cov_x_i_a * (scaling_factor ** iteration)
    current_cov_x_j_a = initial_cov_x_j_a * (scaling_factor ** iteration)
    current_cov_x_k_b = initial_cov_x_k_b * (scaling_factor ** iteration)
    current_cov_x_l_b = initial_cov_x_l_b * (scaling_factor ** iteration)

    # Draw ellipses for each state
    draw_ellipse(ax, pos["x_i^a"], current_cov_x_i_a, scale=0.5, edgecolor='blue', facecolor='none', linestyle='-', linewidth=2)
    draw_ellipse(ax, pos["x_j^a"], current_cov_x_j_a, scale=0.5, edgecolor='green', facecolor='none', linestyle='-', linewidth=2)
    draw_ellipse(ax, pos["x_k^b"], current_cov_x_k_b, scale=0.5, edgecolor='red', facecolor='none', linestyle='-', linewidth=2)
    draw_ellipse(ax, pos["x_l^b"], current_cov_x_l_b, scale=0.5, edgecolor='purple', facecolor='none', linestyle='-', linewidth=2)

    # Set plot limits and show
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    plt.show()

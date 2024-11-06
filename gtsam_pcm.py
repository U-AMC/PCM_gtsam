#!/usr/bin/env python3
# coding=utf8
#written by Jason Kim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import gtsam
from gtsam import symbol

# Function to draw ellipses based on covariance matrices
def draw_ellipse(ax, pos, cov, scale=0.5, **kwargs):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * scale * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=pos, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

# Adjusted initial covariances with more variability for each node
np.random.seed(42)
initial_covariances = {
    'x_i_a': np.array([[0.7 + 0.3 * np.random.rand(), 0.2 * np.random.rand()],
                       [0.2 * np.random.rand(), 0.5 + 0.3 * np.random.rand()]]),
    'x_j_a': np.array([[0.9 + 0.4 * np.random.rand(), 0.3 * np.random.rand()],
                       [0.3 * np.random.rand(), 0.6 + 0.4 * np.random.rand()]]),
    'x_k_b': np.array([[0.4 + 0.5 * np.random.rand(), 0.15 * np.random.rand()],
                       [0.15 * np.random.rand(), 0.4 + 0.5 * np.random.rand()]]),
    'x_l_b': np.array([[0.8 + 0.6 * np.random.rand(), 0.25 * np.random.rand()],
                       [0.25 * np.random.rand(), 0.7 + 0.5 * np.random.rand()]])
}

# Ensure each matrix is symmetric (as required for covariance matrices)
for key in initial_covariances:
    cov = initial_covariances[key]
    initial_covariances[key] = (cov + cov.T) / 2  # Symmetrize each matrix

# Scaling factor for covariance reduction
scaling_factor = 0.8
num_iterations = 50  # Number of steps to visualize

# Define the initial positions for visualization
positions = {'x_i_a': (0, 1), 'x_j_a': (1, 1), 'x_k_b': (1, 0), 'x_l_b': (0, 0)}

# Define symbolic keys for each node
keys = {
    'x_i_a': symbol('a', 0),  # 'a' represents robot A, 0 is the unique index
    'x_j_a': symbol('a', 1),  # 'a' for robot A, 1 as index
    'x_k_b': symbol('b', 0),  # 'b' represents robot B, 0 as index
    'x_l_b': symbol('b', 1)   # 'b' for robot B, 1 as index
}

# Define measurement noise models
intra_robot_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1]))
inter_robot_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.15, 0.15]))

# Set up the initial estimate for each node
initial_estimate = gtsam.Values()
for key in keys.values():
    initial_estimate.insert(key, gtsam.Point2(0, 0))  # Assuming initial (0,0) for simplicity

# Iterate to create and optimize the graph at each step
for iteration in range(num_iterations):
    # Create a new factor graph for each iteration to update prior covariances
    graph = gtsam.NonlinearFactorGraph()

    # Scale covariance for each prior to simulate reduction in uncertainty
    for node, key in keys.items():
        scaled_cov = initial_covariances[node] * (scaling_factor ** iteration)
        noise_model = gtsam.noiseModel.Gaussian.Covariance(scaled_cov)
        graph.add(gtsam.PriorFactorPoint2(key, gtsam.Point2(0, 0), noise_model))

    # Add intra- and inter-robot factors (edges)
    graph.add(gtsam.BetweenFactorPoint2(keys['x_i_a'], keys['x_j_a'], gtsam.Point2(1, 0), intra_robot_noise))
    graph.add(gtsam.BetweenFactorPoint2(keys['x_k_b'], keys['x_l_b'], gtsam.Point2(-1, 0), intra_robot_noise))
    graph.add(gtsam.BetweenFactorPoint2(keys['x_i_a'], keys['x_k_b'], gtsam.Point2(1, -1), inter_robot_noise))
    graph.add(gtsam.BetweenFactorPoint2(keys['x_j_a'], keys['x_l_b'], gtsam.Point2(0, -1), inter_robot_noise))

    # Optimize
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()

    # Plot current state
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Covariance Propagation - Iteration {iteration + 1}")

    # Draw connections (edges) between nodes
    for (start, end) in [('x_i_a', 'x_j_a'), ('x_k_b', 'x_l_b'), ('x_i_a', 'x_k_b'), ('x_j_a', 'x_l_b')]:
        ax.plot(
            [positions[start][0], positions[end][0]], 
            [positions[start][1], positions[end][1]], 
            'k--', linewidth=1, label=f"{start} to {end}" if iteration == 0 else ""
        )

    # Extract marginal covariances and draw ellipses
    marginals = gtsam.Marginals(graph, result)
    for node, key in keys.items():
        mean = positions[node]
        covariance = marginals.marginalCovariance(key)
        draw_ellipse(ax, mean, covariance, scale=0.5, edgecolor='blue', facecolor='none', linestyle='-', linewidth=2)

    # Draw nodes
    for node, (x, y) in positions.items():
        ax.plot(x, y, 'o', markersize=10, label=node)
        ax.text(x + 0.05, y + 0.05, node, fontsize=12, ha='right')

    # Set plot limits and labels
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.legend(loc="upper left")
    plt.show()

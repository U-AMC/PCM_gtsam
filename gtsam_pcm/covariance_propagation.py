import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gtsam
from gtsam import Pose3, Rot3, Point3, noiseModel, BetweenFactorPose3, NonlinearFactorGraph, Values, LevenbergMarquardtParams, LevenbergMarquardtOptimizer
from math import sin, cos, sqrt
import networkx as nx
from networkx.algorithms.clique import find_cliques
import itertools

# Create a Pose3 given position (x, y, z) and yaw rotation (in rad)
def create_pose(x, y, z, yaw):
    rotation = Rot3.Yaw(yaw)
    translation = Point3(x, y, z)
    return Pose3(rotation, translation)

def compute_pcm_matrix(loop_queue, pcm_threshold=5.0, intensity=1.0):
    pcm_matrix = np.zeros((len(loop_queue), len(loop_queue)), dtype=int)
    
    for i in range(len(loop_queue)):
        idx1, idx2, t_aj, t_bk, z_aj_bk, loop_positive = loop_queue[i]
        for j in range(i + 1, len(loop_queue)):
            idx3, idx4, t_ai, t_bl, z_ai_bl, _ = loop_queue[j]

            z_ai_aj = t_ai.between(t_aj)
            z_bk_bl = t_bk.between(t_bl)
            
            resi = residual_pcm(z_aj_bk, z_ai_bl, z_ai_aj, z_bk_bl, 1)
            pcm_matrix[i, j] = 1 if resi < pcm_threshold else 0
    
    return pcm_matrix

def residual_pcm(inter_jk, inter_il, inner_ij, inner_kl, intensity):
    inter_il_inv = inter_il.inverse()
    res_pose = inner_ij.compose(inter_jk).compose(inner_kl).compose(inter_il_inv)
    res_vec = Pose3.Logmap(res_pose)
    
    v = np.full((6, 1), intensity)
    m_cov = np.diag(v.flatten())
    
    return sqrt(res_vec.transpose().dot(m_cov).dot(res_vec))

def generate_consistency_graph(adjacency_matrix):
    # Generate the consistency graph based on the adjacency matrix.
    loop_count = len(adjacency_matrix)
    graph = nx.Graph()
    graph.add_nodes_from([f"Loop {i}" for i in range(loop_count)])
    for i in range(loop_count):
        for j in range(i + 1, loop_count):
            if adjacency_matrix[i, j] == 1:
                graph.add_edge(f"Loop {i}", f"Loop {j}")
    return graph

def apply_maximum_clique(graph):
    #Apply the maximum clique algorithm to find the largest set of mutually consistent loop closures.
    all_cliques = list(find_cliques(graph))
    max_clique = max(all_cliques, key=len)
    return max_clique

def visualize_inlier_loop_pairs(graph, max_clique, loop_queue):
    # Visualize the graph with the maximum clique highlighted as inliers and display loop pair information.
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
        id_0, id_1, pose_1, pose_2, relative_pos, _ = loop_queue[loop_id]
        print(f"Loop Pair: ({id_0}, {id_1}), Relative Pose: {relative_pos}")
        

def generate_corrected_inlier_loop_pairs(max_clique, loop_queue):
    #Generate the loop pair information for the corrected inlier loop closures.
    corrected_inliers = []
    for idx in max_clique:
        loop_id = int(idx.split()[1])
        id_0, id_1, pose_1, pose_2, relative_pos, _ = loop_queue[loop_id]
        corrected_inliers.append(loop_queue[loop_id])
    return corrected_inliers

def visualize_initial_pose_graph(pose_robot1, pose_robot2, loop_queue, graph):
    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.grid(False)
    ax.set_axis_off()
    # Plot Robot 1 poses
    for i, pose in enumerate(poses_robot1):
        x, y, z = pose.x(), pose.y(), pose.z()
        ax.plot([x], [y], [z], 'bo', label='Robot 1' if i == 0 else "")
        if i > 0:
            x_o,y_o,z_o = old_pose.x(), old_pose.y(), old_pose.z()
            ax.plot([x, x_o], [y, y_o], [z,z_o], color='grey', linestyle='-')
        old_pose = pose

    # Plot Robot 2 poses
    for i, pose in enumerate(poses_robot2):
        x, y, z = pose.x(), pose.y(), pose.z()
        ax.plot([x], [y], [z], 'ro', label='Robot 2' if i == 0 else "")
        if i > 0:
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

def visualize_inlier_only_pose_graph(loop_queue, corrected_inliers):
    # Visualize the pose graph with only inlier loop closures and odometry edges.
    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.grid(False)
    ax.set_axis_off()
    # Plot Robot 1 poses
    for i, pose in enumerate(poses_robot1):
        x, y, z = pose.x(), pose.y(), pose.z()
        ax.plot([x], [y], [z], 'bo', label='Robot 1' if i == 0 else "")
        if i > 0:
            x_o,y_o,z_o = old_pose.x(), old_pose.y(), old_pose.z()
            ax.plot([x, x_o], [y, y_o], [z,z_o], color='grey', linestyle='-')
        old_pose = pose

    # Plot Robot 2 poses
    for i, pose in enumerate(poses_robot2):
        x, y, z = pose.x(), pose.y(), pose.z()
        ax.plot([x], [y], [z], 'ro', label='Robot 2' if i == 0 else "")
        if i > 0:
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

# Example 
if __name__ == "__main__":
    # Parameters
    num_poses = 20
    yaw_increment = 0.01
    max_distance = 5.5 # Define maximum Euclidean distance for valid loop closures
    pcm_threshold=5.0
    intensity=1.0

    # Generate poses with rotation and position for Robot 1 and Robot 2
    poses_robot1 = []
    poses_robot2 = []
    for i in range(num_poses):
        x1, y1, z1 = i * 1.0, sin(i * 0.1)*10, 0.0
        yaw1 = i * yaw_increment
        poses_robot1.append(create_pose(x1, y1, z1, yaw1))
        
        x2, y2, z2 = i * 1.0 + 2.0, (i * 0.1)*10, 0.0
        yaw2 = i * yaw_increment
        poses_robot2.append(create_pose(x2, y2, z2, yaw2))

    # Create a factor graph and add initial poses
    graph = NonlinearFactorGraph()
    initial_estimate = Values()

    # Add poses to initial estimates
    for i, pose in enumerate(poses_robot1):
        initial_estimate.insert(i, pose)
    for i, pose in enumerate(poses_robot2):
        initial_estimate.insert(i + num_poses, pose)

    # Noise model for loop closures (small standard deviation for translation and rotation)
    loop_closure_noise = noiseModel.Diagonal.Sigmas([1e-6,1e-6,1e-6,1e-6,1e-6,1e-6])  # XYZRPY
    faulty_loop_closure_noise = noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) 

    # Generate loop closures with noise, checking Euclidean distance
    np.random.seed(0)
    loop_queue = []
    true_positives = 30
    false_positives = 10
    positive_sort = None

    for i in range(40):
        idx1 = np.random.randint(0, num_poses - 1)
        if i < true_positives:
            step = np.random.choice([1,2])
            # print(step)
            if np.random.rand() > 0.5:
                idx2 = (idx1 + step) % num_poses
            else:
                idx2 = (idx1 - step) % num_poses
            positive_sort = True
            #distamce calculation, only 2D info to filter far euclidean distance in true pair
            p1 = poses_robot1[idx1]
            p2 = poses_robot2[idx2]
            distance = sqrt((p1.x() - p2.x())**2 + (p1.y() - p2.y())**2)
            if abs(idx1 - idx2) > 2.0 or (distance > max_distance):
                positive_sort = False
        else:
            # step = np.random.choice([3,8])
            step = np.random.randint(2, 5)
            idx2 = (idx1 + step) % num_poses if np.random.rand() > 0.5 else (idx1 - step) % num_poses
            positive_sort = False
            p1 = poses_robot1[idx1]
            p2 = poses_robot2[idx2]
            distance = sqrt((p1.x() - p2.x())**2 + (p1.y() - p2.y())**2)
            #remove false-positive pair that are too far
            if (distance > 15.0):
                continue
            elif abs(idx1 - idx2) > 4.0 or (distance < 3.0):
                continue
            
        dx, dy, dz = np.random.uniform(-0.1, 0.1, 3)
        droll, dpitch, dyaw = np.random.uniform(0, 0.1, 3)
        delta_rotation = Rot3.RzRyRx(droll, dpitch, dyaw)
        delta_translation = Point3(dx, dy, dz)
        noisy_transform = Pose3(delta_rotation, delta_translation)
        # Add to loop queue with true/false positive flag
        loop_queue.append((idx1, idx2, p1, p2, noisy_transform, positive_sort))
        # Add between factor to graph
        robot1_key = idx1
        robot2_key = idx2 + num_poses

        #TODO: add distributed pose-graph optimization scheme
        #add noise for false positive and true positive in graph 
        if(positive_sort is False):
            graph.add(BetweenFactorPose3(robot1_key, robot2_key, noisy_transform, faulty_loop_closure_noise))
        else:
            graph.add(BetweenFactorPose3(robot1_key, robot2_key, noisy_transform, loop_closure_noise))

    '''Start processing the pose graph, compare the result of optimization before and after PCM'''
    # Step 1: Visualize
    visualize_initial_pose_graph(poses_robot1, poses_robot2, loop_queue, graph)
    # Step 2: Generate adjacency matrix
    adjacency_matrix = compute_pcm_matrix(loop_queue)
    # Step 3: Generate consistency graph
    consistency_graph = generate_consistency_graph(adjacency_matrix)
    # Step 4: Apply maximum clique problem
    max_clique = apply_maximum_clique(consistency_graph)
    # Step 5: Visualize inlier loop pairs and loop pair information
    visualize_inlier_loop_pairs(consistency_graph, max_clique, loop_queue)
    # Step 6: Generate loop pair information for corrected inlier loop closures 
    corrected_inlier_loop_pairs = generate_corrected_inlier_loop_pairs(max_clique, loop_queue)
    print("\nCorrected Inlier Loop Pairs:")
    for pair in corrected_inlier_loop_pairs:
        print(f"Loop Pair: ({pair[0]}, {pair[1]}), Relative Pose: {pair[2]}")
    # Step 7: Visualize Inlier in Euclidean Space
    visualize_inlier_only_pose_graph(loop_queue, corrected_inlier_loop_pairs)


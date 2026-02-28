import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
from networkx.algorithms.clique import find_cliques


def parse_clique_indices(max_clique):
    indices = []
    for node in max_clique:
        try:
            indices.append(int(str(node).split()[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(indices))


def compute_metrics(selected_indices, truth_labels, total_loops):
    selected_set = set(idx for idx in selected_indices if 0 <= idx < total_loops)
    total_truth = len(truth_labels)
    true_positives_total = int(sum(bool(x) for x in truth_labels))

    tp = int(sum(1 for idx in selected_set if idx < total_truth and truth_labels[idx]))
    fp = int(len(selected_set) - tp)
    fn = int(max(0, true_positives_total - tp))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    rejection_ratio = float(1.0 - (len(selected_set) / total_loops)) if total_loops > 0 else 0.0

    return {
        "total_loops": int(total_loops),
        "predicted_inliers": int(len(selected_set)),
        "true_positives_total": true_positives_total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rejection_ratio": rejection_ratio,
    }


def generate_consistency_graph(adjacency_matrix):
    """Generate the consistency graph based on the adjacency matrix."""
    loop_count = len(adjacency_matrix)
    graph = nx.Graph()
    graph.add_nodes_from([f"Loop {i}" for i in range(loop_count)])
    for i in range(loop_count):
        for j in range(i + 1, loop_count):
            if adjacency_matrix[i, j] == 1:
                graph.add_edge(f"Loop {i}", f"Loop {j}")
    return graph


def apply_maximum_clique(graph):
    """Apply the maximum clique algorithm to find the largest set of mutually consistent loop closures."""
    if graph.number_of_nodes() == 0:
        return []

    all_cliques = list(find_cliques(graph))
    if not all_cliques:
        return []

    max_clique = max(all_cliques, key=len)
    return max_clique


def save_run_artifacts(run_results, output_dir, run_tag):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{run_tag}_{timestamp_utc}"

    json_path = out_dir / f"{run_id}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(run_results, f, indent=2)

    csv_path = out_dir / f"{run_tag}_summary.csv"
    summary_row = {
        "run_id": run_id,
        "timestamp_utc": run_results["timestamp_utc"],
        "seed": run_results["config"]["seed"],
    }
    if "num_poses" in run_results["config"]:
        summary_row["num_poses"] = run_results["config"]["num_poses"]
    summary_row.update({
        "pcm_threshold": run_results["config"]["pcm_threshold"],
        "intensity": run_results["config"]["intensity"],
        "total_loops": run_results["metrics"]["total_loops"],
        "predicted_inliers": run_results["metrics"]["predicted_inliers"],
        "true_positives_total": run_results["metrics"]["true_positives_total"],
        "tp": run_results["metrics"]["tp"],
        "fp": run_results["metrics"]["fp"],
        "fn": run_results["metrics"]["fn"],
        "precision": run_results["metrics"]["precision"],
        "recall": run_results["metrics"]["recall"],
        "f1": run_results["metrics"]["f1"],
        "rejection_ratio": run_results["metrics"]["rejection_ratio"],
        "max_clique_size": run_results["max_clique_size"],
    })

    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)

    return run_id, str(json_path), str(csv_path)

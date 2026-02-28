import json

import networkx as nx
import numpy as np

from net_pcm import (
    NetPCMConfig,
    apply_maximum_clique,
    compute_metrics,
    generate_adjacency_matrix,
    get_example_loop_data,
    parse_clique_indices,
    run_demo,
    save_run_artifacts,
)


def test_parse_clique_indices_handles_invalid_tokens():
    max_clique = ["Loop 3", "bad", "Loop x", "Loop 1", "Loop 3"]
    assert parse_clique_indices(max_clique) == [1, 3]


def test_apply_maximum_clique_empty_graph_returns_empty():
    graph = nx.Graph()
    assert apply_maximum_clique(graph) == []


def test_generate_adjacency_matrix_threshold_behavior():
    loop_queue, _ = get_example_loop_data()
    node_count = len(loop_queue)

    adjacency_loose = generate_adjacency_matrix(loop_queue, pcm_threshold=1e9, intensity=1.0)
    adjacency_strict = generate_adjacency_matrix(loop_queue, pcm_threshold=-1.0, intensity=1.0)

    assert adjacency_loose.shape == (node_count, node_count)
    assert np.allclose(np.diag(adjacency_loose), 0.0)
    assert np.allclose(adjacency_loose, adjacency_loose.T)
    assert int(np.sum(adjacency_loose)) == node_count * (node_count - 1)
    assert int(np.sum(adjacency_strict)) == 0


def test_compute_metrics_known_case():
    _, truth_labels = get_example_loop_data()
    selected_indices = [0, 1, 2, 8]
    metrics = compute_metrics(selected_indices, truth_labels, total_loops=10)

    assert metrics["tp"] == 3
    assert metrics["fp"] == 1
    assert metrics["fn"] == 3
    assert metrics["precision"] == 0.75
    assert metrics["recall"] == 0.5
    assert round(metrics["f1"], 6) == 0.6


def test_run_demo_is_deterministic_with_fixed_config():
    config = NetPCMConfig(visualize=False, save_results=False, seed=42)
    first = run_demo(config)
    second = run_demo(config)

    assert first["selected_indices"] == second["selected_indices"]
    assert first["max_clique_size"] == second["max_clique_size"]
    assert first["metrics"] == second["metrics"]


def test_save_run_artifacts_writes_json_and_appends_csv(tmp_path):
    run_results = {
        "timestamp_utc": "2026-02-28T00:00:00+00:00",
        "config": {"seed": 42, "pcm_threshold": 5.0, "intensity": 1.0},
        "metrics": {
            "total_loops": 10,
            "predicted_inliers": 6,
            "true_positives_total": 6,
            "tp": 5,
            "fp": 1,
            "fn": 1,
            "precision": 0.8333,
            "recall": 0.8333,
            "f1": 0.8333,
            "rejection_ratio": 0.4,
        },
        "max_clique_size": 6,
    }

    _, json_path_1, csv_path = save_run_artifacts(run_results, str(tmp_path), "net_pcm_test")
    _, json_path_2, csv_path_2 = save_run_artifacts(run_results, str(tmp_path), "net_pcm_test")

    assert csv_path == csv_path_2

    with open(json_path_1, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["metrics"]["tp"] == 5

    with open(json_path_2, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["max_clique_size"] == 6

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = [line.strip() for line in f if line.strip()]
    # header + two appended runs
    assert len(rows) == 3

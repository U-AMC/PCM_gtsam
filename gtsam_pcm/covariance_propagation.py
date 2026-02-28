"""Compatibility wrapper for the canonical PCM demo implementation.

This file intentionally re-exports the API from `pcm_common` and `gtsam_pcm`
to avoid maintaining duplicate logic across two modules.
"""

from pcm_common import (
    apply_maximum_clique,
    compute_metrics,
    generate_consistency_graph,
    parse_clique_indices,
    save_run_artifacts,
)

try:
    from .gtsam_pcm import (
        GTSAMLoopClosure,
        GTSAMPCMConfig,
        animate_covariance_progression,
        animate_pose_covariance_nodes,
        build_anchored_base_graph,
        build_graph_from_loop_indices,
        compute_covariance_progression,
        compute_pcm_matrix,
        compute_sliding_window_covariance_states,
        create_pose,
        create_realtime_covariance_plot,
        generate_corrected_inlier_loop_pairs,
        plot_covariance_progression,
        residual_pcm,
        run_demo,
        run_pcm_sliding_window,
        update_realtime_covariance_plot,
        visualize_inlier_loop_pairs,
        visualize_inlier_only_pose_graph,
        visualize_initial_pose_graph,
    )
except ImportError as exc:
    if "attempted relative import" not in str(exc):
        raise
    from gtsam_pcm import (
        GTSAMLoopClosure,
        GTSAMPCMConfig,
        animate_covariance_progression,
        animate_pose_covariance_nodes,
        build_anchored_base_graph,
        build_graph_from_loop_indices,
        compute_covariance_progression,
        compute_pcm_matrix,
        compute_sliding_window_covariance_states,
        create_pose,
        create_realtime_covariance_plot,
        generate_corrected_inlier_loop_pairs,
        plot_covariance_progression,
        residual_pcm,
        run_demo,
        run_pcm_sliding_window,
        update_realtime_covariance_plot,
        visualize_inlier_loop_pairs,
        visualize_inlier_only_pose_graph,
        visualize_initial_pose_graph,
    )


if __name__ == "__main__":
    run_demo()

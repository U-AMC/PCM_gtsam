import pytest

try:
    import gtsam
    from gtsam import Point3, Pose3, Rot3
    import gtsam_pcm.gtsam_pcm as gpcm
    HAS_GTSAM = True
except ModuleNotFoundError:
    HAS_GTSAM = False

pytestmark = pytest.mark.skipif(not HAS_GTSAM, reason="gtsam is not installed")


def test_residual_pcm_identity_is_zero():
    identity = Pose3(Rot3(), Point3(0.0, 0.0, 0.0))
    residual = gpcm.residual_pcm(identity, identity, identity, identity, intensity=1.0)
    assert abs(residual) < 1e-9


def test_compute_pcm_matrix_empty_returns_empty():
    matrix = gpcm.compute_pcm_matrix([])
    assert matrix.shape == (0, 0)


def test_run_demo_is_deterministic_with_fixed_config():
    config = gpcm.GTSAMPCMConfig(
        seed=0,
        num_poses=10,
        num_loop_attempts=12,
        true_positive_attempts=8,
        visualize=False,
        save_results=False,
    )
    first = gpcm.run_demo(config)
    second = gpcm.run_demo(config)

    assert first["selected_indices"] == second["selected_indices"]
    assert first["max_clique_size"] == second["max_clique_size"]
    assert first["metrics"] == second["metrics"]

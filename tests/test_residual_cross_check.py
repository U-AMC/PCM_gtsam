"""Cross-check: net_pcm residual vs GTSAM residual on identical inputs.

If both implementations are correct SE(3), they must agree to within
floating-point tolerance.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

try:
    import gtsam
    from gtsam import Pose3, Rot3, Point3
    from gtsam_pcm.gtsam_pcm import residual_pcm as gtsam_residual_pcm
    HAS_GTSAM = True
except ModuleNotFoundError:
    HAS_GTSAM = False

pytestmark = pytest.mark.skipif(not HAS_GTSAM, reason="gtsam is not installed")


def _list_to_pose3(pose_list):
    """Convert [roll, pitch, yaw, x, y, z] to gtsam.Pose3."""
    rot = Rot3.RzRyRx(*pose_list[:3])
    return Pose3(rot, Point3(*pose_list[3:]))


def _to_rot_trans(pose):
    return R.from_euler('xyz', pose[:3]), np.array(pose[3:], dtype=np.float64)


def se3_compose(p1, p2):
    r1, t1 = _to_rot_trans(p1)
    r2, t2 = _to_rot_trans(p2)
    r = r1 * r2
    t = r1.apply(t2) + t1
    return list(r.as_euler('xyz')) + list(t)


def se3_inverse(p):
    r, t = _to_rot_trans(p)
    r_inv = r.inv()
    return list(r_inv.as_euler('xyz')) + list(-r_inv.apply(t))


def se3_logmap(pose):
    """Proper SE(3) Logmap: [omega, J^{-1}(omega) @ t]."""
    r, t = _to_rot_trans(pose)
    omega = r.as_rotvec()
    theta = np.linalg.norm(omega)

    if theta < 1e-10:
        v = t
    else:
        W = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0],
        ])
        W2 = W @ W
        J_inv = (
            np.eye(3)
            - 0.5 * W
            + (1.0 / theta**2 - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))) * W2
        )
        v = J_inv @ t

    return np.concatenate([omega, v])


def net_residual(inner_ij, inter_jk, inner_kl, inter_il, intensity=1.0):
    """Replicate net_pcm's SE(3) residual computation."""
    inter_il_inv = se3_inverse(inter_il)
    res_pose = se3_compose(
        se3_compose(se3_compose(inner_ij, inter_jk), inner_kl),
        inter_il_inv,
    )
    res_vec = se3_logmap(res_pose)
    m_cov = np.diag(np.full(6, intensity))
    return np.sqrt(res_vec.T @ m_cov @ res_vec)


def _make_test_cases():
    """Return pairs of (inner_ij, inter_jk, inner_kl, inter_il) as pose lists."""
    rng = np.random.default_rng(123)
    cases = []
    for _ in range(10):
        poses = []
        for _ in range(4):
            angles = rng.uniform(-0.3, 0.3, 3).tolist()
            trans = rng.uniform(-2.0, 2.0, 3).tolist()
            poses.append(angles + trans)
        cases.append(tuple(poses))
    return cases


@pytest.mark.parametrize("poses", _make_test_cases(), ids=[f"case_{i}" for i in range(10)])
def test_net_pcm_residual_matches_gtsam(poses):
    """The net_pcm SE(3) residual must match the GTSAM Lie-group residual."""
    inner_ij, inter_jk, inner_kl, inter_il = poses

    net_res = net_residual(inner_ij, inter_jk, inner_kl, inter_il, intensity=1.0)

    g_inner_ij = _list_to_pose3(inner_ij)
    g_inter_jk = _list_to_pose3(inter_jk)
    g_inner_kl = _list_to_pose3(inner_kl)
    g_inter_il = _list_to_pose3(inter_il)

    gtsam_res = gtsam_residual_pcm(g_inter_jk, g_inter_il, g_inner_ij, g_inner_kl, intensity=1.0)

    assert abs(net_res - gtsam_res) < 1e-6, (
        f"net={net_res:.8f} vs gtsam={gtsam_res:.8f}, diff={abs(net_res - gtsam_res):.2e}"
    )

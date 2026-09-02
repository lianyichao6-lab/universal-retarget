import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tools" / "luban_ros_grasp_execute.py"
SPEC = importlib.util.spec_from_file_location("luban_ros_grasp_execute", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_request_reader_rejects_internal_21_joint_command(tmp_path) -> None:
    path = tmp_path / "invalid.npz"
    np.savez(path, base_frame="r_base_link", T_robot_base_l25_hand_target=np.eye(4),
        T_robot_base_arm_flange_pregrasp=np.eye(4), T_robot_base_arm_flange_target=np.eye(4),
        l25_active_positions=np.zeros(21), l25_active_joint_names=np.arange(21))
    try:
        MODULE._read_request(path)
    except ValueError as exc:
        assert "16 finite" in str(exc)
    else:
        raise AssertionError("expected active L25 joint validation failure")

import unittest
from pathlib import Path

import mujoco
import numpy as np

from anydexretarget import Retargeter
from example.output.sim.mujoco_output import ROBOT_HAND_CONFIGS
from example.test.debug_skeleton import estimate_pinocchio_to_mujoco_transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "example" / "config" / "adaptive" / "mediapipe"


class TestDebugSkeletonAlignment(unittest.TestCase):
    @staticmethod
    def load_case(robot_type):
        config_path = CONFIG_ROOT / f"mediapipe_{robot_type}.yaml"
        retargeter = Retargeter.from_yaml(str(config_path), "right")
        hand_cfg = ROBOT_HAND_CONFIGS[robot_type]
        model = mujoco.MjModel.from_xml_path(hand_cfg["model_path"]("right"))
        return retargeter.optimizer, model, hand_cfg

    @staticmethod
    def drive_live_data_away_from_neutral(model, hand_cfg):
        """Reproduce debug_skeleton's actuator initialization."""
        data = mujoco.MjData(model)
        if (
            model.nu > 0
            and hand_cfg.get("qpos_servo_alpha") is None
            and not hand_cfg.get("direct_qpos", False)
        ):
            for actuator_id in range(model.nu):
                if model.actuator_ctrllimited[actuator_id]:
                    low, high = model.actuator_ctrlrange[actuator_id]
                    data.ctrl[actuator_id] = (low + high) / 2.0
                else:
                    data.ctrl[actuator_id] = 0.0
            for _ in range(100):
                mujoco.mj_step(model, data)
        else:
            mujoco.mj_forward(model, data)
        return data

    @staticmethod
    def matched_neutral_points(optimizer, model):
        robot = optimizer.robot
        robot.compute_forward_kinematics(np.zeros(robot.model.nq))
        neutral_data = mujoco.MjData(model)
        mujoco.mj_forward(model, neutral_data)

        names = [optimizer.origin_link_name]
        for attr in ("link1_names", "link3_names", "link4_names", "task_link_names"):
            names.extend(getattr(optimizer, attr, []))

        pin_points = []
        mujoco_points = []
        for name in dict.fromkeys(names):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                continue
            try:
                link_id = robot.get_link_index(name)
            except Exception:
                continue
            pin_points.append(robot.get_link_pose(link_id)[:3, 3].copy())
            mujoco_points.append(neutral_data.xpos[body_id].copy())
        return np.asarray(pin_points), np.asarray(mujoco_points)

    def assert_neutral_alignment(self, robot_type, expected_rotation_degrees):
        optimizer, model, hand_cfg = self.load_case(robot_type)
        live_data = self.drive_live_data_away_from_neutral(model, hand_cfg)

        rotation, translation, diagnostics = estimate_pinocchio_to_mujoco_transform(
            optimizer, model, live_data
        )
        source, target = self.matched_neutral_points(optimizer, model)
        predicted = source @ rotation.T + translation
        errors = np.linalg.norm(predicted - target, axis=1)

        self.assertEqual(diagnostics["method"], "matched-link fit")
        self.assertGreaterEqual(len(diagnostics["matched_names"]), 3)
        self.assertLess(float(errors.max()), 1e-5)

        angle = np.degrees(
            np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
        )
        self.assertAlmostEqual(angle, expected_rotation_degrees, places=3)

    def test_ability_ignores_articulated_live_pose_and_uses_identity_frame(self):
        self.assert_neutral_alignment("ability_hand", 0.0)

    def test_shadow_ignores_articulated_live_pose_and_keeps_z90_frame(self):
        self.assert_neutral_alignment("shadow_hand", 90.0)


if __name__ == "__main__":
    unittest.main()

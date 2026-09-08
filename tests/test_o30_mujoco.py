import importlib.util
import pickle
from pathlib import Path

import mujoco
import numpy as np

from anydexretarget.hand_contract import O30_QPOS_JOINT_NAMES


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_o30_trajectory_maps_into_urdf_mujoco_order(tmp_path: Path) -> None:
    playback = _load_tool("o30_mujoco_playback")
    model = mujoco.MjModel.from_xml_path(str(playback.MODEL_PATH))
    target = np.arange(20, dtype=np.float64) / 100.0
    path = tmp_path / "trajectory.pkl"
    path.write_bytes(pickle.dumps([{
        "target": target,
        "robot_joint_names": list(O30_QPOS_JOINT_NAMES),
    }]))

    frames = playback._load_frames(path, model)

    assert frames.shape == (1, model.nq)
    for index, name in enumerate(O30_QPOS_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert frames[0, model.jnt_qposadr[joint_id]] == target[index]


def test_o30_hop_command_has_the_audited_twenty_axes() -> None:
    hardware = _load_tool("o30_hardware_validate")
    model = mujoco.MjModel.from_xml_path(str(hardware.MODEL_PATH))

    command = hardware._qpos_to_hop_u8(np.zeros(20, dtype=np.float64), model)

    assert command.shape == (20,)
    assert command.dtype == np.uint8

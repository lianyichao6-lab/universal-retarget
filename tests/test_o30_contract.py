import numpy as np

from anydexretarget.luban_contract import (
    O30_ACTIVE_INDICES,
    O30_ACTIVE_JOINT_NAMES,
    O30_QPOS_JOINT_NAMES,
    o30_active_joint_names,
    o30_qpos_to_luban_active,
)
from anydexretarget.o30_retarget_backend import VECTOR_CONFIG
from anydexretarget.retarget import Retargeter


def test_o30_contract_maps_physical_urdf_order_to_hop_controller_order() -> None:
    qpos = np.arange(20, dtype=np.float64)
    result = o30_qpos_to_luban_active(qpos)
    np.testing.assert_array_equal(result, qpos[O30_ACTIVE_INDICES])
    assert result.shape == (20,)
    assert o30_active_joint_names()[0] == "r_hand_thumb_roll"
    assert o30_active_joint_names()[-1] == "r_hand_little_tip"


def test_o30_vector_config_loads_the_twenty_axis_right_hand() -> None:
    retargeter = Retargeter.from_yaml(str(VECTOR_CONFIG), hand_side="right")
    names = tuple(str(name) for name in retargeter.optimizer.robot.dof_joint_names)
    assert names == O30_QPOS_JOINT_NAMES
    assert len(O30_ACTIVE_JOINT_NAMES) == 20

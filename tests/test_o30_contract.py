import numpy as np

from anydexretarget.hand_contract import (
    O30_ACTIVE_INDICES,
    O30_ACTIVE_JOINT_NAMES,
    O30_QPOS_JOINT_NAMES,
    o30_qpos_to_command_order,
)
from anydexretarget.o30_retarget_backend import VECTOR_CONFIG
from anydexretarget.retarget import Retargeter


def test_o30_contract_maps_physical_urdf_order_to_hop_controller_order() -> None:
    qpos = np.arange(20, dtype=np.float64)
    result = o30_qpos_to_command_order(qpos)
    np.testing.assert_array_equal(result, qpos[O30_ACTIVE_INDICES])
    assert result.shape == (20,)
    assert O30_ACTIVE_JOINT_NAMES[0] == "thumb_roll"
    assert O30_ACTIVE_JOINT_NAMES[-1] == "little_tip"


def test_o30_vector_config_loads_the_twenty_axis_right_hand() -> None:
    retargeter = Retargeter.from_yaml(str(VECTOR_CONFIG), hand_side="right")
    names = tuple(str(name) for name in retargeter.optimizer.robot.dof_joint_names)
    assert names == O30_QPOS_JOINT_NAMES
    assert len(O30_ACTIVE_JOINT_NAMES) == 20

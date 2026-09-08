import numpy as np

from anydexretarget.arm_ik import AR5NumericalIK


URDF = "/home/evolabs-5080/lianyichao/luban_framework/src/robot_model/evo_work/AR5-5_08/AR5-5_08R-W4C4A6-ZY2_description/urdf/AR5-5_08R-W4C4A6-ZY2.urdf"


def test_ar5_numerical_ik_recovers_a_reachable_fk_pose() -> None:
    solver = AR5NumericalIK(URDF)
    expected = np.asarray([0.15, -0.30, 0.25, -0.20, 0.10, 0.20, -0.15])
    target = solver.fk.flange_transform(expected)
    result = solver.solve(target, np.zeros(7))
    np.testing.assert_allclose(solver.fk.flange_transform(result.positions), target, atol=2e-3)
    assert result.position_error_m < 0.002
    assert result.orientation_error_rad < 0.0524

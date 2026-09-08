import numpy as np

from anydexretarget.o30_collision import O30CollisionEvaluator


def test_o30_open_pose_is_safe_against_a_distant_object_cloud() -> None:
    evaluator = O30CollisionEvaluator(
        np.array([[10.0, 10.0, 10.0]], dtype=np.float64),
        wrist_position_camera=np.zeros(3, dtype=np.float64),
        canonical_basis_row=np.eye(3, dtype=np.float64),
        mesh_vertices_per_link=32,
    )

    result = evaluator.evaluate(np.zeros(20, dtype=np.float64))
    fraction, close = evaluator.safe_closure(np.zeros(20, dtype=np.float64), steps=4)

    assert result.safe
    assert result.self_collision_count == 0
    assert result.forbidden_object_collision_count == 0
    assert fraction == 1.0
    assert close.safe

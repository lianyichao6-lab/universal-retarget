import numpy as np

from anydexretarget.luban_arm import homogeneous_transform


def test_object_rebase_preserves_relative_hand_coordinates() -> None:
    source_hand = np.eye(4)
    source_hand[:3, 3] = [0.0, 0.0, 0.05]
    target_hand = np.eye(4)
    target_hand[:3, 3] = [0.3, -0.2, 0.6]
    point_in_source_world = np.asarray([0.1, 0.0, 0.1, 1.0])
    rebased = target_hand @ np.linalg.inv(source_hand) @ point_in_source_world
    np.testing.assert_allclose(
        np.linalg.inv(target_hand) @ rebased,
        np.linalg.inv(source_hand) @ point_in_source_world,
    )
    assert homogeneous_transform(target_hand, "target").shape == (4, 4)

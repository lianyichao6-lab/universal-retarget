import numpy as np

from tools.pack_smolvla_episode import pack_episode


def test_pack_episode_builds_28d_action(tmp_path) -> None:
    source = tmp_path / "tactile.npz"
    output = tmp_path / "packed.npz"
    contact = np.zeros((3, 5), dtype=bool)
    contact[1, :2] = True
    np.savez(source, timestamps=np.arange(3), qpos=np.zeros((3, 21)),
             tactile_contact=contact, tactile_wrench=np.zeros((3, 5, 6)))
    report = pack_episode(source, output)
    assert report["action_dim"] == 28
    with np.load(output) as data:
        assert data["action_28"].shape == (3, 28)
        assert data["tactile"].shape == (3, 35)
        assert data["success"].all()

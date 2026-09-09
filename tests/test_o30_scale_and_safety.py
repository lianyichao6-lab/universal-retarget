import importlib.util
from pathlib import Path

import mujoco
import numpy as np

from anydexretarget.hand_contract import O30_ACTIVE_SOURCE_NAMES
from anydexretarget.o30_hardware_profile import O30HardwareProfile, nominal_profile as nominal_hardware_profile
from anydexretarget.o30_mesh_collision import O30MeshCollisionEvaluator
from anydexretarget.o30_scale import O30ScaleProfile, nominal_profile


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def test_o30_nominal_scale_profile_is_hash_bound(tmp_path: Path) -> None:
    path = tmp_path / "scale.json"
    nominal_profile().save(path)
    loaded = O30ScaleProfile.load(path)
    assert loaded.key_vector_scales.shape == (15,)
    assert np.all(loaded.key_vector_scales > 0)


def test_o30_hardware_profile_maps_audited_source_order() -> None:
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/linkerhand_o30/right/linkerhand_o30_right.urdf"))
    profile = nominal_hardware_profile(model)
    command = profile.command_from_qpos(np.zeros(20))
    assert profile.source_joint_names == O30_ACTIVE_SOURCE_NAMES
    assert command.shape == (20,)
    assert command.dtype == np.uint8


def test_o30_hardware_profile_rejects_wrong_joint_contract() -> None:
    invalid = O30HardwareProfile(
        source_joint_names=("wrong",) * 20,
        command_joint_names=("wrong",) * 20,
        qpos_knots_rad=np.tile(np.asarray([-1.0, 0.0, 1.0]), (20, 1)),
        command_knots_u8=np.tile(np.asarray([0.0, 128.0, 255.0]), (20, 1)),
        device_uid="test",
    )
    try:
        invalid.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid O30 joint contract was accepted")


def test_o30_bounded_hop_ramp_never_exceeds_requested_step() -> None:
    spec = importlib.util.spec_from_file_location("o30_hardware_execute", TOOLS / "o30_hardware_execute.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    commands = module._bounded_commands(np.zeros(20), np.full(20, 255), 2)
    assert np.array_equal(commands[-1], np.full(20, 255, dtype=np.uint8))
    assert all(np.max(np.abs(after.astype(int) - before.astype(int))) <= 2 for before, after in zip([np.zeros(20, dtype=np.uint8), *commands], commands))


def test_o30_full_mesh_far_object_has_no_forbidden_collision(tmp_path: Path) -> None:
    import trimesh

    mesh_path = tmp_path / "far.stl"
    mesh = trimesh.creation.box(extents=(0.02, 0.02, 0.02))
    mesh.apply_translation((10.0, 10.0, 10.0))
    mesh.export(mesh_path)
    evaluator = O30MeshCollisionEvaluator(
        mesh_path,
        wrist_position_camera=np.zeros(3),
        canonical_basis_row=np.eye(3),
        vertices_per_link=128,
    )
    result = evaluator.evaluate(np.zeros(20))
    assert not result.self_collision_pairs
    assert not result.forbidden_mesh_collisions
    assert not result.nonpad_clearance_violations


def test_o30_thumb_pad_mask_uses_thumb_long_axis() -> None:
    from anydexretarget.o30_mesh_collision import _distal_pad_mask

    vertices = np.column_stack((np.zeros(40), np.linspace(0.0, 0.026, 40), np.linspace(-0.010, 0.005, 40)))
    mask = _distal_pad_mask("thumb_distal", vertices)
    assert mask.shape == (len(vertices),)
    assert mask[-1]
    assert not mask[0]

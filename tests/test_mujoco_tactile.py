import mujoco
import numpy as np

from anydexretarget.mujoco_tactile import fingertip_geom_ids, tactile_state_from_mujoco


def _model(object_pos: str = "0 0 0") -> mujoco.MjModel:
    geoms = "\n".join(
        f'<geom name="{finger}_distal_visual" type="sphere" size="0.05" pos="{i * 0.2} 0 0" />'
        for i, finger in enumerate(("thumb", "index", "middle", "ring", "pinky"))
    )
    return mujoco.MjModel.from_xml_string(
        f'<mujoco><option gravity="0 0 0"/><worldbody>{geoms}'
        f'<body name="object_body" pos="{object_pos}"><freejoint/><geom name="object" type="sphere" size="0.05" /></body>'
        "</worldbody></mujoco>"
    )


def test_fingertip_ids_and_no_contact_are_stable() -> None:
    model = _model("0 0 0.2")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = fingertip_geom_ids(model)
    state = tactile_state_from_mujoco(model, data, ids, object_geom_ids=[model.geom("object").id])
    assert list(ids) == ["thumb", "index", "middle", "ring", "pinky"]
    np.testing.assert_array_equal(state.contact, np.zeros(5, dtype=bool))


def test_object_contact_marks_only_touching_finger(monkeypatch) -> None:
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = fingertip_geom_ids(model)
    object_id = model.geom("object").id

    def fake_contact_force(_model, _data, _contact_index, output) -> None:
        output[2] = 1.0

    monkeypatch.setattr(mujoco, "mj_contactForce", fake_contact_force)
    state = tactile_state_from_mujoco(
        model,
        data,
        ids,
        object_geom_ids=[object_id],
        contact_force_threshold=0.1,
    )
    assert state.contact[0]
    assert not state.contact[1:].any()

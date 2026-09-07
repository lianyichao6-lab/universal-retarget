# O30 MANUS To Luban

The first O30 path is right-hand Vector retargeting only. It preserves the
existing MANUS adapter and converts the standard 21-point hand into AnyDex's
physical O30 URDF qpos, then reorders that qpos into Luban's 20-axis HOP
controller order.

Record MANUS as before, then run:

```bash
.venv/bin/python tools/retarget_manus_recording.py \
  --input outputs/manus/session_001/raw_manus.npz \
  --robot o30 --backend vector \
  --output outputs/manus/session_001/o30_vector.pkl \
  --canonical-output outputs/manus/session_001/canonical_o30_vector.npz
```

Inspect the command without publishing it:

```bash
/usr/bin/python3 tools/luban_hand_ros_execute.py \
  --trajectory outputs/manus/session_001/o30_vector.pkl \
  --frame -1 --hand-model o30
```

After the O30 Luban controller is active and the hand is clear, publish only
that hand pose:

```bash
/usr/bin/python3 tools/luban_hand_ros_execute.py \
  --trajectory outputs/manus/session_001/o30_vector.pkl \
  --frame -1 --hand-model o30 --execute --confirm O30_HAND_CLEAR
```

The tool publishes to `/right_hand_controller/commands` by default. It does
not publish arm commands. Start Luban O30 hardware or simulation first; its
controller order is `thumb_roll` through `little_tip` as defined in
`config/control/controllers_o30.yaml`.

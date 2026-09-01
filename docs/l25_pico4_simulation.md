# Pico 4 to L25 Simulation

This path is simulation only. It starts no LinkerHand SDK, CAN, serial,
`ros2_control` hardware, or actuator process.

```text
Pico 4 hand tracking (21 x 3)
        |
        v
Pico4 input -> AnyDexRetarget vector/adaptive -> L25 qpos
        |
        +-> MuJoCo live viewer
        +-> optional RViz offline trajectory playback
```

## Live MuJoCo

Use `vector` first. It was faster in the local L25 replay check and usually
looks smoother than adaptive before L25-specific calibration.

Start the Pico relay daemon in one terminal:

```bash
cd /path/to/universal-retarget/example
../.venv/bin/python input/pico4_daemon.py
```

Start the L25 MuJoCo simulation in a second terminal:

```bash
cd /path/to/universal-retarget/example
../.venv/bin/python teleop_sim.py \
  --input pico4 --pico4-mode relay \
  --robot l25 --optimizer vector --hand right
```

The direct mode is an alternative that does not start a relay daemon:

```bash
cd /path/to/universal-retarget/example
../.venv/bin/python teleop_sim.py \
  --input pico4 --pico4-mode direct \
  --robot l25 --optimizer vector --hand right
```

Both commands select `pico4_linkerhand_l25.yaml` automatically. For the
pinch-aware solver, change `--optimizer vector` to `--optimizer adaptive`.

## RViz Playback

RViz currently plays a saved L25 simulation trajectory, which is useful for
inspecting the actual URDF and its mimic joints. It is not a hardware launch.

```bash
cd /path/to/universal-retarget
ros2 launch launch/l25_rviz_playback.launch.py \
  trajectory:=outputs/l25/l25_vector_high_sim.pkl fps:=30.0
```

## Calibration Status

The new Pico4 L25 configurations use the upstream L20 Pico coordinate-frame
correction as a bootstrap. Their L25 segment scales and key-vector scales are
not yet calibrated against a captured Pico4 L25 skeleton comparison. They are
valid for exercising the live simulation path, not a final quality claim.

After confirming that live tracking reaches MuJoCo, run the repository's
visual skeleton/calibration tools with the L25 Pico4 configuration before
tuning any individual joint:

```bash
cd /path/to/universal-retarget/example
../.venv/bin/python test/debug_skeleton.py \
  --input pico4 --pico4-mode relay --robot l25 --hand right
```

Do not use `teleop_real.py` for this simulation workflow.

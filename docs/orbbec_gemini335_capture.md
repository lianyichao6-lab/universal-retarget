# Orbbec Gemini 335 Capture

This capture path supplies the RGB-D input required by HUG:

```text
Gemini 335
  -> registered RGB + depth + color intrinsics
  -> rgb.png + depth.png + intrinsics.txt
  -> object mask + object point cloud
  -> HUG candidate grasps
```

The depth image is saved as `uint16` millimetres and must be registered to the
RGB image. The `capture_orbbec_rgbd.py` command intentionally uses the ROS2
system Python, not this repository's `.venv`.

## USB Check

Use a direct USB 3.x data port and cable. Before capture, verify the camera is
not on a 480M USB 2 bus:

```bash
lsusb -t
```

For this machine the Gemini should appear below the `20000M/x2` root hub (or
another SuperSpeed hub). A `480M` camera entry is insufficient for a reliable
RGB-D stream.

## Start The Driver

In terminal A:

```bash
cd /path/to/hand_gesture_custom_primitives
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

ros2 launch orbbec_camera gemini_330_series.launch.py \
  camera_name:=scene_camera \
  device_num:=1 \
  connection_delay:=0 \
  depth_registration:=true \
  align_mode:=SW \
  align_target_stream:=COLOR \
  enable_depth_scale:=true \
  enable_color:=true \
  enable_depth:=true \
  enable_point_cloud:=true \
  enable_ldp:=false \
  log_level:=none
```

The required topics are:

```text
/scene_camera/color/image_raw
/scene_camera/depth/image_raw
/scene_camera/color/camera_info
```

Verify them in terminal B with `ros2 topic list` after sourcing the same ROS2
environment.

## Preview, Compose, And Capture One HUG Frame

Place a single opaque object in view, keep the camera fixed, then run in
terminal B:

```bash
cd /path/to/universal-retarget
source /opt/ros/jazzy/setup.bash
source /path/to/hand_gesture_custom_primitives/install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

/usr/bin/python3 tools/capture_orbbec_rgbd.py \
  --output external/hug/data/gemini335_cup
```

For interactive capture, use:

```bash
/usr/bin/python3 tools/capture_orbbec_rgbd.py \
  --preview \
  --timeout 0 \
  --output external/hug/data/gemini335_cup
```

The live window opens first. Press `SPACE` to save the synchronized RGB-D frame, or `Q`/`ESC` to cancel.

It waits for synchronized RGB, depth, and Color CameraInfo, then writes:

```text
external/hug/data/gemini335_cup/
  rgb.png
  depth.png
  intrinsics.txt
  capture_metadata.json
```

`capture_metadata.json` records the non-zero median depth in millimetres. For
an object approximately 0.3--1.5 m from the camera, a value approximately
300--1500 is expected. Do not feed a capture to HUG if RGB and depth have
different resolutions or if the depth is visibly offset from the color image.

## Continue To HUG

After visually confirming the files, select the object and turn its RGB-D
surface into a point cloud:

```bash
.venv/bin/python tools/create_object_mask.py \
  --rgb external/hug/data/gemini335_cup/rgb.png \
  --output outputs/grasp/gemini335_cup/object_mask.png

.venv/bin/python tools/object_mask_to_pointcloud.py \
  --rgb external/hug/data/gemini335_cup/rgb.png \
  --depth external/hug/data/gemini335_cup/depth.png \
  --intrinsics external/hug/data/gemini335_cup/intrinsics.txt \
  --mask outputs/grasp/gemini335_cup/object_mask.png \
  --output outputs/grasp/gemini335_cup/object_pointcloud.npz \
  --ply outputs/grasp/gemini335_cup/object_pointcloud.ply
```

Draw a rectangle around whichever object you want in the saved RGB image, then press `SPACE` or `ENTER`. To pin a known foreground pixel, add a real coordinate such as `--point 640 360`; it must lie inside the rectangle. The next command is
`tools/generate_hug_candidates.py`; it consumes this aligned RGB-D frame and
the object point cloud, then generates and ranks full MANO grasp candidates.

This is a single-view HUG grasp workflow. A separate multi-view reconstruction
stage is needed before treating the resulting point cloud as a full 3D object
model.

## Multi-view capture for object geometry

Keep the camera fixed and rotate a single opaque object by roughly 45--60
degrees between captures. Start with six to eight views. Keep the table,
lighting, and camera range unchanged throughout the session.

```bash
cd /path/to/universal-retarget
source /opt/ros/jazzy/setup.bash
source /path/to/hand_gesture_custom_primitives/install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

/usr/bin/python3 tools/capture_orbbec_rgbd.py \
  --preview --multi-view --timeout 0 \
  --output outputs/reconstruction/cup_session
```

`SPACE` saves `view_000`, `view_001`, and so on. Rotate the object before
each save; `Q` or `ESC` finishes the session while preserving saved views.
Each view directory contains `rgb.png`, `depth.png`, `intrinsics.txt`, and
capture metadata. Do not use transparent, glossy, or featureless black objects
for the first reconstruction experiment.

For each saved view, draw a mask around the same object and back-project it.
Use `view_000` as the anchor; it is the RGB-D view that HUG will retain for
its image and clicked target point.

```bash
.venv/bin/python tools/create_object_mask.py \
  --rgb outputs/reconstruction/cup_session/view_000/rgb.png \
  --output outputs/reconstruction/cup_session/view_000/object_mask.png

.venv/bin/python tools/object_mask_to_pointcloud.py \
  --rgb outputs/reconstruction/cup_session/view_000/rgb.png \
  --depth outputs/reconstruction/cup_session/view_000/depth.png \
  --intrinsics outputs/reconstruction/cup_session/view_000/intrinsics.txt \
  --mask outputs/reconstruction/cup_session/view_000/object_mask.png \
  --output outputs/reconstruction/cup_session/view_000/object_pointcloud.npz
```

Repeat those two commands for `view_001` through `view_007`, then register
and fuse the masked clouds. The output uses the `view_000` camera frame, so it
can be passed to HUG without moving the anchor click.

```bash
.venv/bin/python tools/reconstruct_object_multiview.py \
  --pointcloud outputs/reconstruction/cup_session/view_000/object_pointcloud.npz \
  --pointcloud outputs/reconstruction/cup_session/view_001/object_pointcloud.npz \
  --pointcloud outputs/reconstruction/cup_session/view_002/object_pointcloud.npz \
  --pointcloud outputs/reconstruction/cup_session/view_003/object_pointcloud.npz \
  --pointcloud outputs/reconstruction/cup_session/view_004/object_pointcloud.npz \
  --pointcloud outputs/reconstruction/cup_session/view_005/object_pointcloud.npz \
  --output outputs/reconstruction/cup_session/fused_object_pointcloud.npz \
  --ply outputs/reconstruction/cup_session/fused_object_pointcloud.ply
```

The fusion metadata records the RMSE for each view. Inspect and recapture any
view marked `warning_high_rmse`; do not proceed with a bad registration.

## HUG point-cloud experiment

The pretrained HUG checkpoint was trained with a single RGB-D point cloud.
Run the baseline and the fused-cloud experiment with identical RGB-D anchor,
selected point, seed range, and HUG sampling steps. This is an experiment, not
a claim that the checkpoint is already multi-view trained.

```bash
# Baseline: HUG rebuilds its PointNeXt input from anchor depth.
.venv/bin/python tools/generate_hug_candidates.py \
  --rgb outputs/reconstruction/cup_session/view_000/rgb.png \
  --depth outputs/reconstruction/cup_session/view_000/depth.png \
  --intrinsics outputs/reconstruction/cup_session/view_000/intrinsics.txt \
  --pointcloud outputs/reconstruction/cup_session/fused_object_pointcloud.npz \
  --robot l25 --optimizer vector --candidates 10 --seed-start 0 \
  --output outputs/grasp/cup_session/hug_single_view --dry-run

# Experimental: inject the fused anchor-frame cloud into HUG PointNeXt.
.venv/bin/python tools/generate_hug_candidates.py \
  --rgb outputs/reconstruction/cup_session/view_000/rgb.png \
  --depth outputs/reconstruction/cup_session/view_000/depth.png \
  --intrinsics outputs/reconstruction/cup_session/view_000/intrinsics.txt \
  --pointcloud outputs/reconstruction/cup_session/fused_object_pointcloud.npz \
  --hug-pointcloud outputs/reconstruction/cup_session/fused_object_pointcloud.npz \
  --robot l25 --optimizer vector --candidates 10 --seed-start 0 \
  --output outputs/grasp/cup_session/hug_fused_pointcloud --dry-run
```

Both commands use the same fused model for geometric scoring. Only the second
changes HUG's PointNeXt input. Compare corresponding candidate seeds before
ranking, then compare their contact distances, MANO/mesh intersection, L25
solver cost, and joint saturation. No command in this workflow sends hardware
commands.

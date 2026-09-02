# 3DGenerationPipeline reconstruction adapter

This integration keeps Hunyuan and 3DGenerationPipeline as interchangeable
reconstruction backends. Downstream HUG and L25 tools consume the same files:

```text
reconstruction/
  object_mesh_anchor.ply
  mesh_surface_anchor.npz
  object_surface_anchor.npz
  alignment_report.json
  reconstruction_metadata.json
```

`object_surface_anchor.npz` is a hybrid cloud. Measured Gemini RGB-D points are
kept verbatim with confidence 1.0. Mesh-only completion points are added with a
lower confidence (0.5 by default). The current pretrained HUG consumes the
points and colors but does not consume the confidence field.

## Export from 3DGenerationPipeline

Prefer the Fine pipeline and provide the measured maximum OBB edge in
millimetres. Export a GLB or the axis-aligned OBB-centred OBJ. The generated
mesh is not yet in the Gemini optical frame; metric scale and 6D pose are two
different requirements.

For the first integration test, automatically estimate only a rigid pose. This
preserves the metric scale exported by 3DGenerationPipeline:

```bash
cd /home/evolabs-5080/lianyichao/AnyDexRetarget

SCENE=/home/evolabs-5080/lianyichao/AnyDexRetarget/outputs/reconstruction/cup_session_run1
GEN_MESH=/absolute/path/to/3dgeneration_fine_metric.glb

PYTHONPATH="$PWD" \
  .venv/bin/python \
  tools/import_3dgeneration_reconstruction.py \
    --mesh "$GEN_MESH" \
    --anchor-pointcloud "$SCENE/view_000/object_pointcloud.npz" \
    --auto-align \
    --backend 3dgenerationpipeline-fine \
    --anchor-frame hand_camera_color_optical_frame \
    --output-dir "$SCENE/3dgeneration/reconstruction"
```

If the exported file is not in metres, specify `--source-unit mm` or
`--source-unit cm`. If the GUI export did not apply the measured size, use
`--known-max-dimension-mm VALUE`. Do not use this option merely to make the
overlay look better; `VALUE` must be a physical measurement.

Automatic partial-surface registration is provisional. Inspect the overlay
and reject solutions with the wrong side or orientation:

```bash
PYTHONPATH="$PWD" \
  .venv/bin/python \
  tools/project_mesh_to_rgb.py \
    --mesh "$SCENE/3dgeneration/reconstruction/object_mesh_anchor.ply" \
    --rgb "$SCENE/view_000/rgb.png" \
    --intrinsics "$SCENE/view_000/intrinsics.txt" \
    --output "$SCENE/3dgeneration/reconstruction/alignment_overlay.png"
```

For deployment, use a measured or pose-estimator-produced rigid transform.
The matrix convention is `T_anchor_object`: it maps metre-valued object-frame
points into `hand_camera_color_optical_frame`.

```bash
PYTHONPATH="$PWD" \
  .venv/bin/python \
  tools/import_3dgeneration_reconstruction.py \
    --mesh "$GEN_MESH" \
    --anchor-pointcloud "$SCENE/view_000/object_pointcloud.npz" \
    --transform /absolute/path/to/T_anchor_object.json \
    --backend 3dgenerationpipeline-fine \
    --anchor-frame hand_camera_color_optical_frame \
    --output-dir "$SCENE/3dgeneration/reconstruction" \
    --overwrite
```

The JSON may be either a raw 4x4 array or:

```json
{"T_anchor_object": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0, 1]]}
```

## Downstream boundary

HUG uses the standardized hybrid surface:

```bash
--hug-pointcloud "$SCENE/3dgeneration/reconstruction/object_surface_anchor.npz"
```

Contact and collision planning use the standardized mesh:


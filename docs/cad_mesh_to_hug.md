# CAD Mesh To HUG

This route supplies a complete, manually created or CAD-derived object geometry to
HUG's point-cloud branch. It does not claim that a single RGB-D image reconstructs
hidden geometry.

## Required Inputs

- An anchor RGB-D capture: `rgb.png`, registered `depth.png`, and `intrinsics.txt`.
- A triangle mesh in OBJ, PLY, or STL format.
- `model_to_camera.txt`: a finite 4x4 rigid transform from the mesh's **meter-scaled**
  model coordinates into the anchor RGB-D camera coordinates.

The matrix format is four whitespace-separated rows:

```text
r00 r01 r02 tx
r10 r11 r12 ty
r20 r21 r22 tz
0   0   0   1
```

The mesh must overlay the object in the anchor camera view at its real position,
scale, and orientation. A mesh placed at the Blender world origin is not usable
until this transform is known.

## Convert A Mesh

For a Blender/CAD mesh expressed in millimeters:

```bash
.venv/bin/python tools/mesh_to_hug_pointcloud.py \
  --mesh assets/cad/cup.obj \
  --model-to-camera assets/cad/cup_model_to_camera.txt \
  --model-unit mm \
  --samples 20000 \
  --color 185 70 50 \
  --output outputs/cad/cup_hug_pointcloud.npz
```

The output contains `points_camera` in meters and `colors_rgb`, which is the
format accepted by `--hug-pointcloud`.

## Verify Alignment

Overlay the observed masked RGB-D cloud with the sampled CAD points before HUG:

```bash
.venv/bin/python tools/view_object_pointcloud.py \
  --pointcloud outputs/interactive/cup/object_pointcloud.npz \
  --overlay-pointcloud outputs/cad/cup_hug_pointcloud.npz \
  --port 8086
```

The observed RGB-D cloud uses its original colors; the CAD samples are amber.
They must occupy the same object surface. Correct the transform in Blender/CAD
when the two clouds differ in position, scale, or orientation.

## Run HUG Candidates With CAD Geometry

Keep the observed masked cloud as `--pointcloud`; it is used to rank visible
surface contact. Pass the aligned CAD cloud separately as the experimental HUG
PointNeXt input:

```bash
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    .venv/bin/python tools/generate_hug_candidates.py \
      --rgb outputs/interactive/cup/rgb.png \
      --depth outputs/interactive/cup/depth.png \
      --intrinsics outputs/interactive/cup/intrinsics.txt \
      --pointcloud outputs/interactive/cup/object_pointcloud.npz \
      --hug-pointcloud outputs/cad/cup_hug_pointcloud.npz \
      --robot l25 --optimizer vector \
      --candidates 10 --sampling-steps 50 \
      --frames 60 --fps 30 \
      --output outputs/interactive/cup/hug_candidates_cad \
      --dry-run
```

The HUG checkpoint was trained with single-view RGB-D point clouds. CAD input is
therefore an experimental geometry-conditioned input, not a guarantee that an
unseen surface produces a physically valid grasp. Compare it against the
single-view baseline using the same RGB-D capture and random seeds.

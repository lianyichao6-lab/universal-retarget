# Gaia Hand20 assets

The URDF and STL meshes in this directory were copied from the Gaia Hand20 ROS
packages supplied with the project integration request:

- `gaiahand20_right`
- `gaiahand20_left`

The package metadata identifies the author as Stellarobot and declares the BSD
license. The original package documentation lists `support@stellarobotic.com`.

Local changes:

- ROS `package://` mesh URLs were rewritten to repository-relative paths.
- Five fixed fingertip frames were added to each URDF because the source model
  ends at each distal-link origin rather than defining fingertip frames.
- The MuJoCo models are generated from these URDFs by `generate_mjcf.py`. All 21
  source visual meshes (palm plus 20 finger links) are retained.

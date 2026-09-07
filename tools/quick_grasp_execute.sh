#!/usr/bin/env bash
set -euo pipefail
#
# Simplified end-to-end: ROS perception → single HUG grasp → L25 retarget → hardware execute.
#
# Skips: 50-candidate generation, 4-backend benchmark, collision-aware planning.
# Produces one grasp, one trajectory, and drives the G20 hand directly.
#
# Prerequisites:
#   - ROS perception publishing on /clawbot_cam_head/...
#   - CAN0 UP with LinkerHand connected
#   - source /opt/ros/jazzy/setup.bash (already done)
#
# Usage:
#   bash tools/quick_grasp_execute.sh --tag <object_tag> --scene <output_name>
#
# Example:
#   bash tools/quick_grasp_execute.sh --tag P5636_36A__20250920001 --scene run_motor_quick

usage() {
    cat <<'HELP'
Usage: quick_grasp_execute.sh --tag TAG --scene SCENE [options]

Required:
  --tag TAG           Perception mesh_cloud object tag (e.g. P5636_36A__20250920001)
  --scene SCENE       Output directory name under outputs/perception_bridge/

Options:
  --seed SEED         HUG random seed (default: 42)
  --optimizer OPT     Retarget backend: vector|adaptive (default: vector)
  --mesh-shrink PCT   Shrink object point cloud by PCT% toward centroid (default: 0)
                      e.g. --mesh-shrink 10 = shrink 10% for tighter grip
  --grip-tighten PCT  Post-retarget: close finger flexion joints PCT% more
                      toward max (default: 0). e.g. --grip-tighten 30
  --max-step N        Hardware max SDK units per step, 1..5 (default: 5)
  --rate-hz HZ        Hardware command rate, 2..20 (default: 10)
  --speed N           Finger speed 10..80 (default: 30)
  --torque N          Finger torque 10..80 (default: 40)
  --skip-capture      Skip step 1+2 if perception data already captured
  --dry-run           Stop before hardware execution (default: execute)
  -h|--help           Show this help
HELP
    exit 1
}

# ── defaults ──────────────────────────────────────────────────────────────────
TAG=""
SCENE=""
SEED=42
OPTIMIZER=vector
MESH_SHRINK=0
GRIP_TIGHTEN=0
MAX_STEP=5
RATE_HZ=10
SPEED=30
TORQUE=40
SKIP_CAPTURE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)           TAG="$2";       shift 2 ;;
        --scene)         SCENE="$2";     shift 2 ;;
        --seed)          SEED="$2";      shift 2 ;;
        --optimizer)     OPTIMIZER="$2"; shift 2 ;;
        --mesh-shrink)   MESH_SHRINK="$2"; shift 2 ;;
        --grip-tighten)  GRIP_TIGHTEN="$2"; shift 2 ;;
        --max-step)      MAX_STEP="$2";  shift 2 ;;
        --rate-hz)       RATE_HZ="$2";   shift 2 ;;
        --speed)         SPEED="$2";     shift 2 ;;
        --torque)        TORQUE="$2";    shift 2 ;;
        --skip-capture)  SKIP_CAPTURE=true; shift ;;
        --dry-run)       DRY_RUN=true;   shift ;;
        -h|--help)       usage ;;
        *)               echo "Unknown option: $1"; usage ;;
    esac
done

[[ -z "$TAG" ]]   && { echo "ERROR: --tag is required"; usage; }
[[ -z "$SCENE" ]] && { echo "ERROR: --scene is required"; usage; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv/bin/python"
PERCEPTION_PY="$HOME/miniconda3/envs/robot_perception/bin/python"
OUT="$ROOT/outputs/perception_bridge/$SCENE"

echo "============================================"
echo " Quick Grasp Pipeline"
echo " TAG:   $TAG"
echo " SCENE: $SCENE"
echo " OUT:   $OUT"
echo "============================================"

# ── Step 1: Capture perception data ──────────────────────────────────────────
if [[ "$SKIP_CAPTURE" == "false" ]]; then
    if [[ -d "$OUT" ]]; then
        echo ""
        echo "Cleaning previous run: $OUT"
        rm -rf "$OUT"
    fi

    echo ""
    echo "=== Step 1/5: Capture perception (auto-save on first valid frame) ==="
    /usr/bin/python3 "$ROOT/tools/capture_ros_perception.py" \
        --output "$OUT"

    echo ""
    echo "=== Step 2/5: Capture object mesh ==="
    "$PERCEPTION_PY" "$ROOT/tools/capture_ros_object_mesh.py" \
        --cloud-topic "/clawbot_cam_head/perception/mesh_cloud/obj_${TAG}" \
        --output "$OUT/object_mesh_camera.ply"
else
    echo ""
    echo "=== Step 1-2: Skipped (--skip-capture) ==="
    for f in rgb.png depth.png intrinsics.txt mask.png object_pointcloud.npz; do
        [[ -f "$OUT/$f" ]] || { echo "ERROR: $OUT/$f not found; remove --skip-capture"; exit 1; }
    done
fi

# ── Step 3: Single HUG grasp + L25 retarget ─────────────────────────────────
echo ""
echo "=== Step 3/5: HUG grasp + L25 retarget (seed=$SEED, optimizer=$OPTIMIZER) ==="

# Read foreground point from mask metadata
POINT=$(/usr/bin/python3 -c "
import json
meta = json.load(open('$OUT/mask.json'))
print(meta['foreground_point'][0], meta['foreground_point'][1])
")

# Optionally shrink the object point cloud for tighter grip
HUG_PCL_FLAG=()
if [[ "$MESH_SHRINK" != "0" ]]; then
    SHRUNK_PCL="$OUT/object_pointcloud_shrunk.npz"
    echo "  Shrinking point cloud by ${MESH_SHRINK}% toward centroid..."
    /usr/bin/python3 -c "
import numpy as np
pct = float('$MESH_SHRINK') / 100.0
with np.load('$OUT/object_pointcloud.npz', allow_pickle=False) as d:
    pts = d['points_camera'].astype(np.float64)
    centroid = pts.mean(axis=0)
    pts_shrunk = centroid + (1.0 - pct) * (pts - centroid)
    np.savez('$SHRUNK_PCL',
        points_camera=pts_shrunk.astype(np.float32),
        colors_rgb=d['colors_rgb'],
        pixels_uv=d['pixels_uv'],
        intrinsics=d['intrinsics'],
        source_rgb=d['source_rgb'],
        source_depth=d['source_depth'],
        source_mask=d['source_mask'])
print(f'  Shrunk {len(pts)} points by {pct*100:.0f}%')
"
    HUG_PCL_FLAG=(--hug-pointcloud "$SHRUNK_PCL")
fi

env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$VENV" "$ROOT/tools/grasp_object.py" \
    --rgb "$OUT/rgb.png" \
    --depth "$OUT/depth.png" \
    --intrinsics "$OUT/intrinsics.txt" \
    --point $POINT \
    --robot l25 --optimizer "$OPTIMIZER" \
    --seed "$SEED" --frames 60 --dry-run \
    --overwrite \
    "${HUG_PCL_FLAG[@]}" \
    --output "$OUT/quick_grasp"

TRAJECTORY="$OUT/quick_grasp/trajectory.pkl"
if [[ ! -f "$TRAJECTORY" ]]; then
    echo "ERROR: trajectory not generated at $TRAJECTORY"
    exit 1
fi

# ── Optional: Tighten grip (post-retarget) ──────────────────────────────────
if [[ "$GRIP_TIGHTEN" != "0" ]]; then
    echo ""
    echo "  Tightening grip by ${GRIP_TIGHTEN}% (closing flexion joints toward max)..."
    "$VENV" -c "
import numpy as np, pickle
pct = float('$GRIP_TIGHTEN') / 100.0
# Flexion qpos indices (21-dim) and their upper sim limits
FLEXION = {
    2: 0.83, 3: 1.25,       # thumb pitch, mcp
    6: 1.22, 7: 1.75,       # index pitch, pip
    10: 1.22, 11: 1.75,     # middle pitch, pip
    14: 1.22, 15: 1.75,     # ring pitch, pip
    18: 1.22, 19: 1.75,     # pinky pitch, pip
}
# Modify trajectory.pkl
pkl_path = '$OUT/quick_grasp/trajectory.pkl'
with open(pkl_path, 'rb') as f:
    records = pickle.load(f)
for rec in records:
    q = np.asarray(rec['target'], dtype=np.float64)
    for idx, qmax in FLEXION.items():
        q[idx] = min(q[idx] + pct * (qmax - q[idx]), qmax)
    rec['target'] = q
with open(pkl_path, 'wb') as f:
    pickle.dump(records, f)
# Modify trajectory.npz (used by show_quick_grasp.py)
npz_path = '$OUT/quick_grasp/trajectory.npz'
data = dict(np.load(npz_path, allow_pickle=False))
qpos = data['robot_qpos'].astype(np.float64)
for i in range(len(qpos)):
    for idx, qmax in FLEXION.items():
        qpos[i, idx] = min(qpos[i, idx] + pct * (qmax - qpos[i, idx]), qmax)
data['robot_qpos'] = qpos.astype(np.float32)
np.savez(npz_path, **data)
print(f'  Tightened {len(records)} frames by {pct*100:.0f}%')
"
fi

echo ""
echo "=== Step 4/5: Trajectory ready ==="
echo "  $TRAJECTORY"

# ── Step 5: Hardware execute ─────────────────────────────────────────────────
if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "=== Step 5/5: Dry run — skipping hardware execution ==="
    echo "To execute on hardware, re-run without --dry-run"
    echo ""
    echo "Or manually:"
    echo "  /usr/bin/python3 tools/g20_hardware_execute.py \\"
    echo "    --sdk-package \"\$LINKERHAND_SDK_PACKAGE\" \\"
    echo "    --trajectory $TRAJECTORY \\"
    echo "    --frame 0 --hardware --confirm G20_RIGHT_CLEAR --read-state \\"
    echo "    --channels 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \\"
    echo "    --replay-target --rate-hz $RATE_HZ --max-step $MAX_STEP \\"
    echo "    --speed $SPEED --torque $TORQUE \\"
    echo "    --report $OUT/g20_execute.json"
    exit 0
fi

if [[ -z "${LINKERHAND_SDK_PACKAGE:-}" ]]; then
    echo "ERROR: LINKERHAND_SDK_PACKAGE not set"
    echo "  export LINKERHAND_SDK_PACKAGE=~/linker_hand_ros2_sdk/linker_hand_ros2_sdk/linker_hand_ros2_sdk"
    exit 1
fi

echo ""
echo "=== Step 5/5: Hardware execution ==="
echo "  max-step=$MAX_STEP  rate-hz=$RATE_HZ  speed=$SPEED  torque=$TORQUE"
echo ""

/usr/bin/python3 "$ROOT/tools/g20_hardware_execute.py" \
    --sdk-package "$LINKERHAND_SDK_PACKAGE" \
    --trajectory "$TRAJECTORY" \
    --frame 0 \
    --hardware --confirm G20_RIGHT_CLEAR \
    --read-state \
    --channels 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
    --replay-target --rate-hz "$RATE_HZ" --max-step "$MAX_STEP" \
    --speed "$SPEED" --torque "$TORQUE" \
    --report "$OUT/g20_execute.json"

echo ""
echo "============================================"
echo " Done!"
echo " Trajectory: $TRAJECTORY"
echo " Report:     $OUT/g20_execute.json"
echo "============================================"

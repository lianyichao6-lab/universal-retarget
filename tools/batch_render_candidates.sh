#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --scene <run_dir> [--backend vector] [--width 960] [--height 720] [--jobs 4]"
    echo ""
    echo "Batch build MuJoCo scenes and render grasp PNGs for all candidates."
    echo ""
    echo "  --scene     Run directory, e.g. outputs/perception_bridge/run_001"
    echo "  --backend   Retarget backend (default: vector)"
    echo "  --width     Render width  (default: 960)"
    echo "  --height    Render height (default: 720)"
    echo "  --jobs      Parallel render jobs (default: 4)"
    exit 1
}

SCENE=""
BACKEND="vector"
WIDTH=960
HEIGHT=720
JOBS=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)   SCENE="$2";   shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --width)   WIDTH="$2";   shift 2 ;;
        --height)  HEIGHT="$2";  shift 2 ;;
        --jobs)    JOBS="$2";    shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

[[ -z "$SCENE" ]] && usage

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv/bin/python"
PLAN_DIR="$SCENE/backend_benchmark_50/$BACKEND"
OUTPUT_DIR="$SCENE/mujoco_view"
MESH_PROXY="$SCENE/backend_benchmark_50/_shared/object_collision_proxy.ply"

if [[ ! -d "$PLAN_DIR" ]]; then
    echo "ERROR: plan directory not found: $PLAN_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

PLANS=("$PLAN_DIR"/candidate_*/l25_collision_aware_plan.npz)
TOTAL=${#PLANS[@]}
echo "Found $TOTAL candidates in $PLAN_DIR"

MESH_FLAG=()
if [[ -f "$MESH_PROXY" ]]; then
    MESH_FLAG=(--mesh-proxy "$MESH_PROXY")
    echo "Using shared mesh proxy: $MESH_PROXY"
fi

render_one() {
    local plan="$1"
    local candidate
    candidate="$(basename "$(dirname "$plan")")"
    local out_dir="$OUTPUT_DIR/$candidate"
    local png="$out_dir/grasp.png"

    if [[ -f "$png" ]]; then
        echo "SKIP  $candidate (already rendered)"
        return 0
    fi

    "$VENV" "$ROOT/tools/build_l25_object_relative_scene.py" \
        --plan "$plan" \
        --output-dir "$out_dir" \
        "${MESH_FLAG[@]}" \
        > /dev/null 2>&1

    "$VENV" "$ROOT/tools/render_l25_scene.py" \
        --scene "$out_dir/l25_object_relative_scene.xml" \
        --plan "$plan" \
        --output "$png" \
        --width "$WIDTH" --height "$HEIGHT" \
        > /dev/null 2>&1

    echo "OK    $candidate -> $png"
}
export -f render_one
export VENV ROOT OUTPUT_DIR MESH_PROXY MESH_FLAG WIDTH HEIGHT

DONE=0
FAIL=0
for plan in "${PLANS[@]}"; do
    if render_one "$plan"; then
        DONE=$((DONE + 1))
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  $(basename "$(dirname "$plan")")" >&2
    fi
    printf "\r[%d/%d] " "$((DONE + FAIL))" "$TOTAL"
done
echo ""

echo "Rendered $DONE/$TOTAL candidates ($FAIL failures)"
echo "Output:  $OUTPUT_DIR/"

# Build a contact sheet if montage (ImageMagick) is available.
if command -v montage &>/dev/null && [[ $DONE -gt 0 ]]; then
    SHEET="$OUTPUT_DIR/contact_sheet.png"
    montage "$OUTPUT_DIR"/candidate_*/grasp.png \
        -tile 10x -geometry "${WIDTH}x${HEIGHT}+4+4" \
        -background '#1a1a1a' \
        "$SHEET" 2>/dev/null \
    && echo "Contact sheet: $SHEET" \
    || echo "(montage failed; individual PNGs still available)"
fi

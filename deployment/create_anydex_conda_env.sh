#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ANYDEX_CONDA_ENV:-anydex}"
CONDA_CMD="${CONDA_EXE:-conda}"

if ! command -v "$CONDA_CMD" >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda/Miniforge or set CONDA_EXE." >&2
  exit 1
fi

"$CONDA_CMD" env create \
  --name "$ENV_NAME" \
  --file "$REPO_ROOT/deployment/environment-anydex.conda.yml"

pushd "$REPO_ROOT" >/dev/null

"$CONDA_CMD" run --name "$ENV_NAME" python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1

"$CONDA_CMD" run --name "$ENV_NAME" python -m pip install \
  --find-links https://data.pyg.org/whl/torch-2.9.0+cu128.html \
  torch-cluster==1.6.3+pt29cu128

"$CONDA_CMD" run --name "$ENV_NAME" python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --find-links https://data.pyg.org/whl/torch-2.9.0+cu128.html \
  --requirement deployment/environment-anydex-nosomehand.freeze.txt

popd >/dev/null

echo "Created Conda environment: $ENV_NAME"
echo "Activate it with: conda activate $ENV_NAME"

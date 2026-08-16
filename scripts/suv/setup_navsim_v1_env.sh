#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_ENV="${SOURCE_ENV:-py312torch210cu126}"
TARGET_ENV="${TARGET_ENV:-suv-navsim1}"
NAVSIM_V1_ROOT="${NAVSIM_V1_ROOT:-${REPO_ROOT}/output/suv/navsim_v1_devkit}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"

if ! conda env list | awk '{print $1}' | grep -Fxq "${TARGET_ENV}"; then
  conda create \
    -n "${TARGET_ENV}" \
    --clone "${SOURCE_ENV}" \
    --override-channels \
    -c conda-forge \
    -y
fi

mkdir -p "$(dirname "${NAVSIM_V1_ROOT}")"
if [[ ! -e "${NAVSIM_V1_ROOT}/.git" ]]; then
  git -C "${REPO_ROOT}/third_party/navsim" worktree add \
    --force --detach "${NAVSIM_V1_ROOT}" v1.1
fi

EXPECTED_COMMIT="$(git -C "${REPO_ROOT}/third_party/navsim" rev-parse v1.1^{commit})"
ACTUAL_COMMIT="$(git -C "${NAVSIM_V1_ROOT}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "NAVSIM v1 worktree points to ${ACTUAL_COMMIT}, expected ${EXPECTED_COMMIT}." >&2
  exit 2
fi

conda run -n "${TARGET_ENV}" python -m pip uninstall -y navsim || true
conda run -n "${TARGET_ENV}" python -m pip install \
  --no-build-isolation --no-deps -e "${NAVSIM_V1_ROOT}"
conda run -n "${TARGET_ENV}" python -m pip install \
  --no-build-isolation --no-deps -e "${REPO_ROOT}/third_party/SUV"
conda run -n "${TARGET_ENV}" python -m pip install -i "${PYPI_INDEX_URL}" \
  control==0.9.1 \
  einops==0.8.1 \
  huggingface-hub==0.29.2 \
  hydra-core==1.3.2 \
  modelscope==1.34.0 \
  omegaconf==2.3.0 \
  opencv-python-headless==4.9.0.80 \
  pandas==2.2.3 \
  pillow==12.0.0 \
  positional-encodings==6.0.1 \
  pyarrow \
  pytorch-lightning==2.2.1 \
  regex==2025.11.3 \
  rich==14.2.0 \
  rtree \
  safetensors==0.5.3 \
  scikit-learn==1.4.2 \
  setuptools==80.9.0 \
  tqdm==4.66.5 \
  transformers==4.49.0 \
  ujson

cd "${REPO_ROOT}"
conda run -n "${TARGET_ENV}" python -m tools.suv.evaluate_navsim_v1 doctor

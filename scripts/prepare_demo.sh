#!/usr/bin/env bash
# Prepare the default mentor demo end-to-end.
#
# Usage:
#   ./scripts/prepare_demo.sh
#   ./scripts/prepare_demo.sh --all   # also build image and run import smoke test

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SMOKE_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --all) SMOKE_ARGS+=(--all) ;;
    -h|--help)
      sed -n '1,6p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [--all]"
      exit 2
      ;;
  esac
done

./scripts/prepare_dataset.sh
./scripts/prepare_models.sh
./scripts/docker_smoke_test.sh "${SMOKE_ARGS[@]}"

echo ""
echo "[prepare_demo] Ready. Run:"
echo "  docker compose up"

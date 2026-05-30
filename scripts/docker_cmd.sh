#!/usr/bin/env bash
# Shared Docker command resolver for setup scripts.
#
# Exports:
#   DOCKER=(docker)       or DOCKER=(sudo docker)
#   DOCKER_SUDO=0|1

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This file is meant to be sourced by other scripts."
  exit 2
fi

if ! command -v docker >/dev/null; then
  echo "[ERROR] docker is required."
  exit 1
fi

DOCKER=(docker)
DOCKER_SUDO=0

if ! docker info >/dev/null 2>&1; then
  if command -v sudo >/dev/null && sudo -n docker info >/dev/null 2>&1; then
    DOCKER=(sudo docker)
    DOCKER_SUDO=1
    echo "[docker] Using sudo docker because current user cannot access Docker directly."
  else
    docker_user="${SUDO_USER:-$USER}"
    echo "[ERROR] Docker is installed but not usable by the current user."
    echo "Try one of:"
    echo "  sudo ./scripts/prepare_demo.sh"
    echo "  sudo usermod -aG docker $docker_user   # then log out/in"
    echo "  sudo docker run --rm hello-world"
    exit 1
  fi
fi

if ! "${DOCKER[@]}" compose version >/dev/null 2>&1; then
  echo "[ERROR] docker compose is required."
  exit 1
fi

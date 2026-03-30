#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ISAAC_VERSION_EXPECTED="${ISAAC_VERSION_EXPECTED:-4.2.0}"

resolve_workspace_root() {
  if [ -n "${WORKSPACE_ROOT:-}" ]; then
    printf '%s' "${WORKSPACE_ROOT}"
    return
  fi

  if [ -n "${COLCON_CURRENT_PREFIX:-}" ]; then
    cd -- "${COLCON_CURRENT_PREFIX}/.." && pwd
    return
  fi

  if [ -d "${DEFAULT_REPO_ROOT}/isaac_app" ]; then
    cd -- "${DEFAULT_REPO_ROOT}/.." && pwd
    return
  fi

  cd -- "${DEFAULT_REPO_ROOT}/../../../.." && pwd
}

WORKSPACE_ROOT="$(resolve_workspace_root)"
DEFAULT_ISAAC_ROOT="${WORKSPACE_ROOT}/third_party/isaac-sim-${ISAAC_VERSION_EXPECTED}"

resolve_source_repo_root() {
  if [ -n "${SOURCE_REPO_ROOT:-}" ] && [ -d "${SOURCE_REPO_ROOT}/isaac_app" ]; then
    printf '%s' "${SOURCE_REPO_ROOT}"
    return
  fi

  if [ -d "${WORKSPACE_ROOT}/repo/isaac_app" ]; then
    printf '%s' "${WORKSPACE_ROOT}/repo"
    return
  fi

  printf '%s' "${DEFAULT_REPO_ROOT}"
}

REPO_ROOT="$(resolve_source_repo_root)"

resolve_isaac_root() {
  printf '%s' "${ISAAC_ROOT:-${DEFAULT_ISAAC_ROOT}}"
}

read_isaac_version() {
  local isaac_root="$1"
  if [ -f "${isaac_root}/VERSION" ]; then
    head -n 1 "${isaac_root}/VERSION"
  fi
}

validate_isaac_install() {
  local isaac_root="$1"
  local version=""

  if [ ! -d "${isaac_root}" ]; then
    echo "Isaac Sim not found at ${isaac_root}." >&2
    echo "Expected a native 4.2.0 install at ${DEFAULT_ISAAC_ROOT}, or set ISAAC_ROOT explicitly." >&2
    return 1
  fi

  if [ ! -x "${isaac_root}/python.sh" ]; then
    echo "Isaac Sim install at ${isaac_root} is missing python.sh." >&2
    return 1
  fi

  version="$(read_isaac_version "${isaac_root}")"
  if [ -n "${version}" ] && [[ "${version}" != "${ISAAC_VERSION_EXPECTED}"* ]] && [ "${ALLOW_UNVERIFIED_ISAAC:-0}" != "1" ]; then
    echo "Isaac Sim version mismatch at ${isaac_root}: found '${version}', expected '${ISAAC_VERSION_EXPECTED}.x'." >&2
    echo "Set ISAAC_ROOT to a 4.2.0 install, or set ALLOW_UNVERIFIED_ISAAC=1 to bypass the version guard." >&2
    return 1
  fi
}

sanitize_path() {
  local path_value="$1"
  local sanitized=""
  local segment=""
  local old_ifs="${IFS}"
  IFS=':'
  for segment in ${path_value}; do
    case "${segment}" in
      *miniconda*|*anaconda*)
        continue
        ;;
    esac
    if [[ -z "${sanitized}" ]]; then
      sanitized="${segment}"
    else
      sanitized="${sanitized}:${segment}"
    fi
  done
  IFS="${old_ifs}"
  printf '%s' "${sanitized}"
}

build_isaac_ros_bridge_paths() {
  local isaac_root="$1"
  local python_paths=()
  local library_paths=()
  local candidate=""

  for candidate in \
    "${isaac_root}/exts/isaacsim.ros2.bridge/humble/rclpy" \
    "${isaac_root}/exts/omni.isaac.ros2_bridge/humble/rclpy"
  do
    if [ -d "${candidate}" ]; then
      python_paths+=("${candidate}")
    fi
  done

  for candidate in \
    "${isaac_root}/exts/isaacsim.ros2.bridge/humble/lib" \
    "${isaac_root}/exts/omni.isaac.ros2_bridge/humble/lib"
  do
    if [ -d "${candidate}" ]; then
      library_paths+=("${candidate}")
    fi
  done

  printf '%s\n' "${python_paths[*]:-}"
  printf '%s\n' "${library_paths[*]:-}"
}

build_ros_python_paths() {
  local python_version=""
  local candidates=()
  local candidate=""

  python_version="$(/usr/bin/python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

  candidates+=(
    "${WORKSPACE_ROOT}/install_sys/chess_manipulator/lib/python${python_version}/site-packages"
    "${WORKSPACE_ROOT}/install_sys/chess_manipulator_msgs/lib/python${python_version}/site-packages"
    "${WORKSPACE_ROOT}/install_sys/chess_manipulator_msgs/local/lib/python${python_version}/dist-packages"
    "/opt/ros/humble/lib/python${python_version}/site-packages"
    "/opt/ros/humble/local/lib/python${python_version}/dist-packages"
  )

  for candidate in "${candidates[@]}"; do
    if [ -d "${candidate}" ]; then
      printf '%s\n' "${candidate}"
    fi
  done | awk '!seen[$0]++'
}

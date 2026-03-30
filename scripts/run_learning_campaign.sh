#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"

TOTAL_GAMES="${TOTAL_GAMES:-2000}"
START_GAME="${START_GAME:-1}"
BATCH_SIZE="${BATCH_SIZE:-50}"
MATCH_FORMAT="${MATCH_FORMAT:-short}"
MAX_PLIES="${MAX_PLIES:-}"
THINK_TIME_SEC="${THINK_TIME_SEC:-0.1}"
CAMPAIGN_ID="${CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${REPO_ROOT}/results/campaigns/${CAMPAIGN_ID}}"
GAME_LOG_DIR="${GAME_LOG_DIR:-${CAMPAIGN_ROOT}/game_logs}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${CAMPAIGN_ROOT}/run_logs}"
VIDEO_DIR="${VIDEO_DIR:-${CAMPAIGN_ROOT}/videos}"
OFFLINE_DIR="${OFFLINE_DIR:-${CAMPAIGN_ROOT}/offline_learning}"
PROMOTED_CHECKPOINT="${PROMOTED_CHECKPOINT:-${REPO_ROOT}/results/demo/learned_player/black_dqn.pt}"
CAPTURE_MILESTONES="${CAPTURE_MILESTONES:-1}"
VIDEO_DISPLAY="${VIDEO_DISPLAY:-${DISPLAY:-:0.0}}"
VIDEO_SIZE="${VIDEO_SIZE:-1920x1080}"
VIDEO_FPS="${VIDEO_FPS:-30}"
GAME_TIMEOUT_SEC="${GAME_TIMEOUT_SEC:-}"
OFFLINE_OPTIMIZATION_STEPS="${OFFLINE_OPTIMIZATION_STEPS:-250}"
SELF_PLAY_EPISODES="${SELF_PLAY_EPISODES:-0}"
POST_SELF_PLAY_OPTIMIZATION_STEPS="${POST_SELF_PLAY_OPTIMIZATION_STEPS:-50}"
MIN_LEGAL_RATE="${MIN_LEGAL_RATE:-1.0}"
MIN_AGREEMENT_RATE="${MIN_AGREEMENT_RATE:-0.0}"

mkdir -p "${GAME_LOG_DIR}" "${RUN_LOG_DIR}" "${VIDEO_DIR}" "${OFFLINE_DIR}"

if [ -z "${MAX_PLIES}" ]; then
  case "${MATCH_FORMAT}" in
    short)
      MAX_PLIES=8
      ;;
    full)
      MAX_PLIES=0
      ;;
    *)
      echo "Unsupported MATCH_FORMAT: ${MATCH_FORMAT}. Use 'short' or 'full'." >&2
      exit 1
      ;;
  esac
fi

if [ -z "${GAME_TIMEOUT_SEC}" ]; then
  if [ "${MATCH_FORMAT}" = "full" ] || [ "${MAX_PLIES}" -le 0 ]; then
    GAME_TIMEOUT_SEC=5400
  else
    GAME_TIMEOUT_SEC=600
  fi
fi

if [ ! -f "${PROMOTED_CHECKPOINT}" ]; then
  echo "Promoted checkpoint not found: ${PROMOTED_CHECKPOINT}" >&2
  echo "Run ./repo/scripts/train_baseline_learned_player.sh first." >&2
  exit 1
fi

set +u
source /opt/ros/humble/setup.bash
source "${WORKSPACE_ROOT}/install_sys/setup.bash"
set -u

cleanup_processes() {
  if [ -n "${VIDEO_PID:-}" ] && kill -0 "${VIDEO_PID}" >/dev/null 2>&1; then
    kill -INT "${VIDEO_PID}" >/dev/null 2>&1 || true
    wait "${VIDEO_PID}" >/dev/null 2>&1 || true
  fi
  "${REPO_ROOT}/scripts/stop_demo.sh" >/dev/null 2>&1 || true
  if [ -n "${DEMO_PID:-}" ] && kill -0 "${DEMO_PID}" >/dev/null 2>&1; then
    wait "${DEMO_PID}" >/dev/null 2>&1 || true
  fi
  VIDEO_PID=""
  DEMO_PID=""
}

trap cleanup_processes EXIT

is_milestone() {
  local game_index="$1"
  if [ "${game_index}" -eq 1 ]; then
    return 0
  fi
  if [ $((game_index % BATCH_SIZE)) -eq 0 ]; then
    return 0
  fi
  return 1
}

start_video_capture() {
  local output_path="$1"
  local window_id=""
  local geometry=""
  local x=""
  local y=""
  local width=""
  local height=""
  local attempts=0
  if [ "${CAPTURE_MILESTONES}" != "1" ]; then
    return 0
  fi
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found; skipping video capture." >&2
    return 0
  fi
  if ! command -v xwininfo >/dev/null 2>&1; then
    echo "xwininfo not found; skipping video capture." >&2
    return 0
  fi
  while [ "${attempts}" -lt 30 ]; do
    window_id="$(xwininfo -root -tree 2>/dev/null | awk '/"Isaac Sim"/ {print $1; exit}')"
    if [ -n "${window_id}" ]; then
      break
    fi
    attempts=$((attempts + 1))
    sleep 1
  done
  if [ -z "${window_id}" ]; then
    echo "Isaac Sim window not found; skipping video capture." >&2
    return 0
  fi
  geometry="$(xwininfo -id "${window_id}" 2>/dev/null || true)"
  x="$(printf '%s\n' "${geometry}" | awk '/Absolute upper-left X:/ {print $4; exit}')"
  y="$(printf '%s\n' "${geometry}" | awk '/Absolute upper-left Y:/ {print $4; exit}')"
  width="$(printf '%s\n' "${geometry}" | awk '/Width:/ {print $2; exit}')"
  height="$(printf '%s\n' "${geometry}" | awk '/Height:/ {print $2; exit}')"
  if [ -z "${x}" ] || [ -z "${y}" ] || [ -z "${width}" ] || [ -z "${height}" ]; then
    echo "Unable to resolve Isaac Sim window geometry; skipping video capture." >&2
    return 0
  fi
  ffmpeg -y \
    -video_size "${width}x${height}" \
    -framerate "${VIDEO_FPS}" \
    -f x11grab \
    -i "${VIDEO_DISPLAY}+${x},${y}" \
    -pix_fmt yuv420p \
    "${output_path}" \
    >"${output_path%.mp4}.ffmpeg.log" 2>&1 &
  VIDEO_PID=$!
}

run_single_game() {
  local game_index="$1"
  local run_log="${RUN_LOG_DIR}/game_${game_index}.log"
  local existing_count
  local archived_log
  local start_time
  local headless=true

  existing_count=$(find "${GAME_LOG_DIR}" -maxdepth 1 -type f -name 'game_*.json' | wc -l)
  start_time=$(date +%s)

  if is_milestone "${game_index}" && [ "${CAPTURE_MILESTONES}" = "1" ]; then
    headless=false
  fi

  printf '\n=== Running game %s/%s (headless=%s) ===\n' "${game_index}" "${TOTAL_GAMES}" "${headless}"

  env \
    SIM_BACKEND=isaac \
    ISAAC_HEADLESS="${headless}" \
    LEARNED_CHECKPOINT="${PROMOTED_CHECKPOINT}" \
    MATCH_FORMAT="${MATCH_FORMAT}" \
    MATCH_MAX_PLIES="${MAX_PLIES}" \
    "${REPO_ROOT}/scripts/run_engine_vs_learned_demo.sh" \
    game_log_dir:="${GAME_LOG_DIR}" \
    think_time_sec:="${THINK_TIME_SEC}" \
    >"${run_log}" 2>&1 &
  DEMO_PID=$!

  if [ "${headless}" = "false" ]; then
    start_video_capture "${VIDEO_DIR}/game_${game_index}.mp4"
  fi

  while true; do
    local current_count
    current_count=$(find "${GAME_LOG_DIR}" -maxdepth 1 -type f -name 'game_*.json' | wc -l)
    if [ "${current_count}" -gt "${existing_count}" ]; then
      break
    fi
    if ! kill -0 "${DEMO_PID}" >/dev/null 2>&1; then
      echo "Demo process exited before producing a new game log. See ${run_log}" >&2
      cleanup_processes
      return 1
    fi
    if [ $(( $(date +%s) - start_time )) -ge "${GAME_TIMEOUT_SEC}" ]; then
      echo "Timed out waiting for game ${game_index} to finish. See ${run_log}" >&2
      cleanup_processes
      return 1
    fi
    sleep 2
  done

  archived_log="${GAME_LOG_DIR}/campaign_game_$(printf '%04d' "${game_index}").json"
  latest_log="$(find "${GAME_LOG_DIR}" -maxdepth 1 -type f -name 'game_*.json' | sort | tail -n 1)"
  if [ -n "${latest_log}" ] && [ -f "${latest_log}" ]; then
    mv "${latest_log}" "${archived_log}"
  fi

  sleep 2
  cleanup_processes
  return 0
}

run_offline_update() {
  local game_index="$1"
  local run_id="games_${game_index}"
  local candidate="${OFFLINE_DIR}/${run_id}/black_dqn_candidate.pt"
  mkdir -p "${OFFLINE_DIR}/${run_id}"
  printf '\n=== Offline update after game %s ===\n' "${game_index}"
  env \
    GAME_LOG_DIR="${GAME_LOG_DIR}" \
    RUN_ID="${run_id}" \
    OUTPUT_DIR="${OFFLINE_DIR}/${run_id}" \
    CANDIDATE_CHECKPOINT="${candidate}" \
    PROMOTED_CHECKPOINT="${PROMOTED_CHECKPOINT}" \
    INITIAL_CHECKPOINT="${PROMOTED_CHECKPOINT}" \
    OFFLINE_OPTIMIZATION_STEPS="${OFFLINE_OPTIMIZATION_STEPS}" \
    SELF_PLAY_EPISODES="${SELF_PLAY_EPISODES}" \
    POST_SELF_PLAY_OPTIMIZATION_STEPS="${POST_SELF_PLAY_OPTIMIZATION_STEPS}" \
    MIN_LEGAL_RATE="${MIN_LEGAL_RATE}" \
    MIN_AGREEMENT_RATE="${MIN_AGREEMENT_RATE}" \
    "${REPO_ROOT}/scripts/run_offline_learning_cycle.sh"
}

printf 'Starting learned-player campaign.\n'
printf '  start_game:         %s\n' "${START_GAME}"
printf '  total_games:        %s\n' "${TOTAL_GAMES}"
printf '  batch_size:         %s\n' "${BATCH_SIZE}"
printf '  match_format:       %s\n' "${MATCH_FORMAT}"
printf '  max_plies:          %s\n' "${MAX_PLIES}"
printf '  think_time_sec:     %s\n' "${THINK_TIME_SEC}"
printf '  game_timeout_sec:   %s\n' "${GAME_TIMEOUT_SEC}"
printf '  campaign_root:      %s\n' "${CAMPAIGN_ROOT}"
printf '  promoted_checkpoint:%s\n' "${PROMOTED_CHECKPOINT}"

for game_index in $(seq "${START_GAME}" "${TOTAL_GAMES}"); do
  run_single_game "${game_index}"
  if [ $((game_index % BATCH_SIZE)) -eq 0 ]; then
    run_offline_update "${game_index}"
  fi
done

/usr/bin/python3 "${REPO_ROOT}/scripts/summarize_campaign.py" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --output-json "${CAMPAIGN_ROOT}/summary.json" \
  --output-markdown "${CAMPAIGN_ROOT}/summary.md"

printf '\nCampaign complete.\n'
printf '  logs:   %s\n' "${GAME_LOG_DIR}"
printf '  videos: %s\n' "${VIDEO_DIR}"
printf '  offline:%s\n' "${OFFLINE_DIR}"
printf '  summary:%s\n' "${CAMPAIGN_ROOT}/summary.md"

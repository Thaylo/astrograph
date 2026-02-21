#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.tripwire_scratch}"
IMAGE="${IMAGE:-astrograph:local-arm64}"
PLATFORM="${PLATFORM:-linux/arm64}"
ITERATIONS="${ITERATIONS:-50}"
CONTAINER_NAME="astrograph-stress-$$"
LOG_PATH="${LOG_PATH:-/tmp/tripwire_metrics_$$.jsonl}"

echo "tripwire stress: container=$CONTAINER_NAME log=$LOG_PATH iterations=$ITERATIONS"

uv run python scripts/tripwire.py \
  --interval 1 \
  --breach-count 3 \
  --max-ram-percent 88 \
  --max-rss-mb 3000 \
  --max-io-mbps 200 \
  --max-gpu-util 95 \
  --docker-name "${CONTAINER_NAME}" \
  --log-path "${LOG_PATH}" \
  -- bash -lc "uv run python scripts/provoke_event_driven.py --root ${ROOT} --files 500 --rate 40 --duration 60 --aggressive & \
  uv run python scripts/stress_docker_mcp.py \
    --iterations ${ITERATIONS} \
    --pause 0.5 \
    --timeout 120 \
    --image ${IMAGE} \
    --platform ${PLATFORM} \
    --container-name ${CONTAINER_NAME}; \
  wait"

echo "tripwire stress done. metrics: ${LOG_PATH}"

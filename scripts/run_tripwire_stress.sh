#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.tripwire_scratch}"
IMAGE="${IMAGE:-astrograph:local-arm64}"
PLATFORM="${PLATFORM:-linux/arm64}"

python3 scripts/tripwire.py \
  --interval 1 \
  --breach-count 3 \
  --max-ram-percent 88 \
  --max-rss-mb 3000 \
  --max-io-mbps 200 \
  --max-gpu-util 95 \
  -- bash -lc "python3 scripts/provoke_event_driven.py --root ${ROOT} --files 500 --rate 40 --duration 30 --aggressive & \
  python3 scripts/stress_docker_mcp.py --iterations 5 --pause 0.5 --timeout 120 --image ${IMAGE} --platform ${PLATFORM}; \
  wait"

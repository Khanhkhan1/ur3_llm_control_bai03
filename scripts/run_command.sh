#!/usr/bin/env bash
# Run task_runner.py with one natural-language command, e.g.:
#   ./run_command.sh "Put the red cube in zone C."
#
# Invokes python3 directly (not `ros2 run`) and sets DYLD_LIBRARY_PATH
# explicitly: on macOS/RoboStack, going through `ros2 run`'s own exec
# indirection was observed to have macOS SIP strip DYLD_LIBRARY_PATH,
# so rclpy's dlopen() of this package's rosidl typesupport .dylib (needed
# to create the /skill/* service clients) fails with "type_support is
# null" even though the library is right there in install/.../lib. No-op
# on Linux, where this isn't needed.
set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: $0 \"<natural language command>\""
  exit 1
fi

PREFIX="$(ros2 pkg prefix ur3_llm_control)"

if [ "$(uname)" = "Darwin" ] && [ -n "${CONDA_PREFIX:-}" ]; then
  export DYLD_LIBRARY_PATH="$PREFIX/lib:$CONDA_PREFIX/lib:${DYLD_LIBRARY_PATH:-}"
  PY="$CONDA_PREFIX/bin/python3"
else
  PY="python3"
fi

if [ -z "${NINEROUTER_API_KEY:-}" ]; then
  echo "Warning: NINEROUTER_API_KEY is not set; falling back to config/llm_config.yaml's llm_api_key." >&2
fi

exec "$PY" "$PREFIX/lib/ur3_llm_control/task_runner.py" --ros-args \
  --params-file "$PREFIX/share/ur3_llm_control/config/llm_config.yaml" \
  -p command:="$1"

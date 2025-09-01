#!/usr/bin/env bash
# Start the LangGraph API server for the example research graph on a different port
# Keeps existing ClickHouse infra untouched. Uses .env in this folder for LANGFUSE_* keys.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# Default port (different from 8123 which may be used by langfuse/clickhouse infra)
PORT=${1:-8124}

if [ ! -f "langgraph.json" ]; then
  echo "langgraph.json not found in $ROOT_DIR"
  exit 1
fi

echo "Starting LangGraph API server using config: $ROOT_DIR/langgraph.json on port $PORT"

# Load env (if present)
if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# If a trusted CA file exists next to this script, copy/point build tooling to it so Docker/PIP can use it.
LOCAL_CERTS=("zscaler_root.crt" "trusted_certs.crt")
CERT_IN_FOLDER=""
for c in "${LOCAL_CERTS[@]}"; do
  if [ -f "$ROOT_DIR/$c" ]; then
    CERT_IN_FOLDER="$ROOT_DIR/$c"
    break
  fi
done

if [ -n "$CERT_IN_FOLDER" ]; then
  echo "Found local CA cert: $CERT_IN_FOLDER"
  # Export common env vars so both host and some build steps see the cert path.
  export CURL_CA_BUNDLE="$CERT_IN_FOLDER"
  export REQUESTS_CA_BUNDLE="$CERT_IN_FOLDER"
  export PIP_CERT="$CERT_IN_FOLDER"
  export GRPC_DEFAULT_SSL_ROOTS_FILE_PATH="$CERT_IN_FOLDER"

  # Provide build-arg hints for Docker builds (LangGraph may pick these up during build)
  export LANGGRAPH_DOCKER_BUILD_ARGS="--build-arg REQUESTS_CA_BUNDLE=/certs/$(basename "$CERT_IN_FOLDER") --build-arg PIP_CERT=/certs/$(basename "$CERT_IN_FOLDER")"

  # Also copy the cert into a small build-context folder that can be used by Docker
  BUILD_CTX="$ROOT_DIR/.langgraph_build_ctx"
  mkdir -p "$BUILD_CTX"
  cp "$CERT_IN_FOLDER" "$BUILD_CTX/"
  echo "Copied cert into build context: $BUILD_CTX/$(basename "$CERT_IN_FOLDER")"
else
  echo "No local CA cert found in $ROOT_DIR; builds may fail behind a corporate proxy."
fi

# Run langgraph CLI. We export the LANGGRAPH_DOCKER_BUILD_ARGS variable which can be picked up by
# the LangGraph CLI/build process to pass --build-arg to docker builds if supported.
# Build a local wheel for the repository and place it in the build context so the container build
# can install the package without hitting PyPI.
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
if command -v python >/dev/null 2>&1; then
  echo "Building local wheel from repo root ($REPO_ROOT) into build context..."
  # Build wheel from the repository root so pyproject.toml is picked up.
  python -m pip wheel "$REPO_ROOT" -w "$BUILD_CTX" || echo "Wheel build failed; container build may still attempt PyPI."
  echo "Wheels in build context:"
  ls -l "$BUILD_CTX" || true
fi

# If build context contains wheels, add a build-arg so pip inside the container will look there first.
if [ -d "$BUILD_CTX" ] && [ "$(ls -A "$BUILD_CTX")" ]; then
  LANGGRAPH_DOCKER_BUILD_ARGS="${LANGGRAPH_DOCKER_BUILD_ARGS:-} --build-arg PIP_FIND_LINKS=/deps/__outer_research/.langgraph_build_ctx"
fi

echo "Starting LangGraph (port=$PORT) with build args: ${LANGGRAPH_DOCKER_BUILD_ARGS:-<none>}"
langgraph up -c langgraph.json -p "$PORT"

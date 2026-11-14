#!/usr/bin/env bash
# Starts HydraDB (local storage, small caches for a 512 MB box), waits for it, then the web app.
set -euo pipefail
ROOT=${HYDRA_HOME:-/tmp/hydradb}
mkdir -p "$ROOT/store" "$ROOT/cache"
TOKEN=${HYDRA_TOKEN:-$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')}
printf '%s\n' "$TOKEN" > "$ROOT/auth-token"
export HYDRA_TOKEN="$TOKEN"

export CLOUD_PROVIDER=local LOCAL_PATH="$ROOT/store" GRAPH_NAMESPACE=default GRAPH_ID=default \
  GRAPH_CELL_ID=cell-0 GRAPH_CELLS=cell-0 GRAPH_NODE_ID=node-0 \
  GRAPH_BOLT_NODE_ADDRESSES=node-0=127.0.0.1:7687 GRAPH_ADVERTISED_BOLT_ADDR=127.0.0.1:7687 \
  GRAPH_DATA_CACHE_DIR="$ROOT/cache" GRAPH_AUTH_TOKEN_FILE="$ROOT/auth-token" GRAPH_ALLOW_PLAINTEXT=true \
  RUST_MIN_STACK=33554432 \
  GRAPH_SLATE_DB_CACHE_BYTES=${GRAPH_SLATE_DB_CACHE_BYTES:-67108864} \
  GRAPH_DATA_CACHE_BYTES=${GRAPH_DATA_CACHE_BYTES:-268435456} \
  GRAPH_L0_SST_SIZE_BYTES=${GRAPH_L0_SST_SIZE_BYTES:-8388608} \
  GRAPH_MAX_UNFLUSHED_BYTES=${GRAPH_MAX_UNFLUSHED_BYTES:-33554432}

graph-node > "$ROOT/graph-node.log" 2>&1 &
HYDRA_PID=$!
for i in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:8443/healthz >/dev/null 2>&1; then break; fi
  if ! kill -0 "$HYDRA_PID" 2>/dev/null; then echo "graph-node exited"; tail -50 "$ROOT/graph-node.log"; exit 1; fi
  sleep 0.5
done
echo "HydraDB is up"
exec /venv/bin/uvicorn app.server:app --host 0.0.0.0 --port "${PORT:-10000}" --workers 1

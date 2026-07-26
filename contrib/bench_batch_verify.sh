#!/usr/bin/env bash
# A/B FD_BATCH_VERIFY on the full pipeline, plus optional crypto microbench.
#
# Headline for the tile change is NOT replay tps (downstream can limit).
# We scrape prometheus verify_txn_result{success} during each run — that is
# the ingest/verify rate (net→quic→verify), independent of pack/replay.
#
# Usage:
#   sudo ./contrib/bench_batch_verify.sh
#   sudo ./contrib/bench_batch_verify.sh --duration 90
#   ./contrib/bench_batch_verify.sh --micro-only          # no sudo; one binary
#   sudo ./contrib/bench_batch_verify.sh --pipeline-only
#
# Note: firedancer-dev bench has no built-in duration flag; we wrap with
# `timeout`. Prefer `sudo --preserve-env=PATH` if you need a custom PATH.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DURATION=60
CFG=src/app/firedancer/config/bench-zen5-24core.toml
JOBS="${JOBS:-$(nproc)}"
BUILD_NOBATCH=build-nobatch
BUILD_BATCH=build-batch
METRICS_URL="${METRICS_URL:-http://127.0.0.1:7999/metrics}"
RUN_PIPELINE=1
RUN_MICRO=1
EXTRA_BENCH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration|-d)   DURATION="$2"; shift 2 ;;
    --config|-c)     CFG="$2"; shift 2 ;;
    --jobs|-j)       JOBS="$2"; shift 2 ;;
    --micro-only)    RUN_PIPELINE=0; shift ;;
    --pipeline-only) RUN_MICRO=0; shift ;;
    --no-quic)       EXTRA_BENCH_ARGS+=(--no-quic); shift ;;
    --no-watch)      EXTRA_BENCH_ARGS+=(--no-watch); shift ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

log() { printf '\n==> %s\n' "$*"; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing required command: $1" >&2; exit 1; }
}

need_cmd make
need_cmd timeout
need_cmd curl
need_cmd awk

if [[ "$RUN_PIPELINE" -eq 1 && "$(id -u)" -ne 0 ]]; then
  echo "full-pipeline bench needs root (hugetlbfs / XDP). re-run with sudo, or pass --micro-only." >&2
  exit 1
fi

OUT_DIR="${OUT_DIR:-$ROOT/build/bench-batch-verify-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT_DIR"
log "logs -> $OUT_DIR"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

if [[ "$RUN_PIPELINE" -eq 1 ]]; then
  log "build baseline (FD_BATCH_VERIFY=0) -> $BUILD_NOBATCH"
  make -j"$JOBS" BUILDDIR="$BUILD_NOBATCH" firedancer-dev \
    2>&1 | tee "$OUT_DIR/build-nobatch.log"

  log "build batch (FD_BATCH_VERIFY=1) -> $BUILD_BATCH"
  make -j"$JOBS" BUILDDIR="$BUILD_BATCH" EXTRA_CPPFLAGS='-DFD_BATCH_VERIFY=1' \
    firedancer-dev \
    2>&1 | tee "$OUT_DIR/build-batch.log"
fi

# Microbench binary is independent of FD_BATCH_VERIFY — one build is enough.
if [[ "$RUN_MICRO" -eq 1 ]]; then
  log "build microbench -> $BUILD_NOBATCH (test_ed25519_x8)"
  make -j"$JOBS" BUILDDIR="$BUILD_NOBATCH" test_ed25519_x8 \
    2>&1 | tee "$OUT_DIR/build-micro.log"
fi

BIN_NOBATCH="$ROOT/build/$BUILD_NOBATCH/bin/firedancer-dev"
BIN_BATCH="$ROOT/build/$BUILD_BATCH/bin/firedancer-dev"
MICRO_BIN="$ROOT/build/$BUILD_NOBATCH/unit-test/test_ed25519_x8"

# ---------------------------------------------------------------------------
# Prometheus: sum verify successes across all verify tiles
#   verify_txn_result_total{kind="verify",...,verify_tile_result="success"}
# ---------------------------------------------------------------------------

verify_success_total() {
  curl -sf "$METRICS_URL" 2>/dev/null \
    | awk '/^verify_txn_result_total\{/ && /verify_tile_result="success"/ { s+=$NF }
           END { printf "%.0f\n", s+0 }'
}

# Sample steady-state verify TPS over the middle third of the run.
sample_verify_tps() {
  local label="$1"
  local out="$OUT_DIR/verify-tps-${label}.txt"
  local warmup=$(( DURATION / 3 ))
  local window=$(( DURATION / 3 ))
  [[ "$window" -lt 5 ]] && window=5

  # Wait for metrics endpoint.
  local ready=0
  for _ in $(seq 1 60); do
    if curl -sf "$METRICS_URL" >/dev/null 2>&1; then ready=1; break; fi
    sleep 1
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "verify_tps[${label}]=NA (metrics never came up)" | tee "$out"
    return 0
  fi

  sleep "$warmup"
  local c0 t0 c1 t1
  c0=$(verify_success_total)
  t0=$(date +%s%N)
  sleep "$window"
  c1=$(verify_success_total)
  t1=$(date +%s%N)

  # Snapshot batch-size hist BEFORE the timed bench is killed / metrics die.
  curl -sf "$METRICS_URL" 2>/dev/null \
    | grep -E '^verify_batch_(txn|sig)_cnt_total\{' \
    > "$OUT_DIR/verify-batch-hist-${label}.txt" || true

  awk -v c0="$c0" -v c1="$c1" -v t0="$t0" -v t1="$t1" \
      -v label="$label" -v warmup="$warmup" 'BEGIN {
    dt = (t1 - t0) / 1e9;
    if (dt <= 0) { printf "verify_tps[%s]=NA\n", label; exit }
    printf "verify_tps[%s]=%.0f  (delta=%.0f over %.1fs; skipped first %ds)\n",
           label, (c1 - c0) / dt, c1 - c0, dt, warmup
  }' | tee "$out"

  if [[ -s "$OUT_DIR/verify-batch-hist-${label}.txt" ]]; then
    echo "--- batch flush sizes [${label}] (summed across verify tiles) ---"
    python3 - "$OUT_DIR/verify-batch-hist-${label}.txt" <<'PY'
import re, sys
txn, sig = [0]*9, [0]*9
for line in open(sys.argv[1]):
    m = re.search(r'verify_batch_(txn|sig)_cnt_total\{[^}]*verify_batch_size="size(\d+)"[^}]*\}\s+(\d+)', line)
    if not m: continue
    kind, n, v = m.group(1), int(m.group(2)), int(m.group(3))
    (txn if kind=="txn" else sig)[n] += v
print("  txns/flush:", " ".join(f"{i}={txn[i]}" for i in range(1,9)))
print("  sigs/flush:", " ".join(f"{i}={sig[i]}" for i in range(1,9)))
PY
  fi
}

run_pipeline() {
  local label="$1"
  local bin="$2"
  local logf="$OUT_DIR/pipeline-${label}.log"

  log "pipeline bench [${label}] for ${DURATION}s (config=$CFG)"
  log "scraping ${METRICS_URL} for verify_txn_result success rate (ingest isolation)"

  sample_verify_tps "$label" &
  local sampler_pid=$!

  set +e
  timeout --signal=INT --kill-after=30s "${DURATION}s" \
    "$bin" bench --config "$CFG" "${EXTRA_BENCH_ARGS[@]}" \
    2>&1 | tee "$logf"
  local rc=${PIPESTATUS[0]}
  set -e

  wait "$sampler_pid" 2>/dev/null || true

  if [[ "$rc" -eq 124 || "$rc" -eq 0 ]]; then
    log "pipeline [${label}] finished (rc=$rc)"
    [[ -f "$OUT_DIR/verify-tps-${label}.txt" ]] && cat "$OUT_DIR/verify-tps-${label}.txt"
  else
    echo "pipeline [${label}] failed rc=$rc (see $logf)" >&2
    local fdlog
    fdlog=$(grep -oE '/tmp/fd-[^[:space:]"]+' "$logf" | head -1 || true)
    if [[ -n "${fdlog:-}" && -f "$fdlog" ]]; then
      echo "--- last ERR/CRIT from $fdlog ---" >&2
      grep -nE 'ERR|CRIT' "$fdlog" | tail -20 >&2 || true
      cp -f "$fdlog" "$OUT_DIR/fdlog-${label}.log" || true
    fi
    exit "$rc"
  fi
}

if [[ "$RUN_PIPELINE" -eq 1 ]]; then
  run_pipeline nobatch "$BIN_NOBATCH"
  sleep 5
  run_pipeline batch "$BIN_BATCH"

  log "verify ingest summary (compare these, not replay tps)"
  cat "$OUT_DIR"/verify-tps-*.txt 2>/dev/null || true

  log "batch-size histograms (from $OUT_DIR/verify-batch-hist-*.txt)"
  for f in "$OUT_DIR"/verify-batch-hist-*.txt; do
    [[ -s "$f" ]] || continue
    echo "--- $(basename "$f") ---"
    cat "$f"
  done
fi

# ---------------------------------------------------------------------------
# Microbench — one binary; FD_BATCH_VERIFY does not affect it
# ---------------------------------------------------------------------------

if [[ "$RUN_MICRO" -eq 1 ]]; then
  log "microbench ($MICRO_BIN --bench)"
  "$MICRO_BIN" --bench 2>&1 | tee "$OUT_DIR/micro.log"
fi

log "done. artifacts in $OUT_DIR"
ls -la "$OUT_DIR"

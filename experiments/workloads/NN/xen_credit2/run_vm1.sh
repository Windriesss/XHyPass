#!/bin/bash
set -u

DURATION=${DURATION:-600}
SEED=${SEED:-12345}
WORKERS=${WORKERS:-8}
MODEL_THREADS=${MODEL_THREADS:-6}
CPU_CYCLIC=${CPU_CYCLIC:-6}
WORKLOAD_CPUS=${WORKLOAD_CPUS:-0-5}
RUN_LEVELS=${RUN_LEVELS:-"light medium heavy"}
LAUNCH_DELAY=${LAUNCH_DELAY:-0}
WARMUP_SECONDS=${WARMUP_SECONDS:-0}
WARMUP_QPS=${WARMUP_QPS:-1}
SYNC_TOKEN=${SYNC_TOKEN:-}
SYNC_TIMEOUT_SECONDS=${SYNC_TIMEOUT_SECONDS:-900}
CT_INTERVAL_US=${CT_INTERVAL_US:-1000}
CT_PRIORITY=${CT_PRIORITY:-99}
HISTOGRAM_LIMIT_US=${HISTOGRAM_LIMIT_US:-10000}
START_DELAY=${START_DELAY:-2}
LIGHT_INCEPTION_QPS=${LIGHT_INCEPTION_QPS:-0.2}
MEDIUM_INCEPTION_QPS=${MEDIUM_INCEPTION_QPS:-0.2}
HEAVY_INCEPTION_QPS=${HEAVY_INCEPTION_QPS:-0.2}

BASE_DIR=${BASE_DIR:-..}
LOADGEN="${BASE_DIR}/loadgen"
BENCHMARK="${BASE_DIR}/benchmark_http_infer"
CYCLICTEST="${BASE_DIR}/cyclictest"
MODEL="${BASE_DIR}/inception_v4_299_quant.tflite"
RESULT_DIR="./results"
PID_SRV=""

sync_wait_for_peer() {
    LEVEL=$1
    [ -n "${SYNC_TOKEN}" ] || return 0
    SYNC_DIR="/tmp/xy_nn_sync_${SYNC_TOKEN}"
    mkdir -p "${SYNC_DIR}"
    : > "${SYNC_DIR}/vm1.${LEVEL}.ready"
    echo "[NN/SYNC] vm1 ready for ${LEVEL}; waiting for peer"
    DEADLINE=$(( $(date +%s) + SYNC_TIMEOUT_SECONDS ))
    while [ ! -f "${SYNC_DIR}/${LEVEL}.go" ]; do
        [ "$(date +%s)" -lt "${DEADLINE}" ] || {
            echo "[ERROR] Timed out at vm1/${LEVEL} synchronization barrier"
            return 1
        }
        sleep 0.05
    done
    date '+%s.%N' > "${RESULT_DIR}/sync_start_${LEVEL}_vm1.txt"
    echo "[NN/SYNC] vm1 released for ${LEVEL}"
}

sync_mark_finished() {
    LEVEL=$1
    [ -n "${SYNC_TOKEN}" ] || return 0
    date '+%s.%N' > "${RESULT_DIR}/sync_finish_${LEVEL}_vm1.txt"
    : > "/tmp/xy_nn_sync_${SYNC_TOKEN}/vm1.${LEVEL}.done"
}

cleanup() {
    [ -z "${PID_SRV}" ] || kill "${PID_SRV}" 2>/dev/null || true
    killall loadgen cyclictest benchmark_http_infer 2>/dev/null || true
}
trap cleanup EXIT INT TERM

killall loadgen cyclictest benchmark_http_infer 2>/dev/null || true
sleep 1
rm -rf "${RESULT_DIR}"
mkdir -p "${RESULT_DIR}"
for file in "${LOADGEN}" "${BENCHMARK}" "${CYCLICTEST}"; do
    [ -x "${file}" ] || { echo "[ERROR] Missing executable ${file}"; exit 1; }
done
[ -f "${MODEL}" ] || { echo "[ERROR] Missing model ${MODEL}"; exit 1; }
command -v taskset >/dev/null || { echo "[ERROR] Missing taskset"; exit 1; }

cat > "${RESULT_DIR}/dom0_config.txt" <<EOF
DURATION=${DURATION}
SEED=${SEED}
WORKERS=${WORKERS}
MODEL_THREADS=${MODEL_THREADS}
CPU_CYCLIC=${CPU_CYCLIC}
WORKLOAD_CPUS=${WORKLOAD_CPUS}
RUN_LEVELS=${RUN_LEVELS}
LAUNCH_DELAY=${LAUNCH_DELAY}
WARMUP_SECONDS=${WARMUP_SECONDS}
WARMUP_QPS=${WARMUP_QPS}
START_DELAY=${START_DELAY}
EOF

[ "${LAUNCH_DELAY}" -le 0 ] || {
    echo "[NN/DOM0] Delaying service launch ${LAUNCH_DELAY}s"
    sleep "${LAUNCH_DELAY}"
}
taskset -c "${WORKLOAD_CPUS}" "${BENCHMARK}" \
    --model "${MODEL}" --threads "${MODEL_THREADS}" --port 8080 \
    > "${RESULT_DIR}/inception_server.log" 2>&1 &
PID_SRV=$!
sleep 5
kill -0 "${PID_SRV}" 2>/dev/null || { echo "[ERROR] Inception server exited"; exit 1; }
sleep "${START_DELAY}"

if [ "${WARMUP_SECONDS}" -gt 0 ]; then
    echo "[NN/DOM0] Warming up inference for ${WARMUP_SECONDS}s"
    WARMUP_FILE="/tmp/xy_nn_warmup_vm1.csv"
    rm -f "${WARMUP_FILE}"
    nice -n -20 taskset -c "${WORKLOAD_CPUS}" "${LOADGEN}" \
        --url http://127.0.0.1:8080/infer --duration "${WARMUP_SECONDS}" \
        --seed "${SEED}" --workers 1 --constant-qps "${WARMUP_QPS}" \
        --out "${WARMUP_FILE}" || { echo "[ERROR] dom0 warm-up failed"; exit 1; }
    awk -F, 'NR > 1 && $3 < 0 { bad++ } END { exit(bad > 0) }' \
        "${WARMUP_FILE}" || { echo "[ERROR] dom0 warm-up inference failed"; exit 1; }
    echo "[NN/DOM0] Warm-up completed"
fi

run_case() {
    LEVEL=$1
    QPS=$2
    sync_wait_for_peer "${LEVEL}" || return $?
    echo "[NN/DOM0] Starting ${LEVEL}"
    taskset -c "${CPU_CYCLIC}" "${CYCLICTEST}" -t1 -p "${CT_PRIORITY}" -m \
        -i "${CT_INTERVAL_US}" -D "${DURATION}s" -h "${HISTOGRAM_LIMIT_US}" \
        --histfile="${RESULT_DIR}/cyclictest_${LEVEL}_vcpu${CPU_CYCLIC}_vm1.txt" -q &
    PID_CT=$!
    nice -n -20 taskset -c "${WORKLOAD_CPUS}" "${LOADGEN}" \
        --url http://127.0.0.1:8080/infer \
        --duration "${DURATION}" --seed "${SEED}" --workers "${WORKERS}" \
        --constant-qps "${QPS}" --out "${RESULT_DIR}/inception_${LEVEL}.csv" &
    PID_LOAD=$!
    RC=0
    wait "${PID_LOAD}" || RC=$?
    wait "${PID_CT}" || RC=$?
    sync_mark_finished "${LEVEL}"
    [ "${RC}" -eq 0 ] || return "${RC}"
    awk -F, 'NR > 1 && $3 < 0 { bad++ } END { exit(bad > 0) }' \
        "${RESULT_DIR}/inception_${LEVEL}.csv" || return 1
    echo "[NN/DOM0] Finished ${LEVEL}"
}

for LEVEL in ${RUN_LEVELS}; do
    case "${LEVEL}" in
        light)  run_case light "${LIGHT_INCEPTION_QPS}" || exit $? ;;
        medium) run_case medium "${MEDIUM_INCEPTION_QPS}" || exit $? ;;
        heavy)  run_case heavy "${HEAVY_INCEPTION_QPS}" || exit $? ;;
        *) echo "[ERROR] Unknown run level: ${LEVEL}"; exit 2 ;;
    esac
done
printf '0\n' > "${RESULT_DIR}/run_complete_vm1"
echo "[NN/DOM0] All cases completed"

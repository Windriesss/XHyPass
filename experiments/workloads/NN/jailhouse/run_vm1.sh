#!/bin/bash
set -u

DURATION=${DURATION:-600}
SEED=${SEED:-12345}
WORKERS=${WORKERS:-8}
MODEL_THREADS=${MODEL_THREADS:-3}
CPU_CYCLIC=${CPU_CYCLIC:-6}
CT_INTERVAL_US=${CT_INTERVAL_US:-1000}
CT_PRIORITY=${CT_PRIORITY:-99}
HISTOGRAM_LIMIT_US=${HISTOGRAM_LIMIT_US:-10000}
START_DELAY=${START_DELAY:-2}
LIGHT_INCEPTION_QPS=${LIGHT_INCEPTION_QPS:-0.2}
MEDIUM_INCEPTION_QPS=${MEDIUM_INCEPTION_QPS:-0.2}
HEAVY_INCEPTION_QPS=${HEAVY_INCEPTION_QPS:-0.2}

BASE_DIR=".."
LOADGEN="${BASE_DIR}/loadgen"
BENCHMARK="${BASE_DIR}/benchmark_http_infer"
CYCLICTEST="${BASE_DIR}/cyclictest"
MODEL="${BASE_DIR}/inception_v4_299_quant.tflite"
RESULT_DIR="./results"
PID_SRV=""

cleanup() {
    [ -z "${PID_SRV}" ] || kill "${PID_SRV}" 2>/dev/null || true
    killall loadgen cyclictest benchmark_http_infer 2>/dev/null || true
}
trap cleanup EXIT INT TERM

killall loadgen cyclictest benchmark_http_infer 2>/dev/null || true
sleep 1
rm -rf "${RESULT_DIR}"
mkdir -p "${RESULT_DIR}"
for file in "${LOADGEN}" "${BENCHMARK}" "${CYCLICTEST}" "${MODEL}"; do
    [ -e "${file}" ] || { echo "[ERROR] Missing ${file}"; exit 1; }
done

cat > "${RESULT_DIR}/rootcell_config.txt" <<EOF
DURATION=${DURATION}
SEED=${SEED}
WORKERS=${WORKERS}
MODEL_THREADS=${MODEL_THREADS}
CPU_CYCLIC=${CPU_CYCLIC}
START_DELAY=${START_DELAY}
EOF

"${BENCHMARK}" --model "${MODEL}" --threads "${MODEL_THREADS}" --port 8080 \
    > "${RESULT_DIR}/inception_server.log" 2>&1 &
PID_SRV=$!
sleep 5
kill -0 "${PID_SRV}" 2>/dev/null || { echo "[ERROR] Inception server exited"; exit 1; }
sleep "${START_DELAY}"

run_case() {
    LEVEL=$1
    QPS=$2
    echo "[NN/ROOTCELL] Starting ${LEVEL}"
    taskset -c "${CPU_CYCLIC}" "${CYCLICTEST}" -t1 -p "${CT_PRIORITY}" -m \
        -i "${CT_INTERVAL_US}" -D "${DURATION}s" -h "${HISTOGRAM_LIMIT_US}" \
        --histfile="${RESULT_DIR}/cyclictest_${LEVEL}_cpu${CPU_CYCLIC}.txt" -q &
    PID_CT=$!
    nice -n -20 "${LOADGEN}" --url http://127.0.0.1:8080/infer \
        --duration "${DURATION}" --seed "${SEED}" --workers "${WORKERS}" \
        --constant-qps "${QPS}" --out "${RESULT_DIR}/inception_${LEVEL}.csv" &
    PID_LOAD=$!
    RC=0
    wait "${PID_LOAD}" || RC=$?
    wait "${PID_CT}" || RC=$?
    [ "${RC}" -eq 0 ] || return "${RC}"
    awk -F, 'NR > 1 && $3 < 0 { bad++ } END { exit(bad > 0) }' \
        "${RESULT_DIR}/inception_${LEVEL}.csv" || return 1
    echo "[NN/ROOTCELL] Finished ${LEVEL}"
}

run_case light "${LIGHT_INCEPTION_QPS}" || exit $?
run_case medium "${MEDIUM_INCEPTION_QPS}" || exit $?
run_case heavy "${HEAVY_INCEPTION_QPS}" || exit $?
echo "[NN/ROOTCELL] All cases completed"

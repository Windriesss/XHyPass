#!/bin/bash
set -e

# -------------------------------------------------
# Pre-clean
# -------------------------------------------------
echo "[INFO] Cleaning up existing benchmark_http_infer/loadgen..."
killall benchmark_http_infer 2>/dev/null || true
killall loadgen 2>/dev/null || true
sleep 1

# -------------------------------------------------
# Global configuration
# -------------------------------------------------
DURATION=600          # seconds, 可改为30做快速验证
SEED=12345
WORKERS=8
CPU_CYCLIC=6

BASE_DIR=".."
LOADGEN="${BASE_DIR}/loadgen"
BENCHMARK="${BASE_DIR}/benchmark_http_infer"
CYCLICTEST="${BASE_DIR}/cyclictest"
MODEL="${BASE_DIR}/inception_v4_299_quant.tflite"
PORT=8080

RESULT_DIR="./results"
rm -r "${RESULT_DIR}"
mkdir -p "${RESULT_DIR}"

# -------------------------------------------------
# Sanity checks
# -------------------------------------------------
for f in "${LOADGEN}" "${BENCHMARK}" "${CYCLICTEST}" "${MODEL}"; do
    [[ -e "$f" ]] || { echo "[ERROR] Missing $f"; exit 1; }
done

echo "=========================================="
echo " VM-A: Inception Test"
echo " Duration: ${DURATION}s"
echo "=========================================="

# -------------------------------------------------
# Start server (常驻)
# -------------------------------------------------
${BENCHMARK} \
    --model ${MODEL} \
    --threads 6 \
    --port ${PORT} \
    > ${RESULT_DIR}/inception_server.log 2>&1 &

PID_SRV=$!
sleep 5

# -------------------------------------------------
# Run loadgen cases
# -------------------------------------------------
run_case() {
    LEVEL=$1
    QPS=$2

    echo ""
    echo "------------------------------------------"
    echo "Running load case: ${LEVEL}"
    echo "------------------------------------------"

    # 启动 cyclictest
    taskset -c ${CPU_CYCLIC} ${CYCLICTEST} \
        -t1 -p 99 -m -i 1000 \
        -D ${DURATION}s \
        -h 10000 \
        --histfile="${RESULT_DIR}/cyclictest_${LEVEL}_cpu${CPU_CYCLIC}_vm1.txt" \
        -q &

    PID_CT=$!
    sleep 2

    # 启动 loadgen
    nice -n -20 ${LOADGEN} \
        --url http://127.0.0.1:${PORT}/infer \
        --duration ${DURATION} \
        --seed ${SEED} \
        --workers ${WORKERS} \
        --constant-qps ${QPS} \
        --out "${RESULT_DIR}/inception_${LEVEL}.csv" &

    PID_LOAD=$!

    # 等待 loadgen 和 cyclictest 完成
    wait ${PID_LOAD}
    wait ${PID_CT}

    echo "[INFO] Load case ${LEVEL} finished."
}

run_case light 0.2
run_case medium 0.2
run_case heavy 0.2

# -------------------------------------------------
# Cleanup server
# -------------------------------------------------
kill ${PID_SRV}
wait ${PID_SRV} 2>/dev/null || true

echo "=========================================="
echo " VM-A finished."
echo "=========================================="


#!/bin/bash
set -e

# -------------------------------------------------
# Pre-clean
# -------------------------------------------------
killall benchmark_http_infer 2>/dev/null || true
killall loadgen 2>/dev/null || true
sleep 1

# -------------------------------------------------
# Global configuration
# -------------------------------------------------
DURATION=600          # seconds
SEED=12345
WORKERS=8
CPU_CYCLIC=6

BASE_DIR=".."
LOADGEN="${BASE_DIR}/loadgen"
BENCHMARK="${BASE_DIR}/benchmark_http_infer"
CYCLICTEST="${BASE_DIR}/cyclictest"
MODEL="${BASE_DIR}/mnasnet_1.3_224.tflite"
PORT=8081

RESULT_DIR="./results"
rm -r "${RESULT_DIR}"
mkdir -p "${RESULT_DIR}"

# -------------------------------------------------
# Sanity checks
# -------------------------------------------------
for f in "${LOADGEN}" "${BENCHMARK}" "${CYCLICTEST}" "${MODEL}"; do
    [[ -e "$f" ]] || { echo "[ERROR] Missing $f"; exit 1; }
done


# -------------------------------------------------
# Start server (常驻)
# -------------------------------------------------
${BENCHMARK} \
    --model ${MODEL} \
    --threads 3 \
    --port ${PORT} \
    > ${RESULT_DIR}/mnasnet_server.log 2>&1 &

PID_SRV=$!
sleep 5

# -------------------------------------------------
# Run loadgen cases
# -------------------------------------------------
run_case() {
    LEVEL=$1
    PEAK_QPS=$2
    POISSON_QPS=$3
    CONST_QPS=$4



    # 启动 loadgen
    nice -n -20 ${LOADGEN} \
        --url http://127.0.0.1:${PORT}/infer \
        --duration ${DURATION} \
        --period-sec 60 \
        --peak-qps ${PEAK_QPS} \
        --qps ${POISSON_QPS} \
        --constant-qps ${CONST_QPS} \
        --seed ${SEED} \
        --workers ${WORKERS} \
        --out "${RESULT_DIR}/mnasnet_${LEVEL}.csv" &

    PID_LOAD=$!
    sleep 2 

    # 启动 cyclictest
    taskset -c ${CPU_CYCLIC} ${CYCLICTEST} \
        -t1 -p 99 -m -i 1000 \
        -D ${DURATION}s \
        -h 10000 \
        --histfile="${RESULT_DIR}/cyclictest_${LEVEL}_cpu${CPU_CYCLIC}_vm2.txt" \
        -q &

    PID_CT=$!
    sleep 2
    # 等待 loadgen 和 cyclictest 完成
    wait ${PID_LOAD}
    wait ${PID_CT}

}

run_case light 5 3 3
run_case medium 12 6 6
run_case heavy 20 9 9

# -------------------------------------------------
# Cleanup server
# -------------------------------------------------
kill ${PID_SRV}
wait ${PID_SRV} 2>/dev/null || true



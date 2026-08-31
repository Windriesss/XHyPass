#!/bin/bash
set -e


DURATION=30
SEED=12345
WORKERS=8

INCEPTION_URL="http://127.0.0.1:8080/infer"
MNASNET_URL="http://127.0.0.1:8081/infer"

RESULT_DIR="./results"
mkdir -p "${RESULT_DIR}"

echo "=========================================="
echo " Dual HTTP Service Load Test"
echo " Results will be saved to ${RESULT_DIR}"
echo "=========================================="

run_case() {
    LEVEL=$1
    INCEPTION_QPS=$2
    PEAK_QPS=$3
    POISSON_QPS=$4
    CONST_QPS=$5

    echo ""
    echo "------------------------------------------"
    echo "Running case: ${LEVEL}"
    echo "------------------------------------------"

    # Inception: fixed QPS
    nice -n -20 ./loadgen \
        --url ${INCEPTION_URL} \
        --duration ${DURATION} \
        --seed ${SEED} \
        --workers ${WORKERS} \
        --constant-qps ${INCEPTION_QPS} \
        --out "${RESULT_DIR}/inception_${LEVEL}.csv" &

    PID_INCEPTION=$!

    # MnasNet: fluctuating load
    nice -n -20 ./loadgen \
        --url ${MNASNET_URL} \
        --duration ${DURATION} \
        --period-sec 30 \
        --peak-qps ${PEAK_QPS} \
        --qps ${POISSON_QPS} \
        --constant-qps ${CONST_QPS} \
        --seed ${SEED} \
        --workers ${WORKERS} \
        --out "${RESULT_DIR}/mnasnet_${LEVEL}.csv" &

    PID_MNASNET=$!

    # Wait for both to finish
    wait ${PID_INCEPTION}
    wait ${PID_MNASNET}

    echo "Case ${LEVEL} finished."
}

# -------------------------------
# Run all cases
# -------------------------------
# Inception QPS: light=0.4, medium=0.8, heavy=1.2
run_case light   0.3  5   3   3
run_case medium  0.6 12   6   6
run_case heavy   0.9 20   9   9

echo ""
echo "=========================================="
echo " All experiments completed successfully."
echo " Results are saved in ${RESULT_DIR}"
echo "=========================================="


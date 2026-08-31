#!/bin/bash
set -u

# Parameters are supplied by run_nn_bare.py. Defaults also allow manual use.
DURATION=${DURATION:-600}
SEED=${SEED:-12345}
WORKERS=${WORKERS:-8}
INCEPTION_THREAD=${INCEPTION_THREAD:-6}
MNAS_THREAD=${MNAS_THREAD:-6}
CT_INTERVAL_US=${CT_INTERVAL_US:-1000}
CT_PRIORITY=${CT_PRIORITY:-99}
HISTOGRAM_LIMIT_US=${HISTOGRAM_LIMIT_US:-10000}
PERIOD_SEC=${PERIOD_SEC:-60}
WORKLOAD_CPUS=${WORKLOAD_CPUS:-0-5}
CGROUP_WEIGHT=${CGROUP_WEIGHT:-100}
CGROUP_SHARES=${CGROUP_SHARES:-1024}

LIGHT_INCEPTION_QPS=${LIGHT_INCEPTION_QPS:-0.2}
LIGHT_PEAK_QPS=${LIGHT_PEAK_QPS:-5}
LIGHT_POISSON_QPS=${LIGHT_POISSON_QPS:-3}
LIGHT_CONSTANT_QPS=${LIGHT_CONSTANT_QPS:-3}
MEDIUM_INCEPTION_QPS=${MEDIUM_INCEPTION_QPS:-0.2}
MEDIUM_PEAK_QPS=${MEDIUM_PEAK_QPS:-12}
MEDIUM_POISSON_QPS=${MEDIUM_POISSON_QPS:-6}
MEDIUM_CONSTANT_QPS=${MEDIUM_CONSTANT_QPS:-6}
HEAVY_INCEPTION_QPS=${HEAVY_INCEPTION_QPS:-0.2}
HEAVY_PEAK_QPS=${HEAVY_PEAK_QPS:-20}
HEAVY_POISSON_QPS=${HEAVY_POISSON_QPS:-9}
HEAVY_CONSTANT_QPS=${HEAVY_CONSTANT_QPS:-9}

BASE_DIR=".."
LOADGEN="${BASE_DIR}/loadgen"
BENCHMARK="${BASE_DIR}/benchmark_http_infer"
CYCLICTEST="${BASE_DIR}/cyclictest"
INCEPTION_MODEL="${BASE_DIR}/inception_v4_299_quant.tflite"
MNASNET_MODEL="${BASE_DIR}/mnasnet_1.3_224.tflite"
RESULT_DIR="./results"

PID_INCEPTION_SRV=""
PID_MNASNET_SRV=""
CGROUP_KIND=""
CGROUP_INCEPTION=""
CGROUP_MNASNET=""

setup_cpu_cgroups() {
    SUFFIX="xy_nn_bare_$$"
    if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
        CGROUP_KIND="cgroup-v2"
        grep -qw cpu /sys/fs/cgroup/cgroup.controllers || {
            echo "[ERROR] cgroup v2 CPU controller is unavailable"
            return 1
        }
        if ! grep -qw cpu /sys/fs/cgroup/cgroup.subtree_control; then
            echo +cpu > /sys/fs/cgroup/cgroup.subtree_control || {
                echo "[ERROR] Could not enable the cgroup v2 CPU controller"
                return 1
            }
        fi
        CGROUP_INCEPTION="/sys/fs/cgroup/${SUFFIX}_inception"
        CGROUP_MNASNET="/sys/fs/cgroup/${SUFFIX}_mnasnet"
        mkdir "${CGROUP_INCEPTION}" "${CGROUP_MNASNET}" || return 1
        echo "${CGROUP_WEIGHT}" > "${CGROUP_INCEPTION}/cpu.weight" || return 1
        echo "${CGROUP_WEIGHT}" > "${CGROUP_MNASNET}/cpu.weight" || return 1
    else
        CPU_ROOT=""
        for candidate in /sys/fs/cgroup/cpu /sys/fs/cgroup/cpu,cpuacct \
                         /sys/fs/cgroup/cpuacct,cpu; do
            if [ -f "${candidate}/cpu.shares" ]; then
                CPU_ROOT="${candidate}"
                break
            fi
        done
        [ -n "${CPU_ROOT}" ] || {
            echo "[ERROR] cgroup v1 CPU controller is unavailable"
            return 1
        }
        CGROUP_KIND="cgroup-v1"
        CGROUP_INCEPTION="${CPU_ROOT}/${SUFFIX}_inception"
        CGROUP_MNASNET="${CPU_ROOT}/${SUFFIX}_mnasnet"
        mkdir "${CGROUP_INCEPTION}" "${CGROUP_MNASNET}" || return 1
        echo "${CGROUP_SHARES}" > "${CGROUP_INCEPTION}/cpu.shares" || return 1
        echo "${CGROUP_SHARES}" > "${CGROUP_MNASNET}/cpu.shares" || return 1
    fi
    echo "[NN/CGROUP] ${CGROUP_KIND}: equal-weight groups are ready"
}

move_process_to_cgroup() {
    PID=$1
    GROUP=$2
    kill -0 "${PID}" 2>/dev/null || {
        echo "[ERROR] Cannot move exited process ${PID} to ${GROUP}"
        return 1
    }
    if [ -f "${GROUP}/cgroup.procs" ]; then
        echo "${PID}" > "${GROUP}/cgroup.procs"
    else
        for task in /proc/"${PID}"/task/*; do
            [ -e "${task}" ] || continue
            echo "${task##*/}" > "${GROUP}/tasks" || return 1
        done
    fi
}

snapshot_cpu_accounting() {
    LABEL=$1
    FILE=$2
    {
        echo "[${LABEL}]"
        echo "inception=${CGROUP_INCEPTION}"
        if [ -f "${CGROUP_INCEPTION}/cpu.stat" ]; then
            sed 's/^/inception /' "${CGROUP_INCEPTION}/cpu.stat"
            sed 's/^/mnasnet /' "${CGROUP_MNASNET}/cpu.stat"
        elif [ -f "${CGROUP_INCEPTION}/cpuacct.usage" ]; then
            echo "inception cpuacct.usage=$(cat "${CGROUP_INCEPTION}/cpuacct.usage")"
            echo "mnasnet cpuacct.usage=$(cat "${CGROUP_MNASNET}/cpuacct.usage")"
        fi
    } >> "${FILE}"
}

cleanup_cpu_cgroups() {
    [ -z "${CGROUP_INCEPTION}" ] || rmdir "${CGROUP_INCEPTION}" 2>/dev/null || true
    [ -z "${CGROUP_MNASNET}" ] || rmdir "${CGROUP_MNASNET}" 2>/dev/null || true
}

cleanup() {
    [ -z "${PID_INCEPTION_SRV}" ] || kill "${PID_INCEPTION_SRV}" 2>/dev/null || true
    [ -z "${PID_MNASNET_SRV}" ] || kill "${PID_MNASNET_SRV}" 2>/dev/null || true
    killall loadgen cyclictest benchmark_http_infer 2>/dev/null || true
    sleep 1
    cleanup_cpu_cgroups
}
trap cleanup EXIT INT TERM

killall loadgen cyclictest benchmark_http_infer 2>/dev/null || true
sleep 1
rm -rf "${RESULT_DIR}"
mkdir -p "${RESULT_DIR}"

for file in "${LOADGEN}" "${BENCHMARK}" "${CYCLICTEST}" \
            "${INCEPTION_MODEL}" "${MNASNET_MODEL}"; do
    [ -e "${file}" ] || { echo "[ERROR] Missing ${file}"; exit 1; }
done
command -v taskset >/dev/null || { echo "[ERROR] Missing taskset"; exit 1; }
setup_cpu_cgroups || { echo "[ERROR] Equal-weight CPU cgroup setup failed"; exit 1; }

cat > "${RESULT_DIR}/experiment_config.txt" <<EOF
DURATION=${DURATION}
SEED=${SEED}
WORKERS=${WORKERS}
INCEPTION_THREAD=${INCEPTION_THREAD}
MNAS_THREAD=${MNAS_THREAD}
CT_INTERVAL_US=${CT_INTERVAL_US}
PERIOD_SEC=${PERIOD_SEC}
WORKLOAD_CPUS=${WORKLOAD_CPUS}
CPU_FAIRNESS=equal-cgroup-weight
CGROUP_KIND=${CGROUP_KIND}
CGROUP_WEIGHT=${CGROUP_WEIGHT}
CGROUP_SHARES=${CGROUP_SHARES}
EOF

taskset -c "${WORKLOAD_CPUS}" "${BENCHMARK}" \
    --model "${INCEPTION_MODEL}" --threads "${INCEPTION_THREAD}" \
    --port 8080 > "${RESULT_DIR}/inception_server.log" 2>&1 &
PID_INCEPTION_SRV=$!
move_process_to_cgroup "${PID_INCEPTION_SRV}" "${CGROUP_INCEPTION}" || exit 1
taskset -c "${WORKLOAD_CPUS}" "${BENCHMARK}" \
    --model "${MNASNET_MODEL}" --threads "${MNAS_THREAD}" \
    --port 8081 > "${RESULT_DIR}/mnasnet_server.log" 2>&1 &
PID_MNASNET_SRV=$!
move_process_to_cgroup "${PID_MNASNET_SRV}" "${CGROUP_MNASNET}" || exit 1
sleep 5
kill -0 "${PID_INCEPTION_SRV}" 2>/dev/null || { echo "[ERROR] Inception server exited"; exit 1; }
kill -0 "${PID_MNASNET_SRV}" 2>/dev/null || { echo "[ERROR] MnasNet server exited"; exit 1; }

run_case() {
    LEVEL=$1
    INCEPTION_QPS=$2
    PEAK_QPS=$3
    POISSON_QPS=$4
    CONST_QPS=$5
    echo "[NN] Starting ${LEVEL}: duration=${DURATION}s"
    ACCOUNTING_FILE="${RESULT_DIR}/cgroup_cpu_${LEVEL}.txt"
    : > "${ACCOUNTING_FILE}"
    snapshot_cpu_accounting "before" "${ACCOUNTING_FILE}"

    taskset -c 6 "${CYCLICTEST}" -t1 -p "${CT_PRIORITY}" -m \
        -i "${CT_INTERVAL_US}" -D "${DURATION}s" -h "${HISTOGRAM_LIMIT_US}" \
        --histfile="${RESULT_DIR}/cyclictest_${LEVEL}_cpu6.txt" -q &
    PID_CT1=$!
    taskset -c 7 "${CYCLICTEST}" -t1 -p "${CT_PRIORITY}" -m \
        -i "${CT_INTERVAL_US}" -D "${DURATION}s" -h "${HISTOGRAM_LIMIT_US}" \
        --histfile="${RESULT_DIR}/cyclictest_${LEVEL}_cpu7.txt" -q &
    PID_CT2=$!

    nice -n -20 taskset -c "${WORKLOAD_CPUS}" "${LOADGEN}" \
        --url http://127.0.0.1:8080/infer \
        --duration "${DURATION}" --seed "${SEED}" --workers "${WORKERS}" \
        --constant-qps "${INCEPTION_QPS}" \
        --out "${RESULT_DIR}/inception_${LEVEL}.csv" &
    PID_INCEPTION_LOAD=$!
    move_process_to_cgroup "${PID_INCEPTION_LOAD}" "${CGROUP_INCEPTION}" || return 1
    nice -n -20 taskset -c "${WORKLOAD_CPUS}" "${LOADGEN}" \
        --url http://127.0.0.1:8081/infer \
        --duration "${DURATION}" --period-sec "${PERIOD_SEC}" \
        --peak-qps "${PEAK_QPS}" --qps "${POISSON_QPS}" \
        --constant-qps "${CONST_QPS}" --seed "${SEED}" --workers "${WORKERS}" \
        --out "${RESULT_DIR}/mnasnet_${LEVEL}.csv" &
    PID_MNASNET_LOAD=$!
    move_process_to_cgroup "${PID_MNASNET_LOAD}" "${CGROUP_MNASNET}" || return 1

    RC=0
    wait "${PID_INCEPTION_LOAD}" || RC=$?
    wait "${PID_MNASNET_LOAD}" || RC=$?
    wait "${PID_CT1}" || RC=$?
    wait "${PID_CT2}" || RC=$?
    snapshot_cpu_accounting "after" "${ACCOUNTING_FILE}"
    [ "${RC}" -eq 0 ] || { echo "[ERROR] ${LEVEL} failed: rc=${RC}"; return "${RC}"; }

    for csv in "${RESULT_DIR}/inception_${LEVEL}.csv" "${RESULT_DIR}/mnasnet_${LEVEL}.csv"; do
        [ -s "${csv}" ] || { echo "[ERROR] Missing result ${csv}"; return 1; }
        awk -F, 'NR > 1 && $3 < 0 { bad++ } END { exit(bad > 0) }' "${csv}" || {
            echo "[ERROR] Failed HTTP inference recorded in ${csv}"; return 1;
        }
    done
    echo "[NN] Finished ${LEVEL}"
}

run_case light  "${LIGHT_INCEPTION_QPS}"  "${LIGHT_PEAK_QPS}"  "${LIGHT_POISSON_QPS}"  "${LIGHT_CONSTANT_QPS}" || exit $?
run_case medium "${MEDIUM_INCEPTION_QPS}" "${MEDIUM_PEAK_QPS}" "${MEDIUM_POISSON_QPS}" "${MEDIUM_CONSTANT_QPS}" || exit $?
run_case heavy  "${HEAVY_INCEPTION_QPS}"  "${HEAVY_PEAK_QPS}"  "${HEAVY_POISSON_QPS}"  "${HEAVY_CONSTANT_QPS}" || exit $?

echo "[NN] All cases completed"

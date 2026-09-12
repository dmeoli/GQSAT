#!/usr/bin/env bash
# Re-evaluate the trained checkpoints, logging BOTH MiniSat baselines per problem
# (see gqsat/utils.py): the resulting .tsv can then be read against the run with
# restarts or against the one without, with no further evaluation.
#
# The logs go to runs/<run>/reeval/, next to the 2021 ones, which are left alone.
# Already written files are skipped, so the script can be stopped and resumed.
#
# Locally:  bash reeval.sh tables
# On Colab: OUT_ROOT=/content/gdrive/MyDrive/neuroSAT_ckpts/reeval DEVICE_FLAG= \
#               MIN_FREE_MB=0 THREADS=4 bash reeval.sh tables
#
# Usage: bash reeval.sh <phase>
#   tables    : what the tables and the bar figures need, i.e. cap 500 everywhere
#               plus the two further caps of the budget table
#   colouring : the colouring-trained runs on the flat families, every cap
#   transfer  : the random-trained runs on the flat families, cap 500
#   random    : the random-trained runs on the random families, every cap
set -u

PY="${PY:-../.venv/bin/python}"
# OUT_ROOT: where the logs go; empty means runs/<run>/reeval, which is what one
# wants locally, while on Colab it should point into the mounted Drive, the
# runtime being thrown away at the end of the session.
OUT_ROOT="${OUT_ROOT:-}"
# DEVICE_FLAG: --no-cuda on a machine without a GPU, empty to use the GPU.
DEVICE_FLAG="${DEVICE_FLAG:---no-cuda}"
# one thread and the lowest priority: the machine is shared, and an evaluation
# that is killed halfway costs more than one that takes longer
export OMP_NUM_THREADS=${THREADS:-2} MKL_NUM_THREADS=${THREADS:-2} \
       OPENBLAS_NUM_THREADS=${THREADS:-2} TORCH_NUM_THREADS=${THREADS:-2}
FLAT="flat30-60 flat50-115 flat75-180 flat100-239 flat125-301 flat150-360 flat175-417 flat200-479"
RAND="uf50-218 uf100-430 uf250-1065 uuf50-218 uuf100-430 uuf250-1065"
CAPS="10 50 100 300 500 1000"
COLOURING_RUNS="Dec08_08-39-57_e63e47f25457 Dec09_12-16-16_d4e65e7af705"
RANDOM_RUNS="Dec21_01-59-59_6bed2aa9b612 Dec21_14-55-50_5eccdc34d583 \
             Dec23_01-48-54_90582559eea7 Dec23_14-42-44_90582559eea7 \
             Nov12_14-06-54_c42e8ad320d8 Nov12_20-35-32_c42e8ad320d8 \
             Nov13_03-55-51_c42e8ad320d8 Nov14_03-26-36_54337a27a809"

data_path() {  # dataset -> directory under ../data
    case "$1" in
        flat50-115) echo "../data/graph-coloring/test/flat50-115" ;;
        flat*)      echo "../data/graph-coloring/$1" ;;
        u*250-1065) echo "../data/uniform-random-3-sat/$1" ;;
        u*)         echo "../data/uniform-random-3-sat/test/$1" ;;
    esac
}

model_of() {  # run directory -> graphqsat | gatqsat
    if grep -aq "use_attention: true" "runs/$1/model.yaml"; then echo gatqsat; else echo graphqsat; fi
}

wait_for_memory() {  # the machine is shared: do not start if it is already tight
    local need="${MIN_FREE_MB:-2500}" avail
    [ "$need" -eq 0 ] && return 0
    while :; do
        avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
        [ "$avail" -ge "$need" ] && return 0
        echo "[$(date +%H:%M:%S)] only ${avail} MB available, waiting"
        sleep 60
    done
}

one() {  # run dataset cap
    local run="$1" ds="$2" cap="$3"
    local model out
    model="$(model_of "$run")"
    local dir="runs/$run/reeval"
    [ -n "$OUT_ROOT" ] && dir="$OUT_ROOT/$run"
    out="$dir/$ds-$model-max$cap.tsv"
    [ -s "$out" ] && return 0
    mkdir -p "$dir"
    wait_for_memory
    echo "[$(date +%H:%M:%S)] $run $ds cap $cap"
    nice -n 10 "$PY" evaluate.py \
        --env-name sat-v0 --core-steps -1 --eps-final 0.0 --no_restarts $DEVICE_FLAG \
        --test_time_max_decisions_allowed "$cap" \
        --eval-problems-paths "$(data_path "$ds")" \
        --model-dir "runs/$run" --model-checkpoint model_50000.chkp \
        2>/dev/null | grep -E "^(sec to solve|[0-9])" > "$out.part" \
        && mv "$out.part" "$out" || rm -f "$out.part"
}

case "${1:?usage: reeval.sh <tables|colouring|transfer|random>}" in
    tables)
        for r in $COLOURING_RUNS; do for d in $FLAT; do one "$r" "$d" 500; done; done
        for r in $COLOURING_RUNS; do for c in 50 1000; do one "$r" flat200-479 "$c"; done; done
        for r in Dec21_01-59-59_6bed2aa9b612 Dec23_01-48-54_90582559eea7 \
                 Nov12_14-06-54_c42e8ad320d8 Nov13_03-55-51_c42e8ad320d8; do
            for d in $FLAT; do one "$r" "$d" 500; done
        done ;;
    colouring) for r in $COLOURING_RUNS; do for d in $FLAT; do for c in $CAPS; do one "$r" "$d" "$c"; done; done; done ;;
    transfer)  for r in $RANDOM_RUNS;    do for d in $FLAT; do one "$r" "$d" 500; done; done ;;
    random)    for r in $RANDOM_RUNS;    do for d in $RAND; do for c in $CAPS; do one "$r" "$d" "$c"; done; done; done ;;
    *) echo "unknown phase: $1" >&2; exit 1 ;;
esac
echo "done."

#!/usr/bin/bash

MODEL_PATH="/dev/shm/Qwen3-30B-A3B/"
CONCURRENCIES="1 4 8 16 32 64 128"

# === Multi GPU ===
# Parallel plan: dp4-ep
bash bench_e2e_offload.sh -m ${MODEL_PATH} --concurrencies "${CONCURRENCIES}" -dp 4 -tp 1 -ep -pc -cpu-gbs "0" -cd "0,1,2,3" -p 8000

# Parallel plan: tp4-ep
# bash bench_e2e_offload.sh -m ${MODEL_PATH} --concurrencies "${CONCURRENCIES}" -dp 1 -tp 4 -ep -pc -cpu-gbs "2 4 8 16" -cd "0,1,2,3" -p 8000

# Parallel plan: tp4
# bash bench_e2e_offload.sh -m ${MODEL_PATH} --concurrencies "${CONCURRENCIES}" -dp 1 -tp 4 -pc -cpu-gbs "0" -cd "0,1,2,3" -p 8000

# === Single GPU ===
# bash bench_e2e_offload.sh -m ${MODEL_PATH} --concurrencies "${CONCURRENCIES}" -dp 1 -tp 4 -pc -cpu-gbs "30" -cd "0" -p 8000

# === Test ===
# bash bench_e2e_offload.sh -m ${MODEL_PATH} --concurrencies "${CONCURRENCIES}" -dp 1 -tp 2 -pc -cpu-gbs "0" -cd "2,3" -p 8000


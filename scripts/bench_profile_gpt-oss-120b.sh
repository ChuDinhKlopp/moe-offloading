#!/usr/bin/bash
MODEL_PATH="/dev/shm/gpt-oss-120b/"
CONCURRENCIES="1 4 8 16 32 64 128"
# CONCURRENCIES="1"

# === Multi GPU ===
# Parallel plan: dp4-ep
bash bench_profile.sh -m ${MODEL_PATH} --concurrencies "${CONCURRENCIES}" -dp 4 -tp 1 -ep -pc -cpu-gbs "0" -cd "0,1,2,3" -p 8000

# Parallel plan: tp4-ep
# bash bench_profile.sh -m ${MODEL_PATH} --concurrencies "${CONCURRENCIES}" -dp 1 -tp 4 -ep -pc -cpu-gbs "2 4 8 16" -cd "0,1,2,3" -p 8000

# Parallel plan: tp4
# bash bench_profile.sh -m ${MODEL_PATH} --concurrencies "${CONCURRENCIES}" -dp 1 -tp 4 -pc -cpu-gbs "0" -cd "0,1,2,3" -p 8000

# === Single GPU ===
# bash bench_profile.sh -m ${MODEL_PATH} --concurrencies "${CONCURRENCIES}" -dp 1 -tp 4 -pc -cpu-gbs "30" -cd "0" -p 8000

# === Test ===
# bash bench_profile.sh -m /dev/shm/gpt-oss-20b/ --concurrencies "${CONCURRENCIES}" -dp 2 -tp 1 -ep -pc -cpu-gbs "0" -cd "0,3" -p 8002
# bash bench_profile.sh -m /dev/shm/gpt-oss-20b --concurrencies "${CONCURRENCIES}" -dp 1 -tp 2 -ep -pc -cpu-gbs "0" -cd "2,3" -p 8000

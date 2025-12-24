#!/usr/bin/bash
set -e 

DUMP_DIR=./dump
LOG_DIR=./log

SERVER_DIR=server
BENCH_DIR=bench
DEVICE_DIR=device
OBS_DIR=observability

VLLM_PATH=./thirdparty/vllm-hpclab

mkdir -p $DUMP_DIR
mkdir -p $LOG_DIR

mkdir -p $DUMP_DIR/$SERVER_DIR/
mkdir -p $DUMP_DIR/$BENCH_DIR/
mkdir -p $DUMP_DIR/$OBS_DIR/

mkdir -p $LOG_DIR/$BENCH_DIR/
mkdir -p $LOG_DIR/$DEVICE_DIR/
mkdir -p $LOG_DIR/$OBS_DIR/

MAX_RUN_SECONDS=3600
WORKLOADS=(
	"ShareGPT ShareGPT"
	# "4096 1024"
	# "16384 1024"
	# "32768 1024"
)

# === Parse CLI args ===
while [[ "$#" -gt 0 ]]; do
	case $1 in
		--model | -m) MODEL_PATH="$2"; shift;;
		--concurrencies) CONCURRENCIES=($2); shift;;
		--data-parallel-size | -dp) DP_SIZE="$2"; shift;;
		--tensor-parallel-size | -tp) TP_SIZE="$2"; shift;;
		--enable-expert-parallel | -ep) ENABLE_EP=1 ;;
		--enable-prefix-caching | -pc) ENABLE_PC=1 ;;
		--offload-gbs | -cpu-gbs) CPU_GBS=($2); shift ;;
		--cuda-devices | -cd) CUDA_DEVICES="$2"; shift;;
		--port | -p ) PORT="$2"; shift;;
		--help | -h )
			echo -e "Add later, okay? ^^"
			exit 0 ;;
		*)
			echo "[ERROR] Unknown option: $1"
			exit 1 ;;
	esac
	shift
done

# === Validate required args ===
if [[ -z "$MODEL_PATH" ]]; then
	echo "[ERROR] --model is required."
	exit 1
fi

if [[ ${#CONCURRENCIES[@]} -eq 0 ]]; then
	echo "[ERROR] --concurrencies is required (e.g., \"1 4 8\")"
	exit 1
fi

MODEL_NAME=$(basename "$MODEL_PATH")

# === Start server ===
start_server() {
	local max_concurrency=$1
	local tp=$2
	local dp=$3
	local offload_gb=$4

	echo -e "[INFO] Starting vLLM server with:"
	echo -e "\tmax_num_seqs \t\t= $max_concurrency (reqs)"
	echo -e "\ttensor-parallel-size \t= $tp"
	echo -e "\tdata-parallel-size \t= $dp"
	echo -e "\tenable-expert-parallel \t= $ENABLE_EP"
	echo -e "\tcpu-offload-gb \t\t= $offload_gb (GB)"
	
	local ep_arg=()
	if [[ "$ENABLE_EP" -eq 1 ]]; then
		ep_arg=(--enable-expert-parallel)
	fi

	local pc_arg=()
	if [[ "$ENABLE_PC" -eq 1 ]]; then
		pc_arg=(--enable-prefix-caching)
	fi

	local offload_arg=()
	if [[ -n "${offload_gb:-}" ]]; then
		offload_args=(--cpu-offload-gb "$offload_gb")
	fi

	# --enforce-eager is enabled because offloading + parallelism does not work with CUDA graph
	LOG_SUFFIX=$TIMING_LOG_SUFFIX ENABLE_EXPERT_ACTIVATION_PROFILE=0 ENABLE_LATENCY_PROFILE=1 CUDA_VISIBLE_DEVICES=$CUDA_DEVICES vllm serve "$MODEL_PATH" \
		--max-num-seqs "$max_concurrency" \
		--data-parallel-size "$dp" \
		--tensor-parallel-size "$tp" \
		"${ep_arg[@]}" \
		"${pc_arg[@]}" \
		"${offload_args[@]}" \
		--gpu-memory-utilization 0.9 \
		--enable-chunked-prefill \
		--enforce-eager \
		--port "$PORT" > "$SERVER_LOG" 2>&1 &
	
	SERVER_PID=$! # $! is a special bash variable that stores the PID of the most recently executed background command
	set +e
	echo "[INFO] Waiting for server to become healthy on port $PORT..."
	until [ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health)" -eq 200 ]; do
		sleep 1
	done
	echo "[INFO] Server is ready (PID=$SERVER_PID)"
}

start_observability() {
	echo "[INFO] Starting Prometheus-Grafana stack"
	sudo docker compose -f $VLLM_PATH/examples/online_serving/prometheus_grafana/docker-compose.yaml up > $DUMP_DIR/$OBS_DIR/obs.dump 2>&1 & 
	echo "[INFO] Observability is ready"
}

stop_server() {
	echo "[INFO] Stopping server (PID=$SERVER_PID)..."
	kill "$SERVER_PID"
	wait "$SERVER_PID" 2>/dev/null || true
	echo "[INFO] Server stopped."
}

for OFFLOAD_GB in "${CPU_GBS[@]}"; do
	for CONCURRENCY in "${CONCURRENCIES[@]}"; do
		NUM_PROMPTS=$((3 * CONCURRENCY))
		LOG_SUFFIX="${MODEL_NAME}_tp${TP_SIZE}_dp${DP_SIZE}_ep${ENABLE_EP}_off${OFFLOAD_GB}_con${CONCURRENCY}"
		SERVER_LOG="$DUMP_DIR/$SERVER_DIR/server_${LOG_SUFFIX}.log"
		# TORCH_TRACE_DIR=/dev/shm/ducct/trace-logs/torch-profiler/$LOG_SUFFIX

		for workload in "${WORKLOADS[@]}"; do
			# run vLLM
			read -r INPUT_LEN OUTPUT_LEN <<< "$workload"
			TIMING_LOG_SUFFIX="${LOG_SUFFIX}_IN${INPUT_LEN}_OUT${OUTPUT_LEN}"
			# TIMING_LOG_SUFFIX=""
			start_server "$CONCURRENCY" "$TP_SIZE" "$DP_SIZE" "$OFFLOAD_GB"
			# run Prometheus-Grafana
			start_observability


			echo "==============================="              
			echo "[RUN] Input=$INPUT_LEN | Output=$OUTPUT_LEN | CONCURRENCY=$CONCURRENCY | NUM PROMPTS=$NUM_PROMPTS"
			echo "==============================="              
			RESULT_DIR="$LOG_DIR/$BENCH_DIR/report_in${INPUT_LEN}_out${OUTPUT_LEN}_${LOG_SUFFIX}"
			CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python device_monitor.py \
				--interval 1 \
				--logfile "$LOG_DIR/$DEVICE_DIR/gpu_in${INPUT_LEN}_out${OUTPUT_LEN}_${LOG_SUFFIX}.csv" &
			GPU_MONITOR_PID=$!

			set +e
			# Increase limit (temporary for current shell)
			ulimit -n 65536
			# timeout "$MAX_RUN_SECONDS" vllm bench serve \
			# vllm bench serve \
			# 	--model "$MODEL_PATH" \
			# 	--dataset-name random \
			# 	--num-prompts "$NUM_PROMPTS" \
			# 	--max-concurrency "$CONCURRENCY" \
			# 	--request-rate inf \
			# 	--random-input-len "$INPUT_LEN" \
			# 	--random-output-len "$OUTPUT_LEN" \
			# 	--save-result \
			# 	--save-detailed \
			# 	--result-dir "$RESULT_DIR" \
			# 	--ignore-eos --seed 0 \
			# 	--port $PORT \
			# 	> "$DUMP_DIR/$BENCH_DIR/summary_in${INPUT_LEN}_out${OUTPUT_LEN}_${LOG_SUFFIX}.txt"

			vllm bench serve \
				--model "$MODEL_PATH" \
				--dataset-name sharegpt \
				--num-prompts "$NUM_PROMPTS" \
				--max-concurrency "$CONCURRENCY" \
				--request-rate inf \
				--dataset-path ./data/ShareGPT_V3_unfiltered_cleaned_split.json \
				--save-result \
				--save-detailed \
				--result-dir "$RESULT_DIR" \
				--seed 0 \
				--port $PORT \
				> "$DUMP_DIR/$BENCH_DIR/summary_in${INPUT_LEN}_out${OUTPUT_LEN}_${LOG_SUFFIX}.txt"
			bench_status=$?
			set -e

			if [[ $bench_status -eq 124 ]]; then
				echo "[WARN] Benchmark timed out after ${MAX_RUN_SECONDS}s (1 hour). Skipping to the next configuration."
				break
			elif [[ $bench_status -ne 0 ]]; then
				echo "[ERROR] Benchmark failed with status $bench_status."
				stop_server
				exit 1
			fi

			# test query prometheus
			python observability.py \
				--logfile "$LOG_DIR/$OBS_DIR/gpu_in${INPUT_LEN}_out${OUTPUT_LEN}_${LOG_SUFFIX}.csv"
			
			# Stop the server and move on to the next iteration
			kill "$GPU_MONITOR_PID" 2>/dev/null || true # stop
			wait "$GPU_MONITOR_PID" 2>/dev/null || true # wait for device_monitor.py to exit
			stop_server
			echo "[INFO] Sleeping 5s before next run..."
			sleep 5
		done
	done
done

echo
echo "[DONE] All benchmarks completed successfully"

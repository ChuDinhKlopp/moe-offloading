# Clone the vllm-hpclab
```
cd thirdparty
git clone https://github.com/ChuDinhKlopp/vllm-hpclab.git
```

# Download ShareGPT dataset
```
wget -P data/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

# Expert activation pattern
To record the expert activation pattern, set the environment variable `ENABLE_EXPERT_ACTIVATION_PROFILE=1` of `vllm serve` command in `scipts/bench_e2e_offload.sh`

```
LOG_SUFFIX=$TIMING_LOG_SUFFIX ENABLE_EXPERT_ACTIVATION_PROFILE=1 ENABLE_LATENCY_PROFILE=0 CUDA_VISIBLE_DEVICES=$CUDA_DEVICES vllm serve "$MODEL_PATH" \
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

```

# Run
Execute the script `bench_gpt-oss-120b.sh`
```
./bench_gpt-oss-120b.sh
```

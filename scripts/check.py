import torch
from torch.profiler import profile, ProfilerActivity, record_function
from vllm.model_executor.models.gpt_oss import TransformerBlock

from vllm.config import (  
    VllmConfig, ModelConfig, CacheConfig, ParallelConfig,  
    SchedulerConfig, DeviceConfig, LoadConfig, AttentionConfig,  
    StructuredOutputsConfig, ObservabilityConfig, CompilationConfig  
)  
from vllm.config import set_current_vllm_config  
from vllm.distributed import init_distributed_environment, initialize_model_parallel  
import tempfile  
  
# Initialize distributed environment  
temp_file = tempfile.mkstemp()[1]  
init_distributed_environment(  
    world_size=1,  
    rank=0,  
    distributed_init_method=f"file://{temp_file}",  
    local_rank=0,  
    backend="nccl",  
)  
initialize_model_parallel(tensor_model_parallel_size=1)    
model_config = ModelConfig(  
    model="/dev/shm/gpt-oss-20b",  
    tokenizer="/dev/shm/gpt-oss-20b",  
    dtype="bfloat16",  
    quantization="mxfp4",  
    trust_remote_code=False,  
    seed=0,  
    max_model_len=131072,  
)  
  
cache_config = CacheConfig(  
    block_size=16,  # Default, platform-dependent  
    gpu_memory_utilization=0.9,  
    swap_space=0,  
    cache_dtype="auto",  
    enable_prefix_caching=True,  
)  
  
parallel_config = ParallelConfig(  
    tensor_parallel_size=1,  
    pipeline_parallel_size=1,  
    data_parallel_size=1,  
    disable_custom_all_reduce=False,  
)  
  
scheduler_config = SchedulerConfig.default_factory(  
    max_model_len=131072,  
    enable_chunked_prefill=True,  
)  
  
device_config = DeviceConfig(device="cuda")  
load_config = LoadConfig(load_format="auto")  
attention_config = AttentionConfig()  
structured_outputs_config = StructuredOutputsConfig(  
    backend="auto",  
    reasoning_parser="openai_gptoss",  
)  
observability_config = ObservabilityConfig()  
compilation_config = CompilationConfig(  
    mode="none",  
    backend="inductor",  
    custom_ops=["all"],  
    splitting_ops=[],  
    compile_sizes=[],  
    compile_ranges_split_points=[2048],  
    cudagraph_mode="none",  
    dynamic_shapes_config={  
        "type": "backed",  
        "evaluate_guards": False,  
        "assume_32_bit_indexing": True,  
    },  
)  
  
vllm_config = VllmConfig(  
    model_config=model_config,  
    cache_config=cache_config,  
    parallel_config=parallel_config,  
    scheduler_config=scheduler_config,  
    device_config=device_config,  
    load_config=load_config,  
    attention_config=attention_config,  
    structured_outputs_config=structured_outputs_config,  
    observability_config=observability_config,  
    compilation_config=compilation_config,  
    quant_config=None,  # Set by quantization="mxfp4" in ModelConfig  
)

# Wrap TransformerBlock creation in the context manager  
with set_current_vllm_config(vllm_config):  
    block = TransformerBlock(  
        vllm_config=vllm_config,  
        quant_config=vllm_config.quant_config,  
        prefix="model.layers.0"  
    )

torch.cuda.init()
torch.cuda.synchronize()

N_EXPERTS = 4
CACHE_SIZE = 8
INTER_DIM = 2880
HIDDEN_DIM = 2880

# Pre-create GPU weight (avoid randn/alloc inside compute)
mat = torch.randn(N_EXPERTS, HIDDEN_DIM, HIDDEN_DIM, device="cuda")

def compute(data):
    return data @ mat

copy_s = torch.cuda.Stream()
compute_s = torch.cuda.default_stream()

current_data = torch.randn(N_EXPERTS, INTER_DIM, HIDDEN_DIM, device="cuda")
next_data = torch.randn(N_EXPERTS, INTER_DIM, HIDDEN_DIM, pin_memory=True, device="cpu")
gpu_cache = torch.zeros(CACHE_SIZE, INTER_DIM, HIDDEN_DIM, device="cuda")
cached_expert_ids = torch.tensor([0,1,2,3,4,5], device="cuda")
predicted_expert_ids = torch.tensor([5,8,2,7], device="cuda")

# Warmup (avoid first-call initialization dominating trace)
for _ in range(3):
    with torch.cuda.stream(copy_s):
        gpu_cache[:4].copy_(next_data, non_blocking=True)
    _ = compute(current_data)
torch.cuda.synchronize()

mask = torch.isin(predicted_expert_ids, cached_expert_ids)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # This run on CPU
    if not mask.all():
        with torch.cuda.stream(copy_s), record_function("H2D_copy"):
            gpu_cache[:4].copy_(next_data, non_blocking=True)
    
    # This run on GPU
    with record_function("compute"):
        output = 
    
# Export after profiling
prof.export_chrome_trace("trace.json")
print("wrote trace.json, res:", res.shape)


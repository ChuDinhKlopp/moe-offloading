from vllm import LLM, SamplingParams  
from vllm.engine.arg_utils import EngineArgs  
  
# engine_args = EngineArgs(  
#     model="/dev/shm/gpt-oss-20b",  
#     profiler_config='{"profiler": "torch", "torch_profiler_dir": "/dev/shm/ducct/trace-logs"}',  
# )  

llm = LLM(
     model="/dev/shm/gpt-oss-20b",  
     profiler_config='{"profiler": "torch", "torch_profiler_dir": "/dev/shm/ducct/trace-logs"}',
     )
llm.start_profile()  
outputs = llm.generate(["Hello, how u doin"], SamplingParams(max_tokens=10))  
llm.stop_profile()

# # SPDX-License-Identifier: Apache-2.0
# """
# This file demonstrates the example usage of cpu offloading
# with LMCache.

# Note that `pip install lmcache` is needed to run this example.
# Learn more about LMCache in https://github.com/LMCache/LMCache.
# """
# import os
# import time

# from lmcache.experimental.cache_engine import LMCacheEngineBuilder
# from lmcache.integration.vllm.utils import ENGINE_NAME
# ENGINE_NAME = "vllm-instance"

# from vllm import LLM, SamplingParams
# from vllm.config import KVTransferConfig

# # LMCache-related environment variables
# # Use experimental features in LMCache
# os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
# # LMCache is set to use 256 tokens per chunk
# os.environ["LMCACHE_CHUNK_SIZE"] = "256"
# # Enable local CPU backend in LMCache
# os.environ["LMCACHE_LOCAL_CPU"] = "True"
# # Set local CPU memory limit to 5.0 GB
# os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5.0"

# # This example script runs two requests with a shared prefix.
# shared_prompt = "Hello, how are you?" * 1000
# first_prompt = [
#     shared_prompt + "Hello, my name is",
# ]
# second_prompt = [
#     shared_prompt + "Tell me a very long story",
# ]

# sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

# ktc = KVTransferConfig.from_cli(
#     '{"kv_connector":"LMCacheConnector", "kv_role":"kv_both"}')
# # Set GPU memory utilization to 0.8 for an A40 GPU with 40GB
# # memory. Reduce the value if your GPU has less memory.
# # Note that LMCache is not compatible with chunked prefill for now.
# llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2",
#           kv_transfer_config=ktc,
#           max_model_len=8000,
#           enable_chunked_prefill=False,
#           gpu_memory_utilization=0.8)

# outputs = llm.generate(first_prompt, sampling_params)
# for output in outputs:
#     generated_text = output.outputs[0].text
#     print(f"Generated text: {generated_text!r}")
# print("First request done.")

# time.sleep(1)

# outputs = llm.generate(second_prompt, sampling_params)
# for output in outputs:
#     generated_text = output.outputs[0].text
#     print(f"Generated text: {generated_text!r}")
# print("Second request done.")

# # Clean up lmcache backend
# LMCacheEngineBuilder.destroy(ENGINE_NAME)


# SPDX-License-Identifier: Apache-2.0
# from vllm import LLM
# from vllm.sampling_params import SamplingParams
# import pynvml
# from vllm.config import KVTransferConfig
# from vllm import LLM, SamplingParams
# import os
# import time
# import psutil  # 用于监控内存
# import pynvml  # 用于监控 GPU 内存

# from lmcache.experimental.cache_engine import LMCacheEngineBuilder
# from lmcache.integration.vllm.utils import ENGINE_NAME
# ENGINE_NAME = "vllm-instance"


# # 初始化 GPU 内存监控
# pynvml.nvmlInit()
# gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(1)


# def get_gpu_memory_usage():
#     """获取 GPU 内存使用量（MB）"""
#     memory_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
#     return memory_info.used / 1024 / 1024  # 转换为 MB


# # LMCache 环境变量
# os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
# os.environ["LMCACHE_CHUNK_SIZE"] = "256"
# os.environ["LMCACHE_LOCAL_CPU"] = "True"
# os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5.0"

# # 两个请求的共享前缀
# shared_prompt = "Hello, how are you?" * 150  # 约 5000 token
# first_prompt = [shared_prompt + "Hello, my name is"]
# second_prompt = [shared_prompt + "Tell me a very long story"]

# sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)


# def run_inference(use_lmcache=True):
#     """运行推理，比较启用和禁用 LMCache 的效果"""
#     # 配置 KVTransferConfig
#     if use_lmcache:
#         ktc = KVTransferConfig.from_cli(
#             '{"kv_connector":"LMCacheConnector", "kv_role":"kv_both"}')
#         print("Running with LMCache enabled...")
#     else:
#         ktc = None  # 禁用 LMCache
#         print("Running with LMCache disabled...")

#     # 初始化 LLM
#     llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2",
#               kv_transfer_config=ktc,
#               max_model_len=8000,
#               enable_chunked_prefill=False,
#               gpu_memory_utilization=0.8)

#     # 第一个请求
#     start_time = time.time()
#     outputs = llm.generate(first_prompt, sampling_params)
#     first_duration = time.time() - start_time
#     first_memory = get_gpu_memory_usage()

#     for output in outputs:
#         print(f"First request generated text: {output.outputs[0].text!r}")

#     print(f"First request time: {first_duration:.2f} seconds")
#     print(f"First request GPU memory usage: {first_memory:.2f} MB")

#     time.sleep(1)

#     # 第二个请求
#     start_time = time.time()
#     outputs = llm.generate(second_prompt, sampling_params)
#     second_duration = time.time() - start_time
#     second_memory = get_gpu_memory_usage()

#     for output in outputs:
#         print(f"Second request generated text: {output.outputs[0].text!r}")

#     print(f"Second request time: {second_duration:.2f} seconds")
#     print(f"Second request GPU memory usage: {second_memory:.2f} MB")

#     # 清理
#     if use_lmcache:
#         LMCacheEngineBuilder.destroy(ENGINE_NAME)

#     return first_duration, second_duration, first_memory, second_memory


# # 运行对比实验
# print("=== Experiment 1: With LMCache ===")
# with_lmcache = run_inference(use_lmcache=True)

# print("\n=== Experiment 2: Without LMCache ===")
# without_lmcache = run_inference(use_lmcache=False)

# # 对比结果
# print("\n=== Comparison ===")
# print(
#     f"First request time (with LMCache): {with_lmcache[0]:.2f} s, (without LMCache): {without_lmcache[0]:.2f} s")
# print(
#     f"Second request time (with LMCache): {with_lmcache[1]:.2f} s, (without LMCache): {without_lmcache[1]:.2f} s")
# print(
#     f"First request GPU memory (with LMCache): {with_lmcache[2]:.2f} MB, (without LMCache): {without_lmcache[2]:.2f} MB")
# print(
#     f"Second request GPU memory (with LMCache): {with_lmcache[3]:.2f} MB, (without LMCache): {without_lmcache[3]:.2f} MB")


import pynvml
import os
import time
from vllm import LLM
from vllm.config import KVTransferConfig
from vllm.sampling_params import SamplingParams

from lmcache.experimental.cache_engine import LMCacheEngineBuilder
from lmcache.integration.vllm.utils import ENGINE_NAME
ENGINE_NAME = "vllm-instance"


pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(1)


def get_gpu_memory_usage():
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    return memory_info.used / 1024 / 1024


# LMCache 环境变量
os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
os.environ["LMCACHE_LOCAL_CPU"] = "True"
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5.0"

# 两个请求的共享前缀
shared_prompt = "Hello, how are you?" * 150  # 约 5000 token
first_prompt = [shared_prompt + "Hello, my name is"]
second_prompt = [shared_prompt + "Tell me a very long story"]

sample_parameter = SamplingParams(temperature=0, top_p=0.95, max_tokens=128)


def run_inference(use_lmcache: bool):

    if use_lmcache is True:
        ktc = KVTransferConfig.from_cli(
            '{"kv_connector":"LMCacheConnector", "kv_role":"kv_both"}')
        print("lmcache is enable....")
    else:
        ktc = None
        print("lmcache is disable....")

    model = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.8, kv_transfer_config=ktc,
                max_model_len=8000,
                enable_chunked_prefill=False)
    
    start_time = time.time()
    outputs = model.generate(first_prompt, sampling_params=sample_parameter)
    end_time = time.time()
    first_duration = end_time - start_time
    first_memory_usage = get_gpu_memory_usage()
    
    for output in outputs:
        print(output.outputs[0].text)
        
    time.sleep(1)
    
    
    start_time = time.time()
    outputs = model.generate(second_prompt, sampling_params=sample_parameter)
    end_time = time.time()
    second_duration = end_time - start_time
    second_memory_usage = get_gpu_memory_usage()
    
    for output in outputs:
        print(output.outputs[0].text)
    
    if use_lmcache:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)
        
    return first_duration, second_duration, first_memory_usage, second_memory_usage


# 运行对比实验
print("=== Experiment 1: With LMCache ===")
with_lmcache = run_inference(use_lmcache=True)

print("\n=== Experiment 2: Without LMCache ===")
without_lmcache = run_inference(use_lmcache=False)

# 对比结果
print("\n=== Comparison ===")
print(
    f"First request time (with LMCache): {with_lmcache[0]:.2f} s, (without LMCache): {without_lmcache[0]:.2f} s")
print(
    f"Second request time (with LMCache): {with_lmcache[1]:.2f} s, (without LMCache): {without_lmcache[1]:.2f} s")
print(
    f"First request GPU memory (with LMCache): {with_lmcache[2]:.2f} MB, (without LMCache): {without_lmcache[2]:.2f} MB")
print(
    f"Second request GPU memory (with LMCache): {with_lmcache[3]:.2f} MB, (without LMCache): {without_lmcache[3]:.2f} MB")


    
    
    
    

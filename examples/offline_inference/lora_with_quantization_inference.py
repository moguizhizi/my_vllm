# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use LoRA with different quantization techniques
for offline inference.

Requires HuggingFace credentials for access.
"""

import gc
from typing import Optional

import torch
from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest


def create_test_prompts(
        lora_path: str
) -> list[tuple[str, SamplingParams, Optional[LoRARequest]]]:
    return [
        # this is an example of using quantization without LoRA
        ("My name is",
         SamplingParams(temperature=0.0,
                        logprobs=1,
                        prompt_logprobs=1,
                        max_tokens=128), None),
        # the next three examples use quantization with LoRA
        ("my name is",
         SamplingParams(temperature=0.0,
                        logprobs=1,
                        prompt_logprobs=1,
                        max_tokens=128),
         LoRARequest("lora-test-1", 1, lora_path)),
        ("The capital of USA is",
         SamplingParams(temperature=0.0,
                        logprobs=1,
                        prompt_logprobs=1,
                        max_tokens=128),
         LoRARequest("lora-test-2", 1, lora_path)),
        ("The capital of France is",
         SamplingParams(temperature=0.0,
                        logprobs=1,
                        prompt_logprobs=1,
                        max_tokens=128),
         LoRARequest("lora-test-3", 1, lora_path)),
    ]


def process_requests(engine: LLMEngine,
                     test_prompts: list[tuple[str, SamplingParams,
                                              Optional[LoRARequest]]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            request_id += 1

        request_outputs: list[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                print("----------------------------------------------------")
                print(f"Prompt: {request_output.prompt}")
                print(f"Output: {request_output.outputs[0].text}")


# def initialize_engine(model: str, quantization: str,
#                       lora_repo: Optional[str]) -> LLMEngine:
#     """Initialize the LLMEngine."""

#     if quantization == "bitsandbytes":
#         # QLoRA (https://arxiv.org/abs/2305.14314) is a quantization technique.
#         # It quantizes the model when loading, with some config info from the
#         # LoRA adapter repo. So need to set the parameter of load_format and
#         # qlora_adapter_name_or_path as below.
#         engine_args = EngineArgs(model=model,
#                                  quantization=quantization,
#                                  qlora_adapter_name_or_path=lora_repo,
#                                  enable_lora=True,
#                                  max_lora_rank=64)
#     else:
#         engine_args = EngineArgs(model=model,
#                                  quantization=quantization,
#                                  enable_lora=True,
#                                  max_loras=4)
        
    
#     return LLMEngine.from_engine_args(engine_args)

def initialize_engine(model: str, quantization: Optional[str],
                      lora_repo: Optional[str]) -> LLMEngine:
    """Initialize the LLMEngine and print debug info."""

    if quantization == "bitsandbytes":
        engine_args = EngineArgs(
            model=model,
            quantization=quantization,
            qlora_adapter_name_or_path=lora_repo,
            enable_lora=True,
            max_lora_rank=64
        )
    else:
        engine_args = EngineArgs(
            model=model,
            quantization=quantization,
            enable_lora=True,
            max_loras=4
        )

    engine = LLMEngine.from_engine_args(engine_args)

    # >>>>> 打印底层模型结构 <<<<<
    try:
        model_obj = engine.model_executor.driver_worker.model_runner.model
        print(f"[Debug] lm_head weight shape: {model_obj.lm_head.weight.shape}")
        print(f"[Debug] embedding weight shape: {model_obj.get_input_embeddings().weight.shape}")
    except Exception as e:
        print("[Warning] Failed to access internal model:", e)

    return engine



def main():
    """Main function that sets up and runs the prompt processing."""

    # test_configs = [{
    #     "name": "qlora_inference_example",
    #     'model': "huggyllama/llama-7b",
    #     'quantization': "bitsandbytes",
    #     'lora_repo': 'timdettmers/qlora-flan-7b'
    # }, {
    #     "name": "AWQ_inference_with_lora_example",
    #     'model': 'TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ',
    #     'quantization': "awq",
    #     'lora_repo': 'jashing/tinyllama-colorist-lora'
    # }, {
    #     "name": "GPTQ_inference_with_lora_example",
    #     'model': 'TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ',
    #     'quantization': "gptq",
    #     'lora_repo': 'jashing/tinyllama-colorist-lora'
    # }]
    
    
    test_configs = [{
        "name": "AWQ_inference_with_lora_example",
        'model': 'PY007/TinyLlama-1.1B-Chat-v0.3',
        'quantization': None,
        'lora_repo': 'jashing/tinyllama-colorist-lora'
    }, {
        "name": "GPTQ_inference_with_lora_example",
        'model': 'PY007/TinyLlama-1.1B-Chat-v0.3',
        'quantization':None,
        'lora_repo': 'jashing/tinyllama-colorist-lora'
    }]

    for test_config in test_configs:
        print(
            f"~~~~~~~~~~~~~~~~ Running: {test_config['name']} ~~~~~~~~~~~~~~~~"
        )
        engine = initialize_engine(test_config['model'],
                                   test_config['quantization'],
                                   test_config['lora_repo'])
        lora_path = snapshot_download(repo_id=test_config['lora_repo'])
        print(lora_path)

        exit(0)

        test_prompts = create_test_prompts(lora_path)
        process_requests(engine, test_prompts)

        # Clean up the GPU memory for the next test
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
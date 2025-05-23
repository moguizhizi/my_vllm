# SPDX-License-Identifier: Apache-2.0

# from vllm import LLM, SamplingParams

# # Sample prompts.
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
# # Create a sampling params object.
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# # Create an LLM.
# llm = LLM(model="facebook/opt-125m")
# # Generate texts from the prompts. The output is a list of RequestOutput objects
# # that contain the prompt, generated text, and other information.
# outputs = llm.generate(prompts, sampling_params)
# # Print the outputs.
# print("\nGenerated Outputs:\n" + "-" * 60)
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt:    {prompt!r}")
#     print(f"Output:    {generated_text!r}")
#     print("-" * 60)

from vllm import LLM, SamplingParams

params = SamplingParams(temperature=0.8, top_p=0.95)
model = LLM(model="facebook/opt-125m")

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

outputs = model.generate(prompts, params)

for output in outputs:
    print(f"prompt:{output.prompt}")
    print(f"text:{output.outputs[0].text}")
    print("*" * 60)
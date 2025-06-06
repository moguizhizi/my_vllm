# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams

prompts = [
    "A robot may not injure a human being",
    "It is only with the heart that one can see rightly;",
    "The greatest glory in living lies not in never falling,",
]
answers = [
    " or, through inaction, allow a human being to come to harm.",
    " what is essential is invisible to the eye.",
    " but in rising every time we fall.",
]
N = 1
# Currently, top-p sampling is disabled. `top_p` should be 1.0.
sampling_params = SamplingParams(temperature=0, top_p=1.0, n=N, max_tokens=16)


def main():
    # Set `enforce_eager=True` to avoid ahead-of-time compilation.
    # In real workloads, `enforace_eager` should be `False`.
    llm = LLM(model="/data1/temp/huggingface/Mistral-7B-Instruct-v0.2",
              max_num_batched_tokens=64,
              max_num_seqs=4)
    outputs = llm.generate(prompts, sampling_params)
    print("-" * 50)
    for output, answer in zip(outputs, answers):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        assert generated_text.startswith(answer)
        print("-" * 50)


if __name__ == "__main__":
    main()
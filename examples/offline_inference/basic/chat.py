# # SPDX-License-Identifier: Apache-2.0

# from vllm import LLM, EngineArgs
# from vllm.utils import FlexibleArgumentParser


# def main(args: dict):
#     # Pop arguments not used by LLM
#     max_tokens = args.pop("max_tokens")
#     temperature = args.pop("temperature")
#     top_p = args.pop("top_p")
#     top_k = args.pop("top_k")
#     chat_template_path = args.pop("chat_template_path")

#     # Create an LLM
#     llm = LLM(**args)

#     # Create sampling params object
#     sampling_params = llm.get_default_sampling_params()
#     if max_tokens is not None:
#         sampling_params.max_tokens = max_tokens
#     if temperature is not None:
#         sampling_params.temperature = temperature
#     if top_p is not None:
#         sampling_params.top_p = top_p
#     if top_k is not None:
#         sampling_params.top_k = top_k

#     def print_outputs(outputs):
#         print("\nGenerated Outputs:\n" + "-" * 80)
#         for output in outputs:
#             prompt = output.prompt
#             generated_text = output.outputs[0].text
            # print(f"Prompt: {prompt!r}\n")
            # print(f"Generated text: {generated_text!r}")
            # print("-" * 80)

#     print("=" * 80)

#     # In this script, we demonstrate how to pass input to the chat method:
#     conversation = [
#         {
#             "role": "system",
#             "content": "You are a helpful assistant"
#         },
#         {
#             "role": "user",
#             "content": "Hello"
#         },
#         {
#             "role": "assistant",
#             "content": "Hello! How can I assist you today?"
#         },
#         {
#             "role": "user",
#             "content":
#             "Write an essay about the importance of higher education.",
#         },
#     ]
#     outputs = llm.chat(conversation, sampling_params, use_tqdm=False)
#     print_outputs(outputs)

#     # You can run batch inference with llm.chat API
#     conversations = [conversation for _ in range(10)]

#     # We turn on tqdm progress bar to verify it's indeed running batch inference
#     outputs = llm.chat(conversations, sampling_params, use_tqdm=True)
#     print_outputs(outputs)

#     # A chat template can be optionally supplied.
#     # If not, the model will use its default chat template.
#     # if chat_template_path is not None:
#     #     with open(chat_template_path) as f:
#     #         chat_template = f.read()

#     #     outputs = llm.chat(
#     #         conversations,
#     #         sampling_params,
#     #         use_tqdm=False,
#     #         chat_template=chat_template,
#     #     )

#     chat_template = (
#         "{% for message in messages %}"
#         "{% if message.role == 'system' %}"
#         "System: {{ message.content }}\n"
#         "{% elif message.role == 'user' %}"
#         "User: {{ message.content }}\n"
#         "{% elif message.role == 'assistant' %}"
#         "Assistant: {{ message.content }}\n"
#         "{% endif %}"
#         "{% endfor %}"
#     )

#     outputs = llm.chat(
#         conversations,
#         sampling_params,
#         use_tqdm=False,
#         chat_template=chat_template,
#     )

#     print_outputs(outputs)


# if __name__ == "__main__":
#     parser = FlexibleArgumentParser()
#     # Add engine args
#     engine_group = parser.add_argument_group("Engine arguments")
#     EngineArgs.add_cli_args(engine_group)
#     engine_group.set_defaults(model="LLM-Research/Llama-3.2-1B-Instruct")
#     # Add sampling params
#     sampling_group = parser.add_argument_group("Sampling parameters")
#     sampling_group.add_argument("--max-tokens", type=int)
#     sampling_group.add_argument("--temperature", type=float)
#     sampling_group.add_argument("--top-p", type=float)
#     sampling_group.add_argument("--top-k", type=int)
#     # Add example params
#     parser.add_argument("--chat-template-path", type=str)
#     args: dict = vars(parser.parse_args())
#     main(args)



from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


def print_outputs(outputs):
    for output in outputs:
        print(f"prompt:{output.prompt!r}")
        print(f"output:{output.outputs[0].text!r}")


def main(args:dict):
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_k = args.pop("top_k")
    top_p = args.pop("top_p")
    
    model = LLM(**args)
    sample_parameter = model.get_default_sampling_params()
    if max_tokens is not None:
        sample_parameter.max_tokens = max_tokens
    if temperature is not None:
        sample_parameter.temperature = temperature
    if top_k is not None:
        sample_parameter.top_k = top_k
    if top_p is not None:
        sample_parameter.top_p = top_p
        
        
    print("=" * 80)

    # In this script, we demonstrate how to pass input to the chat method:
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": "Hello"
        },
        {
            "role": "assistant",
            "content": "Hello! How can I assist you today?"
        },
        {
            "role": "user",
            "content":
            "Write an essay about the importance of higher education.",
        },
    ]
    
    outputs = model.chat(conversation, sample_parameter, use_tqdm=False)
    print_outputs(outputs)
    
    conversations = [conversation for _ in range(10)]
    outputs = model.chat(conversations, sample_parameter, use_tqdm=True)
    print_outputs(outputs)
    
    
    
    

if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    engine_group = parser.add_argument_group("Engine Groups")
    EngineArgs.add_cli_args(engine_group)
    engine_group.set_defaults(model="LLM-Research/Llama-3.2-1B-Instruct")
    
    sample_parameter = parser.add_argument_group("Sample Groups")
    sample_parameter.add_argument("--max_tokens", type=int)
    sample_parameter.add_argument("--temperature",type=float)
    sample_parameter.add_argument("--top_k",type=int)
    sample_parameter.add_argument("--top_p",type=float)
    
    args:dict = vars(parser.parse_args())
    main(args=args)
# # SPDX-License-Identifier: Apache-2.0

# from argparse import Namespace

# from vllm import LLM, EngineArgs
# from vllm.utils import FlexibleArgumentParser


# def main(args: Namespace):
#     # Sample prompts.
#     prompts = [
#         "Hello, my name is",
#         "The president of the United States is",
#         "The capital of France is",
#         "The future of AI is",
#     ]

#     # Create an LLM.
#     # You should pass task="embed" for embedding models
#     model = LLM(**vars(args))

#     # Generate embedding. The output is a list of EmbeddingRequestOutputs.
#     outputs = model.embed(prompts)

#     # Print the outputs.
#     print("\nGenerated Outputs:\n" + "-" * 60)
#     for prompt, output in zip(prompts, outputs):
#         embeds = output.outputs.embedding
#         embeds_trimmed = ((str(embeds[:16])[:-1] +
#                            ", ...]") if len(embeds) > 16 else embeds)
#         print(f"Prompt: {prompt!r} \n"
#               f"Embeddings: {embeds_trimmed} (size={len(embeds)})")
#         print("-" * 60)


# if __name__ == "__main__":
#     parser = FlexibleArgumentParser()
#     parser = EngineArgs.add_cli_args(parser)
#     # Set example specific arguments
#     parser.set_defaults(model="intfloat/e5-mistral-7b-instruct",
#                         task="embed",
#                         enforce_eager=True)
#     args = parser.parse_args()
#     main(args)


from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser

from argparse import Namespace


def main(args: Namespace):
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    model = LLM(**vars(args))
    outputs = model.embed(prompts)

    for prompt, output in zip(prompts, outputs):
        print(f"prompt:{prompt}")
        print(f"embed:{output.outputs.embedding}")

        print("*" * 60)


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="intfloat/e5-mistral-7b-instruct", task="embed", enforce_eager=True)
    args = parser.parse_args()
    main(args)

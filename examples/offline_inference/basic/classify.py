# SPDX-License-Identifier: Apache-2.0

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
#     # You should pass task="classify" for classification models
#     model = LLM(**vars(args))

#     # Generate logits. The output is a list of ClassificationRequestOutputs.
#     outputs = model.classify(prompts)

#     # Print the outputs.
#     print("\nGenerated Outputs:\n" + "-" * 60)
#     for prompt, output in zip(prompts, outputs):
#         probs = output.outputs.probs
#         probs_trimmed = ((str(probs[:16])[:-1] +
#                           ", ...]") if len(probs) > 16 else probs)
#         print(f"Prompt: {prompt!r} \n"
#               f"Class Probabilities: {probs_trimmed} (size={len(probs)})")
#         print("-" * 60)


# if __name__ == "__main__":
#     parser = FlexibleArgumentParser()
#     parser = EngineArgs.add_cli_args(parser)
#     # Set example specific arguments
#     parser.set_defaults(model="jason9693/Qwen2.5-1.5B-apeach",
#                         task="classify",
#                         enforce_eager=True)
#     args = parser.parse_args()
#     main(args)
























from vllm.utils import FlexibleArgumentParser
from vllm import LLM, EngineArgs

from argparse import Namespace



def main(parser:Namespace):
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    model = LLM(**vars(parser))
    
    outputs = model.classify(prompts)
    
    for prompt, output in zip(prompts, outputs):
        print(f"prompt:{prompt}")
        print(f"prob:{output.outputs.probs}")
    pass







if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="jason9693/Qwen2.5-1.5B-apeach", task="classify", enforce_eager=True)
    args = parser.parse_args()
    main(args)
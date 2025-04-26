# import ray

# ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")
# print(ds.take(5))  # 打印前 5 行

# from vllm import LLM, SamplingParams
# from vllm.inputs import TokensPrompt, TextPrompt

# llm = LLM(
#     model="mistralai/Mistral-7B-Instruct-v0.2"
# )

# tokenizer = llm.llm_engine.get_tokenizer_group()

# prompt_text = "The capital of France is"
# token_ids = tokenizer.encode(prompt_text)

# prompt_obj = TokensPrompt(prompt_token_ids=token_ids)
# print(prompt_obj)

# prompt = TextPrompt(prompt="The president of the United States is")
# print(prompt)

from dataclasses import asdict, dataclass

@dataclass
class Person:
    name: str
    age: int
    address:str

person = Person(name="525", age=65, address="8956")
print(asdict(person))

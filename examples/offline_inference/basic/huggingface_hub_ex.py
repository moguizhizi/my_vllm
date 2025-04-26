# from huggingface_hub import hf_hub_download

# repo_id = "bartowski/Phi-3-medium-4k-instruct-GGUF"
# filename = "Phi-3-medium-4k-instruct-IQ2_M.gguf"

# print(hf_hub_download(repo_id, filename=filename))


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

eagle_dir = "jinaai/jina-embeddings-v3"

tokenizer = AutoTokenizer.from_pretrained(eagle_dir)
model = AutoModelForCausalLM.from_pretrained(eagle_dir)



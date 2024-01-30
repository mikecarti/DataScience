import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer, pipeline


class LLM:
    def __init__(self, model_id: str = None, bnb_config=None):
        if model_id is None:
            model_id = "openlm-research/open_llama_3b"
        if bnb_config is None:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0})
        self.model = pipeline('text-generation', model=model, tokenizer=self.tokenizer, max_new_tokens=20)
        print(f"Model {model_id} loaded successfully")

    def generate(self, input_: str) -> str:
        return self.model(input_)[0].get("generated_text")


class LLMAPI:
    """
    Just for testing by myself, not used in the code
    """

    def __init__(self):
        from openai import OpenAI
        import os
        os.environ["OPENAI_API_KEY"] = ""

        self.client = OpenAI()

    def generate(self, input_: str) -> str:
        message = [{"role": "assistant", "content": ""}, {"role": "user", "content": input_}]
        temperature = 0.2
        max_tokens = 256
        frequency_penalty = 0.0

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty
        )
        return response.choices[0].message.content

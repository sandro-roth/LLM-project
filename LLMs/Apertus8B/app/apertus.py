from typing import Optional, List

from langchain_core.language_models import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM


from utils import timeit
from utils import setup_logging

setup_logging(app_name='apertus-inference', to_stdout=True, retention=30)

class ApertusInferenceLLM(LLM):
    def __init__(self, model_path:str, tokenizer_path:str, temperatur:float, top_p:float, max_tokens:int):
        super().__init__()
        object.__setattr__(self, "_model", AutoModelForCausalLM.from_pretrained(model_path))
        object.__setattr__(self, "_tokenizer", AutoTokenizer.from_pretrained(tokenizer_path))
        object.__setattr__(self, "_temperatur", temperatur)
        object.__setattr__(self, "_top_p", top_p)
        object.__setattr__(self, "_max_tokens", max_tokens)

        object.__setattr__(self, "_systemmessage", {
            'role': 'system', 'content':'Du bist ein präziser, detailorientierter medizinischer Schreibassistent.'
        })

    @timeit
    def _call(self):
        pass

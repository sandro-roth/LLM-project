from langchain_core.language_models import LLM

from transformers import AutoTokenizer, AutoModelForCausalLM



class MeditronInstanceLLM(LLM):
    def __init__(self, model_path: str, tokenizer_path: str):
        super.__init__()
        object.__setattr__(self, '_model', AutoModelForCausalLM.from_pretrained(model_path))
        object.__setattr__(self, '_tokenizer', AutoTokenizer.from_pretrained(model_path))

    def __str__(self):
        return self._model
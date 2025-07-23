from langchain_core.language_models import LLM

from transformers import AutoTokenizer, AutoModelForCausalLM



class MeditronInstanceLLM(LLM):
    def __init__(self, model_path: str, temperature: float=0.0, max_tokens: int = 1000):
        super().__init__()
        object.__setattr__(self, '_model', AutoModelForCausalLM.from_pretrained(model_path))
        object.__setattr__(self, '_tokenizer', AutoTokenizer.from_pretrained(model_path))
        object.__setattr__(self, '_temperature', temperature)
        object.__setattr__(self, '_max_tokens', max_tokens)



    @property
    def _llm_type(self) -> str:
        return 'meditron-7b'



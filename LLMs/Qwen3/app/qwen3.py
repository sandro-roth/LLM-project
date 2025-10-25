from typing import Optional, List, ClassVar, Iterator
from pathlib import Path

from langchain_core.language_models import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import threading


from utils import timeit
from utils import setup_logging

LOGGER = setup_logging(app_name='qwen-inference', to_stdout=True, retention=30)

class ApertusInferenceLLM(LLM):
    device: ClassVar[str] = 'cuda'
    def __init__(self, model_path:Path, tokenizer_path:Path, temperature:float, top_p:float, max_tokens:int):
        super().__init__()
        object.__setattr__(self, "_model", AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(self.device))
        object.__setattr__(self, "_tokenizer", AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True))
        object.__setattr__(self, "_temperature", temperature)
        object.__setattr__(self, "_top_p", top_p)
        object.__setattr__(self, "_max_tokens", max_tokens)

        object.__setattr__(self, "_systemmessage", {
            'role': 'system', 'content':'Du bist ein präziser, detailorientierter medizinischer Schreibassistent.'
        })

    @property
    def _llm_type(self) -> str:
        return 'qwen-inference'


    def _effective_params(self, temperature:Optional[float], top_p:Optional[float], max_tokens:Optional[int]):
        """ helper function for _call and stream """
        temp = self._temperature if temperature is None else float(temperature)
        nucleus = self._top_p if top_p is None else float(top_p)
        max_new = self._max_tokens if max_tokens is None else int(max_tokens)
        return temp, nucleus, max_new, (temp > 0.0)


    def _build_inputs(self, prompt: str, system_prompt: Optional[str]):
        """ helper function for _call and stream """
        sys_msg = {'role': 'system', 'content': system_prompt} if system_prompt else self._systemmessage
        messages = [sys_msg, {'role': 'user', 'content': prompt}]
        return self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt'
        ).to(self._model.device)
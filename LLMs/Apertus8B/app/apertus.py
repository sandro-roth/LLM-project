from typing import Optional, List, ClassVar
from pathlib import Path

from langchain_core.language_models import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


from utils import timeit
from utils import setup_logging

LOGGER = setup_logging(app_name='apertus-inference', to_stdout=True, retention=30)

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
        return 'apertus-inference'

    @timeit
    def _call(self,prompt: str, system_prompt: Optional[str] = None, stop: Optional[List[str]] = None) -> str:
        system_message = {'role': 'system', 'content': system_prompt} if system_prompt else self._systemmessage
        messages = [
            system_message,
            {'role': 'user', 'content': prompt}
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors='pt'
        ).to(self._model.device)

        LOGGER.info(f'Sampling parameters: max_tokens = {self._max_tokens}, temperature = {self._temperature}, top_p = {self._top_p}')
        do_sample = self._temperatur > 0.0
        with torch.no_grad():
            outputs = self._model.generate(**inputs,
                                           max_new_tokens=self._max_tokens,
                                           temperature=self._temperature,
                                           top_p=self._top_p,
                                           do_sample=do_sample,
                                           pad_token_id=self._tokenizer.eos_token_id)

        

        # LOGGER.warning for stop condition
        # define stop


        #decoded_output = self._tokenizer.decode(outputs[0][inputs["inputs_ids"].shape[-1]:])
        decoded_output = outputs[0][len(inputs.input_ids[0]):]
        return decoded_output

    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return self._call(prompt=prompt, system_prompt=system_prompt)
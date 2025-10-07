from typing import Optional, List, ClassVar, Iterator
from pathlib import Path

from langchain_core.language_models import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import threading


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


    def _gen_kwargs(self, inputs, max_new: int, temp: float, nucleus: float, do_sample: bool, *, streamer=None):
        return dict(
            **inputs,
            max_new_tokens=max_new,
            temperature=temp,
            top_p=nucleus,
            do_sample=do_sample,
            pad_token_id=self._tokenizer.eos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            **({"streamer": streamer} if streamer is not None else {})
        )


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
        do_sample = self._temperature > 0.0
        with torch.no_grad():
            outputs = self._model.generate(**inputs,
                                           max_new_tokens=self._max_tokens,
                                           temperature=self._temperature,
                                           top_p=self._top_p,
                                           do_sample=do_sample,
                                           pad_token_id=self._tokenizer.eos_token_id,
                                           eos_token_id=self._tokenizer.eos_token_id)


        # only newly generated tokens
        input_len = inputs.input_ids.shape[-1]
        new_tokens = outputs[0][input_len:]

        # LOGGER.warning for stop condition
        # define stop

        decoded_output = self._tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return decoded_output

    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return self._call(prompt=prompt, system_prompt=system_prompt)

    @timeit
    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        """
        Gibt inkrementell Textstücke zurück (Token/Chunks), sobald sie generiert werden.
        Verwendet HuggingFace TextIteratorStreamer.
        """
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

        # Streamer: prompt wird nicht wiederholt, Sondersymbole werden übersprungen
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        do_sample = self._temperature > 0.0
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            do_sample=do_sample,
            pad_token_id=self._tokenizer.eos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            streamer=streamer
        )

        # generate in separatem Thread starten; Haupt-Thread liest aus streamer
        def _worker():
            with torch.no_grad():
                self._model.generate(**generate_kwargs)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        try:
            for text in streamer:
                # text kann 1+ Token enthalten; gib es direkt weiter
                if text:
                    yield text
        finally:
            # Aufräumen/Join ist optional (Streamer beendet sich)
            t.join(timeout=0.1)
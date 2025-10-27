from typing import Optional, List, ClassVar, Iterator
from pathlib import Path
import os

from langchain_core.language_models import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import threading


from utils import timeit
from utils import setup_logging

LOGGER = setup_logging(app_name='qwen-inference', to_stdout=True, retention=30)

def _primary_device_of(model: torch.nn.Module) -> torch.device:
    for p in model.parameter():
        return p.device
    return torch.device('cpu')

class ApertusInferenceLLM(LLM):
    device: ClassVar[str] = 'auto'

    def __init__(self, model_path:Path, tokenizer_path:Path, temperature:float,
                 top_p:float, max_tokens:int, offload_folder:Path):

        super().__init__()

        model_id = str(model_path)
        tok_id = str(tokenizer_path)

        # Config from .env
        use_4bit = os.getenv('LOAD_IN_4BIT', 'false').lower() == 'true'
        use_8bit = os.getenv('LOAD_IN_8BIT', 'false').lower() == 'true' and not use_4bit
        torch_dtype = torch.bfloat16 if os.getenv('TORCH_DTYPE', 'bf16').lower() in ("bf16","bfloat16") else torch.float16

        # Memory budget
        per_gpu_gib = int(os.getenv('MAX_VRAM_PER_GPU', '45'))
        cpu_gib = int(os.getenv('CPU_RAM_BUDGET_BIG', '180'))

        # device_map + max_memory construct
        n_gpu = torch.cuda.device_count()
        max_memory = {f"cuda:{i}": f"{per_gpu_gib}GiB" for i in range(n_gpu)}
        max_memory["cpu"] = f"{cpu_gib}GiB"

        os.makedirs(offload_folder, exist_ok=True)

        # ------------------ Quantization --------------------------
        nb_config = None
        load_kwargs = {}
        if use_4bit:
            pass
        elif use_8bit:
            pass
        else:
            load_kwargs['torch_dtype'] = torch_dtype

        # Full model downloaded already make sure to only local processing
        local_only = os.getenv('HF_LOCAL_ONLY', 'true').lower() == 'true'

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tok_id, local_files_only=local_only, use_fast=True, trust_remote_code=True,)

        # Model with offload
        model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=local_only,
            device_map="auto", max_memory=max_memory, offload_folder=str(offload_folder),
            low_cpu_mem_usage=True, trust_remote_code=True, **load_kwargs)
        model.eval()

        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(self, "_temperature", temperature)
        object.__setattr__(self, "_top_p", top_p)
        object.__setattr__(self, "_max_tokens", max_tokens)

        object.__setattr__(self, "_systemmessage", {
            'role': 'system', 'content':'Du bist ein präziser, detailorientierter medizinischer Schreibassistent.'
        })

        # Log
        try:
            dev = _primary_device_of(self._model)
            LOGGER.info(f"Model primary device: {dev}; n_gpu={n_gpu}; max_memory={max_memory}")
        except Exception as e:
            LOGGER.warning(f"Could not infer primary device: {e}")


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


    def _stream_chunks(self, prompt: str, system_prompt: Optional[str],
            *, temperature: Optional[float], top_p: Optional[float], max_tokens: Optional[int]) -> Iterator[str]:
        temp, nucleus, max_new, do_sample = self._effective_params(temperature, top_p, max_tokens)
        LOGGER.info(f"Sampling: max_new_tokens={max_new}, temperature={temp}, top_p={nucleus}")
        inputs = self._build_inputs(prompt, system_prompt)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        generate_kwargs = self._gen_kwargs(inputs, max_new, temp, nucleus, do_sample, streamer=streamer)

        def _worker():
            with torch.no_grad():
                self._model.generate(**generate_kwargs)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        try:
            for text in streamer:
                if text:
                    yield text
        finally:
            t.join(timeout=0.1)


    @timeit
    def _call(
        self, prompt:str, system_prompt:Optional[str]=None, stop:Optional[List[str]]=None,
        *, temperature:Optional[float]=None, top_p:Optional[float]=None, max_tokens:Optional[int]=None
    ) -> str:

        # definiere stop Kriterium

        # nutze denselben Streamer unter der Haube, aber sammle die Chunks
        parts = []
        for chunk in self._stream_chunks(
            prompt, system_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens
        ):
            parts.append(chunk)
        return "".join(parts)


    def invoke(self, prompt: str, system_prompt: Optional[str] = None,
               *, temperature: Optional[float] = None, top_p: Optional[float] = None,
               max_tokens: Optional[int] = None) -> str:
        return self._call(prompt, system_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)


    @timeit
    def stream(self, prompt:str, system_prompt:Optional[str]=None,
               *, temperature:Optional[float]=None, top_p:Optional[float]=None, max_tokens:Optional[int]=None) -> Iterator[str]:
        yield from self._stream_chunks(
            prompt, system_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
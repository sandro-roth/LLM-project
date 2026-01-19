from typing import Optional, List, ClassVar, Iterator
from pathlib import Path
import threading

from langchain_core.language_models import LLM

from llama_cpp import Llama

from utils import timeit
from utils import setup_logging

LOGGER = setup_logging(app_name='apertus70-inference', to_stdout=True, retention=30)

class LLM_inference(LLM):
    device: ClassVar[str] = "cuda"
    def __init__(self, model_path: Path, temperature: float, top_p: float, max_tokens: int,
                 n_ctx: int = 8192, n_gpu_layers: int = -1):
        super().__init__()
        object.__setattr__(self, '_llm', Llama(model_path=str(model_path), n_ctx=int(n_ctx),
                                               n_gpu_layers=int(n_gpu_layers), verbose=False))
        object.__setattr__(self, '_temperature', float(temperature))
        object.__setattr__(self, '_top_p', float(top_p))
        object.__setattr__(self, '_max_tokens', int(max_tokens))
        object.__setattr__(self, '_systemmessage', "Du bist ein präziser, detailorientierter medizinischer Schreibassistent.")
        object.__setattr__(self, "_lock", threading.Lock())

    @property
    def _llm_type(self):
        return "apertus70-llama_cpp"

    def _effective_params(self, temperature: Optional[float], top_p: Optional[float], max_tokens: Optional[int]):
        temp = self._temperature if temperature is None else float(temperature)
        nucleus = self._top_p if top_p is None else float(top_p)
        max_new = self._max_tokens if max_tokens is None else int(max_tokens)
        do_sample = temp > 0.0
        return temp, nucleus, max_new, do_sample

    def _build_prompt(self, prompt: str, system_prompt: Optional[str]) -> str:
        sys_text = (system_prompt or self._systemmessage).strip()
        user_text = (prompt or "").strip()

        return (
            f"[SYSTEM]\n{sys_text}\n[/SYSTEM]\n\n"
            f"[USER]\n{user_text}\n[/USER]\n\n"
            f"[ASSISTANT]\n"
        )

    def _stream_chunks(self, prompt: str, system_prompt: Optional[str],
                       *, temperature: Optional[float], top_p: Optional[float],
                       max_tokens: Optional[int]) -> Iterator[str]:
        temp, nucleus, max_new, _ = self._effective_params(temperature, top_p, max_tokens)
        full_prompt = self._build_prompt(prompt, system_prompt)

        LOGGER.info(f'Sampling: max_tokens={max_new}, temperature={temp}, top_p={nucleus}')

        # Single GPU, big model: serialize generations to avoid thrashing
        with self._lock:
            for chunk in self._llm(full_prompt, max_tokens=max_new, temperature=temp,
                    top_p=nucleus, stream=True,
            ):
                text = chunk["choices"][0]["text"] or ""
                if text:
                    yield text


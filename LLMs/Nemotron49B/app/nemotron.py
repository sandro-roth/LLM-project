from typing import Optional, List, ClassVar, Iterator
from pathlib import Path
import threading

from langchain_core.language_models import LLM

from llama_cpp import Llama

from utils import timeit
from utils import setup_logging

LOGGER = setup_logging(app_name='nemotron49B-inference', to_stdout=True, retention=30)

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
        return "nemotron49B-llama_cpp"

    def _effective_params(self, temperature: Optional[float], top_p: Optional[float], max_tokens: Optional[int]):
        temp = self._temperature if temperature is None else float(temperature)
        nucleus = self._top_p if top_p is None else float(top_p)
        if temp == 0.0:
            nucleus = 1.0
        max_new = self._max_tokens if max_tokens is None else int(max_tokens)
        do_sample = temp > 0.0
        return temp, nucleus, max_new, do_sample

    def _build_messages(self, prompt: str, system_prompt: Optional[str]) -> list[dict]:
        sys_text = (system_prompt or self._systemmessage).strip()
        user_text = (prompt or "").strip()

        if "Korrigiere den folgenden Text" in sys_text:
            sys_text = "/no_think\n" + sys_text

        return [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_text},
        ]

    def _stream_chunks(self, prompt: str, system_prompt: Optional[str],
                       *, temperature: Optional[float], top_p: Optional[float],
                       max_tokens: Optional[int]) -> Iterator[str]:
        temp, nucleus, max_new, _ = self._effective_params(temperature, top_p, max_tokens)
        messages = self._build_messages(prompt, system_prompt)

        LOGGER.info(f"Sampling: max_tokens={max_new}, temperature={temp}, top_p={nucleus}")
        stop = ["</think>", "<|eot_id|>", "<|end_of_text|>"]

        # Single GPU, big model: serialize generations to avoid thrashing
        with self._lock:
            for chunk in self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_new,
                    temperature=temp,
                    top_p=nucleus,
                    stream=True,
                    stop=stop
            ):
                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content") or ""
                if text:
                    yield text

    @timeit
    def _call(self, prompt: str, system_prompt: Optional[str] = None, stop: Optional[List[str]] = None,
              *, temperature: Optional[float] = None, top_p: Optional[float] = None,
              max_tokens: Optional[int] = None) -> str:
        parts = []
        for chunk in self._stream_chunks(prompt, system_prompt, temperature=temperature,
                                         top_p=top_p, max_tokens=max_tokens):
            parts.append(chunk)
        return "".join(parts)

    def invoke(self, prompt: str, system_prompt: Optional[str] = None, *, temperature: Optional[float] = None,
               top_p: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        return self._call(prompt, system_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    @timeit
    def stream(self, prompt: str, system_prompt: Optional[str] = None, *, temperature: Optional[float] = None,
               top_p: Optional[float] = None, max_tokens: Optional[int] = None) -> Iterator[str]:
        yield from self._stream_chunks(prompt, system_prompt, temperature=temperature,
                                       top_p=top_p, max_tokens=max_tokens)
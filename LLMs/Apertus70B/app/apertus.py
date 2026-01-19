from typing import Optional, List, ClassVar
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



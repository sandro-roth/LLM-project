from typing import Optional, Generator
from pathlib import Path

from app import LLM_inference
from sympy.physics.units import temperature

from LLMs.Mistral7B.app.server import model_path

BASE_DIR = Path(__file__).resolve().parent.parent
model_file = Path(BASE_DIR / 'model8bit' / 'swiss-ai_Apertus-70B-Instruct-2509-Q8_0')

llm = LLM_inference(
    model_path=model_file,
    temperature=0.8,
    top_p=0.9,
    max_tokens=200,
    n_ctx=8192,
    n_gpu_layers=-1
)
from typing import Optional, Generator
from pathlib import Path

from pydantic import BaseModel
from fastapi import FastAPI

from app import LLM_inference


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

app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

class PromptRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None

class ConfigOut(BaseModel):
    model: str = "Apertus70B-8Bit"
    defaults: dict


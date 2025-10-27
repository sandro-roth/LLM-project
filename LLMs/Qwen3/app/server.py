from typing import Optional, Generator
from pathlib import Path
import os
import json

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app import ApertusInferenceLLM

BASE_DIR = Path(__file__).resolve().parent.parent

# Paths can be set in .env
model_dir = Path(os.getenv('MODEL_DIR') or (BASE_DIR / 'model_inference'))
token_dir = Path(os.getenv("TOKEN_DIR", str(model_dir)))
offload_dir = Path(os.getenv("OFFLOAD_FOLDER") or (BASE_DIR / "offload"))
offload_dir.mkdir(parents=True, exist_ok=True)


llm = ApertusInferenceLLM(
    model_path=model_dir,
    tokenizer_path=token_dir,
    temperature=0.8,
    top_p=0.9,
    max_tokens=200,
    offload_folder=offload_dir
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
    model: str = 'Qwen3'
    defaults: dict


@app.post("/generate")
def generate_text(request:PromptRequest):
    response = llm.invoke(prompt=request.prompt,
                          system_prompt=request.system_prompt,
                          temperature=request.temperature,
                          top_p=request.top_p,
                          max_tokens=request.max_tokens)
    return {"response": response}


@app.get("/config")
def get_config() -> ConfigOut:
    return ConfigOut(
        defaults={
            'temperature':llm._temperature,
            'top_p':llm._top_p,
            'max_tokens':llm._max_tokens,
        }
    )
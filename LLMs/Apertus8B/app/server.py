from typing import Optional
from pathlib import Path

from pydantic import BaseModel
from fastapi import FastAPI

from app import ApertusInferenceLLM

BASE_DIR = Path(__file__).resolve().parent.parent
model_dir = Path(BASE_DIR / "base_model")
token_dir = model_dir

llm = ApertusInferenceLLM(
    model_path=model_dir,
    tokenizer_path=token_dir,
    temperature=0.8,
    top_p=0.9,
    max_tokens=200
)

app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

class PromptRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None

@app.post("/generate")
def generate_text(request:PromptRequest):
    response = llm.invoke(prompt=request.prompt, system_prompt=request.system_prompt)
    return {"response": response}
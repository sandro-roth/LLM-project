from typing import Optional, Generator
import os
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import TransformersLLM

MODEL_ID = os.getenv("MODEL_ID", "/models/current")
MODEL_NAME = os.getenv("MODEL_NAME", "TransformersModel")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "bfloat16")
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"
HF_LOCAL_ONLY = os.getenv("HF_LOCAL_ONLY", "true").lower() == "true"

llm = TransformersLLM(
    model_id_or_path=MODEL_ID,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS,
    torch_dtype=TORCH_DTYPE,
    trust_remote_code=TRUST_REMOTE_CODE,
    local_files_only=HF_LOCAL_ONLY
)

app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

class PromptRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    disable_think: Optional[bool] = False

class ConfigOut(BaseModel):
    model: str
    defaults: dict

@app.post("/generate")
def generate_text(request: PromptRequest):
    response = llm.invoke(
        prompt=request.prompt,
        system_prompt=request.system_prompt,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        disable_think=bool(request.disable_think),
    )
    return {"response": response}

def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

@app.post("/generate_stream")
def generate_text_stream(request: PromptRequest):
    def token_generator() -> Generator[bytes, None, None]:
        try:
            for tok in llm.stream(
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                disable_think=bool(request.disable_think),
            ):
                if tok:
                    yield sse_event({"token": tok}).encode("utf-8")

            yield sse_event({"finished": True}).encode("utf-8")

        except GeneratorExit:
            return
        except Exception as e:
            yield sse_event({"error": str(e)}).encode("utf-8")

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/config")
def get_config() -> ConfigOut:
    return ConfigOut(
        model=MODEL_NAME,
        defaults={
            "temperature": llm.temperature,
            "top_p": llm.top_p,
            "max_tokens": llm.max_tokens,
        },
    )

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}
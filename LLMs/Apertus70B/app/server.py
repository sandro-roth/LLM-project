from typing import Optional, Generator
from pathlib import Path
import json

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.response import StreamingResponse

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

@app.post("/generate")
def generate_text(request: PromptRequest):
    response = llm.invoke(
        prompt=request.prompt,
        system_prompt=request.system_prompt,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens
    )
    return {"response": response}

def sse_event(data: dict) -> str:
    return f'data: {json.dumps(data, ensure_ascii=False)}\n\n'

@app.post("/generate_stream")
def generate_text_stream(request: PromptRequest):
    def token_generator() -> Generator[bytes, None, None]:
        try:
            for tok in llm.stream(
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens
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
            "X-Accel-Buffering": "no"
        }
    )
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from app import MistralInferenceLLM

model_path = 'mistral-7B-Instruct-v0.3'
tokenizer_path = 'mistral-7B-Instruct-v0.3/tokenizer.model.v3'

llm = MistralInferenceLLM(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        temperature=0.0,
        max_tokens=1800
)

app = FastAPI(
        docs_url=None,
        redoc_url=None,
        openapi_url=None)

class PromptRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None

@app.post("/generate")
def generate_text(request:PromptRequest):
    response = llm.invoke(prompt=request.prompt, system_prompt=request.system_prompt)
    return {"response": response}

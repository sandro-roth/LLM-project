from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

from app import MeditronInstanceLLM


model_path = Path(__file__).resolve().parents[1] / 'model'
#model_path = Path("/model")

llm = MeditronInstanceLLM(
    model_path=model_path
)

app = FastAPI(
    title='Meditron LLM API',
    version='1.0.0',
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

class PromptRequest(BaseModel):
    prompt: str
    # change to take optional system_prompt

@app.post("/generate")
def generate_text(request:PromptRequest):
    response=llm.invoke(request.prompt)
    # change to take optional system_prompt
    return {'response': response}
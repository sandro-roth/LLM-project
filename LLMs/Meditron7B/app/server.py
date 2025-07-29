from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

from meditron import MeditronInstanceLLM


model_path = Path(__file__).resolve().parents[1] / 'model'

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

@app.post("/generate")
def generate_text(request:PromptRequest):
    response=llm.invoke(request.prompt)
    return {'response': response}

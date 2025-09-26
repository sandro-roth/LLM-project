from typing import Optional

from langchain_core.language_models import LLM


from utils import timeit
from utils import setup_logging

setup_logging(app_name='apertus-inference', to_stdout=True, retention=30)

class ApertusInferenceLLM(LLM):
    def __init__(self):
        pass

    @timeit
    def _call(self):
        pass
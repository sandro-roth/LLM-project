from langchain_core.language_models import LLM
from langchain.prompts import PromptTemplate

from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
from pathlib import Path

class MeditronInstanceLLM(LLM):
    def __init__(self, model_path: str, temperature: float=0.0, max_tokens: int = 1000):
        super().__init__()
        object.__setattr__(self, '_model', AutoModelForCausalLM.from_pretrained(model_path,
                                                                                low_cpu_mem_usage=True,
                                                                                device_map='cuda:0',
                                                                                local_files_only=True))
        object.__setattr__(self, '_tokenizer', AutoTokenizer.from_pretrained(model_path), local_files_only=True)
        object.__setattr__(self, '_temperature', temperature)
        object.__setattr__(self, '_max_tokens', max_tokens)


    @property
    def _llm_type(self) -> str:
        return 'meditron-7b'

    def _call(self, prmt: str, s_msg: Optional[str] = None) -> str:

        if s_msg is None:
            s_msg = (
            "Du bist ein präziser, detailorientierter medizinischer Schreibassistent. "
            "Du erstellst strukturierte medizinische Berichte aus Untersuchungsnotizen. "
            "Verwende immer klare Abschnitte: Patientendaten, Klinische Befunde, Diagnose, Empfehlungen. "
            "Nutze korrekte medizinische Terminologie und vermeide Vermutungen, wenn Daten fehlen. "
            "Du antwortest ausschliesslich in korrektem, fachlich präzisem Deutsch."
            )

        p_temp = PromptTemplate.from_template(
            "<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{prompt_lines} [/INST]"
        )

        full_prompt = p_temp.format(system_message=s_msg, prompt_lines=prmt)
        return self._tokenizer(full_prompt, return_tensor='pt')


    def invoke(self, prmt:str) -> str:
        return self._call(prmt)

if __name__ == '__main__':
    m_path = Path(__file__).resolve().parents[1] / 'model'
    Instance = MeditronInstanceLLM(m_path)
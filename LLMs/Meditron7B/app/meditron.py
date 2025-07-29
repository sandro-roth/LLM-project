from langchain_core.language_models import LLM
from langchain.prompts import PromptTemplate

from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from pathlib import Path

class MeditronInstanceLLM(LLM):
    def __init__(self, model_path: Path, temperature: float=0.0, max_tokens: int = 1000):
        super().__init__()
        object.__setattr__(self, '_model', AutoModelForCausalLM.from_pretrained(model_path,
                                                                                low_cpu_mem_usage=True,
                                                                                torch_dtype=torch.float16,
                                                                                device_map='cuda:0',
                                                                                local_files_only=True))
        object.__setattr__(self, '_tokenizer', AutoTokenizer.from_pretrained(model_path, local_files_only=True))
        object.__setattr__(self, '_temperature', temperature)
        object.__setattr__(self, '_max_tokens', max_tokens)


    @property
    def _llm_type(self) -> str:
        return 'meditron-7b'

    def _call(self, prmt: str, s_msg: Optional[str] = None) -> str:

        if s_msg is None:
            s_msg = (
                "Du bist ein medizinischer Schreibassistent mit klinischer Expertise. Deine Aufgabe ist es, aus kurzen "
                "Untersuchungshinweisen und Symptombeschreibungen präzise, strukturierte medizinische Berichte zu erstellen.\n\n"
                "Der Bericht muss stets folgende Abschnitte enthalten:\n"
                "1. Patientendaten\n"
                "2. Klinische Befunde\n"
                "3. Diagnose\n"
                "4. Empfehlungen\n\n"
                "Beachte dabei:\n"
                "- Verwende korrekte, standardisierte medizinische Terminologie.\n"
                "- Berichte sachlich, fachlich präzise und ohne Interpretation über den gegebenen Informationsstand hinaus.\n"
                "- Falls Informationen fehlen, gib keine Mutmassungen an, sondern dokumentiere dies entsprechend "
                "(\"keine Angaben\" oder \"nicht dokumentiert\").\n"
                "- Achte auf klare, gut gegliederte Sprache im Stil ärztlicher Dokumentation.\n"
                "- Schreibe ausschliesslich auf Deutsch.\n\n"
                "Beispielhafte Einträge:\n\n"
                "Patientendaten:\n"
                "- Alter: 56 Jahre\n"
                "- Geschlecht: männlich\n\n"
                "Klinische Befunde:\n"
                "- ...\n\n"
                "Diagnose:\n"
                "- ...\n\n"
                "Empfehlungen:\n"
                "- ..."
            )

        #full_prompt = f"<s>[INST] <<SYS>>\n{s_msg}\n<</SYS>>\n\n{prmt} [/INST]\n"
        full_prompt = f"{s_msg}\n\n{prmt}"

        device = next(self._model.parameters()).device
        input = self._tokenizer(full_prompt, return_tensors='pt').to(device)

        output = self._model.generate(
            **input,
            max_new_tokens=self._max_tokens,
            temperature=self._temperature,
            do_sample=self._temperature > 0.0,
            top_p=0.95,
            pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            eos_token_id=self._tokenizer.eos_token_id
        )

        decode = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return decode

    def invoke(self, prmt:str) -> str:
        return self._call(prmt)

if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1] / 'model'
    Instance = MeditronInstanceLLM(path)

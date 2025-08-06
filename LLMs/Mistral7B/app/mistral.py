import os

from langchain_core.language_models import LLM
from typing import Optional, List, Any, Mapping

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

class MistralInferenceLLM(LLM):
    def __init__(self, model_path: str, tokenizer_path: str, temperature: float = 0.0, max_tokens: int = 1000):
        super().__init__()
        object.__setattr__(self, "_model", Transformer.from_folder(model_path))
        object.__setattr__(self, "_tokenizer",  MistralTokenizer.from_file(tokenizer_path))
        object.__setattr__(self, "_temperature", temperature)
        object.__setattr__(self, "_max_tokens", max_tokens)


        # Default system message (Can also be pass externally)
        object.__setattr__(self, "_system_message", SystemMessage(
                role='system',
                content=(
                    "Du bist ein präziser, detailorientierter medizinischer Schreibassistent."
                    "Du erstellst strukturierte medizinische Berichte aus Untersuchungsnotizen."
                    "Verwende immer klare Abschnitte: Patientendaten, Klinische Befunde, Diagnose, Empfehlungen."
                    "Nutze korrekte medizinische Terminologie und vermeide Vermutungen, wenn Daten fehlen."
                    "Du antwortest ausschliesslich in korrektem, fachlich präzisem Deutsch."
                )
        ))

    @property
    def _llm_type(self) -> str:
        return 'mistral_inference'

    def _call(self, prompt: str, system_prompt: Optional[str] = None, stop: Optional[List[str]] = None) -> str:
        system_message = SystemMessage(role='system', content=system_prompt) if system_prompt else self._system_message

        user_message = UserMessage(role='user', content=prompt)
        completion_request = ChatCompletionRequest(messages=[system_message, user_message])
        tokens = self._tokenizer.encode_chat_completion(completion_request).tokens

        eos_id = self._tokenizer.instruct_tokenizer.tokenizer.eos_id

        # Text generieren
        output = generate([tokens],
                          self._model,
                          max_tokens=self._max_tokens,
                          temperature=self._temperature,
                          eos_id=eos_id)

        decoded_output = self._tokenizer.decode(output[0])
        if isinstance(decoded_output, list):
            decoded_output = ''.join(str(x) for x in decoded_output)

        stop_words = stop or ["\n\n", "###", "ENDE"]
        for stop_word in stop_words:
            if stop_word in decoded_output:
                decoded_output = decoded_output.split(stop_word)[0]
                break

        return decoded_output.strip()


    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return self._call(prompt=prompt, system_prompt=system_prompt)

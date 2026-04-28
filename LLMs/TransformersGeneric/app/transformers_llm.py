from typing import Optional, List, ClassVar, Iterator
import threading
import torch

from langchain_core.language_models import LLM

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

from utils import timeit
from utils import setup_logging


LOGGER = setup_logging(
    app_name="transformers-inference",
    to_stdout=True,
    retention=30,
)


class TransformersLLM(LLM):
    device: ClassVar[str] = "cuda"

    def __init__(
        self,
        model_id_or_path: str,
        temperature: float = 0.8,
        top_p: float = 0.9,
        max_tokens: int = 512,
        system_message: str = "Du bist ein präziser, detailorientierter medizinischer Schreibassistent.",
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        local_files_only: bool = True,
    ):
        super().__init__()

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "auto": "auto",
        }

        dtype = dtype_map.get(torch_dtype.lower(), torch.bfloat16)

        LOGGER.info(f"Loading tokenizer from: {model_id_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )

        try:
            LOGGER.info(f"Loading processor from: {model_id_or_path}")
            processor = AutoProcessor.from_pretrained(
                model_id_or_path,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            LOGGER.warning(f"No processor loaded: {exc}")
            processor = None

        LOGGER.info(
            f"Loading model from: {model_id_or_path}, dtype={torch_dtype}, local_files_only={local_files_only}"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )

        model.eval()

        object.__setattr__(self, "model_id_or_path", model_id_or_path)
        object.__setattr__(self, "tokenizer", tokenizer)
        object.__setattr__(self, "processor", processor)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "_temperature", float(temperature))
        object.__setattr__(self, "_top_p", float(top_p))
        object.__setattr__(self, "_max_tokens", int(max_tokens))
        object.__setattr__(self, "_systemmessage", system_message)
        object.__setattr__(self, "_lock", threading.Lock())

    @property
    def _llm_type(self):
        return "transformers-causal-lm"

    def _effective_params(self, temperature: Optional[float], top_p: Optional[float], max_tokens: Optional[int]):
        temp = self._temperature if temperature is None else float(temperature)
        nucleus = self._top_p if top_p is None else float(top_p)
        max_new = self._max_tokens if max_tokens is None else int(max_tokens)

        do_sample = temp > 0.0
        if not do_sample:
            nucleus = 1.0

        return temp, nucleus, max_new, do_sample

    def _build_messages(self, prompt: str, system_prompt: Optional[str], disable_think: bool) -> list[dict]:
        sys_text = (system_prompt or self._systemmessage).strip()
        user_text = (prompt or "").strip()

        if disable_think:
            sys_text = "/no_think\n" + sys_text

        return [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_text},
        ]

    def _tokenize_messages(self, messages):
        tokenizer_or_processor = self.processor or self.tokenizer

        has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))

        if hasattr(tokenizer_or_processor, "apply_chat_template") and has_chat_template:
            inputs = tokenizer_or_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            system_text = messages[0]["content"].strip()
            user_text = messages[1]["content"].strip()

            text = (
                f"System:\n{system_text}\n\n"
                f"User:\n{user_text}\n\n"
                f"Assistant:\n"
            )

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
            )

        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _remove_thinking(self, text: str) -> str:
        while "<think>" in text and "</think>" in text:
            start = text.find("<think>")
            end = text.find("</think>") + len("</think>")
            text = text[:start] + text[end:]
        return text

    @timeit
    def _call(self, prompt: str, system_prompt: Optional[str] = None, *, temperature: Optional[float] = None,
              top_p: Optional[float] = None, max_tokens: Optional[int] = None, disable_think: bool = False) -> str:

        messages = self._build_messages(prompt, system_prompt, disable_think)
        inputs = self._tokenize_messages(messages)

        temp, nucleus, max_new, do_sample = self._effective_params(
            temperature,
            top_p,
            max_tokens,
        )

        LOGGER.info(
            f"Sampling: max_tokens={max_new}, temperature={temp}, top_p={nucleus}, do_sample={do_sample}"
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            generation_kwargs["temperature"] = temp
            generation_kwargs["top_p"] = nucleus

        with self._lock:
            with torch.inference_mode():
                output = self.model.generate(**generation_kwargs)

        input_len = inputs["input_ids"].shape[-1]
        generated = output[0][input_len:]

        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        if disable_think:
            text = self._remove_thinking(text)

        return text.strip()

    def invoke(self, prompt: str, system_prompt: Optional[str] = None, *, temperature: Optional[float] = None,
               top_p: Optional[float] = None, max_tokens: Optional[int] = None, disable_think: bool = False) -> str:

        return self._call(
            prompt,
            system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            disable_think=disable_think,
        )

    @timeit
    def stream(self, prompt: str, system_prompt: Optional[str] = None, *, temperature: Optional[float] = None,
               top_p: Optional[float] = None, max_tokens: Optional[int] = None, disable_think: bool = False) -> Iterator[str]:

        messages = self._build_messages(prompt, system_prompt, disable_think)
        inputs = self._tokenize_messages(messages)

        temp, nucleus, max_new, do_sample = self._effective_params(
            temperature,
            top_p,
            max_tokens,
        )

        LOGGER.info(
            f"Streaming: max_tokens={max_new}, temperature={temp}, top_p={nucleus}, do_sample={do_sample}"
        )

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            generation_kwargs["temperature"] = temp
            generation_kwargs["top_p"] = nucleus

        def generate():
            with self._lock:
                with torch.inference_mode():
                    self.model.generate(**generation_kwargs)

        thread = threading.Thread(target=generate)
        thread.start()

        in_think = False

        for text in streamer:
            if not text:
                continue

            if disable_think:
                if "<think>" in text:
                    in_think = True
                    continue
                if "</think>" in text:
                    in_think = False
                    continue
                if in_think:
                    continue

            yield text

        thread.join()
from typing import Optional, Iterator
import threading
import torch
import os

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

class TransformersLLM:
    def __init__(
        self,
        model_id_or_path: str,
        temperature: float = 0.8,
        top_p: float = 0.9,
        max_tokens: int = 512,
        system_message: str = "Du bist ein präziser, detailorientierter medizinischer Schreibassistent.",
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ):
        self.model_id_or_path = model_id_or_path
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.system_message = system_message
        self.lock = threading.Lock()

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto",
        }

        local_files_only = os.getenv('HF_LOCAL_ONLY', 'true').lower() == 'true'
        dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only
        )

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id_or_path,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            self.processor = None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only
        )

        self.model.eval()

    def _effective_params(
        self,
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
    ):
        temp = self.temperature if temperature is None else float(temperature)
        nucleus = self.top_p if top_p is None else float(top_p)
        max_new = self.max_tokens if max_tokens is None else int(max_tokens)

        do_sample = temp > 0.0
        if not do_sample:
            nucleus = 1.0

        return temp, nucleus, max_new, do_sample

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str],
        disable_think: bool,
    ):
        sys_text = (system_prompt or self.system_message).strip()
        user_text = (prompt or "").strip()

        if disable_think:
            sys_text = "/no_think\n" + sys_text

        return [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_text},
        ]

    def _tokenize_messages(self, messages):
        tokenizer_or_processor = self.processor or self.tokenizer

        if hasattr(tokenizer_or_processor, "apply_chat_template"):
            inputs = tokenizer_or_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self.tokenizer(text, return_tensors="pt")

        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        disable_think: bool = False,
    ) -> str:
        messages = self._build_messages(prompt, system_prompt, disable_think)
        inputs = self._tokenize_messages(messages)
        temp, nucleus, max_new, do_sample = self._effective_params(
            temperature, top_p, max_tokens
        )

        with self.lock:
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    temperature=temp if do_sample else None,
                    top_p=nucleus if do_sample else None,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        input_len = inputs["input_ids"].shape[-1]
        generated = output[0][input_len:]

        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        if disable_think:
            text = self._remove_thinking(text)

        return text.strip()

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        disable_think: bool = False,
    ) -> Iterator[str]:
        messages = self._build_messages(prompt, system_prompt, disable_think)
        inputs = self._tokenize_messages(messages)
        temp, nucleus, max_new, do_sample = self._effective_params(
            temperature, top_p, max_tokens
        )

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new,
            temperature=temp if do_sample else None,
            top_p=nucleus if do_sample else None,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        def generate():
            with self.lock:
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

    def _remove_thinking(self, text: str) -> str:
        while "<think>" in text and "</think>" in text:
            start = text.find("<think>")
            end = text.find("</think>") + len("</think>")
            text = text[:start] + text[end:]
        return text
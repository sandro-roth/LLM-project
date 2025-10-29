from queue import Queue
from typing import Optional, List, ClassVar, Iterator
from pathlib import Path
from queue import Queue, Empty
from packaging import version
import os

from langchain_core.language_models import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, AutoConfig
import torch
import threading

try:
    import accelerate
    ACC_VERSION = version.parse(getattr(accelerate, '__version__', '0.0.0'))
except Exception:
    ACC_VERSION = version.parse('0.0.0')
#
# try:
#     from accelerate.hooks import cpu_offload as _cpu_offload
# except Exception:
#     try:
#         from accelerate import cpu_offload as _cpu_offload
#     except Exception:
#         _cpu_offload = None

try:
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
except Exception:
    init_empty_weights = None
    load_checkpoint_and_dispatch = None

from utils import timeit
from utils import setup_logging

LOGGER = setup_logging(app_name='qwen-inference', to_stdout=True, retention=30)
LOGGER.info("VERSIONS torch=%s accelerate=%s",
            getattr(torch,'__version__','?'),
            getattr(accelerate,'__version__','?'))
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def _primary_device_of(model: torch.nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    return torch.device('cpu')

class QwenInferenceLLM(LLM):

    def __init__(self, model_path:Path, tokenizer_path:Path, temperature:float,
                 top_p:float, max_tokens:int, offload_folder:Path):

        super().__init__()

        model_id = str(model_path)
        tok_id = str(tokenizer_path)

        # Config from .env
        use_4bit = os.getenv('LOAD_IN_4BIT', 'false').lower() == 'true'
        use_8bit = os.getenv('LOAD_IN_8BIT', 'false').lower() == 'true' and not use_4bit
        dtype = torch.bfloat16 if os.getenv('TORCH_DTYPE', 'bf16').lower() in ("bf16","bfloat16") else torch.float16

        # Memory budget
        per_gpu_gib = int(os.getenv('MAX_VRAM_PER_GPU', '45'))
        cpu_gib = int(os.getenv('CPU_RAM_BUDGET_GIB', '180'))

        # device_map + max_memory construct
        n_gpu = torch.cuda.device_count()
        max_memory = {i: f"{per_gpu_gib}GiB" for i in range(n_gpu)}
        max_memory["cpu"] = f"{cpu_gib}GiB"

        os.makedirs(offload_folder, exist_ok=True)

        # ------------------ Quantization --------------------------
        nb_config = None
        load_kwargs = {}
        if use_4bit:
            pass
        elif use_8bit:
            pass
        else:
            load_kwargs['dtype'] = dtype

        # Full model downloaded already make sure to only local processing
        local_only = os.getenv('HF_LOCAL_ONLY', 'true').lower() == 'true'

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tok_id, local_files_only=local_only, use_fast=True, trust_remote_code=True,)

        # Model with offload
        # model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=local_only, device_map='cpu',
        #                                              low_cpu_mem_usage=True, trust_remote_code=True, **load_kwargs)
        # model.to('cpu')
        # model.eval()

        if load_checkpoint_and_dispatch and init_empty_weights and ACC_VERSION >= version.parse('0.26.0'):
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, **load_kwargs)

            # qwen classes are named differently depending on release (few candidates)
            no_split = ["Qwen2DecoderLayer", "QwenBlock", "TransformerLayer", "Block"]

            model = load_checkpoint_and_dispatch(
                model,
                model_id,
                device_map={'': 'cpu'},
                no_split_module_classes=no_split,
                dtype=load_kwargs.get('dtype', None),
                offload_folder=str(offload_folder),
                offload_state_dict=True,
                max_memory=max_memory,
                sequential_cpu_offload=True
            )
            model.eval()

        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                local_files_only=local_only,
                device_map="auto",
                max_memory=max_memory,
                offload_folder=str(offload_folder),
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **load_kwargs
            )
            model.eval()
        # if _cpu_offload is None:
        #     LOGGER.warning('accelerate.cpu_offload nicht gefunden!')
        # else:
        #     exec_device = torch.device('cuda:0')
        #     try:
        #         _cpu_offload(model, exec_device, offload_buffers=True, pin_memory=True)
        #     except TypeError:
        #         try:
        #             _cpu_offload(model, exec_device)
        #         except TypeError:
        #             _cpu_offload(model, 0)

        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(self, "_temperature", temperature)
        object.__setattr__(self, "_top_p", top_p)
        object.__setattr__(self, "_max_tokens", max_tokens)

        object.__setattr__(self, "_systemmessage", {
            'role': 'system', 'content':'Du bist ein präziser, detailorientierter medizinischer Schreibassistent.'
        })

        # Log
        try:
            dev = _primary_device_of(self._model)
            LOGGER.info(f"Model primary device: {dev}; n_gpu={n_gpu}; max_memory={max_memory}")
        except Exception as e:
            LOGGER.warning(f"Could not infer primary device: {e}")


    @property
    def _llm_type(self) -> str:
        return 'qwen-inference'


    def _effective_params(self, temperature:Optional[float], top_p:Optional[float], max_tokens:Optional[int]):
        """ helper function for _call and stream """
        temp = self._temperature if temperature is None else float(temperature)
        nucleus = self._top_p if top_p is None else float(top_p)
        max_new = self._max_tokens if max_tokens is None else int(max_tokens)
        return temp, nucleus, max_new, (temp > 0.0)


    def _build_inputs(self, prompt: str, system_prompt: Optional[str]):
        """ helper function for _call and stream """
        sys_msg = {'role': 'system', 'content': system_prompt} if system_prompt else self._systemmessage
        messages = [sys_msg, {'role': 'user', 'content': prompt}]
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True,return_tensors='pt'
        )
        return inputs


    def _gen_kwargs(self, inputs, max_new: int, temp: float, nucleus: float, do_sample: bool, *, streamer=None):
        return dict(
            **inputs,
            max_new_tokens=max_new,
            temperature=temp,
            top_p=nucleus,
            do_sample=do_sample,
            use_cache=True,
            pad_token_id=self._tokenizer.eos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            **({"streamer": streamer} if streamer is not None else {})
        )


    def _stream_chunks(self, prompt: str, system_prompt: Optional[str],
            *, temperature: Optional[float], top_p: Optional[float], max_tokens: Optional[int]) -> Iterator[str]:
        temp, nucleus, max_new, do_sample = self._effective_params(temperature, top_p, max_tokens)
        LOGGER.info(f"Sampling: max_new_tokens={max_new}, temperature={temp}, top_p={nucleus}")
        inputs = self._build_inputs(prompt, system_prompt)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        generate_kwargs = self._gen_kwargs(inputs, max_new, temp, nucleus, do_sample, streamer=streamer)

        err_q: Queue[BaseException] = Queue(maxsize=1)
        def _worker():
            try:
                with torch.no_grad():
                    self._model.generate(**generate_kwargs)
            except BaseException as e:
                err_q.put(e)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        try:
            while True:
                try:
                    e = err_q.get_nowait()
                    raise e
                except Empty:
                    pass

                chunk = next(streamer, None)
                if chunk is None:
                    break
                if chunk:
                    yield chunk

            try:
                e = err_q.get_nowait()
                raise e
            except Empty:
                pass

        finally:
            t.join(timeout=0.1)


    @timeit
    def _call(
        self, prompt:str, system_prompt:Optional[str]=None, stop:Optional[List[str]]=None,
        *, temperature:Optional[float]=None, top_p:Optional[float]=None, max_tokens:Optional[int]=None
    ) -> str:

        parts = []
        for chunk in self._stream_chunks(
            prompt, system_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens
        ):
            parts.append(chunk)
        return "".join(parts)


    def invoke(self, prompt: str, system_prompt: Optional[str] = None,
               *, temperature: Optional[float] = None, top_p: Optional[float] = None,
               max_tokens: Optional[int] = None) -> str:
        return self._call(prompt, system_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)


    @timeit
    def stream(self, prompt:str, system_prompt:Optional[str]=None,
               *, temperature:Optional[float]=None, top_p:Optional[float]=None, max_tokens:Optional[int]=None) -> Iterator[str]:
        yield from self._stream_chunks(
            prompt, system_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
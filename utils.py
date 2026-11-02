import gc
import os
import json
import re
import torch
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes
import litellm
from litellm import completion_cost


_JSON_DECODER = json.JSONDecoder()

_THINK_NAMES = r"(?:think|thinking|thought|reasoning|reflection|scratchpad|cot)"

# (opening-tag regex, closing-tag regex)
_THINK_PAIRS = [
    (rf"<\s*{_THINK_NAMES}\b[^>]*>",            rf"<\s*/\s*{_THINK_NAMES}\s*>"),
    (r"\[\s*think(?:ing)?\s*\]",                r"\[\s*/\s*think(?:ing)?\s*\]"),
    (r"<\|\s*begin_of_thought\s*\|>",           r"<\|\s*end_of_thought\s*\|>"),
    (r"◁\s*think\s*▷",                          r"◁\s*/\s*think\s*▷"),
]

# Marker tokens to delete but whose content is kept (e.g. the solution part)
_MARKER_ONLY = re.compile(r"<\|\s*(?:begin|end)_of_solution\s*\|>", re.IGNORECASE)
load_dotenv(override=True)
litellm.drop_params = True

litellm.completion_cost = completion_cost

if os.environ.get('LANGFUSE_PUBLIC_KEY') and os.environ.get('LANGFUSE_SECRET_KEY'):
    LANGFUSE = get_client()
else:
    LANGFUSE = None
    print('WARNING: LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set - nothing will be logged to Langfuse')


# vLLM engine settings shared by every task; main.py overrides them via the CLI.
# max_num_seqs must stay below the KV/Mamba cache capacity of the GPU.
VLLM_ENGINE_ARGS = {
    "max_num_seqs": 256,
    "gpu_memory_utilization": 0.90,
    "max_model_len": None,
    "trust_remote_code": True,
    "enforce_eager": False,
}

# Number of concurrent requests LiteLLMModel.generate_batch sends to the API.
API_CONCURRENCY = 8

# One vLLM engine per model, reused across tasks so the GPU is not re-allocated.
_ENGINE_CACHE = {}


def set_vllm_engine_args(**kwargs):
    VLLM_ENGINE_ARGS.update(kwargs)


def set_api_concurrency(concurrency):
    global API_CONCURRENCY
    API_CONCURRENCY = max(1, int(concurrency))


# Extra variables passed to the tokenizer's chat template (local backend only).
CHAT_TEMPLATE_KWARGS = {}


def set_thinking(mode="default", effort=None):
    """mode: 'on' | 'off' | 'default' (leave the template's own behaviour).
    effort: reasoning_effort template variable (e.g. low/medium/xhigh), or None."""
    CHAT_TEMPLATE_KWARGS.clear()
    if mode == "on":
        CHAT_TEMPLATE_KWARGS["enable_thinking"] = True
    elif mode == "off":
        CHAT_TEMPLATE_KWARGS["enable_thinking"] = False
    elif mode != "default":
        raise ValueError(f"Unknown thinking mode: {mode!r}. Use 'on', 'off' or 'default'.")
    if effort is not None:
        CHAT_TEMPLATE_KWARGS["reasoning_effort"] = effort


# A prompt ending in an opening reasoning tag means the model's output starts
# mid-reasoning; the tag is re-attached so strip_thinking can detect truncation.
_TRAILING_THINK_OPEN = re.compile(
    "(?:" + "|".join(open_re for open_re, _ in _THINK_PAIRS) + r")\s*$",
    re.IGNORECASE,
)





def strip_thinking(text):
    """
    Remove chain-of-thought / reasoning sections from LLM output.
    Handles closed blocks, orphan closing tags (when the opening tag was
    in the prompt), and unclosed blocks (truncated generation).
    Returns the cleaned text (may be '' if the output was only thinking).
    """
    if not text:
        return text

    flags = re.DOTALL | re.IGNORECASE
    for open_re, close_re in _THINK_PAIRS:
        # 1) Fully closed blocks: <think> ... </think>
        text = re.sub(rf"{open_re}.*?{close_re}", "", text, flags=flags)

        # 2) Orphan closing tag: everything before it was thinking
        closers = list(re.finditer(close_re, text, flags=flags))
        if closers:
            text = text[closers[-1].end():]

        # 3) Unclosed opening tag: the rest is (truncated) thinking
        opener = re.search(open_re, text, flags=flags)
        if opener:
            text = text[:opener.start()]

    text = _MARKER_ONLY.sub("", text)
    return text.strip()

class ModelInterface:

    def generate(self, prompt: str, n: int = 1, temperature: float = 0.7) -> list[str]:
        raise NotImplementedError

    def generate_batch(self, user_prompts, sys_prompts, n=1) -> list[list[str]]:
        raise NotImplementedError


class LiteLLMModel(ModelInterface):

    def __init__(self, model_name: str, max_new_tokens: int = 50, max_retries: int = 6, **kwargs):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.cost = 0.0
        self.kwargs = kwargs
        self._cost_lock = threading.Lock()

    def _generate_parameters(self,sys_prompt,user_prompt):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt, "cache_control": {"type": "ephemeral"}}]},
            {"role": "user",   "content": user_prompt},
        ]

        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.kwargs.get("temperature", 0.0),
            # "max_tokens":  self.max_new_tokens,
        }

        return params

    def generate(self, sys_prompt,user_prompt, n = 1) -> list[str]:
        params = self._generate_parameters(sys_prompt,user_prompt)
        
        outputs = []
        for _ in range(n):
            text = ""
            for attempt in range(self.max_retries):
                try:

                    litellm.success_callback = ["langfuse"] if LANGFUSE else []
                    litellm.failure_callback = ["langfuse"] if LANGFUSE else []
                    
                    response = litellm.completion(**params)
                    
                    with self._cost_lock:
                        self.cost += float(response._hidden_params.get('response_cost', 0))
                    text = response.choices[0].message.content.strip()
                    break
                except Exception as e:
                    print(f"API error on attempt {attempt+1}/{self.max_retries}: {e}")
                    time.sleep(2) # Exponential backoff can be added here
            outputs.append(text)
            time.sleep(0.5)                         
        return outputs

    def generate_batch(self, user_prompts, sys_prompts, n=1) -> list[list[str]]:
        from tqdm import tqdm

        def _one(pair):
            user_prompt, sys_prompt = pair
            return self.generate(sys_prompt=sys_prompt, user_prompt=user_prompt, n=n)

        with ThreadPoolExecutor(max_workers=API_CONCURRENCY) as pool:
            return list(tqdm(
                pool.map(_one, zip(user_prompts, sys_prompts)),
                total=len(user_prompts),
                desc=self.model_name,
            ))


class HuggingFaceModel(ModelInterface):

    def __init__(self, temperature, model_name, max_new_tokens=50):
        self.temperature = temperature
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._warned_no_template = False
        self.last_truncated = 0

        if model_name not in _ENGINE_CACHE:
            engine_args = {k: v for k, v in VLLM_ENGINE_ARGS.items() if v is not None}
            print(f"[*] Loading vLLM engine for {model_name} with {engine_args}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=engine_args.get("trust_remote_code", True)
            )
            _ENGINE_CACHE[model_name] = (tokenizer, LLM(model=model_name, **engine_args))
        self.tokenizer, self.llm = _ENGINE_CACHE[model_name]



    def build_chat_prompt(self,user_prompt,sys_prompt):
    
        messages = [
        {
            "role": "system",
            "content": sys_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **CHAT_TEMPLATE_KWARGS,
        )

        return chat_prompt



    def _prompt_text(self, user_prompt, sys_prompt):
        try:
            return self.build_chat_prompt(user_prompt, sys_prompt)
        except Exception as e: #incase model does not support chat template
            if not self._warned_no_template:
                print(f"Error in building chat prompt, falling back to raw text: {e}")
                self._warned_no_template = True
            return sys_prompt + user_prompt

    def generate_batch(self, user_prompts, sys_prompts, n=1) -> list[list[str]]:
        prompts = [self._prompt_text(u, s) for u, s in zip(user_prompts, sys_prompts)]
        params = SamplingParams(
            n=n,
            temperature=self.temperature,
            top_p=1.0,
            max_tokens=self.max_new_tokens,
        )
        results = self.llm.generate(prompts, params, use_tqdm=True)

        all_outputs = []
        total = truncated = unfinished = 0
        for prompt, out in zip(prompts, results):
            m = _TRAILING_THINK_OPEN.search(prompt)
            open_tag = m.group(0).strip() + "\n" if m else ""
            texts = []
            for c in out.outputs:
                text = open_tag + c.text
                total += 1
                if c.finish_reason == "length":
                    truncated += 1
                    if not strip_thinking(text):
                        unfinished += 1
                texts.append(text.strip())
            all_outputs.append(texts)

        self.last_truncated = truncated
        if truncated:
            print(f"[!] {truncated}/{total} outputs hit max_new_tokens "
                  f"({unfinished} with unfinished reasoning)")
        return all_outputs

    def generate(self, user_prompt, sys_prompt, n=1) -> list[str]:
        return self.generate_batch([user_prompt], [sys_prompt], n=n)[0]



def extract_json_answer(text, key="answer"):
    """
    Robustly pull the value of `key` out of arbitrary LLM output.
    Handles: surrounding prose, ```json fences, multiple/nested braces,
    invalid escapes (e.g. SMILES backslashes), single quotes, trailing
    commas, and truncated output. Returns a string, or None if not found.
    """
    if not text:
        return None

    text = strip_thinking(text)
    def _clean(val):
        if isinstance(val, (list, tuple)):
            val = val[0] if val else None
        if val is None:
            return None
        return str(val).strip()

    # Escape lone backslashes that aren't valid JSON escapes (SMILES etc.)
    def _fix_escapes(s):
        return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", s)

    # 1) Try raw_decode starting at every '{'
    for variant in (text, _fix_escapes(text)):
        for m in re.finditer(r"\{", variant):
            try:
                obj, _ = _JSON_DECODER.raw_decode(variant[m.start():])
            except ValueError:
                continue
            if isinstance(obj, dict) and key in obj:
                out = _clean(obj[key])
                if out is not None:
                    return out

    # 2) Lenient regex: "answer": "...", 'answer': '...', or unquoted value
    k = re.escape(key)
    patterns = [
        rf'["\']?{k}["\']?\s*:\s*"((?:[^"\\]|\\.)*)"',    # double-quoted
        rf"[\"']?{k}[\"']?\s*:\s*'((?:[^'\\]|\\.)*)'",    # single-quoted
        rf'["\']?{k}["\']?\s*:\s*"([^"\n]*)$',            # truncated, no closing quote
        rf'["\']?{k}["\']?\s*:\s*([^,}}\n]+)',            # unquoted value
    ]
    for p in patterns:
        m = re.search(p, text, re.DOTALL | re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if val:
                return val

    return None
import gc
import os
import json
import re
import torch
import time
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


class LiteLLMModel(ModelInterface):

    def __init__(self, model_name: str, max_new_tokens: int = 50, max_retries: int = 6, **kwargs):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.cost = 0.0
        self.kwargs = kwargs

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
                    
                    self.cost += float(response._hidden_params.get('response_cost', 0))
                    text = response.choices[0].message.content.strip()
                    break
                except Exception as e:
                    print(f"API error on attempt {attempt+1}/{self.max_retries}: {e}")
                    time.sleep(2) # Exponential backoff can be added here
            outputs.append(text)
            time.sleep(0.5)                         
        return outputs


class HuggingFaceModel(ModelInterface):

    def __init__(self, temperature, model_name, max_new_tokens=50):
        self.temperature = temperature
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm = LLM(model=self.model_name)



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
        )

        return chat_prompt



    def generate(self, user_prompt, sys_prompt, n=1) -> list[str]:
        outputs = []

        try:
            prompt = self.build_chat_prompt(user_prompt, sys_prompt)
        except Exception as e: #incase model does not support chat template
            print(f"Error in building chat prompt: {e}")
            prompt = sys_prompt + user_prompt

        for _ in range(n):
            

            params = SamplingParams(
                temperature=self.temperature,
                top_p=1.0,
                max_tokens=self.max_new_tokens,
            )

            result = self.llm.generate(prompt, params)

            for i, output in enumerate(result):
                generated_text = output.outputs[0].text
                outputs.append(generated_text.strip())

        return outputs 



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
"""Minimal Hugging Face causal-LM loader for public evaluation."""

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class LMClass:
    def __init__(self, args):
        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.torch_dtype = torch.bfloat16 if args.use_bfloat16 else torch.float16

        config = AutoConfig.from_pretrained(
            self.model_name,
            attn_implementation=args.attn_implementation,
            trust_remote_code=args.trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False,
            legacy=False,
            trust_remote_code=args.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=config,
            device_map="cpu",
            torch_dtype=self.torch_dtype,
            trust_remote_code=args.trust_remote_code,
        )
        self.model.eval()

    @property
    def device(self):
        return self._device

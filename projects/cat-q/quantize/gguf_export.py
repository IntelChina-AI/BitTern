"""Write a CAT-Q checkpoint out as a packed ternary GGUF file.

The quantized projections are stored as `Q2_0`, the group-128 ternary weight
type of the Bonsai llama.cpp fork (https://github.com/PrismML-Eng/llama.cpp,
branch `prism`); see `quantize/q2_0.py` for the block layout.  Everything the
CAT-Q recipe leaves in floating point - token embeddings, the LM head, every
norm and the MoE router - is written as F16 (or `--gguf_float_type`) and F32
exactly as a stock conversion would.

Only the `gguf` PyPI package is required.  Producing the file has no dependency
on a llama.cpp checkout; llama.cpp is needed to *run* the result, not to build
it.

Usage is through `main.py --export_gguf_path`, which feeds `capture()` from the
checkpoint restoration in `quantize.merge` and then calls `write()`; see
`deployment/README.md` for the export-and-serve walkthrough.
"""

import json
import logging
from pathlib import Path

import gguf
import numpy as np
import torch

from .q2_0 import LLAMA_FTYPE_MOSTLY_Q2_0, pack_q2_0, register_q2_0
from .ternary_export import iter_layer_ternary

logger = logging.getLogger(__name__)

FLOAT_TYPES = {
    "f16": gguf.GGMLQuantizationType.F16,
    "bf16": gguf.GGMLQuantizationType.BF16,
    "f32": gguf.GGMLQuantizationType.F32,
}

# Tokenizer pre-tokenizer identifiers used by llama.cpp.  Guessing one silently
# changes tokenization, so only architectures that have been checked are listed.
BPE_PRE_TOKENIZERS = {
    "qwen3": "qwen2",
    "qwen3moe": "qwen2",
}


class _Arch:
    """Per-architecture conversion rules."""

    def __init__(self, name, arch, vocab, permute_qk=False, rope_dim=False, vocab_size=False):
        self.name = name
        self.arch = arch
        self.vocab = vocab
        self.permute_qk = permute_qk
        self.rope_dim = rope_dim
        self.vocab_size = vocab_size


ARCHITECTURES = {
    "Qwen3ForCausalLM": _Arch("qwen3", gguf.MODEL_ARCH.QWEN3, "bpe"),
    "Qwen3MoeForCausalLM": _Arch("qwen3moe", gguf.MODEL_ARCH.QWEN3MOE, "bpe"),
    "LlamaForCausalLM": _Arch(
        "llama", gguf.MODEL_ARCH.LLAMA, "spm", permute_qk=True, rope_dim=True, vocab_size=True
    ),
}


def _permute(tensor, n_head, n_head_kv):
    """The q/k row interleaving llama.cpp undoes for LLaMA-style RoPE."""
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (
        tensor.reshape(n_head, 2, tensor.shape[0] // n_head // 2, *tensor.shape[1:])
        .swapaxes(1, 2)
        .reshape(tensor.shape)
    )


def _pin_malloc_thresholds():
    """Keep glibc handing large blocks back to the kernel.

    The exporter frees a whole decoder layer worth of float tensors on every
    iteration.  glibc grows its dynamic mmap threshold whenever such a block is
    released, after which same-sized allocations come from the heap and their
    memory is never returned, so a long export slowly accumulates hundreds of
    gigabytes of freed-but-resident memory.  Pinning the threshold disables that
    heuristic.
    """
    import ctypes

    M_MMAP_THRESHOLD = -3
    M_TRIM_THRESHOLD = -1
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.mallopt(ctypes.c_int(M_MMAP_THRESHOLD), ctypes.c_int(128 * 1024))
        libc.mallopt(ctypes.c_int(M_TRIM_THRESHOLD), ctypes.c_int(128 * 1024))
    except OSError:  # pragma: no cover - non-glibc platforms
        logger.debug("Could not pin the malloc thresholds; this is glibc-only")


class TernaryGGUFExporter:
    """Turn a restored CAT-Q model into a single `Q2_0` GGUF file.

    `capture` is handed to `merge_catq_checkpoint`, which calls it once per
    decoder layer while the ternary codes and scales still exist.  `write` then
    walks the merged model and swaps the packed bytes in for the corresponding
    float weights.
    """

    def __init__(self, model_dir, outfile, float_type="f16", low_memory=False):
        if float_type not in FLOAT_TYPES:
            raise ValueError(f"Unknown float type {float_type!r}; expected one of {sorted(FLOAT_TYPES)}")
        self.model_dir = Path(model_dir)
        self.outfile = Path(outfile)
        self.float_type = FLOAT_TYPES[float_type]
        self.low_memory = low_memory
        if low_memory:
            _pin_malloc_thresholds()
        # AutoConfig rather than raw config.json: llama.cpp reads the same
        # normalised view, so defaults such as `head_dim` and `rope_theta` are
        # filled in for older checkpoints that omit them.
        from transformers import AutoConfig

        self.hparams = AutoConfig.from_pretrained(self.model_dir).to_dict()

        architectures = self.hparams.get("architectures") or []
        if not architectures or architectures[0] not in ARCHITECTURES:
            raise ValueError(
                f"Packed ternary export does not know architecture {architectures}; "
                f"supported: {sorted(ARCHITECTURES)}"
            )
        self.arch = ARCHITECTURES[architectures[0]]
        self.block_count = self.hparams["num_hidden_layers"]
        self.n_head = self.hparams["num_attention_heads"]
        self.n_head_kv = self.hparams.get("num_key_value_heads", self.n_head)

        self.qtype_q2_0 = register_q2_0()
        self.tensor_map = gguf.get_tensor_name_map(self.arch.arch, self.block_count)
        # In low-memory mode the packed bytes are spooled to a temporary file as
        # soon as they are handed to the writer, so only one tensor at a time has
        # to be resident while the GGUF is being assembled.
        self.writer = gguf.GGUFWriter(path=None, arch=self.arch.name, use_temp_file=low_memory)
        self.packed = {}  # hf weight name -> packed uint8 array
        self.n_packed = 0

    # ------------------------------------------------------------------
    # collection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def capture(self, layer_id, qlayer):
        """Pack every ternary projection of one restored decoder layer.

        Returns the Hugging Face weight names that were packed, so that the
        caller can drop the corresponding floating-point weights (see
        `quantize.merge`).
        """
        names = []
        for name, ternary in iter_layer_ternary(layer_id, qlayer):
            codes = ternary.codes.cpu().numpy()
            scales = ternary.scales.cpu().numpy().reshape(codes.shape[:-1] + (-1,))
            self.packed[name] = self._pack(name, codes, scales)
            names.append(name)
        self.n_packed += len(names)
        logger.info("Packed %d ternary tensors from layer %d", len(names), layer_id)
        return names

    def _pack(self, name, codes, scales):
        """Pack one weight, undoing the LLaMA q/k row interleaving if needed.

        The scales are laid out as one column per group, so the row permutation
        applies to them unchanged.
        """
        if self.arch.permute_qk and name.endswith("q_proj.weight"):
            codes = _permute(codes, self.n_head, self.n_head)
            scales = _permute(scales, self.n_head, self.n_head)
        elif self.arch.permute_qk and name.endswith("k_proj.weight"):
            codes = _permute(codes, self.n_head, self.n_head_kv)
            scales = _permute(scales, self.n_head, self.n_head_kv)
        return pack_q2_0(codes, scales)

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------
    def _set_parameters(self):
        writer, hparams = self.writer, self.hparams
        writer.add_block_count(self.block_count)
        writer.add_context_length(hparams["max_position_embeddings"])
        writer.add_embedding_length(hparams["hidden_size"])
        writer.add_feed_forward_length(hparams["intermediate_size"])
        writer.add_head_count(self.n_head)
        writer.add_head_count_kv(self.n_head_kv)

        rope = hparams.get("rope_parameters") or hparams.get("rope_scaling") or {}
        rope_type = rope.get("rope_type", rope.get("type"))
        factor = rope.get("factor")
        if rope_type == "linear" and factor is not None:
            writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
            writer.add_rope_scaling_factor(factor)
        elif rope_type == "yarn" and factor is not None:
            writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            writer.add_rope_scaling_factor(factor)
            writer.add_rope_scaling_orig_ctx_len(rope["original_max_position_embeddings"])
        elif rope_type not in (None, "default"):
            raise ValueError(f"Unsupported rope_scaling type {rope_type!r} for packed ternary export")

        writer.add_rope_freq_base(hparams.get("rope_theta") or rope["rope_theta"])
        writer.add_layer_norm_rms_eps(hparams["rms_norm_eps"])

        if (n_experts := hparams.get("num_experts")) is not None:
            writer.add_expert_count(n_experts)
        if (n_used := hparams.get("num_experts_per_tok")) is not None:
            writer.add_expert_used_count(n_used)
        if (moe_ff := hparams.get("moe_intermediate_size")) is not None:
            writer.add_expert_feed_forward_length(moe_ff)

        head_dim = hparams.get("head_dim")
        if head_dim is not None:
            writer.add_key_length(head_dim)
            writer.add_value_length(head_dim)
        if self.arch.vocab_size:
            writer.add_vocab_size(hparams["vocab_size"])
        if self.arch.rope_dim:
            writer.add_rope_dimension_count(head_dim or hparams["hidden_size"] // self.n_head)

        writer.add_file_type(LLAMA_FTYPE_MOSTLY_Q2_0)

    def _set_vocab(self):
        if self.arch.vocab == "bpe":
            self._set_vocab_bpe()
        else:
            self._set_vocab_spm()
        gguf.SpecialVocab(self.model_dir, load_merges=self.arch.vocab == "bpe").add_to_gguf(self.writer)

    def _looks_special(self, token):
        """Added tokens that llama.cpp treats as control tokens even when the
        Hugging Face tokenizer does not flag them as special."""
        if isinstance(token, (bytes, bytearray)):
            token = token.decode("utf-8")
        return (
            token in ("<pad>", "<mask>", "<2mass>", "[@BOS@]")
            or (token.startswith("<|") and token.endswith("|>"))
            or (token.startswith("<\uff5c") and token.endswith("\uff5c>"))
            or (token.startswith("<unused") and token.endswith(">"))
        )

    def _set_vocab_bpe(self):
        from transformers import AutoTokenizer

        pre = BPE_PRE_TOKENIZERS.get(self.arch.name)
        if pre is None:
            raise ValueError(f"No known llama.cpp pre-tokenizer for architecture {self.arch.name!r}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        vocab_size = self.hparams.get("vocab_size", len(tokenizer.vocab))
        reverse_vocab = {index: token for token, index in tokenizer.vocab.items()}
        added_vocab = tokenizer.get_added_vocab()
        added_decoder = tokenizer.added_tokens_decoder

        tokens, toktypes = [], []
        for index in range(vocab_size):
            if index not in reverse_vocab:
                tokens.append(f"[PAD{index}]")
                toktypes.append(gguf.TokenType.UNUSED)
                continue
            token = reverse_vocab[index]
            if token in added_vocab:
                if not added_decoder[index].normalized:
                    # llama.cpp expects added tokens in their normalized form.
                    token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False))
                if added_decoder[index].special or self._looks_special(token):
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    token = token.replace("\u2581", " ")
                    toktypes.append(gguf.TokenType.USER_DEFINED)
            else:
                toktypes.append(gguf.TokenType.NORMAL)
            tokens.append(token)

        self.writer.add_tokenizer_model("gpt2")
        self.writer.add_tokenizer_pre(pre)
        self.writer.add_token_list(tokens)
        self.writer.add_token_types(toktypes)

    def _set_vocab_spm(self):
        from sentencepiece import SentencePieceProcessor

        model_file = self.model_dir / "tokenizer.model"
        if not model_file.is_file():
            raise FileNotFoundError(
                f"{model_file} is required for a SentencePiece vocabulary; "
                "BPE-only LLaMA variants are not supported by this exporter"
            )
        sp = SentencePieceProcessor()
        sp.LoadFromFile(str(model_file))

        vocab_size = self.hparams.get("vocab_size", sp.vocab_size())
        tokens = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores = [-10000.0] * vocab_size
        toktypes = [gguf.TokenType.UNUSED] * vocab_size

        for index in range(min(sp.vocab_size(), vocab_size)):
            tokens[index] = sp.IdToPiece(index).encode("utf-8")
            scores[index] = sp.GetScore(index)
            if sp.IsUnknown(index):
                toktypes[index] = gguf.TokenType.UNKNOWN
            elif sp.IsControl(index):
                toktypes[index] = gguf.TokenType.CONTROL
            elif sp.IsUnused(index):
                toktypes[index] = gguf.TokenType.UNUSED
            elif sp.IsByte(index):
                toktypes[index] = gguf.TokenType.BYTE
            else:
                toktypes[index] = gguf.TokenType.NORMAL

        for index, token in self._added_tokens().items():
            if index >= vocab_size:
                continue
            content = token["content"]
            if token.get("special") or self._looks_special(content):
                toktypes[index] = gguf.TokenType.CONTROL
            else:
                content = content.replace("\u2581", " ")
                toktypes[index] = gguf.TokenType.USER_DEFINED
            tokens[index] = content.encode("utf-8")
            scores[index] = -1000.0

        self.writer.add_tokenizer_model("llama")
        self.writer.add_tokenizer_pre("default")
        self.writer.add_token_list(tokens)
        self.writer.add_token_scores(scores)
        self.writer.add_token_types(toktypes)

    def _added_tokens(self):
        added = {}
        path = self.model_dir / "added_tokens.json"
        if path.is_file():
            for content, index in json.loads(path.read_text(encoding="utf-8")).items():
                added[int(index)] = {"content": content}
        path = self.model_dir / "tokenizer_config.json"
        if path.is_file():
            config = json.loads(path.read_text(encoding="utf-8"))
            for index, token in config.get("added_tokens_decoder", {}).items():
                added[int(index)] = token
        return added

    # ------------------------------------------------------------------
    # tensors
    # ------------------------------------------------------------------
    def _gguf_name(self, name):
        new_name = self.tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
        if new_name is None:
            raise ValueError(f"No GGUF name for tensor {name!r}")
        return new_name

    def _is_router(self, new_name):
        for key in (gguf.MODEL_TENSOR.FFN_GATE_INP, gguf.MODEL_TENSOR.FFN_GATE_INP_SHEXP):
            if key not in gguf.MODEL_TENSORS[self.arch.arch]:
                continue
            template = gguf.TENSOR_NAMES[key]
            for bid in range(self.block_count):
                if new_name == template.format(bid=bid) + ".weight":
                    return True
        return False

    def _float_dtype(self, new_name, n_dims):
        # 1-D tensors, norms and the MoE router stay in F32, as in a stock
        # llama.cpp conversion; everything else follows --gguf_float_type.
        if n_dims <= 1 or new_name.endswith("_norm.weight") or self._is_router(new_name):
            return gguf.GGMLQuantizationType.F32
        return self.float_type

    def _add_float_tensor(self, new_name, tensor):
        dtype = self._float_dtype(new_name, tensor.ndim)
        if dtype == gguf.GGMLQuantizationType.F32:
            data = tensor.to(torch.float32).numpy()
        elif dtype == gguf.GGMLQuantizationType.F16:
            data = tensor.to(torch.float16).numpy()
        else:
            data = tensor.to(torch.bfloat16).view(torch.uint16).numpy()
        self.writer.add_tensor(new_name, data, raw_shape=tuple(tensor.shape), raw_dtype=dtype)

    def _expert_slot(self, name):
        """`...experts.<slot>.<proj>.weight` -> (merged key, slot), else None."""
        parts = name.split(".")
        if "experts" not in parts:
            return None
        index = parts.index("experts")
        if index + 1 >= len(parts) or not parts[index + 1].isdigit():
            return None
        slot = int(parts[index + 1])
        merged = ".".join(parts[:index + 1] + parts[index + 2:])
        return merged, slot

    @torch.no_grad()
    def write(self, model):
        if not self.packed:
            raise RuntimeError("No ternary tensors were captured; nothing to pack")

        state_dict = model.state_dict()
        tied = self.hparams.get("tie_word_embeddings", False)
        experts = {}
        n_ternary = 0

        for name, tensor in state_dict.items():
            if name.endswith(".inv_freq") or (tied and name == "lm_head.weight"):
                continue

            slot = self._expert_slot(name)
            if slot is not None:
                merged, index = slot
                bucket = experts.setdefault(merged, {})
                bucket[index] = self.packed.pop(name, None)
                if bucket[index] is None:
                    bucket[index] = tensor
                if len(bucket) < self.hparams["num_experts"]:
                    continue
                stacked = [bucket[i] for i in range(self.hparams["num_experts"])]
                new_name = self._gguf_name(merged)
                if isinstance(stacked[0], np.ndarray):
                    n_ternary += len(stacked)
                    stacked = np.stack(stacked)
                    del experts[merged], bucket
                    self.writer.add_tensor(new_name, stacked, raw_dtype=self.qtype_q2_0)
                    del stacked
                else:
                    self._add_float_tensor(new_name, torch.stack(stacked))
                    del experts[merged]
                continue

            new_name = self._gguf_name(name)
            if name in self.packed:
                self.writer.add_tensor(new_name, self.packed.pop(name), raw_dtype=self.qtype_q2_0)
                n_ternary += 1
            else:
                self._add_float_tensor(new_name, tensor)

        if experts:
            raise RuntimeError(f"Incomplete expert groups: {sorted(experts)}")
        if n_ternary != self.n_packed:
            raise RuntimeError(
                f"{self.n_packed - n_ternary} captured ternary tensors are absent from the model: "
                f"{sorted(self.packed)[:4]}"
            )

        self._write_metadata()
        self.writer.write_header_to_file(path=self.outfile)
        self.writer.write_kv_data_to_file()
        self.writer.write_tensors_to_file(progress=True)
        self.writer.close()
        logger.info(
            "Wrote %s (%d tensors, %d ternary)",
            self.outfile,
            len(self.writer.tensors[0]),
            n_ternary,
        )
        return self.outfile

    def _write_metadata(self):
        total_params, shared_params, expert_params, expert_count = self.writer.get_total_parameter_count()
        metadata = gguf.Metadata.load(None, self.model_dir, None, total_params)
        if metadata.name is None:
            metadata.name = self.model_dir.name
        if metadata.size_label is None and total_params > 0:
            metadata.size_label = gguf.size_label(total_params, shared_params, expert_params, expert_count)
        metadata.set_gguf_meta_model(self.writer)
        self.writer.add_type(gguf.GGUFType.MODEL)
        self._set_parameters()
        self._set_vocab()
        self.writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

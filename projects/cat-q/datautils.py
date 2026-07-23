"""Public evaluation-dataset loaders."""

from datasets import load_dataset


def get_test_loader(name, tokenizer, seqlen):
    if name == "wikitext2":
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        return tokenizer("\n\n".join(dataset["text"]), return_tensors="pt").input_ids
    if name == "ptb":
        dataset = load_dataset("ptb_text_only", "penn_treebank", split="validation")
        return tokenizer("\n\n".join(dataset["sentence"]), return_tensors="pt").input_ids
    if name == "c4":
        dataset = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        encoded = tokenizer(" ".join(dataset[:1100]["text"]), return_tensors="pt").input_ids
        return encoded[:, : 256 * seqlen]
    raise ValueError(f"Unsupported perplexity dataset: {name}. Use wikitext2, ptb, or c4.")

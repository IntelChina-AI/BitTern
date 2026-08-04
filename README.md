<div align="center">
  <img src="images/logo.png" alt="BitTern logo" width="220">

  <h3>An Open Toolkit for Post-Training Ternary Quantization: Research, Models, and Systems.</h3>

  <p>
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg">
    <a href="https://huggingface.co/IntelLabsChina"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-IntelLabsChina-FFD21E.svg"></a>
  </p>
</div>

**BitTern** aims to provide low-cost, high-accuracy post-training ternary quantization tools, as well as 1.58-bit models across diverse architectures, model scales, and reasoning tasks. Its goal is to lower the barrier to entry for developing 1.58-bit models, enabling broader community participation and allowing everyone can contribute and benefit from shared tools and models.

## Projects

| Project | Venue | Public Release |
| --- | --- | --- |
| [CAT-Q](projects/cat-q) | ICML 2026 Oral | Model checkpoints, inference, and evaluation code |

## Latest News

- `[Stay tuned]` We are preparing to release the CAT-Q training code, etc. 
- `[04/08/2026]` [The technical report](https://arxiv.org/abs/2608.01078) "**Attend to Your Own Thoughts: Breaking the Barrier for Post-Training Quantization of Reasoning LLMs through the Lens of 1.58-Bit Quantization**" is now available on arXiv.
- `[22/07/2026]` The [CAT-Q](projects/cat-q) model checkpoints (including Qwen3-1.7B/4B/8B/14B/32B, Llama2-7B, Qwen3-30B-A3B and Qwen3-235B-A22B), inference, and evaluation code are now available.
- `[25/06/2026]` [The CAT-Q paper](https://arxiv.org/abs/2606.26650) is now available on arXiv.
- `[01/05/2026]` 🎉🎉🎉Our paper "**CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs**" is accepted to ICML 2026 as an oral. The project page for our sliding-layer reconstruction framework used in CAT-Q is available at [SliderQuant (ICLR 2026)](https://github.com/deep-optimization/SliderQuant).

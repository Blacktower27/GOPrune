# HIVTP: A Training-Free Method to Improve VLMs Efficiency via Hierarchical Visual Token Pruning Using Middle-Layer-Based Importance Score
1. Our HIVTP demonstrates superiority over prior training-free methods in improving VLMs' inference efficiency while preserving accuracy.
2. HIVTP can reduce the time-to-first token (TTFT) of LLaVA-v1.5-7B and LLaVA-Next-7B by up to 50.0% and 55.1%, respectively, and improve the token generation throughput by up to 60.9% and 47.3%, without sacrificing accuracy, and even achieving improvements on certain benchmarks.
3. The local retaining stage in HIVTP helps mitigate hallucinations in VLMs by reducing visual uncertainty.
4. [paper link](https://arxiv.org/abs/2509.23663)
## Overview
![Overview of HIVTP](Framework.png)
---

## Installation

Clone this repository and install:

```bash
git clone https://github.com/Blacktower27/HIVTP.git
cd GOPrune

pip install -e .
```

## Usage with lmms-eval

First, follow the official [lmms-eval installation guide](https://github.com/EvolvingLMMs-Lab/lmms-eval) to set up the environment. To integrate **HIVTP** into lmms-eval with LLaVA, you need to modify the `class Llava(lmms)` inside `lmms_eval/models/simple/llava.py`. Specifically, in the `__init__` function after loading the pretrained model, we insert the following code:

```python
try:
    # Try to load the model with the multimodal argument
    self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
        pretrained, None, model_name, device_map=self.device_map, **llava_model_args
    )
    from goprune import goprune
    self._model = goprune(self._model)
except TypeError:
    # for older versions of LLaVA that don't have multimodal argument
    llava_model_args.pop("multimodal", None)
    self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
        pretrained, None, model_name, device_map=self.device_map, **llava_model_args
    )
    from goprune import goprune
    self._model = goprune(self._model)
```    
After this modification, you can test our method on LLaVA using lmms-eval’s provided script:
```
bash examples/models/llava_next.sh
```
If you want to change the dataset, please directly modify this bash script in lmms-eval.
## Cite us
```bash
@misc{xu2025hivtptrainingfreemethodimprove,
      title={HIVTP: A Training-Free Method to Improve VLMs Efficiency via Hierarchical Visual Token Pruning Using Middle-Layer-Based Importance Score}, 
      author={Jingqi Xu and Jingxi Lu and Chenghao Li and Sreetama Sarkar and Peter A. Beerel},
      year={2025},
      eprint={2509.23663},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.23663}, 
}
```


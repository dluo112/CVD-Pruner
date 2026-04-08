# 💽 CVD-Pruner

*A training-free and plug-and-play visual token pruning method for efficient multi-image reasoning in multimodal large language models.*

<!-- [📄 Paper](#) [🌐 Project Page](#) -->

## 📰 News

🔥 **[2026/04/08]** Initial code release of **CVD-Pruner**.

---

## 👁️ Overview

Multimodal large language models (MLLMs) often incur substantial inference overhead in **multi-image reasoning**, where the number of visual tokens grows rapidly with the number of input images. Existing pruning methods typically focus on either token importance or feature similarity, but they are often suboptimal in this setting: importance-only methods may retain many **duplicate tokens**, while similarity-only methods may overlook the **context-dependent relevance** induced by the current question.

**CVD-Pruner** is a **training-free** and **plug-and-play** visual token pruning method designed for efficient multi-image reasoning. Instead of selecting tokens solely by saliency or removing redundancy solely by pairwise similarity, CVD-Pruner jointly considers:

- **self relevance** of each token,
- **cross-image distinctiveness** to reduce inter-image redundancy, and
- **intra-image diversity** to preserve complementary information.

Under a fixed token budget, CVD-Pruner retains a compact yet informative subset of visual tokens, improving efficiency while preserving strong reasoning performance.

<!-- Add your teaser figure here -->
<!-- ![teaser](assets/teaser.png) -->

---
## ⚙️ Setup

### 🏝️ Environment

The experiments in this repository are based on the following environment:

- **OS**: Ubuntu 22.04
- **Python**: 3.10
- **CUDA**: 12.1
- **GPU**: NVIDIA RTX 4090
- **PyTorch**: 2.5.1 + cu121

### 📦 Installation

1. Clone this repository.

```bash
git clone https://github.com/dluo112/CVD-Pruner.git
cd CVD-Pruner
```

2. Create and activate a conda environment.

```bash
conda create -n cvd-pruner python=3.10 -y
conda activate cvd-pruner
```

3. Install dependencies.

```bash
bash install.sh
```

### ⚠️ FlashAttention Installation Note

In practice, **FlashAttention may fail to install directly via `install.sh`** on some environments.  
If this happens, please manually download and install the prebuilt wheel:

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

pip install flash_attn-2.7.4.post1+cu12torch2.5cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

After that, you can continue using the repository normally.

---

## 📊 Data

All benchmarks are evaluated through **`lmms-eval`**.

For most evaluation settings, the framework will automatically handle the required model loading and downloading.  
However, some datasets still need to be prepared manually.

### MIBench

The [**MIBench**](https://huggingface.co/datasets/StarBottle/MIBench) dataset needs to be downloaded manually and placed under:

```bash
data/mibench/
```

Please make sure the directory structure is correct before running the corresponding evaluation.

> For other datasets, please follow the dataset preparation instructions used by the `lmms-eval` tasks in your setup.

---
## 📋 Evaluation

All evaluations are conducted through **`lmms-eval`**.

### Basic Usage

```bash
cd CVD-Pruner

COMPRESSOR=cvd accelerate launch --num_processes=8 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,device_map=auto,model_name=llava_qwen \
    --tasks=muirbench,mirb,mantis,mibench \
    --batch_size=1
```
```bash
cd CVD-Pruner

COMPRESSOR=cvd IMG_KEEP_PER_IMAGE=128 ablation_mode=full accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=262144,attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks muirbench,mirb,mantis,mibench \
    --batch_size 1
```
---

<!-- ## 📈 Results

CVD-Pruner is designed to reduce computation and memory overhead while maintaining competitive performance on challenging multi-image reasoning benchmarks.

A recommended way to present results is:

| Method | Token Budget | Benchmark | Score / Accuracy | Latency | Memory |
|--------|--------------|-----------|------------------|---------|--------|
| Baseline | Full | Mantis-Eval | - | - | - |
| CVD-Pruner | 128 | Mantis-Eval | - | - | - |
| CVD-Pruner | 32 | Mantis-Eval | - | - | - |

> Replace the table above with your final numbers. -->

<!-- --- -->

## 🎟️ License

This project is released under the [Apache 2.0 license](LICENSE).

---

## 🏅 Acknowledgement

We appreciate the open-source efforts of the following projects and communities:

- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)

If your implementation additionally builds upon other repositories, please add them here as well.
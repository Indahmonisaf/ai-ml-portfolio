# Module to Create Fine-Tuning

A **modular Jupyter workflow** for fine-tuning LLMs on your own dataset. Supports both **full fine-tuning** and **parameter-efficient training (LoRA/QLoRA)** using Hugging Face **Transformers + PEFT**.

---

## Table of Contents
- [Features](#features)
- [Data Format](#data-format)
- [Environment](#environment)
- [Quick Start](#quick-start)
- [Typical Hyperparameters](#typical-hyperparameters)
- [Training](#training)
- [Evaluation](#evaluation)
- [Saving & Inference](#saving--inference)
- [Tips](#tips)
- [Roadmap](#roadmap)
- [Author](#author)

---

## Features
- 🧩 **Modular pipeline:** data loading → prompt formatting → tokenization → model building → training → evaluation → export.  
- ⚙️ **PEFT (LoRA/QLoRA)** support for low-VRAM training.  
- 🧪 **Validation loop** with extendable metrics (loss by default; easy to add Accuracy/F1/ROUGE).  
- 💾 **Checkpointing:** save adapters or full weights; export an inference-ready artifact.  
- 🔁 **Prompt templates:** instruction / input / output format for instruction-tuning.

---

## Data Format
Use **JSONL/JSON/CSV** with fields like:
```json
{"instruction": "Rewrite in formal English", "input": "pls send asap", "output": "Please send it as soon as possible."}
````

Common alternatives:

```json
{"question": "...", "answer": "..."}            // Q&A
{"system": "...", "user": "...", "assistant": "..."}  // Dialogue
```

> Adjust the **prompt formatting** cell in the notebook to map your schema into a single training text.

---

## Environment

Install base dependencies:

```bash
pip install \
  transformers datasets peft bitsandbytes \
  accelerate torch \
  numpy pandas scikit-learn tqdm jsonlines
```

> If CUDA is available, install the matching **PyTorch** build from pytorch.org.

---

## Quick Start

1. Put your dataset (e.g., `train.jsonl`, `valid.jsonl`) in the project folder.
2. Open **`Module to Create Fine Tuning.ipynb`** and run cells in order:

   * **Config & Hyperparameters** (model name, epochs, batch sizes, lr, max_seq_length)
   * **Load & Preprocess Data** (prompt formatting + tokenization)
   * **Build Model** (base or PEFT/LoRA/QLoRA)
   * **Train** (Trainer/Accelerate loop)
   * **Evaluate** (validation loss; add metrics if needed)
   * **Save / Export** (adapter or full model)
3. *(Optional)* Push to **Hugging Face Hub**.

---

## Typical Hyperparameters

```python
model_name        = "meta-llama/Llama-3.1-8B-Instruct"   # or mistral, qwen, T5, etc.
max_seq_length    = 1024
epochs            = 3
batch_size        = 2
gradient_accum    = 8
learning_rate     = 2e-4
weight_decay      = 0.01
warmup_steps      = 100

# LoRA/QLoRA (PEFT)
lora_r            = 16
lora_alpha        = 32
lora_dropout      = 0.05
load_8bit         = True   # bitsandbytes
load_4bit         = False  # set True for QLoRA if VRAM-constrained
```

> Toggle `load_8bit` / `load_4bit` based on available GPU memory. Use **QLoRA (4-bit)** for very constrained VRAM.

---

## Training

* Uses Hugging Face **Trainer** (or an equivalent Accelerate loop) with **mixed precision** and **gradient accumulation**.
* For **LoRA/QLoRA**, only adapter parameters are updated—base model remains frozen, enabling training on consumer GPUs.

---

## Evaluation

* **Default:** validation **loss**.
* Extend with `compute_metrics` to report **Accuracy/F1** (classification) or **ROUGE/BLEU** (generation).
* You can also do **manual spot-checks**: feed held-out inputs → compare model outputs vs. references.

---

## Saving & Inference

**Adapters (PEFT):** Save LoRA weights only, then load with the same base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(model_name)
tok  = AutoTokenizer.from_pretrained(model_name, use_fast=True)
ft   = PeftModel.from_pretrained(base, "path/to/lora_adapter")

prompt = "Rewrite politely: close the door"
inputs = tok(prompt, return_tensors="pt")
out = ft.generate(**inputs, max_new_tokens=128)
print(tok.decode(out[0], skip_special_tokens=True))
```

**Full fine-tuned model:** load directly via:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok  = AutoTokenizer.from_pretrained("path/to/checkpoint", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("path/to/checkpoint")
```

---

## Tips

* Keep your **prompt template** consistent between **training** and **inference**.
* Prefer **truncate** (not pad) to `max_seq_length` to avoid OOM; try **sequence packing** when appropriate.
* Start with a **small learning rate** for stability; increase **epochs** only if underfitting.
* If outputs look **repetitive**, tune **temperature / top-p** at inference and consider longer context or more examples.

---

## Roadmap

* Add **metric suite** (Accuracy/F1 or ROUGE/BLEU depending on task)
* Add **early stopping** and **best-val checkpointing**
* Provide **trainer configs** for multiple base models (LLaMA, Mistral, Qwen, T5)
* **CLI script** version (`train.py`) for headless runs
* *(Optional)* **Weights & Biases** logging

---

## Author

Built by **Indah Monisa Firdiantika (M.S.)** — a modular, low-VRAM-friendly fine-tuning pipeline for quickly adapting LLMs to domain tasks.

```
```

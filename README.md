

<p align="center">
  📌 <a href="#-overview">Overview</a> • 📊 <a href="#-evaluating-fpcc-with-llms">Evaluation</a> • 🧠 <a href="#-method-overview">Method</a> • 🚀 <a href="#-effectiveness-of-our-method-hpcl">HPCL Results</a> • ⚡ <a href="#-quick-start">Quick Start</a>
</p>

---


# 🧠 Evaluating and Improving Framework-Based Parallel Code Completion with Large Language Models

This repository contains the dataset, models, and evaluation scripts for our paper.


## 🧩 Overview

With the growing complexity of modern computing workloads, **parallel programming** has become essential for exploiting the performance potential of multicore CPUs, GPUs, and distributed clusters.  
However, **a vast amount of legacy code in scientific and engineering domains remains serial**, failing to utilize available parallel hardware.

To address this gap, developers increasingly face the need to **transform sequential functions into parallel code**, often using multiple frameworks simultaneously — such as combining **OpenMP**, **MPI**, and **CUDA** within a single application.

To better understand this real-world demand, we analyzed open-source HPC repositories on GitHub:


**Key finding:**

- **59.6%** of repositories combine two or more parallel frameworks (e.g., OpenMP + MPI + CUDA).
- Only **40.4%** of repositories use a single framework.

This shows that real-world HPC development frequently involves **multi-framework integration**, tailored to meet diverse performance goals and heterogeneous hardware constraints.



### 🔍 We Introduce: Framework-Based Parallel Code Completion (FPCC)

Despite the increasing demand for automatic parallelization, **existing datasets and methods** overwhelmingly focus on narrow, single-framework settings.  
This significantly limits their **practical relevance** in assisting real-world development workflows.

To bridge this gap, we propose **Framework-Based Parallel Code Completion (FPCC)** — a realistic code generation task that requires large language models to:

1. 🧠 **Identify insertion points** for parallel directives in serial code,  
2. ⚙️ **Select the appropriate parallel framework** (e.g., OpenMP, MPI, CUDA),  
3. 🧾 **Generate complete and correct directive statements**, including proper clause usage.


FPCC moves beyond basic API generation tasks and frames parallelization as a **structured transformation problem** — requiring **multi-step reasoning over program structure, control flow, and data dependencies**.

By aligning evaluation with real-world code transformation needs, FPCC offers a principled benchmark for building next-generation intelligent assistants for parallel programming.



## 📊 Evaluating FPCC with LLMs

We conduct a systematic empirical study to evaluate the performance of large language models (LLMs) on the FPCC task.

Our evaluation covers:
- ⚙️ Zero-shot prompting using popular open-source and closed-source LLMs
- 🧪 Multi-stage fine-tuning under task decomposition
- 📈 Metrics including line-level F1, framework accuracy, and CodeBLEU

To support reproducibility, we provide our **prompt templates** used for zero-shot and 3-shot evaluation in `data/prompt/`:

### 🧾  Zero-shot and 3-shot Performance on FPCC

| Model                        | EM (0) | EM (3) | STF (0) | STF (3) | IP (0) | IP (3) | FW (0) | FW (3) | DIR (0) | DIR (3) |
|-----------------------------|--------|--------|---------|---------|--------|--------|--------|--------|---------|---------|
| Llama3.1-8B                 | 0.03   | 0.09   | 0.02    | 0.06    | 2.29   | 2.35   | 8.32   | 25.38  | 0.76    | 1.32    |
| **GPT-4**                   | **0.43** | **1.81** | **0.37**  | **1.36**  | **6.83** | **7.66** | **32.29** | 44.61 |3.97  | **5.87**  |
| Qwen2.5-Coder-32B-Instruct  | 0.12   | 0.42   | 0.13    | 0.37    | 3.97   | 5.44   | 23.15  | 42.95  | **4.02**    | 5.27    |
| Qwen2.5-Coder-7B-Instruct   | 0.00   | 0.30   | 0.00    | 0.19    | 5.19   | 4.93   | 31.90  | **70.32**  | 3.73    | 5.54    |
| StarCoder2-7B               | 0.00   | 0.00   | 0.00    | 0.00    | 0.24   | 0.21   | 0.16   | 0.50   | 0.09    | 0.11    |
| CodeLlama-7B-hf             | 0.00   | 0.00   | 0.00    | 0.00    | 0.25   | 0.30   | 0.20   | 0.52   | 0.10    | 0.13    |

### 📐 Single-Framework vs Multi-Framework Evaluation

We compare model performance on single-framework vs multi-framework samples using the Qwen2.5-Coder-7B-Instruct model and its HPCL-enhanced variant.

| Approach                          | Sample Type       | EM    | SFT   | IP     | FW     | DIR    |
|----------------------------------|-------------------|-------|-------|--------|--------|--------|
| Qwen2.5-Coder-7B-Instruct         | Single-Framework  | 0.30  | 0.19  | 4.93   | 70.32  | 5.54   |
|                                  | Multi-Framework   | 0.00  | 0.00  | 0.34   | 38.56  | 0.76   |
| Qwen2.5-Coder-7B-Instruct + HPCL | Single-Framework  | **46.14** | **46.00** | **56.75** | **99.33** | **44.43** |
|                                  | Multi-Framework   | 9.94  | 9.68  | 19.42  | 96.81  | 7.79   |




## 🧪 Method Overview

Our approach consists of a structured pipeline to build and evaluate a realistic benchmark for multi-framework parallel code completion.

<p align="center">
  <img src="images/overview.jpg" alt="Method Overview" width="850"/>
</p>

- 🔍 **Data Construction**: We extract and normalize parallel code from GitHub, ensuring license compliance and quality. Each sample is labeled with insertion points and target frameworks (OpenMP, MPI, CUDA, etc).
- 📊 **Sample Difficulty Estimation**: We assess instance complexity (e.g., cyclomatic complexity, loop depth) to support curriculum learning.
- 🧠 **Multi-Stage Training**: We decompose the FPCC task into four sub-tasks and train LLMs accordingly.
- 🔍 **Limitation Analysis**: We analyze typical model failure cases to inform future improvements.


## 🚀 Effectiveness of Our Method: HPCL

We evaluate the impact of our proposed Hierarchical Progressive Curriculum Learning (HPCL) across four representative LLMs.
Results show that HPCL consistently improves performance across all sub-tasks in the FPCC benchmark.

| Model                       | EM     | SFT    | IP     | FW     | DIR    |
|----------------------------|--------|--------|--------|--------|--------|
| Llama3.1-8B                | 0.09   | 0.06   | 2.35   | 25.38  | 1.32   |
| + SFT                     | 36.85  | 36.72  | 49.32  | 98.81  | 40.35  |
| + HPCL                    | **43.21**  | **43.05**  | **54.80**  | 98.79  | **43.88**  |
| StarCoder2-7B             | 0.00   | 0.00   | 0.21   | 0.50   | 0.11   |
| + SFT                     | 31.05  | 30.95  | 43.19  | 98.68  | 38.08  |
| + HPCL                    | **42.17**  | **42.05**  | **53.47**  | 98.77  | **43.48**  |
| CodeLlama-7B-hf           | 0.00   | 0.00   | 0.30   | 0.52   | 0.13   |
| + SFT                     | 38.32  | 38.44  | 50.08  | 98.87  | 41.74  |
| + HPCL                    | **44.72**  | **44.64**  | **56.15**  | 98.86  | **44.12**  |
| Qwen2.5-Coder-7B-Instruct | 0.30   | 0.19   | 4.93   | 70.32  | 5.54   |
| + SFT                     | 39.16  | 39.08  | 50.09  | 99.05  | 42.14  |
| + HPCL                    | **46.14**  | **46.00**  | **56.75**  | **99.33**  | **44.43**  |

## ⚡ Quick Start

You can quickly load and use our **HPCL-Coder** model via the 🤗 Hugging Face `transformers` library.

### 🔧 Installation

```bash
pip install transformers accelerate
```
### 🧠 Load the HPCL Model

```bash
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("HPCL/HPCL-Coder")
model = AutoModelForCausalLM.from_pretrained("HPCL/HPCL-Coder")

# Move to GPU if available
import torch
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
```
🚀 Run Inference on a Prompt
```bash

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
📁 You can also use our structured prompt templates in data/prompt/ to guide FPCC-style completions.







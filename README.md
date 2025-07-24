

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



## ⚙️ Model Configuration

### 🔄 Decoding & Sampling Config

| Parameter             | Value            |
|-----------------------|------------------|
| `seed`                | 42               |
| `do_sample`           | `true`           |
| `temperature`         | 0.7              |
| `top_k`               | 20               |
| `top_p`               | 0.8              |
| `repetition_penalty` | 1.1              |
| `transformers_version`| 4.49.0           |

> These decoding settings are consistently used in all evaluations.

---

### 🏋️ Training Config (Shared by HPCL and SFT)

All models were trained using the **same set of hyperparameters**, whether using standard supervised fine-tuning (SFT) or our proposed Hierarchical Progressive Curriculum Learning (HPCL) strategy.

| Parameter                     | Value     |
|-------------------------------|-----------|
| `optimizer`                   | AdamW     |
| `lr_scheduler_type`           | cosine    |
| `learning_rate`               | 5e-5      |
| `num_train_epochs`            | 3         |
| `warmup_ratio`                | 0.03      |
| `per_device_train_batch_size` | 4         |
| `gradient_accumulation_steps` | 8         |
| `weight_decay`                | 0.01      |
| `max_seq_length`              | 2048      |

> ✅ These settings are **fixed across all training experiments** to ensure fair comparison between SFT and HPCL strategies.

### 🖥️ Hardware Platform

All training and evaluation experiments were conducted on the following hardware:


| Component          | Specification                                      |
|--------------------|----------------------------------------------------|
| **CPU**            | 2 × Intel Xeon Platinum 8358 @ 2.60GHz (128 cores) |
| **RAM**            | 512 GB DDR4                                        |
| **GPU**            | 8 × NVIDIA H100 PCIe (80 GB each)                  |
| **CUDA Version**   | 12.4                                               |
| **Driver Version** | 550.54.15                                          |
| **OS**             | Ubuntu 20.04.6 LTS (Linux 5.15 kernel)             |


> ✅ All experiments were run using the same hardware setup to ensure consistency across model variants and tasks.



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



### ❗ Error Type Distribution Comparison (HPCL vs SFT)

We manually annotated a sample of model outputs under both SFT and HPCL training regimes.  
The table below shows the distribution of fine-grained error types under each setting:

| High-Level Error Category   | Fine-Grained Error Type          | Description                                                                                                           |   HPCL (%) |   SFT (%) |
|:----------------------------|:---------------------------------|:----------------------------------------------------------------------------------------------------------------------|-----------:|----------:|
| Insertion Point Error       | Misaligned Directive             | Directive inserted at incorrect line, affects scope or logic.                                                         |      30.19 |     40.19 |
|                             | Scope Violation                  | Directive placed outside its intended loop or region.                                                                 |       6.73 |     10.19 |
|                             | Execution Order Misunderstanding | Incorrect placement disrupts logical execution sequence.                                                              |       4.04 |      6.15 |
| Missing Completion          | Missing Directive                | Expected parallel construct is not generated.                                                                         |       8.85 |     14.42 |
|                             | Missing Clause                   | Directive lacks required clause like num_threads, collapse.                                                            |       1.73 |      2.12 |
|                             | Missing Synchronization          | Omitted necessary barrier or wait, risking race conditions.                                                            |       0.38 |      1.35 |
|                             | Missing Finalization             | Post-execution validation or clean-up steps omitted (e.g., getLastCudaError, cudaFree, MPI_Finalize).                |       0.38 |      1.15 |
| Redundant Completion        | Redundant Directive              | Inserted directive is unnecessary given existing context.                                                             |      14.42 |     20.19 |
|                             | Duplicate Operation              | Same memory transfer or barrier duplicated.                                                                           |       1.15 |      3.08 |
|                             | Hallucinated Code                | Model generated directive/function unrelated to context.                                                              |       2.50 |      6.15 |
| Incorrect Usage             | Wrong Directive Type             | Incorrect parallel directive selected (e.g., MPI_Wait vs MPI_Bcast).                                                  |       9.04 |     13.08 |
|                             | Wrong Framework                  | Used directives from the wrong parallel framework.                                                                    |       0.58 |      2.31 |
|                             | Incorrect Variable or Buffer     | Used wrong variable in directive or data transfer.                                                                    |       5.96 |     10.96 |
|                             | Invalid Argument                 | Function call includes wrong or mismatched arguments.                                                                 |       2.88 |      4.62 |
|                             | Incorrect Clause Specification   | Clause value is invalid (e.g., collapse(-1), if()).                                                                   |       3.85 |      6.54 |
|                             | Invalid Synchronization Usage    | Synchronization APIs used in wrong order or scope.                                                                    |       0.77 |      2.12 |
| Syntax Error                | Malformed Directive              | Directive syntax is incorrect (e.g., collapse(4,)).                                                                   |       4.04 |      5.96 |
|                             | Invalid Clause Syntax            | Clause malformed or includes illegal values.                                                                          |       1.73 |      3.08 |
|                             | Typographical Error              | Code contains undefined symbols, typos, or unclosed expressions.                                                      |       0.77 |      2.31 |


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







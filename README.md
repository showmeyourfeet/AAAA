# Hierarchical-Control-Systems-for-Robotic

This project implements various hierarchical control systems and Vision-Language-Action (VLA) architectures for robotic surgery and manipulation tasks. It explores different ways to decompose control into high-level planning and low-level execution.

## Project Structure

```
/
├── hlp/                  # High-Level Policy (Original Implementation)
│   ├── dataset.py        # HLP data loading
│   ├── model.py          # Swin-Tiny + Transformer/Gated Attention
│   └── train.py          # Training script
├── llp/                  # Low-Level Policy (Original Implementation)
│   ├── model.py          # DETR-VAE + Gated Attention (ACT-based)
│   ├── train.py          # Training script
│   └── ...
├── pi_0_5_torch/         # Pi0.5 Model Implementation (Physical Intelligence)
│   ├── model.py          # PaliGemma + Gemma Expert + Flow Matching
│   ├── train.py          # Training script
│   └── ...
├── SmolVLA_torch/        # SmolVLA Model Implementation
│   ├── model.py          # SmolVLM + Action Expert (Cross-Attention)
│   ├── train.py          # Training script
│   └── ...
├── additional_modules/   # Custom Neural Network Modules
│   └── gated_attention_SDPA.py # Gated Multi-Head Attention
└── ...
```

## 1. Standard Hierarchical Implementation (`hlp/` & `llp/`)

This is the original two-stage approach:

### High-Level Policy (HLP)
Predicts the high-level goal or "stage" of the task.
*   **Architecture:** **Swin-Tiny** image encoder + **Transformer/Gated Attention** decoder.
*   **Input:** Multi-camera image sequences.
*   **Output:** Command/Stage embedding.

### Low-Level Policy (LLP)
Executes high-level commands. This module (`llp/policy.py`) now supports multiple policy architectures:

1.  **ACT (Action Chunking Transformer):**
    *   **Architecture:** DETR-VAE based (CVAE encoder + Transformer Decoder).
    *   **Features:** Supports optional **VQ-VAE** (Vector Quantization) and Gated Attention.
2.  **Diffusion Policy:**
    *   **Architecture:** **Conditional U-Net** (1D) or **DiT** (Diffusion Transformer).
    *   **Mechanism:** Predicts action sequences via iterative denoising (DDIM Scheduler).
    *   **Backbone:** ResNet or EfficientNet (FiLMed for language conditioning).
3.  **Flow Matching Policy:**
    *   **Architecture:** **DiT** (Diffusion Transformer) or **U-Net**.
    *   **Mechanism:** Pi0.5-style Rectified Flow Matching (Euler sampling). Samples timesteps from a Beta distribution ($ \alpha=1.5, \beta=1 $).
    *   **Features:** Efficient inference with **EMA** (Exponential Moving Average) support.

## 2. Pi0.5 Implementation (`pi_0_5_torch/`)

A faithful implementation of the **Pi0.5** architecture from OpenPI (Physical Intelligence).

*   **Core Concept:** Hierarchical VLA with Flow Matching.
*   **Architecture:**
    *   **High-Level:** **PaliGemma** (VLM) for subtask prediction ($ \pi_\theta(\hat{\ell} | o_t, \ell) $).
    *   **Low-Level:** **Gemma** (Action Expert) for action generation ($ \pi_\theta(a_{t:t+H} | o_t, \hat{\ell}) $).
*   **Mechanism:**
    *   Uses **Flow Matching** for action generation (instead of VAE/Diffusion).
    *   Uses **AdaRMS** for time/condition injection.
    *   Supports multi-camera inputs.
*   **Training:** Jointly trains subtask prediction (Cross-Entropy) and action generation (Flow Matching).

## 3. SmolVLA Implementation (`SmolVLA_torch/`)

A lightweight VLA model designed for efficiency.

*   **Architecture:**
    *   **Backbone:** **SmolVLM2-500M-Video-Instruct** (Vision-Language Model).
    *   **Action Expert:** A smaller Transformer Decoder (initialized from VLM config).
*   **Mechanism:**
    *   **Interleaved Attention:** The Action Expert attends to the VLM's hidden states via **Cross-Attention** layers inserted every $N$ layers.
    *   **RoPE:** Uses Rotary Positional Embeddings.
    *   **Efficiency:** Uses a smaller parameter count for the action head while leveraging a pre-trained VLM.

## Custom Modules (`additional_modules/`)

*   **Gated Attention:** Implements **Gated Multihead Attention** (`gated_attention_SDPA.py`) to introduce gating signals into attention mechanisms, improving stability for long-horizon tasks.

## Key Technologies

*   **PyTorch**
*   **Transformers (Hugging Face)**
*   **PaliGemma / Gemma / SmolVLM**
*   **DETR / ACT**
*   **Flow Matching / VAE**

```
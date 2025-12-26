# Hierarchical-Control-Systems-for-Robotic

This project implements a hierarchical control system for robotic surgery/manipulation tasks. It decomposes the control problem into two levels:
1.  **High-Level Policy (HLP):** Determines the current "stage" or high-level command (e.g., "grasp needle", "move to suture") based on visual observations.
2.  **Low-Level Policy (LLP):** Generates precise robot joint actions to execute the command given by the HLP, conditioned on visual feedback and robot state.

## Project Structure

```
/
├── hlp/                  # High-Level Policy
│   ├── dataset.py        # HLP data loading (splitted, composite, etc.)
│   ├── model.py          # HLP architecture (Swin-Tiny + Transformer/Gated Attention)
│   └── train.py          # HLP training script
├── llp/                  # Low-Level Policy
│   ├── model.py          # LLP architecture (DETR-VAE + Gated Attention)
│   ├── train.py          # LLP training script
│   ├── policy.py         # Policy wrapper
│   └── ...               # Backbone implementations (EfficientNet, ResNet)
├── additional_modules/   # Custom Neural Network Modules
│   └── gated_attention_SDPA.py # Gated Multi-Head Attention implementation
|   └── DETRTransformer.py # DETR-style Transformer block implementation
|   └── ...
└── ...
```

## High-Level Policy (HLP)

The HLP predicts the high-level goal or "stage" of the task.

*   **Architecture:** Uses a **Swin-Tiny** image encoder followed by a decoder model. The decoder model can be a standard **DETR Transformer Decoder** or a custom **Gated Attention DETR Decoder**.
*   **Input:** Sequence of images from multiple cameras (e.g., `left_frame`, `right_frame`).
*   **Output:** Instrcuction command embedding, Correction flag embedding, Correction command flag.

### Training

To train the HLP, use `hlp/train.py`. The following command (based on `train_hl.sh`) demonstrates a robust training configuration:

```bash
python hlp/train.py \
  --normal_root_dir datasets/normal_training_datasets \
  --normal_val_root datasets/normal_validation_datasets \
  --stage_embeddings_file utils/stage_embeddings_file.json \
  --stage_texts_file utils/stage_embeddings_file.json \
  --dagger_embeddings_file utils/dagger_embeddings_file.json \
  --ckpt_dir ckpts/hl_test \
  --batch_size 4 \
  --num_epochs 1000 \
  --camera_names left_frame \
  --train_image_encoder \
  --lr 1e-5 \
  --seed 42 \
  --gpu 0 \
  --history_len 4 \
  --history_skip_frame 12 \
  --prediction_offset 15 \
  --use_composite \
  --use_augmentation \
  --sampling_strategy sequential \
  --n_repeats 1 \
  --use_weak_traversal \
  --samples_non_cross_per_stage 2 \
  --samples_cross_per_stage 2 \
  --use_dagger \
  --dagger_root_dir datasets/dagger_training_datasets \
  --dagger_val_root datasets/dagger_validation_datasets \
  --dagger_mix_ratio 0.5 \
  --use_gated_attention \
  --mlp_type swiglu \
  --gate_mode element-wise
```

**Key Arguments:**
*   `--use_gated_attention`: Enables the custom Gated Transformer Encoder.
*   `--train_image_encoder`: Unfreezes the Swin-Tiny backbone for fine-tuning.
*   `--use_composite`: Uses composite dataset sampling (combining runs).
*   `--use_dagger`: Enable dagger training mode.

## Low-Level Policy (LLP)

The LLP executes the high-level commands.

*   **Architecture:** Based on **ACT (Action Chunking Transformer)**. It uses a **DETR**-like encoder-decoder with a **VAE** for generative modeling. It can optionally use **Gated Attention** layers.
*   **Backbone:** Supports **EfficientNet** (e.g., `efficientnet_b3film`) or **ResNet** with FiLM layers for language/command conditioning.
*   **Input:** Images, Robot Joint State (`qpos`), and High-Level Command Embedding.
*   **Output:** A sequence (chunk) of future actions.

### Training

To train the LLP, use `llp/train.py`. The following command (based on `train_ll.sh`) shows a standard configuration:

```bash
python llp/train.py \
  --ckpt_dir ckpts/ll_test \
  --use_language \
  --use_splitted \
  --splitted_root splitted_datasets/training_datasets \
  --val_splitted_root splitted_datasets/validation_datasets \
  --stage_embeddings_file utils/stage_embeddings_file.json \
  --batch_size 8 \
  --seed 42 \
  --kl_weight 20 \
  --num_epochs 10 \
  --lr 2e-5 \
  --hidden_dim 512 \
  --dim_feedforward 3200 \
  --chunk_size 40 \
  --camera_names left_frame right_frame \
  --image_encoder efficientnet_b3film \
  --train_image_encoder \
  --best_model_metric l1_loss \
  --prefetch_factor 1 \
  --num_workers 1 \
  --dec_layers_num 7 \
  --gpu 0 \
  --save_ckpt_every 2 \
  --save_latest_every 5 \
  --constant_lr \
  --shared_backbone true \
  --use_gated_attention
```

**Key Arguments:**
*   `--shared_backbone true`: Shares the image encoder weights across different camera views.
*   `--use_gated_attention`: Replaces standard attention with Gated Attention in the DETR transformer.
*   `--chunk_size`: Length of the predicted action sequence.
*   `--kl_weight`: Weight for the KL divergence loss term (VAE).

## Custom Modules

### Gated Attention (`additional_modules/`)
This project implements a novel **Gated Multihead Attention** mechanism (`gated_attention_SDPA.py`). This mechanism introduces a gating signal to the attention output, potentially improving stability and performance for long-horizon tasks. This custom layer is integrated into both the HLP and LLP transformers when the `--use_gated_attention` flag is set.

## Key Technologies

*   **PyTorch**: Deep learning framework.
*   **Transformers**: Core architecture for both policies.
*   **DETR**: Object detection transformer adapted for action prediction.
*   **VAE/VQ-VAE**: Variational Autoencoder for handling multimodal action distributions.
*   **Swin Transformer**: Hierarchical vision transformer used in HLP.
*   **EfficientNet/ResNet**: CNN backbones used in LLP.

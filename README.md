# AutoSurgical-Algorithms

This project implements a two-tiered policy for controlling a surgical robot. It uses a high-level policy to determine the overall goal from camera images and a low-level policy to generate the specific actions for the robot to execute.

## Project Structure

```
/
├── HLP/
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── LLP/
│   ├── _efficientnet.py
│   ├── _resnet.py
│   ├── backbone_impl.py
│   ├── dataset.py
│   ├── DETRTransformer.py
│   ├── model.py
│   ├── policy.py
│   ├── train.py
│   └── utils.py
```

## High-Level Policy

The high-level policy takes in images from multiple cameras and determines the overall goal or command (e.g., "move to the bag"). It uses a Transformer-based model (`HighLevelModel`) to process image sequences and predict the correct command from a predefined set.

### Training

To train the high-level policy, run the following command:

```bash
python hlp/train.py --splitted_root /home/beta/projs/pys/AutoSurgical-Algorithms/srt-h/splitted_datasets/training_datasets  --val_splitted_root /home/beta/projs/pys/AutoSurgical-Algorithms/srt-h/splitted_datasets/validation_datasets --stage_embeddings_file /home/beta/projs/pys/AutoSurgical-Algorithms/srt-h/splitted_datasets/stage_embeddings_distilbert.json  --stage_texts_file /home/beta/projs/pys/AutoSurgical-Algorithms/srt-h/splitted_datasets/stage_embeddings_distilbert.json   --ckpt_dir ./hlp/ckpts --batch_size 16 --num_epochs 1000 --lr 1e-5 --seed 0 --gpu 0  --history_len 4 --history_skip_frame 10  --prediction_offset 15 --use_composite --n_repeats 10 --use_augmentation
```

## Low-Level Policy

The low-level policy takes the high-level command and the current robot state (images and joint positions) to generate the specific, low-level actions for the robot to execute. It uses a DETR-like model (`DETRVAE`) with a FiLMed backbone to generate a sequence of actions.

### Training

To train the low-level policy, run the following command:

```bash
python llp/train.py --splitted_root /home/beta/projs/pys/AutoSurgical-Algorithms/srt-h/splitted_datasets/training_datasets --val_splitted_root /home/beta/projs/pys/AutoSurgical-Algorithms/srt-h/splitted_datasets/validation_datasets --stage_embeddings_file /home/beta/projs/pys/AutoSurgical-Algorithms/srt-h/splitted_datasets/stage_embeddings_distilbert.json --ckpt_dir ./llp/ckpts --batch_size 16 --seed 42 --num_epochs 1500 --lr 1e-4 --chunk_size 30 --hidden_dim 512 --dim_feedforward 3200 --kl_weight 80 --image_encoder efficientnet_b3film --use_language --language_encoder distilbert --use_splitted --no_encoder --use_augmentation
```

## Key Technologies

*   PyTorch
*   Transformers
*   ResNet and EfficientNet
*   FiLM (Feature-wise Linear Modulation)

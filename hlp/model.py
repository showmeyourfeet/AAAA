import torch
import torch.nn as nn
import numpy as np
import timm
import torchvision.transforms as transforms
import random

from additional_modules.DETRTransformer import TransformerDecoder, TransformerDecoderLayer
from additional_modules.gated_attention_SDPA import GatedDETRDecoderLayer
# Image preprocessing for ImageNet-pretrained encoders (e.g., Swin Tiny)
# NOTE: Currently dataset returns images of various sizes (e.g., 640x480 for cam_high),
# so we need to resize to 224x224 here. If you want to optimize, consider resizing
# in the dataset to 224x224 and removing the Resize here (only keep Normalize).
image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Required: dataset images are not 224x224
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

ONE_HOT = False  # ablation TODO: add argument


class HighLevelModel(nn.Module):
    def __init__(
        self,
        device,
        history_len,
        output_size=768,
        hidden_size=512,
        num_heads=8,
        num_layers=6,
        candidate_embeddings=None,
        candidate_texts=None,
        command_to_index=None,
        num_cameras=4,
        train_image_encoder=False,
        aggregation_mode="last",  # 'last', 'avg', or 'cls'
        use_gated_attention: bool = False,  # whether to use gated attention encoder
        mlp_type: str = "swiglu",  # MLP type for gated attention layers
        gate_mode: str = "element-wise",  # Gate mode for gated attention layers
        num_queries: int = 3,
    ):
        super().__init__()
        self.device = device
        self.num_cameras = num_cameras
        self.history_len = history_len
        self.num_queries = num_queries
        self.candidate_embeddings = candidate_embeddings
        self.candidate_texts = candidate_texts
        self.use_gated_attention = use_gated_attention

        # Load the pretrained Swin-Tiny image encoder (timm)
        # Outputs a pooled feature vector of dimension self.visual_out_dim
        self.image_encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0,         # remove classifier, return pooled features
            global_pool="avg",
        )
        self.visual_out_dim = self.image_encoder.num_features
        
        # Conditionally freeze/unfreeze image encoder parameters
        if not train_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False  # Freeze image encoder parameters

        # Transformer for processing sequences of image embeddings
        # Two options:
        #   1) standard TransformerEncoder (seq-first)
        #   2) custom gated-attention encoder (batch-first)
        self.d_model = self.visual_out_dim
        self.use_gated_attention = use_gated_attention
        if self.use_gated_attention:
            print(f"Using GatedDETRDecoderLayer with mlp_type={mlp_type}, gate_mode={gate_mode}")
            # use the gated attention
            decoder_layer = GatedDETRDecoderLayer(
                d_model=self.d_model,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                dropout=0.1,
                activation="relu",
                normalize_before=False,
                mlp_type=mlp_type,      # pass the MLP type (e.g. SwiGLU)
                gate_mode=gate_mode     # pass the Gate mode (e.g. element-wise)
            )
        else:
            print("Using Vanilla TransformerDecoderLayer")
            decoder_layer = TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                dropout=0.1,
                activation="relu",
                normalize_before=False
            )
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers = num_layers,
            norm=nn.LayerNorm(self.d_model),
            return_intermediate=False,
        )
        self.query_embed = nn.Embedding(num_queries, self.d_model) 

        if num_cameras > 1:
            self.temporal_positional_encoding = self.create_sinusoidal_embeddings(
                self.d_model, history_len + 1
            )
            self.camera_embedding = nn.Embedding(num_cameras, self.d_model)
        else:
            self.positional_encoding = self.create_sinusoidal_embeddings(
                self.d_model, history_len + 1
            )

        # Prediction Heads (Regression Heads)
        self.instruction_proj = nn.Sequential(
            nn.Linear(self.d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        self.correction_flag_proj = nn.Sequential(
            nn.Linear(self.d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.correction_instruction_proj = nn.Sequential(
            nn.Linear(self.d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        
        self.temperature = nn.Parameter(torch.ones(1))

        total, trainable = count_parameters(self)
        print(f"Total parameters: {total / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable / 1e6:.2f}M")
        print(f"Aggregation mode: {aggregation_mode}")
        print(f"Use gated attention: {self.use_gated_attention}")

    def forward(self, images):
        # Given images of shape (bs, ts, k, c, h, w)
        batch_size, timesteps, num_cameras, c, h, w = images.shape

        # Check if padding is required
        if timesteps < self.history_len + 1:
            padding_needed = self.history_len + 1 - timesteps
            padding = torch.zeros(
                (batch_size, padding_needed, num_cameras, c, h, w), device=images.device
            )
            images = torch.cat([padding, images], dim=1)
            timesteps = (
                self.history_len + 1
            )  # Update timesteps to reflect the new length

        # Reshape images to (bs*ts*k, c, h, w) for processing through CLIP
        images_reshaped = images.reshape(batch_size * timesteps * num_cameras, c, h, w)

        # Apply transformations for encoder (resize + ImageNet normalize)
        images_transformed = image_transform(images_reshaped)

        # Get image features from Swin-Tiny
        image_features = self.image_encoder(images_transformed)  # (B*, D)

        # reshape the image features to (bs, ts, k, d)
        memory_features = image_features.reshape(batch_size, timesteps, num_cameras, -1)

        # build the positional encoding
        if num_cameras > 1:
            temporal_pe = self.temporal_positional_encoding[:timesteps, :].to(images.device).unsqueeze(0).unsqueeze(2) # (1, T, 1, D)
            camera_pe = self.camera_embedding.weight[:num_cameras, :].unsqueeze(0).unsqueeze(1) # (1, 1, K, D)
            pos_embed = temporal_pe + camera_pe # Broadcasting (1, T, K, D)
        else:
            pos_embed = self.positional_encoding[:timesteps, :].to(images.device).unsqueeze(0).unsqueeze(2) # (1, T, 1, D)
        # expand the positional encoding to the batch size dimension
        pos_embed = pos_embed.expand(batch_size, -1, -1, -1) # (bs, T*K, D)

        # Prepare the memory features for the Transformer Decoder
        # The expected shape is [seq_len, bs, d]
        # original ([bs, ts, k, d]) --> flatten ([bs, ts*k, d]) --> permute ([ts*k, bs, d])
        memory = memory_features.reshape(batch_size, timesteps * num_cameras, -1).permute(1, 0, 2)
        pos = pos_embed.reshape(batch_size, timesteps * num_cameras, -1).permute(1, 0, 2)

        # Prepare the Queries
        # query embed: [num_queries, d] --> [num_queries, bs, d]
        # target: [num_queries, bs, d], initialized to 0 according to the DETRTransformer
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1) # (num_queries, bs, d)
        tgt = torch.zeros_like(query_pos) # (num_queries, bs, d)

        # Decoer Forward Pass
        decoder_output = self.decoder(
            tgt,
            memory,
            pos=pos,
            memory_key_padding_mask=None, # no padding mask for the memory
            query_pos=query_pos,
        ) # (num_queries, bs, d)

        # Get the last layer output and permute to (bs, num_queries, d)
        final_features = decoder_output[-1].permute(1, 0, 2) # (bs, num_queries, d)

        # Prediction Heads
        feat_instruction = final_features[:, 0, :]
        feat_correction_flag = final_features[:, 1, :]
        feat_correction_cmd = final_features[:, 2, :]

        pred_instruction = self.instruction_proj(feat_instruction) # (bs, output_size)
        pred_correction_flag = self.correction_flag_proj(feat_correction_flag) # (bs, 1)
        pred_correction_cmd = self.correction_instruction_proj(feat_correction_cmd) # (bs, output_size)

        pred_instruction_logits = self.compute_similarities(pred_instruction) / self.temperature.clamp(min=1e-8)
        # pass the raw logits without similarity computation
        pred_correction_flag_logits = pred_correction_flag
        pred_correction_cmd_logits = self.compute_similarities(pred_correction_cmd) / self.temperature.clamp(min=1e-8)

        return pred_instruction_logits, pred_correction_flag_logits, pred_correction_cmd_logits, self.temperature


    def compute_similarities(self, embeddings):
        # Compute the cosine similarities
        cosine_similarities = (embeddings @ self.candidate_embeddings.T) / (
            embeddings.norm(dim=-1, keepdim=True)
            * self.candidate_embeddings.norm(dim=-1, keepdim=True).T
        )

        return cosine_similarities

    def create_sinusoidal_embeddings(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def decode_logits(self, logits, temperature):
        # Compute the probabilities
        probs = (
            logits
            if ONE_HOT
            else torch.nn.functional.softmax(logits / temperature, dim=-1)
        )

        # Find the indices of the max logit for each example in the batch
        _, max_indices = torch.max(probs, dim=-1)

        return [self.candidate_texts[index] for index in max_indices.cpu().numpy()]

    def get_nearest_text(self, embeddings):
        # Compute cosine similarities
        similarities = self.compute_similarities(embeddings)

        # Get the index of the maximum similarity for each prediction
        indices = similarities.argmax(dim=-1)

        # Map the indices back to the actual texts
        return [self.candidate_texts[i] for i in indices.cpu().numpy()]

    def get_nearest_embedding(self, embeddings):
        # Compute cosine similarities
        similarities = self.compute_similarities(embeddings)

        # Get the index of the maximum similarity for each prediction
        indices = similarities.argmax(dim=-1)

        # Print the top 5 candidates
        probs = torch.nn.functional.softmax(similarities, dim=-1)
        top_probs, top_indices = torch.topk(probs[0], 5)
        normalized_top_probs = top_probs / top_probs.sum()
        for i, (index, prob) in enumerate(zip(top_indices, normalized_top_probs)):
            print(
                f"Candidate {i}: {self.candidate_texts[index]}, Normalized Prob: {prob:.4f}"
            )

        # Map the indices back to the embeddings
        return [self.candidate_embeddings[i] for i in indices.cpu().numpy()]

    def get_random_from_top_k(self, embeddings, k=3):
        similarities = (embeddings @ self.candidate_embeddings.T) / (
            embeddings.norm(dim=-1, keepdim=True)
            * self.candidate_embeddings.norm(dim=-1, keepdim=True).T
        )
        top_k_indices = similarities.topk(k, dim=-1)[1]

        # Randomly select one from the top-k for each row
        selected_indices = [
            random.choice(indices_row) for indices_row in top_k_indices.cpu().numpy()
        ]

        return [self.candidate_texts[i] for i in selected_indices]

    def sample_with_temperature(self, embeddings, temperature=1.0):
        similarities = (embeddings @ self.candidate_embeddings.T) / (
            embeddings.norm(dim=-1, keepdim=True)
            * self.candidate_embeddings.norm(dim=-1, keepdim=True).T
        )
        probs = torch.nn.functional.softmax(similarities / temperature, dim=-1)
        sampled_indices = torch.multinomial(
            probs, 1
        ).squeeze()  # Squeezing to potentially remove singleton dimensions
        # Check if sampled_indices is a scalar (0-dim) or an array
        if sampled_indices.ndim == 0:
            # If it's a scalar, we make it a one-element array
            sampled_indices = [sampled_indices.item()]
        else:
            # Otherwise, we convert it to a list
            sampled_indices = sampled_indices.tolist()

        return [self.candidate_texts[i] for i in sampled_indices]


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# Example usage:
if __name__ == "__main__":
    from dataset import load_data

    # Dataset and Dataloader parameters
    dataset_dir = "/scr/lucyshi/dataset/aloha_bag_3_objects"
    num_episodes = 10
    camera_names = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    batch_size_train = 8
    batch_size_val = 8

    # Load the dataloader
    train_dataloader, val_dataloader, _ = load_data(
        dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val
    )

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HighLevelModel(device=device, history_len=5)
    model.to(device)

    # Fetch a batch of data and pass it through the model
    for image_data, language_data, _ in train_dataloader:
        image_data = image_data.to(device)
        predictions = model(image_data)
        print(f"Image data shape: {image_data.shape}")
        print(f"Language data shape: {language_data.shape}")
        print(f"Predictions shape: {predictions.shape}")
        break

import torch
import torch.nn as nn
import numpy as np
import timm
import torchvision.transforms as transforms
import random

from additional_modules.gated_attention_SDPA import GatedTransformerEncoderLayer

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
    ):
        super().__init__()
        
        assert aggregation_mode in ("last", "avg", "cls"), \
            f"aggregation_mode must be 'last', 'avg', or 'cls', got {aggregation_mode}"
        self.aggregation_mode = aggregation_mode
        self.num_cameras = num_cameras

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
        self.use_gated_attention = use_gated_attention
        if self.use_gated_attention:
            # stack several gated encoder layers (batch-first: [B, T, D])
            self.gated_layers = nn.ModuleList(
                [
                    GatedTransformerEncoderLayer(
                        d_model=self.visual_out_dim,
                        num_heads=num_heads,
                        dim_feedforward=hidden_size,
                        mlp_type=mlp_type,
                        gate_mode=gate_mode,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.visual_out_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_size,
                    batch_first=True,
                ),
                num_layers=num_layers,
            )

        if ONE_HOT:
            output_size = len(candidate_texts)

        self.mlp = nn.Sequential(
            nn.Linear(self.visual_out_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1))

        # Positional Encoding: separate temporal and camera encoding for multi-camera
        if num_cameras > 1:
            # Temporal positional encoding (for each timestep)
            self.temporal_positional_encoding = self.create_sinusoidal_embeddings(
                self.visual_out_dim, history_len + 1
            )
            # Learnable camera embedding
            self.camera_embedding = nn.Embedding(num_cameras, self.visual_out_dim)
        else:
            # Single camera: use original flat positional encoding
            self.positional_encoding = self.create_sinusoidal_embeddings(
                self.visual_out_dim, history_len + 1
            )
        
        # CLS token for 'cls' aggregation mode
        if aggregation_mode == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.visual_out_dim))

        self.history_len = history_len
        self.candidate_embeddings = candidate_embeddings
        self.candidate_texts = candidate_texts
        self.command_to_index = command_to_index

        total, trainable = count_parameters(self)
        print(f"Total parameters: {total / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable / 1e6:.2f}M")
        print(f"Aggregation mode: {aggregation_mode}")
        print(f"Use gated attention: {self.use_gated_attention}")

    def forward(self, images):
        # Given images of shape (b, t, k, c, h, w)
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

        # Reshape images to (b*t*k, c, h, w) for processing through CLIP
        images_reshaped = images.reshape(batch_size * timesteps * num_cameras, c, h, w)

        # Apply transformations for encoder (resize + ImageNet normalize)
        images_transformed = image_transform(images_reshaped)

        # Get image features from Swin-Tiny
        image_features = self.image_encoder(images_transformed)  # (B*, D)

        # Reshape the image features to [batch_size, timesteps, cameras, feature_dim]
        image_features_reshaped = image_features.reshape(
            batch_size, timesteps, num_cameras, -1
        ).to(torch.float32)

        # Add positional encoding
        if self.num_cameras > 1:
            # Separate temporal + camera encoding for multi-camera setup
            # temporal_pe: (timesteps, D) -> (1, timesteps, 1, D)
            temporal_pe = self.temporal_positional_encoding[:timesteps, :].to(
                image_features_reshaped.device
            ).unsqueeze(0).unsqueeze(2)
            # camera_pe: (num_cameras, D) -> (1, 1, num_cameras, D)
            camera_pe = self.camera_embedding.weight[:num_cameras, :].unsqueeze(0).unsqueeze(1)
            # Combine: broadcast to (batch, timesteps, cameras, D)
            image_features_reshaped = image_features_reshaped + temporal_pe + camera_pe
        else:
            # Single camera: use flat positional encoding
            # (timesteps, D) -> (1, timesteps, 1, D)
            pos_enc = self.positional_encoding[:timesteps, :].to(
                image_features_reshaped.device
            ).unsqueeze(0).unsqueeze(2)
            image_features_reshaped = image_features_reshaped + pos_enc

        # Flatten to [batch_size, timesteps * cameras, feature_dim]
        image_features_flat = image_features_reshaped.reshape(
            batch_size, timesteps * num_cameras, -1
        )

        # Prepend CLS token if using 'cls' aggregation mode
        if self.aggregation_mode == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(image_features_flat.device)
            image_features_flat = torch.cat([cls_tokens, image_features_flat], dim=1)

        # Pass the concatenated features through the Transformer / gated encoder
        if self.use_gated_attention:
            # batch-first: (B, T, D) is default.
            transformer_out = image_features_flat
            for layer in self.gated_layers:
                transformer_out = layer(transformer_out)
        else:
            # While batch_first is True, the input will be transformed from (seq_len, batch, d_model) to (batch, seq_len, d_model) internally.
            # which is consistent with the general input format.
            transformer_out = self.transformer(image_features_flat)

        # Extract output based on aggregation mode
        if self.aggregation_mode == "cls":
            # Take the CLS token output (position 0)
            final_output = transformer_out[:, 0, :]
        elif self.aggregation_mode == "avg":
            # Average over all sequence positions
            final_output = transformer_out.mean(dim=1)
        else:  # 'last'
            # Take the last token output
            final_output = transformer_out[:, -1, :]

        if ONE_HOT:
            # Directly predict the logits for each command
            logits = self.mlp(final_output)
        else:
            # Predict the command embedding
            command_pred = self.mlp(final_output)
            # Compute the similarity scores as logits
            logits = self.compute_similarities(command_pred) / self.temperature.clamp(
                min=1e-8
            )

        return logits, self.temperature

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

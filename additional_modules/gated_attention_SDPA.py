import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMultiheadAttention(nn.Module):
    """
    Element-wise Gated Multihead Attention module (based on https://arxiv.org/abs/2505.06708).

    Internally operates in batch-first mode: (batch_size, seq_len, d_model),
    but can accept either batch-first or seq-first (seq_len, batch_size, d_model)
    via the `batch_first` flag.
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        dropout: float = 0.1,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.batch_first = batch_first

        # shape trick for query [d_model] --> [d_model * 2], actually [query] --> [query, gate]
        self.q_proj = nn.Linear(d_model, d_model * 2, bias=bias)

        # keep the key and value shape
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        # output projection
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            query, key, value:
                - if batch_first=True:  (batch_size, seq_len, d_model)
                - if batch_first=False: (seq_len, batch_size, d_model)
            attn_mask: (seq_len, seq_len) or broadcastable to attention scores
            key_padding_mask: (batch_size, seq_len) bool mask
        Returns:
            output: same layout as inputs (batch_first or seq_first)
        """
        # Normalize to batch-first inside
        if self.batch_first:
            query_b, key_b, value_b = query, key, value
        else:
            # (S, N, D) -> (N, S, D)
            query_b = query.transpose(0, 1)
            key_b = key.transpose(0, 1)
            value_b = value.transpose(0, 1)

        batch_size, seq_len, _ = query_b.shape

        # q_gate: (batch_size, seq_len, d_model * 2)
        q_gate = self.q_proj(query_b)

        # split the query into query and gate score
        # q: (batch_size, seq_len, d_model)
        # gate_score: (batch_size, seq_len, d_model)
        q, gate_score = q_gate.chunk(2, dim=-1)

        k = self.k_proj(key_b)
        v = self.v_proj(value_b)

        # reshape for multi-head attention
        # [batch_size, seq_len, d_model] --> [batch_size, seq_len, num_heads, head_dim] --> [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # reshape gate to fit multi-head attention
        # [batch_size, seq_len, d_model] --> [batch_size, seq_len, num_heads, head_dim] --> [batch_size, num_heads, seq_len, head_dim]
        gate_score = gate_score.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Build unified attention mask compatible with scaled_dot_product_attention
        final_attn_mask = None

        # 1) Base attn_mask: typically (Tq, Tk)
        if attn_mask is not None:
            if attn_mask.dim() == 2:  # (Tq, Tk)
                final_attn_mask = attn_mask.view(1, 1, attn_mask.size(0), attn_mask.size(1))
            else:
                # Assume user passes something broadcastable to (B, H, Tq, Tk)
                final_attn_mask = attn_mask

        # 2) key_padding_mask: (B, Tk) -> additive mask (B, 1, 1, Tk) with -inf on padded positions
        if key_padding_mask is not None:
            # bool mask: True for padding positions
            pad = key_padding_mask.to(torch.bool).unsqueeze(1).unsqueeze(2)  # (B,1,1,Tk)
            # convert to additive mask: 0 for non-pad, -inf for pad
            pad_add = torch.zeros_like(pad, dtype=q.dtype)
            pad_add = pad_add.masked_fill(pad, float("-inf"))  # (B,1,1,Tk)
            if final_attn_mask is None:
                final_attn_mask = pad_add
            else:
                final_attn_mask = final_attn_mask + pad_add

        # scaled_dot_product_attention will broadcast final_attn_mask to (B, H, Tq, Tk)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=final_attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        # Apply Element-wise Gating Mechanism
        # Output = Attention_out * Sigmoid(Gate_score)
        gate_activation = torch.sigmoid(gate_score)
        attn_output = attn_output * gate_activation

        # Recover the shape
        # [batch_size, num_heads, seq_len_query, head_dim] --> [batch_size, seq_len_query, num_heads, head_dim] --> [batch_size, seq_len_query, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # output projection
        output_b = self.o_proj(attn_output)

        # Convert back to original layout if needed
        if self.batch_first:
            return output_b
        # (N, S, D) for seq-first
        return output_b.transpose(0, 1)


class GatedTransformerEncoderLayer(nn.Module):
    """
    A simple Transformer encoder layer that uses GatedMultiheadAttention.
    This implementation is **batch-first**: inputs are (batch, seq_len, d_model).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = GatedMultiheadAttention(
            num_heads=num_heads,
            d_model=d_model,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # LayerNorms and dropouts
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            src: (batch_size, seq_len, d_model)
            src_mask: (seq_len, seq_len) or broadcastable to attention scores, optional
            src_key_padding_mask: (batch_size, seq_len) bool, optional
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual + norm
        attn_output = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feed-forward with residual + norm
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)
        return src


class GatedTransformerDecoderLayer(nn.Module):
    """
    A standard Transformer decoder layer that uses GatedMultiheadAttention.
    This implementation is **batch-first**: inputs are (batch, seq_len, d_model).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Self-attention and cross-attention both use batch-first gated attention
        self.self_attn = GatedMultiheadAttention(
            num_heads=num_heads,
            d_model=d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = GatedMultiheadAttention(
            num_heads=num_heads,
            d_model=d_model,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # LayerNorms and dropouts
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: (batch_size, tgt_len, d_model)
            memory: (batch_size, src_len, d_model)
            tgt_mask: (tgt_len, tgt_len) or broadcastable to attention scores
            memory_mask: (tgt_len, src_len) or broadcastable
            tgt_key_padding_mask: (batch_size, tgt_len) bool
            memory_key_padding_mask: (batch_size, src_len) bool
        Returns:
            Tensor of shape (batch_size, tgt_len, d_model)
        """
        # Self-attention
        tgt2 = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention over encoder memory
        tgt2 = self.cross_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed-forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class GatedDETREncoderLayer(nn.Module):
    """
    DETR-style encoder layer that swaps nn.MultiheadAttention for GatedMultiheadAttention.
    Expects sequence-first tensors: (seq_len, batch_size, d_model).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ):
        super().__init__()
        # use gated attention instead of nn.MultiheadAttention
        self.self_attn = GatedMultiheadAttention(
            num_heads=nhead,
            d_model=d_model,
            dropout=dropout,
            batch_first=False,  # seq-first for DETR
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None):
        return tensor if pos is None else tensor + pos

    def _sa_block(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None,
        src_key_padding_mask: torch.Tensor | None,
        pos: torch.Tensor | None,
    ) -> torch.Tensor:
        # src: (S, N, E), self.self_attn handles seq-first internally
        q = k = self.with_pos_embed(src, pos)  # (S, N, E)
        src2 = self.self_attn(
            q,
            k,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        return src2

    def forward_post(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src2 = self._sa_block(src, src_mask, src_key_padding_mask, pos)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src2 = self.norm1(src)
        src2 = self._sa_block(src2, src_mask, src_key_padding_mask, pos)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class GatedDETRDecoderLayer(nn.Module):
    """
    DETR-style decoder layer using GatedMultiheadAttention for both self- and cross-attention.
    Expects sequence-first tensors.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ):
        super().__init__()
        self.self_attn = GatedMultiheadAttention(
            num_heads=nhead,
            d_model=d_model,
            dropout=dropout,
        )
        self.multihead_attn = GatedMultiheadAttention(
            num_heads=nhead,
            d_model=d_model,
            dropout=dropout,
            batch_first=False,  # seq-first for DETR
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None):
        return tensor if pos is None else tensor + pos

    def _sa_block(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        tgt_key_padding_mask: torch.Tensor | None,
        query_pos: torch.Tensor | None,
    ) -> torch.Tensor:
        q = k = self.with_pos_embed(tgt, query_pos)
        # self.self_attn handles seq-first when batch_first=False
        out = self.self_attn(
            q,
            k,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        return out

    def _mha_block(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None,
        memory_key_padding_mask: torch.Tensor | None,
        pos: torch.Tensor | None,
        query_pos: torch.Tensor | None,
    ) -> torch.Tensor:
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        v = memory
        out = self.multihead_attn(
            q,
            k,
            v,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        return out

    def forward_post(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt2 = self._sa_block(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self._mha_block(
            tgt,
            memory,
            memory_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt2 = self.norm1(tgt)
        tgt2 = self._sa_block(tgt2, tgt_mask, tgt_key_padding_mask, query_pos)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self._mha_block(
            tgt2,
            memory,
            memory_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


if __name__ == "__main__":
    # quick sanity check
    batch_size, seq_len, d_model, num_heads = 2, 10, 512, 8
    x = torch.randn(batch_size, seq_len, d_model)
    layer = GatedTransformerEncoderLayer(d_model=d_model, num_heads=num_heads)
    out = layer(x)
    print("GatedTransformerEncoderLayer output shape:", out.shape)
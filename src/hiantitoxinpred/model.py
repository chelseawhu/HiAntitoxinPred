from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BRANCH_SPECS = OrderedDict(
    [
        ("dipeptide", {"kernel_size": 2, "padding": 0}),
        ("tripeptide", {"kernel_size": 3, "padding": 1}),
        ("short_loop", {"kernel_size": 5, "padding": 2}),
        ("helix_turn", {"kernel_size": 7, "padding": 3}),
        ("alpha_helix", {"kernel_size": 11, "padding": 5}),
        ("binding_domain", {"kernel_size": 15, "padding": 7}),
    ]
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.pe[: inputs.size(0), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, v)
        return output, attention

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        attn_output, attention = self.scaled_dot_product_attention(q, k, v, mask=mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(attn_output)
        if return_attention:
            return output, attention
        return output


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_attention:
            attn_output, attention = self.self_attention(inputs, inputs, inputs, mask=mask, return_attention=True)
            x = self.norm1(inputs + self.dropout(attn_output))
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
            return x, attention
        attn_output = self.self_attention(inputs, inputs, inputs, mask=mask)
        x = self.norm1(inputs + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DynamicPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attention_pooling = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = features.shape
        scores = self.attention_pooling(features).squeeze(-1)
        if masks is not None:
            final_masks = masks
        elif lengths is not None:
            final_masks = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=features.device)
            for idx, cur_len in enumerate(lengths):
                final_masks[idx, : cur_len] = True
        else:
            final_masks = torch.ones(batch_size, seq_len, dtype=torch.bool, device=features.device)
        scores = scores.masked_fill(~final_masks, float("-inf"))
        attention = F.softmax(scores, dim=-1)
        pooled = torch.bmm(attention.unsqueeze(1), features).squeeze(1)
        if return_attention:
            return pooled, attention
        return pooled


class GatedFeatureFusion(nn.Module):
    def __init__(
        self,
        esm_dim: int,
        one_hot_dim: int,
        aaindex_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
        use_onehot: bool = True,
        use_aaindex: bool = True,
    ) -> None:
        super().__init__()
        self.use_onehot = bool(use_onehot)
        self.use_aaindex = bool(use_aaindex)
        self.one_hot_dim = int(one_hot_dim) if use_onehot else 0
        self.aaindex_dim = int(aaindex_dim) if use_aaindex else 0

        if self.use_onehot and self.one_hot_dim > 0:
            self.one_hot_projection = nn.Sequential(
                nn.Linear(one_hot_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        if self.use_aaindex and self.aaindex_dim > 0:
            self.aaindex_projection = nn.Sequential(
                nn.Linear(aaindex_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        num_features = 1
        if self.use_onehot and self.one_hot_dim > 0:
            num_features += 1
        if self.use_aaindex and self.aaindex_dim > 0:
            num_features += 1

        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features),
            nn.Softmax(dim=-1),
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * num_features, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.residual_projection = nn.Linear(hidden_dim * num_features, hidden_dim)

    def forward(
        self,
        esm_features: torch.Tensor,
        one_hot_features: Optional[torch.Tensor],
        aaindex_features: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = [esm_features]
        if self.use_onehot and self.one_hot_dim > 0 and one_hot_features is not None:
            projected.append(self.one_hot_projection(one_hot_features))
        if self.use_aaindex and self.aaindex_dim > 0 and aaindex_features is not None:
            projected.append(self.aaindex_projection(aaindex_features))

        concatenated = torch.cat(projected, dim=-1)
        gate_weights = self.gate_network(concatenated)
        gated = [feature * gate_weights[:, idx : idx + 1] for idx, feature in enumerate(projected)]
        gated_concat = torch.cat(gated, dim=-1)
        fused = self.fusion_layer(gated_concat)
        residual = self.residual_projection(concatenated)
        return fused + residual, gate_weights


class ChannelLayerNorm1d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(inputs.transpose(1, 2)).transpose(1, 2)


class TopKMotifPooling(nn.Module):
    def __init__(self, hidden_dim: int, top_k: int = 3) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.top_k = int(top_k)
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, branch_tokens: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        scores = self.scorer(branch_tokens).squeeze(-1)
        masked_scores = scores.masked_fill(~masks, torch.finfo(scores.dtype).min)
        k = min(self.top_k, branch_tokens.size(1))
        top_scores, top_indices = torch.topk(masked_scores, k=k, dim=1)
        valid_mask = top_scores > (torch.finfo(top_scores.dtype).min / 2)
        stable_scores = top_scores.masked_fill(~valid_mask, float("-inf"))
        attn = torch.softmax(stable_scores, dim=1)
        attn = torch.where(valid_mask, attn, torch.zeros_like(attn))
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-8)
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, branch_tokens.size(-1))
        gathered_tokens = branch_tokens.gather(1, gather_index)
        pooled = (gathered_tokens * attn.unsqueeze(-1)).sum(dim=1)

        lengths = masks.sum(dim=1).clamp_min(1)
        denom = (lengths - 1).clamp_min(1).float().unsqueeze(1)
        normalized_positions = top_indices.float() / denom
        valid_float = valid_mask.float()
        score_summary = (torch.sigmoid(top_scores) * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp_min(1.0)
        position_summary = (normalized_positions * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp_min(1.0)
        return pooled, {
            "topk_scores": torch.sigmoid(top_scores),
            "topk_positions": top_indices,
            "score_summary": score_summary,
            "position_summary": position_summary,
        }


class SampleAdaptiveBranchGate(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, branch_summaries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size, num_branches, _ = branch_summaries.shape
        context_expanded = context.unsqueeze(1).expand(batch_size, num_branches, -1)
        logits = self.network(torch.cat([branch_summaries, context_expanded], dim=-1)).squeeze(-1)
        return torch.softmax(logits, dim=-1)


class SampleScalarGate(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.2, init_prob: float = 0.1) -> None:
        super().__init__()
        init_prob = float(init_prob)
        init_prob = min(max(init_prob, 1e-4), 1.0 - 1e-4)
        self.gate_logit_bias = nn.Parameter(torch.tensor(np.log(init_prob / (1.0 - init_prob)), dtype=torch.float32))
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, main_summary: torch.Tensor, cnn_summary: torch.Tensor) -> torch.Tensor:
        context_gate = torch.sigmoid(self.network(torch.cat([main_summary, cnn_summary], dim=-1)).squeeze(-1))
        master_gate = torch.sigmoid(self.gate_logit_bias)
        return master_gate * context_gate

    def get_bias_probability(self) -> float:
        return float(torch.sigmoid(self.gate_logit_bias.detach()).cpu().item())


class BranchAwareMultiScaleCNN(nn.Module):
    def __init__(self, input_dim: int, motif_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.branch_names = list(BRANCH_SPECS.keys())
        self.branch_extractors = nn.ModuleDict(
            {
                name: nn.Conv1d(
                    input_dim,
                    motif_dim,
                    kernel_size=spec["kernel_size"],
                    padding=spec["padding"],
                )
                for name, spec in BRANCH_SPECS.items()
            }
        )
        self.branch_enhancers = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv1d(motif_dim, motif_dim, kernel_size=3, padding=1, groups=motif_dim),
                    nn.Conv1d(motif_dim, motif_dim, kernel_size=1),
                    ChannelLayerNorm1d(motif_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for name in self.branch_names
            }
        )

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = inputs.shape
        del batch_size
        x_conv = inputs.transpose(1, 2)
        outputs: Dict[str, torch.Tensor] = {}
        for name in self.branch_names:
            branch = self.branch_extractors[name](x_conv)
            if branch.size(-1) < seq_len:
                branch = F.pad(branch, (0, seq_len - branch.size(-1)))
            elif branch.size(-1) > seq_len:
                branch = branch[:, :, :seq_len]
            branch = self.branch_enhancers[name](branch).transpose(1, 2)
            branch = branch * masks.unsqueeze(-1).to(branch.dtype)
            outputs[name] = branch
        return outputs


class BiologicalMotifExtractor(nn.Module):
    """Original serial motif extractor used before the CLS Transformer path."""

    def __init__(self, input_dim: int, motif_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.branch_names = list(BRANCH_SPECS.keys())
        self.motif_extractors = nn.ModuleDict(
            {
                name: nn.Conv1d(
                    input_dim,
                    motif_dim,
                    kernel_size=spec["kernel_size"],
                    padding=spec["padding"],
                )
                for name, spec in BRANCH_SPECS.items()
            }
        )
        self.channel_weights = nn.Parameter(torch.tensor([0.10, 0.10, 0.15, 0.20, 0.25, 0.20]))
        self.depthwise_convs = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv1d(motif_dim, motif_dim, kernel_size=3, padding=1, groups=motif_dim),
                    nn.Conv1d(motif_dim, motif_dim, kernel_size=1),
                    ChannelLayerNorm1d(motif_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for name in self.branch_names
            }
        )
        self.ln_concat = nn.LayerNorm(motif_dim * len(self.branch_names))
        self.bio_attention = nn.MultiheadAttention(
            embed_dim=motif_dim * len(self.branch_names),
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_out = nn.LayerNorm(motif_dim * len(self.branch_names))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, seq_len, _ = inputs.shape
        x_conv = inputs.transpose(1, 2)
        motif_features: List[torch.Tensor] = []
        for idx, (name, extractor) in enumerate(self.motif_extractors.items()):
            motif = extractor(x_conv)
            if motif.size(-1) < seq_len:
                motif = F.pad(motif, (0, seq_len - motif.size(-1)))
            elif motif.size(-1) > seq_len:
                motif = motif[:, :, :seq_len]
            enhanced = self.depthwise_convs[name](motif) * self.channel_weights[idx]
            motif_features.append(enhanced.transpose(1, 2))
        concatenated = torch.cat(motif_features, dim=-1)
        normalized = self.ln_concat(concatenated)
        key_padding_mask = ~mask if mask is not None else None
        attended, motif_attention = self.bio_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
        )
        residual_output = self.ln_out(attended + normalized)
        return self.dropout(residual_output), motif_attention


def _collect_parameters(modules: List[nn.Module], extra: Optional[List[torch.nn.Parameter]] = None) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    seen = set()
    for module in modules:
        for param in module.parameters():
            if id(param) not in seen:
                params.append(param)
                seen.add(id(param))
    for param in extra or []:
        if id(param) not in seen:
            params.append(param)
            seen.add(id(param))
    return params


class BaseCNNRescueNet(nn.Module):
    def __init__(
        self,
        *,
        esm_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        one_hot_dim: int,
        aaindex_dim: int,
        use_onehot: bool,
        use_aaindex: bool,
        use_cnn: bool,
        use_transformer: bool,
        top_k: int,
        scalar_gate_init_prob: float,
        main_pooling_mode: str = "dynamic",
        cnn_only_mode: bool = False,
    ) -> None:
        super().__init__()
        self.esm_dim = int(esm_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_onehot = bool(use_onehot)
        self.use_aaindex = bool(use_aaindex)
        self.use_cnn = bool(use_cnn)
        self.use_transformer = bool(use_transformer)
        self.top_k = int(top_k)
        self.cnn_only_mode = bool(cnn_only_mode)
        self.main_pooling_mode = str(main_pooling_mode).strip().lower()
        if self.main_pooling_mode not in {"dynamic", "mean", "max"}:
            raise ValueError("main_pooling_mode must be 'dynamic', 'mean', or 'max'")
        self.branch_names = list(BRANCH_SPECS.keys())

        self.esm_projection = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.main_pooling = DynamicPooling(hidden_dim)
        self.feature_fusion = GatedFeatureFusion(
            esm_dim=hidden_dim,
            one_hot_dim=one_hot_dim,
            aaindex_dim=aaindex_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_onehot=use_onehot,
            use_aaindex=use_aaindex,
        )
        if self.use_transformer:
            self.positional_encoding = PositionalEncoding(hidden_dim)
            self.transformer_layers = nn.ModuleList(
                [TransformerLayer(hidden_dim, num_heads, hidden_dim * 4, dropout) for _ in range(num_layers)]
            )

        if self.use_cnn:
            motif_dim = max(hidden_dim // 8, 32)
            self.cnn_extractor = BranchAwareMultiScaleCNN(hidden_dim, motif_dim, dropout=dropout)
            self.branch_poolers = nn.ModuleDict(
                {name: TopKMotifPooling(motif_dim, top_k=top_k) for name in self.branch_names}
            )
            self.branch_summary_projection = nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.Linear(motif_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                    for name in self.branch_names
                }
            )
            self.branch_gate_network = SampleAdaptiveBranchGate(hidden_dim, dropout=dropout)
            self.sample_gate = SampleScalarGate(hidden_dim, dropout=dropout, init_prob=scalar_gate_init_prob)
        else:
            self.cnn_extractor = None

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim // 8),
            nn.Dropout(dropout * 0.75),
            nn.Linear(hidden_dim // 8, 1),
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                if getattr(module, "weight", None) is not None:
                    nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def get_masks(self, esm_features: torch.Tensor, esm_len: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len, _ = esm_features.shape
        if esm_len is None:
            return torch.ones(batch_size, seq_len, dtype=torch.bool, device=esm_features.device)
        masks = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=esm_features.device)
        for idx, cur_len in enumerate(esm_len):
            masks[idx, : int(cur_len)] = True
        return masks

    def masked_mean_pool(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        float_mask = masks.unsqueeze(-1).to(features.dtype)
        return (features * float_mask).sum(dim=1) / float_mask.sum(dim=1).clamp_min(1.0)

    def masked_max_pool(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        expanded_mask = masks.unsqueeze(-1)
        masked = features.masked_fill(~expanded_mask, torch.finfo(features.dtype).min)
        return masked.max(dim=1).values

    def encode_main_path(
        self,
        esm_features: torch.Tensor,
        one_hot_features: Optional[torch.Tensor],
        aaindex_features: Optional[torch.Tensor],
        esm_len: Optional[torch.Tensor],
        *,
        return_attention: bool = False,
        return_residue_attention: bool = False,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor] | None]:
        masks = self.get_masks(esm_features, esm_len)
        esm_projected = self.esm_projection(esm_features)
        token_stream = esm_projected
        attention_weights: List[torch.Tensor] = []
        if self.use_transformer:
            token_stream = self.positional_encoding(esm_projected.transpose(0, 1)).transpose(0, 1)
            for layer in self.transformer_layers:
                if return_attention:
                    token_stream, weights = layer(token_stream, mask=masks, return_attention=True)
                    attention_weights.append(weights)
                else:
                    token_stream = layer(token_stream, mask=masks)
        residue_attention = None
        if self.main_pooling_mode == "mean":
            pooled_main = self.masked_mean_pool(token_stream, masks)
            if return_residue_attention:
                float_mask = masks.to(token_stream.dtype)
                residue_attention = float_mask / float_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        elif self.main_pooling_mode == "max":
            pooled_main = self.masked_max_pool(token_stream, masks)
            if return_residue_attention:
                residue_attention = None
        elif return_residue_attention:
            pooled_main, residue_attention = self.main_pooling(
                token_stream,
                esm_len,
                masks,
                return_attention=True,
            )
        else:
            pooled_main = self.main_pooling(token_stream, esm_len, masks)
        fused_main, gate_weights = self.feature_fusion(pooled_main, one_hot_features, aaindex_features)
        return {
            "masks": masks,
            "esm_projected": esm_projected,
            "token_stream": token_stream,
            "pooled_main": pooled_main,
            "main_summary": fused_main,
            "gate_weights": gate_weights,
            "attention_weights": attention_weights,
            "residue_attention": residue_attention,
        }

    def summarize_cnn_branch(
        self,
        esm_projected: torch.Tensor,
        masks: torch.Tensor,
        context: torch.Tensor,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        if not self.use_cnn or self.cnn_extractor is None:
            raise RuntimeError("CNN branch is disabled for this variant")
        branch_tokens = self.cnn_extractor(esm_projected, masks)
        branch_summaries = []
        branch_score_summaries = []
        branch_position_summaries = []
        branch_topk_scores = []
        branch_topk_positions = []
        for name in self.branch_names:
            pooled, details = self.branch_poolers[name](branch_tokens[name], masks)
            projected = self.branch_summary_projection[name](pooled)
            branch_summaries.append(projected)
            branch_score_summaries.append(details["score_summary"])
            branch_position_summaries.append(details["position_summary"])
            branch_topk_scores.append(details["topk_scores"].mean(dim=1))
            branch_topk_positions.append(details["topk_positions"].float().mean(dim=1))
        branch_stack = torch.stack(branch_summaries, dim=1)
        branch_gates = self.branch_gate_network(branch_stack, context)
        cnn_summary = (branch_stack * branch_gates.unsqueeze(-1)).sum(dim=1)
        scalar_gate = self.sample_gate(context, cnn_summary)
        return {
            "cnn_summary": cnn_summary,
            "branch_gates": branch_gates,
            "scalar_gate": scalar_gate,
            "topk_score_matrix": torch.stack(branch_score_summaries, dim=1),
            "topk_position_matrix": torch.stack(branch_position_summaries, dim=1),
            "motif_attention": {
                "branch_names": list(self.branch_names),
                "topk_scores": torch.stack(branch_topk_scores, dim=1),
                "topk_positions": torch.stack(branch_topk_positions, dim=1),
            },
        }

    def get_cnn_gate_bias_probability(self) -> Optional[float]:
        if not self.use_cnn:
            return None
        return float(self.sample_gate.get_bias_probability())

    def get_parameter_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        groups = {
            "main": _collect_parameters(
                [self.esm_projection, self.main_pooling, self.feature_fusion] + list(getattr(self, "transformer_layers", []))
            ),
            "cnn": _collect_parameters(
                [self.cnn_extractor, self.branch_poolers, self.branch_summary_projection, self.branch_gate_network, self.sample_gate]
                if self.use_cnn
                else []
            ),
            "fusion": _collect_parameters(self.get_fusion_modules()),
            "classifier": _collect_parameters([self.classifier], extra=[self.temperature]),
            "aux": _collect_parameters(self.get_aux_modules()),
        }
        return groups

    def get_fusion_modules(self) -> List[nn.Module]:
        raise NotImplementedError

    def get_aux_modules(self) -> List[nn.Module]:
        return []


class SerialCLSRescueNet(nn.Module):
    """Serial CNN-CLS-Transformer architecture for clean ablation diagnostics."""

    def __init__(
        self,
        *,
        esm_dim: int = 1280,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
        one_hot_dim: int = 1,
        aaindex_dim: int = 1,
        use_onehot: bool = False,
        use_aaindex: bool = False,
        use_cnn: bool = True,
        use_transformer: bool = True,
        top_k: int = 3,
        main_pooling_mode: str = "dynamic",
    ) -> None:
        super().__init__()
        del one_hot_dim, aaindex_dim, top_k, main_pooling_mode
        self.esm_dim = int(esm_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_onehot = bool(use_onehot)
        self.use_aaindex = bool(use_aaindex)
        self.use_cnn = bool(use_cnn)
        self.use_transformer = bool(use_transformer)
        self.branch_names = list(BRANCH_SPECS.keys()) if self.use_cnn else []
        self.cls_init_pooling_mode = "dynamic" if self.use_cnn else "mean"

        self.esm_projection = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dynamic_pooling = DynamicPooling(hidden_dim)
        if self.use_cnn:
            motif_dim = max(hidden_dim // 8, 32)
            self.biological_motif_extractor = BiologicalMotifExtractor(
                input_dim=hidden_dim,
                motif_dim=motif_dim,
                dropout=dropout,
            )
            self.motif_projection = nn.Linear(motif_dim * len(BRANCH_SPECS), hidden_dim)
        else:
            self.biological_motif_extractor = None
            self.motif_projection = None

        if self.use_transformer:
            self.positional_encoding = PositionalEncoding(hidden_dim)
            self.transformer_layers = nn.ModuleList(
                [TransformerLayer(hidden_dim, num_heads, hidden_dim * 4, dropout) for _ in range(num_layers)]
            )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim // 8),
            nn.Dropout(dropout * 0.75),
            nn.Linear(hidden_dim // 8, 1),
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                if getattr(module, "weight", None) is not None:
                    nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def get_masks(self, esm_features: torch.Tensor, esm_len: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len, _ = esm_features.shape
        if esm_len is None:
            return torch.ones(batch_size, seq_len, dtype=torch.bool, device=esm_features.device)
        masks = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=esm_features.device)
        for idx, cur_len in enumerate(esm_len):
            masks[idx, : int(cur_len)] = True
        return masks

    def masked_mean_pool(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        float_mask = masks.unsqueeze(-1).to(features.dtype)
        return (features * float_mask).sum(dim=1) / float_mask.sum(dim=1).clamp_min(1.0)

    def encode_serial_tokens(
        self,
        esm_features: torch.Tensor,
        esm_len: Optional[torch.Tensor],
        *,
        return_motif_attention: bool = False,
        return_residue_attention: bool = False,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor] | None]:
        masks = self.get_masks(esm_features, esm_len)
        esm_projected = self.esm_projection(esm_features)
        motif_attention = None
        residue_attention = None
        if self.use_cnn:
            motif_features, motif_attention = self.biological_motif_extractor(esm_projected, mask=masks)
            token_stream = self.motif_projection(motif_features)
            if return_residue_attention:
                cls_init, residue_attention = self.dynamic_pooling(
                    token_stream,
                    esm_len,
                    masks,
                    return_attention=True,
                )
            else:
                cls_init = self.dynamic_pooling(token_stream, esm_len, masks)
        else:
            token_stream = esm_projected
            cls_init = self.masked_mean_pool(token_stream, masks)
            if return_residue_attention:
                float_mask = masks.to(token_stream.dtype)
                residue_attention = float_mask / float_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return {
            "masks": masks,
            "esm_projected": esm_projected,
            "token_stream": token_stream,
            "cls_init": cls_init,
            "motif_attention": motif_attention if return_motif_attention else None,
            "residue_attention": residue_attention,
        }

    def compute_outputs(
        self,
        esm_features: torch.Tensor,
        one_hot_features: Optional[torch.Tensor],
        aaindex_features: Optional[torch.Tensor],
        *,
        esm_len: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_motif_attention: bool = False,
        return_residue_attention: bool = False,
        return_gate_weights: bool = False,
        return_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor] | List[torch.Tensor] | None]:
        del one_hot_features, aaindex_features, return_gate_weights
        payload = self.encode_serial_tokens(
            esm_features,
            esm_len,
            return_motif_attention=return_motif_attention,
            return_residue_attention=return_residue_attention,
        )
        global_features = payload["cls_init"]
        attention_weights: List[torch.Tensor] = []
        if self.use_transformer:
            cls_token = global_features.unsqueeze(1)
            transformer_input = torch.cat([cls_token, payload["token_stream"]], dim=1)
            transformer_input = self.positional_encoding(transformer_input.transpose(0, 1)).transpose(0, 1)
            cls_mask = torch.ones(transformer_input.size(0), 1, dtype=torch.bool, device=payload["masks"].device)
            extended_masks = torch.cat([cls_mask, payload["masks"]], dim=1)
            for layer in self.transformer_layers:
                if return_attention:
                    transformer_input, weights = layer(transformer_input, mask=extended_masks, return_attention=True)
                    attention_weights.append(weights)
                else:
                    transformer_input = layer(transformer_input, mask=extended_masks)
            global_features = transformer_input[:, 0, :]

        prediction = self.classifier(global_features) / torch.clamp(self.temperature, min=0.1, max=10.0)
        return {
            "logits": prediction,
            "aux_logits": None,
            "attention_weights": attention_weights if return_attention else None,
            "motif_attention": payload["motif_attention"] if return_motif_attention else None,
            "residue_attention": payload["residue_attention"] if return_residue_attention else None,
            "gate_weights": None,
            "diagnostics": None,
        }

    def forward(
        self,
        esm_features: torch.Tensor,
        one_hot_features: Optional[torch.Tensor],
        aaindex_features: Optional[torch.Tensor],
        *,
        esm_len: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_motif_attention: bool = False,
        return_residue_attention: bool = False,
        return_gate_weights: bool = False,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | List[torch.Tensor]:
        payload = self.compute_outputs(
            esm_features,
            one_hot_features,
            aaindex_features,
            esm_len=esm_len,
            return_attention=return_attention,
            return_motif_attention=return_motif_attention,
            return_residue_attention=return_residue_attention,
            return_gate_weights=return_gate_weights,
            return_diagnostics=return_diagnostics,
        )
        outputs: List[torch.Tensor | Dict[str, torch.Tensor] | List[torch.Tensor] | None] = [payload["logits"]]
        if return_attention:
            outputs.append(payload["attention_weights"])
        if return_motif_attention:
            outputs.append(payload["motif_attention"])
        if return_residue_attention:
            outputs.append(payload["residue_attention"])
        if return_gate_weights:
            outputs.append(payload["gate_weights"])
        if return_diagnostics:
            outputs.append(payload["diagnostics"])
        return outputs[0] if len(outputs) == 1 else outputs

    def get_cnn_gate_bias_probability(self) -> Optional[float]:
        return None

    def get_parameter_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        cnn_modules: List[nn.Module] = []
        if self.use_cnn:
            cnn_modules = [self.biological_motif_extractor, self.motif_projection]
        return {
            "main": _collect_parameters(
                [self.esm_projection, self.dynamic_pooling] + list(getattr(self, "transformer_layers", []))
            ),
            "cnn": _collect_parameters(cnn_modules),
            "fusion": [],
            "classifier": _collect_parameters([self.classifier], extra=[self.temperature]),
            "aux": [],
        }


class TrackAMinimalRescueNet(BaseCNNRescueNet):
    def __init__(
        self,
        *,
        esm_dim: int = 1280,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
        one_hot_dim: int = 1,
        aaindex_dim: int = 1,
        use_onehot: bool = False,
        use_aaindex: bool = False,
        use_cnn: bool = True,
        use_transformer: bool = True,
        top_k: int = 3,
        main_pooling_mode: str = "dynamic",
        cnn_only_mode: bool = False,
    ) -> None:
        del cnn_only_mode
        super().__init__(
            esm_dim=esm_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            one_hot_dim=one_hot_dim,
            aaindex_dim=aaindex_dim,
            use_onehot=use_onehot,
            use_aaindex=use_aaindex,
            use_cnn=use_cnn,
            use_transformer=use_transformer,
            top_k=top_k,
            scalar_gate_init_prob=0.1,
            main_pooling_mode=main_pooling_mode,
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def get_fusion_modules(self) -> List[nn.Module]:
        return [self.fusion_mlp]

    def compute_outputs(
        self,
        esm_features: torch.Tensor,
        one_hot_features: Optional[torch.Tensor],
        aaindex_features: Optional[torch.Tensor],
        *,
        esm_len: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_motif_attention: bool = False,
        return_residue_attention: bool = False,
        return_gate_weights: bool = False,
        return_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor] | List[torch.Tensor] | None]:
        main_payload = self.encode_main_path(
            esm_features,
            one_hot_features,
            aaindex_features,
            esm_len,
            return_attention=return_attention,
            return_residue_attention=return_residue_attention,
        )
        main_summary = main_payload["main_summary"]
        motif_attention = None
        diagnostics = None
        if self.use_cnn:
            cnn_payload = self.summarize_cnn_branch(
                main_payload["esm_projected"],
                main_payload["masks"],
                main_summary,
            )
            scalar_gate = cnn_payload["scalar_gate"]
            gated_cnn = scalar_gate.unsqueeze(-1) * cnn_payload["cnn_summary"]
            fusion_delta = self.fusion_mlp(torch.cat([main_summary, cnn_payload["cnn_summary"]], dim=-1))
            global_features = main_summary + scalar_gate.unsqueeze(-1) * fusion_delta
            motif_attention = cnn_payload["motif_attention"]
            diagnostics = {
                "cnn_gate": scalar_gate,
                "branch_gates": cnn_payload["branch_gates"],
                "topk_score_matrix": cnn_payload["topk_score_matrix"],
                "topk_position_matrix": cnn_payload["topk_position_matrix"],
            }
        else:
            global_features = main_summary
        prediction = self.classifier(global_features) / torch.clamp(self.temperature, min=0.1, max=10.0)
        return {
            "logits": prediction,
            "aux_logits": None,
            "attention_weights": main_payload["attention_weights"] if return_attention else None,
            "motif_attention": motif_attention if return_motif_attention else None,
            "residue_attention": main_payload["residue_attention"] if return_residue_attention else None,
            "gate_weights": main_payload["gate_weights"] if return_gate_weights else None,
            "diagnostics": diagnostics if return_diagnostics else None,
        }

    def forward(
        self,
        esm_features: torch.Tensor,
        one_hot_features: Optional[torch.Tensor],
        aaindex_features: Optional[torch.Tensor],
        *,
        esm_len: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_motif_attention: bool = False,
        return_residue_attention: bool = False,
        return_gate_weights: bool = False,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | List[torch.Tensor]:
        payload = self.compute_outputs(
            esm_features,
            one_hot_features,
            aaindex_features,
            esm_len=esm_len,
            return_attention=return_attention,
            return_motif_attention=return_motif_attention,
            return_residue_attention=return_residue_attention,
            return_gate_weights=return_gate_weights,
            return_diagnostics=return_diagnostics,
        )
        outputs: List[torch.Tensor | Dict[str, torch.Tensor] | List[torch.Tensor] | None] = [payload["logits"]]
        if return_attention:
            outputs.append(payload["attention_weights"])
        if return_motif_attention:
            outputs.append(payload["motif_attention"])
        if return_residue_attention:
            outputs.append(payload["residue_attention"])
        if return_gate_weights:
            outputs.append(payload["gate_weights"])
        if return_diagnostics:
            outputs.append(payload["diagnostics"])
        return outputs[0] if len(outputs) == 1 else outputs


class TrackBSecondGenRescueNet(BaseCNNRescueNet):
    def __init__(
        self,
        *,
        esm_dim: int = 1280,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
        one_hot_dim: int = 1,
        aaindex_dim: int = 1,
        use_onehot: bool = False,
        use_aaindex: bool = False,
        use_cnn: bool = True,
        use_transformer: bool = True,
        top_k: int = 3,
        scalar_gate_init_prob: float = 0.1,
        main_pooling_mode: str = "dynamic",
        cnn_only_mode: bool = False,
    ) -> None:
        super().__init__(
            esm_dim=esm_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            one_hot_dim=one_hot_dim,
            aaindex_dim=aaindex_dim,
            use_onehot=use_onehot,
            use_aaindex=use_aaindex,
            use_cnn=use_cnn,
            use_transformer=use_transformer,
            top_k=top_k,
            scalar_gate_init_prob=scalar_gate_init_prob,
            main_pooling_mode=main_pooling_mode,
            cnn_only_mode=cnn_only_mode,
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.use_cnn:
            self.cnn_aux_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.cnn_aux_head = None

    def get_fusion_modules(self) -> List[nn.Module]:
        return [self.fusion_mlp]

    def get_aux_modules(self) -> List[nn.Module]:
        return [self.cnn_aux_head] if self.cnn_aux_head is not None else []

    def compute_outputs(
        self,
        esm_features: torch.Tensor,
        one_hot_features: Optional[torch.Tensor],
        aaindex_features: Optional[torch.Tensor],
        *,
        esm_len: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_motif_attention: bool = False,
        return_residue_attention: bool = False,
        return_gate_weights: bool = False,
        return_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor] | List[torch.Tensor] | None]:
        if self.cnn_only_mode:
            if not self.use_cnn:
                raise RuntimeError("cnn_only_mode requires use_cnn=True")
            masks = self.get_masks(esm_features, esm_len)
            esm_projected = self.esm_projection(esm_features)
            branch_tokens = self.cnn_extractor(esm_projected, masks)
            branch_summaries = []
            branch_score_summaries = []
            branch_position_summaries = []
            branch_topk_scores = []
            branch_topk_positions = []
            for name in self.branch_names:
                pooled, details = self.branch_poolers[name](branch_tokens[name], masks)
                projected = self.branch_summary_projection[name](pooled)
                branch_summaries.append(projected)
                branch_score_summaries.append(details["score_summary"])
                branch_position_summaries.append(details["position_summary"])
                branch_topk_scores.append(details["topk_scores"].mean(dim=1))
                branch_topk_positions.append(details["topk_positions"].float().mean(dim=1))
            branch_stack = torch.stack(branch_summaries, dim=1)
            # Strict CNN-only ablation: branch aggregation uses CNN-derived context only.
            cnn_context = branch_stack.mean(dim=1)
            branch_gates = self.branch_gate_network(branch_stack, cnn_context)
            cnn_summary = (branch_stack * branch_gates.unsqueeze(-1)).sum(dim=1)
            prediction = self.classifier(cnn_summary) / torch.clamp(self.temperature, min=0.1, max=10.0)
            diagnostics = {
                "cnn_gate": torch.ones(cnn_summary.size(0), device=cnn_summary.device, dtype=cnn_summary.dtype),
                "branch_gates": branch_gates,
                "topk_score_matrix": torch.stack(branch_score_summaries, dim=1),
                "topk_position_matrix": torch.stack(branch_position_summaries, dim=1),
            }
            motif_attention = {
                "branch_names": list(self.branch_names),
                "topk_scores": torch.stack(branch_topk_scores, dim=1),
                "topk_positions": torch.stack(branch_topk_positions, dim=1),
            }
            return {
                "logits": prediction,
                "aux_logits": None,
                "attention_weights": None,
                "motif_attention": motif_attention if return_motif_attention else None,
                "residue_attention": None,
                "gate_weights": None,
                "diagnostics": diagnostics if return_diagnostics else None,
            }

        main_payload = self.encode_main_path(
            esm_features,
            one_hot_features,
            aaindex_features,
            esm_len,
            return_attention=return_attention,
            return_residue_attention=return_residue_attention,
        )
        main_summary = main_payload["main_summary"]
        motif_attention = None
        diagnostics = None
        aux_logits = None
        if self.use_cnn:
            cnn_payload = self.summarize_cnn_branch(
                main_payload["esm_projected"],
                main_payload["masks"],
                main_summary,
            )
            scalar_gate = cnn_payload["scalar_gate"]
            gated_cnn = scalar_gate.unsqueeze(-1) * cnn_payload["cnn_summary"]
            fusion_input = torch.cat(
                [
                    main_summary,
                    gated_cnn,
                    main_summary * gated_cnn,
                    torch.abs(main_summary - gated_cnn),
                ],
                dim=-1,
            )
            fusion_delta = self.fusion_mlp(fusion_input)
            global_features = main_summary + scalar_gate.unsqueeze(-1) * fusion_delta
            motif_attention = cnn_payload["motif_attention"]
            diagnostics = {
                "cnn_gate": scalar_gate,
                "branch_gates": cnn_payload["branch_gates"],
                "topk_score_matrix": cnn_payload["topk_score_matrix"],
                "topk_position_matrix": cnn_payload["topk_position_matrix"],
            }
            if self.cnn_aux_head is not None:
                aux_logits = self.cnn_aux_head(gated_cnn)
        else:
            global_features = main_summary
        prediction = self.classifier(global_features) / torch.clamp(self.temperature, min=0.1, max=10.0)
        return {
            "logits": prediction,
            "aux_logits": aux_logits,
            "attention_weights": main_payload["attention_weights"] if return_attention else None,
            "motif_attention": motif_attention if return_motif_attention else None,
            "residue_attention": main_payload["residue_attention"] if return_residue_attention else None,
            "gate_weights": main_payload["gate_weights"] if return_gate_weights else None,
            "diagnostics": diagnostics if return_diagnostics else None,
        }

    def forward(
        self,
        esm_features: torch.Tensor,
        one_hot_features: Optional[torch.Tensor],
        aaindex_features: Optional[torch.Tensor],
        *,
        esm_len: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_motif_attention: bool = False,
        return_residue_attention: bool = False,
        return_gate_weights: bool = False,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | List[torch.Tensor]:
        payload = self.compute_outputs(
            esm_features,
            one_hot_features,
            aaindex_features,
            esm_len=esm_len,
            return_attention=return_attention,
            return_motif_attention=return_motif_attention,
            return_residue_attention=return_residue_attention,
            return_gate_weights=return_gate_weights,
            return_diagnostics=return_diagnostics,
        )
        outputs: List[torch.Tensor | Dict[str, torch.Tensor] | List[torch.Tensor] | None] = [payload["logits"]]
        if return_attention:
            outputs.append(payload["attention_weights"])
        if return_motif_attention:
            outputs.append(payload["motif_attention"])
        if return_residue_attention:
            outputs.append(payload["residue_attention"])
        if return_gate_weights:
            outputs.append(payload["gate_weights"])
        if return_diagnostics:
            outputs.append(payload["diagnostics"])
        return outputs[0] if len(outputs) == 1 else outputs


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1, pos_weight: Optional[float] = None) -> None:
        super().__init__()
        self.smoothing = float(smoothing)
        self.pos_weight = None if pos_weight is None else torch.tensor([float(pos_weight)], dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smoothed = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(logits, smoothed, pos_weight=pos_weight)

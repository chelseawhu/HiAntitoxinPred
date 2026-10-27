"""Standalone effective network of the registered ProtT5 Dynamic model.

This first extraction preserves the registered convolution/padding mathematics.
It does not yet fix the known dependence on batch padding width. See
docs/numerical_behavior.md before changing input geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


BRANCH_SPECS = (
    ("dipeptide", 2, 0),
    ("tripeptide", 3, 1),
    ("short_loop", 5, 2),
    ("helix_turn", 7, 3),
    ("alpha_helix", 11, 5),
    ("binding_domain", 15, 7),
)


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 1024
    hidden_dim: int = 768
    heads: int = 8
    layers: int = 2
    dropout: float = 0.1
    max_length: int = 200

    def __post_init__(self):
        for name in ("input_dim", "hidden_dim", "heads", "layers", "max_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % 8 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by 8 and by heads")
        if not 0 <= self.dropout < 1 or self.max_length != 200:
            raise ValueError("Requires finite dropout in [0,1) and cap200")

    def to_dict(self):
        return asdict(self)


class PositionEncoding(nn.Module):
    def __init__(self, width):
        super().__init__()
        pe = torch.zeros(5000, width)
        positions = torch.arange(5000, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, width, 2).float() * (-np.log(10000.0) / width)
        )
        pe[:, 0::2] = torch.sin(positions * frequencies)
        pe[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        return (x.transpose(0, 1) + self.pe[: x.size(1)]).transpose(0, 1)


class SelfAttention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.width, self.heads, self.head_dim = width, heads, width // heads
        self.w_q = nn.Linear(width, width)
        self.w_k = nn.Linear(width, width)
        self.w_v = nn.Linear(width, width)
        self.w_o = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        batch = x.size(0)
        q = self.w_q(x).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        attention = self.dropout(F.softmax(scores, dim=-1))
        attended = (
            torch.matmul(attention, v)
            .transpose(1, 2)
            .contiguous()
            .view(batch, -1, self.width)
        )
        return self.w_o(attended)


class TransformerLayer(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.self_attention = SelfAttention(width, heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
        )
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.self_attention(x, mask)))
        return self.norm2(x + self.dropout(self.feed_forward(x)))


class AttentionPooling(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.score = nn.Linear(width, 1)

    def forward(self, x, mask):
        scores = self.score(x).squeeze(-1).masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1), weights


class GlobalProjection(nn.Module):
    """The effective post-pooling MLP and residual; no single-input gate."""

    def __init__(self, width, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.LayerNorm(width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.residual = nn.Linear(width, width)

    def forward(self, x):
        return self.network(x) + self.residual(x)


class LocalEncoder(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        channels = max(width // 8, 32)
        self.convolutions = nn.ModuleDict(
            {
                name: nn.Conv1d(width, channels, k, padding=p)
                for name, k, p in BRANCH_SPECS
            }
        )
        self.normalizations = nn.ModuleDict(
            {name: nn.LayerNorm(channels) for name, _, _ in BRANCH_SPECS}
        )
        self.dropouts = nn.ModuleDict(
            {name: nn.Dropout(dropout) for name, _, _ in BRANCH_SPECS}
        )
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(channels, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for name, _, _ in BRANCH_SPECS
            }
        )
        self.scorer = nn.Linear(width, 1)

    def forward(self, x, mask):
        length = x.size(1)
        branches = []
        # Preserve the reference order: compute ALL convolution outputs first,
        # then ALL branch projections. This also defines stochastic call order.
        tokens = {}
        for name, _, _ in BRANCH_SPECS:
            branch = self.convolutions[name](x.transpose(1, 2))
            if branch.size(-1) < length:
                branch = F.pad(branch, (0, length - branch.size(-1)))
            elif branch.size(-1) > length:
                branch = branch[:, :, :length]
            branch = self.normalizations[name](branch.transpose(1, 2))
            branch = self.dropouts[name](F.gelu(branch))
            tokens[name] = branch * mask.unsqueeze(-1).to(branch.dtype)
        for name, _, _ in BRANCH_SPECS:
            valid = tokens[name].masked_fill(~mask.unsqueeze(-1), 0.0)
            denominator = mask.sum(1, keepdim=True).clamp_min(1).to(x.dtype)
            branches.append(self.projections[name](valid.sum(1) / denominator))
        stacked = torch.stack(branches, dim=1)
        weights = self.scorer(stacked).squeeze(-1).softmax(dim=1)
        return (stacked * weights.unsqueeze(-1)).sum(1), weights


class HiAntitoxinPred(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        h, p = c.hidden_dim, c.dropout
        self.projection = nn.Sequential(
            nn.Linear(c.input_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p)
        )
        self.global_pool = AttentionPooling(h)
        self.global_projection = GlobalProjection(h, p)
        self.position = PositionEncoding(h)
        self.transformer = nn.ModuleList(
            [TransformerLayer(h, c.heads, p) for _ in range(c.layers)]
        )
        self.local = LocalEncoder(h, p)
        self.classifier = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.BatchNorm1d(h // 2),
            nn.Dropout(p),
            nn.Linear(h // 2, h // 4),
            nn.GELU(),
            nn.BatchNorm1d(h // 4),
            nn.Dropout(p),
            nn.Linear(h // 4, h // 8),
            nn.GELU(),
            nn.BatchNorm1d(h // 8),
            nn.Dropout(p * 0.75),
            nn.Linear(h // 8, 1),
        )
        self.apply(self._initialize)
        # The source concat fusion uses Linear defaults, after base initialization.
        self.fusion_mlp = nn.Sequential(
            nn.Linear(h * 2, h * 2),
            nn.LayerNorm(h * 2),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(h * 2, h),
            nn.GELU(),
            nn.Dropout(p),
        )
        nn.init.zeros_(self.local.scorer.weight)
        nn.init.zeros_(self.local.scorer.bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _validate(self, features, mask):
        if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
            raise ValueError("features must have shape [batch, width, input_dim]")
        if mask.dtype != torch.bool or mask.shape != features.shape[:2]:
            raise ValueError("mask must be bool [batch, width]")
        if features.shape[0] < 1 or not mask.any(dim=1).all():
            raise ValueError("An empty protein or batch is not supported")
        if not 2 <= features.shape[1] <= self.config.max_length:
            raise ValueError(
                "Input width must be between 2 and 200; pad short inputs explicitly"
            )
        if (
            features.device != mask.device
            or features.dtype != self.projection[0].weight.dtype
        ):
            raise ValueError("Input device/dtype must match the mask and model")
        lengths = mask.sum(1)
        expected = (
            torch.arange(mask.shape[1], device=mask.device)[None] < lengths[:, None]
        )
        if not torch.equal(mask, expected):
            raise ValueError("Valid residues must form a contiguous prefix")
        if not torch.isfinite(features).all():
            raise ValueError("Input features must be finite")
        if torch.count_nonzero(features.masked_select(~mask.unsqueeze(-1))):
            raise ValueError("Padded input features must be zero")

    def explain_features(self, features, mask):
        """Return raw logits, effective summaries, and pooling weights.

        Pooling weights are model internals, not MAIA or causal residue effects.
        """
        self._validate(features, mask)
        projected = self.projection(features)
        stream = self.position(projected)
        for layer in self.transformer:
            stream = layer(stream, mask)
        pooled, attention = self.global_pool(stream, mask)
        global_summary = self.global_projection(pooled)
        local_summary, weights = self.local(projected, mask)
        fused = self.fusion_mlp(torch.cat([global_summary, local_summary], dim=-1))
        return dict(
            logits=self.classifier(fused).squeeze(-1),
            branch_weights=weights,
            global_attention=attention,
            local_summary=local_summary,
            global_summary=global_summary,
            fused=fused,
        )

    def forward(self, features, mask):
        return self.explain_features(features, mask)["logits"]

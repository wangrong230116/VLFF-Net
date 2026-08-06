"""Released vision-language feature-fusion blocks used by VLFF-Net.

The end-to-end model parser, segmentation head, losses, and training runtime are
not included in the review-stage release.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class VCTETextAdapter(nn.Module):
    """Near-identity residual adapter for cached text embeddings."""

    def __init__(self, dim: int = 512, bottleneck: int = 64) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=1e-4)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        delta = self.up(F.gelu(self.down(self.norm(text))))
        return F.normalize(text + delta, dim=-1)


class VCTEVisualAdapter(nn.Module):
    """Extract a normalized global context vector from a visual feature map."""

    def __init__(self, in_channels: int, out_dim: int = 512, reduce_dim: int = 128) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.reduce = nn.Conv2d(in_channels, reduce_dim, 1, bias=False)
        self.dw3 = nn.Conv2d(reduce_dim, reduce_dim, 3, padding=1, groups=reduce_dim, bias=False)
        self.dw5 = nn.Conv2d(reduce_dim, reduce_dim, 5, padding=2, groups=reduce_dim, bias=False)
        self.dw_dilated = nn.Conv2d(
            reduce_dim, reduce_dim, 3, padding=2, dilation=2, groups=reduce_dim, bias=False
        )
        self.channel_mix = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.proj = nn.Linear(reduce_dim, out_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=1e-4)

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        visual = self.reduce(self.norm(visual))
        visual = F.relu(visual + self.dw3(visual) + self.dw5(visual) + self.dw_dilated(visual))
        pooled = visual.mean(dim=(2, 3)) + visual.amax(dim=(2, 3))
        pooled = self.channel_mix(pooled.unsqueeze(1)).squeeze(1)
        return F.normalize(self.proj(pooled), dim=-1)


class VCTETextEnhancer(nn.Module):
    """Fuse class-text features with image-level visual context in a low-rank space."""

    def __init__(
        self,
        text_dim: int = 512,
        visual_dim: int = 512,
        out_dim: int = 512,
        rank: int = 64,
    ) -> None:
        super().__init__()
        self.text_proj = nn.Linear(text_dim, rank, bias=False)
        self.visual_proj = nn.Linear(visual_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=False)
        self.gate = nn.Linear(rank, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.text_proj.weight, std=0.02)
        nn.init.normal_(self.visual_proj.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=1e-4)
        nn.init.normal_(self.gate.weight, std=1e-4)

    def forward(self, text: torch.Tensor, visual_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if text.ndim == 2:
            text = text.unsqueeze(0).expand(visual_context.shape[0], -1, -1)
        text_low = self.text_proj(text)
        visual_low = self.visual_proj(visual_context).unsqueeze(1)
        interaction = F.gelu(text_low + visual_low + text_low * visual_low)
        residual = self.up(interaction)
        gate = torch.sigmoid(self.gate(interaction))
        enhanced = F.normalize(text + self.alpha * gate * residual, dim=-1)
        return enhanced, enhanced.mean(dim=1)


class VCTENeckModulator(nn.Module):
    """Apply text-conditioned FiLM modulation to one feature-pyramid level."""

    def __init__(self, channels: int, text_dim: int = 512, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc = nn.Linear(text_dim, hidden_dim, bias=False)
        self.gamma = nn.Linear(hidden_dim, channels, bias=False)
        self.beta = nn.Linear(hidden_dim, channels, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.gamma.weight, std=1e-4)
        nn.init.normal_(self.beta.weight, std=1e-4)

    def forward(self, feature: torch.Tensor, text_context: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc(text_context))
        gamma = self.gamma(hidden).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(hidden).unsqueeze(-1).unsqueeze(-1)
        return feature * (1.0 + self.alpha * gamma) + self.alpha * beta


"""Contact-aware spatial aggregation and target-specific temporal routing."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

TARGETS = ("max_force_n", "work_j")
DILATIONS = (1, 2, 4, 8, 16, 32)
LAGS = (1, 4, 8)
EXPERTS = ("semantic", "event", "motion")


@dataclass(frozen=True)
class Variant:
    appearance: bool = True
    geometry: bool = True
    casa: bool = False
    experts: tuple[str, ...] = ()
    route: bool = False
    auxiliary: bool = False


VARIANTS = {
    "dino": Variant(geometry=False),
    "da3": Variant(appearance=False),
    "dino_da3": Variant(),
    "casa": Variant(casa=True),
    "tmtr": Variant(casa=True, experts=EXPERTS, route=True),
    "full": Variant(casa=True, experts=EXPERTS, route=True, auxiliary=True),
    "semantic": Variant(casa=True, experts=("semantic",), auxiliary=True),
    "event": Variant(casa=True, experts=("event",), auxiliary=True),
    "motion": Variant(casa=True, experts=("motion",), auxiliary=True),
    "concat": Variant(casa=True, experts=EXPERTS, auxiliary=True),
}


def masked_softmax(logits: Tensor, valid: Tensor) -> Tensor:
    weights = logits.float().masked_fill(~valid, torch.finfo(torch.float32).min).softmax(-1)
    weights = weights * valid
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    return weights.to(logits.dtype)


def lagged_delta(value: Tensor, valid: Tensor, lag: int) -> Tensor:
    if value.shape[1] <= lag:
        return torch.zeros_like(value)
    paired = valid[:, lag:] & valid[:, :-lag]
    delta = value[:, lag:] - value[:, :-lag]
    return F.pad(delta.masked_fill(~paired[..., None], 0), (0, 0, lag, 0))


class CASA(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.appearance_from_geometry = nn.MultiheadAttention(
            dim, 4, dropout=dropout, batch_first=True
        )
        self.geometry_from_appearance = nn.MultiheadAttention(
            dim, 4, dropout=dropout, batch_first=True
        )
        self.appearance_norm = nn.LayerNorm(dim)
        self.geometry_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            2 * dim, 8, dropout=dropout, batch_first=True
        )
        self.fusion = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim))
        self.relevance = nn.Linear(dim, 1)

    def forward(self, appearance: Tensor, geometry: Tensor) -> Tensor:
        updated_appearance, _ = self.appearance_from_geometry(
            appearance, geometry, geometry, need_weights=False
        )
        updated_geometry, _ = self.geometry_from_appearance(
            geometry, appearance, appearance, need_weights=False
        )
        joined = torch.cat(
            (self.appearance_norm(appearance + updated_appearance),
             self.geometry_norm(geometry + updated_geometry)), dim=-1
        )
        attended, _ = self.self_attention(joined, joined, joined, need_weights=False)
        fused = self.fusion(joined + attended)
        weights = self.relevance(fused).softmax(dim=-2)
        return (weights * fused).sum(dim=-2)


class TemporalBlock(nn.Module):
    def __init__(self, dim: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            dim, dim, 3, padding=dilation, dilation=dilation, groups=dim
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(dim, 2 * dim, 1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(2 * dim, dim, 1), nn.Dropout(dropout)
        )

    def forward(self, value: Tensor, valid: Tensor) -> Tensor:
        hidden = self.norm(value).masked_fill(~valid, 0)
        hidden = self.pointwise(self.depthwise(hidden.transpose(1, 2))).transpose(1, 2)
        return (value + hidden).masked_fill(~valid, 0)


class TemporalExpert(nn.Module):
    def __init__(self, input_dim: int, dim: int, dropout: float):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, dim), nn.GELU()
        )
        self.blocks = nn.ModuleList(TemporalBlock(dim, d, dropout) for d in DILATIONS)
        self.norm = nn.LayerNorm(dim)

    def forward(self, value: Tensor, valid: Tensor) -> Tensor:
        mask = valid[..., None]
        value = self.projection(value).masked_fill(~mask, 0)
        for block in self.blocks:
            value = block(value, mask)
        return self.norm(value).masked_fill(~mask, 0)


class TargetHead(nn.Module):
    def __init__(self, input_dim: int, dim: int, prior: float):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 1))
        self.regression = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, dim), nn.GELU(), nn.Linear(dim, 1)
        )
        nn.init.constant_(self.regression[-1].bias, prior + float(
            torch.log(-torch.expm1(torch.tensor(-prior)))
        ))

    def forward(self, value: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        weight = masked_softmax(self.score(value).squeeze(-1), valid)
        pooled = (weight[..., None] * value).sum(dim=1)
        return F.softplus(self.regression(pooled).squeeze(-1).float()), weight


class EgoPhys(nn.Module):
    def __init__(self, variant: str = "full", priors=(10.0, 1.0), dim: int = 192):
        super().__init__()
        self.variant = VARIANTS[variant]
        cfg = self.variant
        self.appearance = nn.Linear(1024, dim) if cfg.casa else None
        self.geometry = nn.Linear(2048, dim) if cfg.casa else None
        self.casa = CASA(dim, 0.1) if cfg.casa else None
        self.cls = nn.Linear(1024, dim) if "semantic" in cfg.experts else None
        self.global_patch = nn.Linear(1024, dim) if (
            "semantic" in cfg.experts or "motion" in cfg.experts
        ) else None
        input_dims = {"semantic": 2 * dim + 32, "event": dim + 32, "motion": 6 * dim + 32}
        self.time = nn.Sequential(nn.Linear(2, 32), nn.GELU()) if cfg.experts else None
        self.experts = nn.ModuleDict({
            name: TemporalExpert(input_dims[name], dim, 0.1) for name in cfg.experts
        })
        feature_dim = dim if cfg.casa else 1024 * cfg.appearance + 2048 * cfg.geometry
        if cfg.experts and not cfg.route:
            feature_dim *= len(cfg.experts)
        self.gates = nn.ModuleDict()
        self.route_norms = nn.ModuleDict()
        if cfg.route:
            self.gates.update({
                target: nn.Sequential(nn.LayerNorm(3 * dim), nn.Linear(3 * dim, 3))
                for target in TARGETS
            })
            self.route_norms.update({target: nn.LayerNorm(dim) for target in TARGETS})
        self.heads = nn.ModuleDict({
            target: TargetHead(feature_dim, dim, prior) for target, prior in zip(TARGETS, priors)
        })
        self.dense_head = nn.Linear(feature_dim, 2) if cfg.auxiliary else None

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        cfg = self.variant
        visual_valid = batch["visual_valid"]
        geometry_valid = batch["geometry_valid"] & visual_valid
        valid = geometry_valid if cfg.geometry and not cfg.appearance else visual_valid
        fused = None
        if self.casa is not None:
            # Pack only frames with both streams and keep the padded sequence shape.
            appearance = batch["dino_grid"][geometry_valid].flatten(1, 2).float()
            geometry = batch["da3_grid"][geometry_valid].flatten(1, 2).float()
            interaction = self.casa(self.appearance(appearance), self.geometry(geometry))
            fused = interaction.new_zeros((*valid.shape, interaction.shape[-1]))
            fused[geometry_valid] = interaction
            if not cfg.experts:
                valid = geometry_valid
        else:
            streams = []
            if cfg.appearance:
                streams.append(batch["dino_mean"].float().masked_fill(~visual_valid[..., None], 0))
            if cfg.geometry:
                geometry = batch["da3_grid"].float().mean(dim=(-3, -2))
                streams.append(geometry.masked_fill(~geometry_valid[..., None], 0))
            fused = torch.cat(streams, dim=-1)

        states = []
        if cfg.experts:
            time = self.time(batch["time"])
            if self.global_patch is not None:
                patch = self.global_patch(batch["dino_mean"].float())
            for name, expert in self.experts.items():
                if name == "semantic":
                    inputs = [self.cls(batch["dino_cls"].float()), patch]
                elif name == "event":
                    inputs = [fused]
                else:
                    inputs = [
                        delta for lag in LAGS for delta in
                        (lagged_delta(fused, geometry_valid, lag), lagged_delta(patch, valid, lag))
                    ]
                states.append(expert(torch.cat([*inputs, time], dim=-1), valid))
            fused = torch.cat(states, dim=-1)

        result = {"frame_valid": valid}
        for target, head in self.heads.items():
            value = fused
            if cfg.route:
                gate = self.gates[target](fused).softmax(dim=-1)
                value = self.route_norms[target](
                    (torch.stack(states, dim=-2) * gate[..., None]).sum(dim=-2)
                )
                result[target + "_gate"] = gate
            result[target], result[target + "_attention"] = head(value, valid)
            if target == "work_j" and self.dense_head is not None:
                mean, envelope = self.dense_head(value).float().unbind(-1)
                result["mean_force_n"] = F.softplus(mean)
                result["envelope_logits"] = envelope
        return result

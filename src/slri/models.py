"""Model factories for NSD, fixed-sheaf controls, and standard baselines."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from sheaf_mpnn.nsd import NSDModel, NSDVariant
from sheaf_mpnn.nsd.nsd_layers import BaseNSDConv
from sheaf_mpnn.utils import apply_orthogonal_norm, cayley
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

SHEAF_VARIANTS = {
    "general",
    "orthogonal",
    "diagonal",
    "identity",
    "frozen_orthogonal",
}
BASELINE_VARIANTS = {"gcn", "gat", "graphsage", "mlp"}
MODEL_VARIANTS = SHEAF_VARIANTS | BASELINE_VARIANTS


class IdentityNSDConv(BaseNSDConv):
    """NSD layer whose node-edge restriction maps are fixed to identity."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        context_dim: int | None = None,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__(
            stalk_dim=stalk_dim,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            alpha=alpha,
            context_dim=context_dim,
            add_self_loops=add_self_loops,
        )
        self.reset_parameters()

    def restriction_maps(
        self,
        num_edges: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Materialize the fixed restriction maps for inspection and tests."""
        return torch.eye(
            self.stalk_dim,
            device=device or self.W1.device,
            dtype=dtype or self.W1.dtype,
        ).expand(num_edges, -1, -1)

    def get_map_products(
        self, x_feat: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[None, torch.Tensor]:
        del x_feat
        return None, self.restriction_maps(
            edge_index.size(1),
            device=edge_index.device,
            dtype=self.W1.dtype,
        )

    def _apply_norm(
        self,
        self_map: None,
        cross_map: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del self_map
        return apply_orthogonal_norm(cross_map, edge_index, num_nodes)

    def message(
        self,
        z_dst: torch.Tensor,
        z_src: torch.Tensor,
        self_map: torch.Tensor,
        cross_map: torch.Tensor,
    ) -> torch.Tensor:
        return self_map * z_dst - torch.matmul(cross_map, z_src)


class FrozenOrthogonalNSDConv(BaseNSDConv):
    """NSD layer with deterministic, edge-specific orthogonal maps."""

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        alpha: float = 1.0,
        context_dim: int | None = None,
        add_self_loops: bool = True,
        map_seed: int = 0,
    ) -> None:
        super().__init__(
            stalk_dim=stalk_dim,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            alpha=alpha,
            context_dim=context_dim,
            add_self_loops=add_self_loops,
        )
        self.register_buffer(
            "map_seed", torch.tensor(int(map_seed), dtype=torch.long)
        )
        self._local_node_ids: torch.Tensor | None = None
        self.reset_parameters()

    def set_local_node_ids(self, local_node_ids: torch.Tensor | None) -> None:
        """Set graph-local node IDs used to make maps batching-invariant."""
        self._local_node_ids = local_node_ids

    def restriction_maps(
        self,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fixed destination and source incidence maps."""
        src, dst = edge_index
        local = self._local_node_ids
        if local is None or local.numel() <= int(edge_index.max().item()):
            local = torch.arange(
                int(edge_index.max().item()) + 1,
                device=edge_index.device,
                dtype=torch.long,
            )
        src_local, dst_local = local[src], local[dst]
        lo = torch.minimum(src_local, dst_local)
        hi = torch.maximum(src_local, dst_local)
        edge_code = lo * 1_000_003 + hi * 97_409 + self.map_seed
        return (
            self._orthogonal_map(edge_code, dst_local),
            self._orthogonal_map(edge_code, src_local),
        )

    def _orthogonal_map(
        self, edge_code: torch.Tensor, endpoint: torch.Tensor
    ) -> torch.Tensor:
        code = edge_code.to(dtype=self.W1.dtype)
        endpoint_code = endpoint.to(dtype=self.W1.dtype)
        if self.stalk_dim == 1:
            sign = torch.where(
                torch.sin(code * 0.017 + endpoint_code * 0.131) >= 0,
                1.0,
                -1.0,
            )
            return sign.view(-1, 1, 1)
        num_params = self.stalk_dim * (self.stalk_dim - 1) // 2
        coordinates = torch.arange(
            1,
            num_params + 1,
            device=code.device,
            dtype=code.dtype,
        )
        params = torch.sin(
            code[:, None] * (0.011 + coordinates * 0.00013)
            + endpoint_code[:, None] * (0.071 + coordinates * 0.00017)
            + coordinates * 1.61803398875
        )
        return cayley(params, self.stalk_dim, clamp_val=10.0)

    def get_map_products(
        self, x_feat: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[None, torch.Tensor]:
        del x_feat
        dst_map, src_map = self.restriction_maps(edge_index)
        return None, dst_map.transpose(-2, -1) @ src_map

    def _apply_norm(
        self,
        self_map: None,
        cross_map: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del self_map
        return apply_orthogonal_norm(cross_map, edge_index, num_nodes)

    def message(
        self,
        z_dst: torch.Tensor,
        z_src: torch.Tensor,
        self_map: torch.Tensor,
        cross_map: torch.Tensor,
    ) -> torch.Tensor:
        return self_map * z_dst - torch.matmul(cross_map, z_src)


class IdentityNSDModel(NSDModel):
    """Upstream NSD architecture with fixed identity restriction maps."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        alpha: float = 1.0,
        add_self_loops: bool = True,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = True,
        jknet: bool = False,
        **_: Any,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stalk_dim=stalk_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            variant=NSDVariant.ORTHOGONAL,
            alpha=alpha,
            add_self_loops=add_self_loops,
            input_dropout=input_dropout,
            dropout=dropout,
            normalize_output=normalize_output,
            jknet=jknet,
        )
        context_dim = stalk_dim * hidden_dim
        self.layers = nn.ModuleList(
            [
                IdentityNSDConv(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    context_dim=context_dim,
                    alpha=alpha,
                    add_self_loops=add_self_loops,
                )
                for _ in range(num_layers)
            ]
        )


class FrozenOrthogonalNSDModel(NSDModel):
    """Upstream NSD architecture with deterministic frozen orthogonal maps."""

    supports_batch_argument = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 4,
        hidden_dim: int = 16,
        num_layers: int = 2,
        alpha: float = 1.0,
        add_self_loops: bool = True,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        normalize_output: bool = True,
        map_seed: int = 0,
        **_: Any,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stalk_dim=stalk_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            variant=NSDVariant.ORTHOGONAL,
            alpha=alpha,
            add_self_loops=add_self_loops,
            input_dropout=input_dropout,
            dropout=dropout,
            normalize_output=normalize_output,
        )
        context_dim = stalk_dim * hidden_dim
        self.layers = nn.ModuleList(
            [
                FrozenOrthogonalNSDConv(
                    stalk_dim=stalk_dim,
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                    context_dim=context_dim,
                    alpha=alpha,
                    add_self_loops=add_self_loops,
                    map_seed=map_seed + 104_729 * layer_index,
                )
                for layer_index in range(num_layers)
            ]
        )

    @staticmethod
    def _local_ids(x: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:
        if batch is None:
            return torch.arange(x.size(0), device=x.device)
        counts = torch.bincount(batch, minlength=int(batch.max().item()) + 1)
        offsets = torch.cumsum(counts, dim=0) - counts
        return torch.arange(x.size(0), device=x.device) - offsets[batch]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_ids = self._local_ids(x, batch)
        for layer in self.layers:
            layer.set_local_node_ids(local_ids)
        try:
            return super().forward(x, edge_index)
        finally:
            for layer in self.layers:
                layer.set_local_node_ids(None)


class BaselineNodeModel(nn.Module):
    """Explicit-layer GCN, GAT, GraphSAGE, or node-wise MLP baseline."""

    supports_batch_argument = True

    def __init__(
        self,
        *,
        family: str,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()
        if family not in BASELINE_VARIANTS:
            raise ValueError(f"Unknown baseline family: {family!r}")
        self.family = family
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.normalize_output = normalize_output
        self.input_dropout_layer = nn.Dropout(input_dropout)
        self.dropout_layer = nn.Dropout(dropout)
        self.encoder = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList(
            [self._make_layer(family, hidden_dim) for _ in range(num_layers)]
        )
        self.decoder = nn.Linear(hidden_dim, out_channels)

    @staticmethod
    def _make_layer(family: str, hidden_dim: int) -> nn.Module:
        if family == "gcn":
            return GCNConv(hidden_dim, hidden_dim, add_self_loops=True)
        if family == "gat":
            return GATConv(
                hidden_dim,
                hidden_dim,
                heads=1,
                concat=False,
                add_self_loops=True,
            )
        if family == "graphsage":
            return SAGEConv(hidden_dim, hidden_dim)
        return nn.Linear(hidden_dim, hidden_dim)

    def initial_state(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.input_dropout_layer(x))

    def layer_step(
        self,
        layer_index: int,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        layer = self.layers[layer_index]
        if self.family == "mlp":
            updated = layer(hidden)
        else:
            updated = layer(hidden, edge_index)
        return torch.relu(updated)

    def decode_state(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.normalize_output:
            hidden = F.normalize(hidden, p=2, dim=-1)
        return self.decoder(hidden)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del batch
        hidden = self.initial_state(x)
        for layer_index in range(self.num_layers):
            hidden = self.layer_step(layer_index, hidden, edge_index)
            hidden = self.dropout_layer(hidden)
        return self.decode_state(hidden)


def build_model(
    *,
    variant: str,
    in_channels: int,
    out_channels: int,
    stalk_dim: int,
    hidden_dim: int,
    num_layers: int,
    alpha: float = 1.0,
    orth_strategy: str = "cayley",
    input_dropout: float = 0.0,
    dropout: float = 0.0,
    normalize_output: bool = True,
    seed: int = 0,
) -> nn.Module:
    """Build any supported sheaf model or conventional baseline."""
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant: {variant!r}")
    if variant in BASELINE_VARIANTS:
        return BaselineNodeModel(
            family=variant,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            input_dropout=input_dropout,
            normalize_output=normalize_output,
        )
    common = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "stalk_dim": stalk_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "alpha": alpha,
        "input_dropout": input_dropout,
        "dropout": dropout,
        "normalize_output": normalize_output,
    }
    if variant == "identity":
        return IdentityNSDModel(**common)
    if variant == "frozen_orthogonal":
        return FrozenOrthogonalNSDModel(**common, map_seed=seed)
    if variant == "orthogonal" and stalk_dim == 1:
        raise ValueError("learned orthogonal NSD with stalk_dim=1 is identity")
    nsd_variant = NSDVariant[variant.upper()]
    return NSDModel(
        **common,
        variant=nsd_variant,
        orth_strategy=orth_strategy,
    )


def build_nsd_model(**kwargs: Any) -> nn.Module:
    """Backward-compatible alias for the unified model factory."""
    return build_model(**kwargs)


def forward_model(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Call local models with graph-batch metadata when they support it."""
    if getattr(model, "supports_batch_argument", False):
        return model(x, edge_index, batch=batch)
    return model(x, edge_index)


__all__ = [
    "BASELINE_VARIANTS",
    "MODEL_VARIANTS",
    "SHEAF_VARIANTS",
    "BaselineNodeModel",
    "FrozenOrthogonalNSDConv",
    "FrozenOrthogonalNSDModel",
    "IdentityNSDConv",
    "IdentityNSDModel",
    "build_model",
    "build_nsd_model",
    "forward_model",
]

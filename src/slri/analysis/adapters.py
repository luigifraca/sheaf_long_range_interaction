"""Differentiable layer-by-layer adapters for analysis."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sheaf_mpnn.nsd import NSDModel
from torch import nn

from slri.models import (
    BaselineNodeModel,
    FrozenOrthogonalNSDConv,
    forward_model,
)


class ModelTraceAdapter:
    """Expose encoder, individual diffusion layers, and decoder."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        if isinstance(model, BaselineNodeModel):
            self.kind = "explicit"
        elif isinstance(model, NSDModel):
            self.kind = "nsd"
        elif all(
            hasattr(model, name)
            for name in (
                "initial_state",
                "layer_step",
                "decode_state",
                "num_layers",
                "hidden_dim",
                "encoder",
                "decoder",
            )
        ):
            self.kind = "explicit"
        else:
            raise TypeError(f"Unsupported trace model: {type(model).__name__}")

    @property
    def num_layers(self) -> int:
        return int(self.model.num_layers)

    @property
    def hidden_size(self) -> int:
        if self.kind == "explicit":
            return int(self.model.hidden_dim)
        return int(self.model.stalk_dim * self.model.hidden_dim)

    def initial_state(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "explicit":
            return self.model.initial_state(x)
        hidden = self.model.encoder(self.model.input_dropout_layer(x))
        return hidden.reshape(x.size(0), -1)

    def layer_step(
        self,
        layer_index: int,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if self.kind == "explicit":
            return self.model.layer_step(layer_index, hidden, edge_index)
        layer = self.model.layers[layer_index]
        if isinstance(layer, FrozenOrthogonalNSDConv):
            layer.set_local_node_ids(
                torch.arange(hidden.size(0), device=hidden.device)
            )
        stalk = hidden.view(
            hidden.size(0), self.model.stalk_dim, self.model.hidden_dim
        )
        try:
            return layer(hidden, stalk, edge_index).reshape(hidden.size(0), -1)
        finally:
            if isinstance(layer, FrozenOrthogonalNSDConv):
                layer.set_local_node_ids(None)

    def decode_state(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.kind == "explicit":
            return self.model.decode_state(hidden)
        if self.model.normalize_output:
            hidden = F.normalize(hidden, p=2, dim=-1)
        return self.model.decoder(hidden)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        return forward_model(self.model, x, edge_index)

    def trace(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Return states H^(0)...H^(L) and final logits."""
        states = [self.initial_state(x)]
        for layer_index in range(self.num_layers):
            states.append(
                self.layer_step(layer_index, states[-1], edge_index)
            )
        return states, self.decode_state(states[-1])

    def encoder_jacobian(self) -> torch.Tensor:
        """Return the node-local encoder Jacobian dH0_u/dX_u."""
        return self.model.encoder.weight

    def decoder_jacobian(
        self, hidden_node: torch.Tensor
    ) -> torch.Tensor:
        """Return the node-local decoder Jacobian dZ_v/dHL_v."""

        def decode_node(value: torch.Tensor) -> torch.Tensor:
            if self.kind == "explicit":
                if self.model.normalize_output:
                    value = F.normalize(value, p=2, dim=-1)
                return self.model.decoder(value)
            if self.model.normalize_output:
                value = F.normalize(value, p=2, dim=-1)
            return self.model.decoder(value)

        return torch.func.jacrev(decode_node)(hidden_node)

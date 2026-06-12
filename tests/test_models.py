import itertools

import pytest
import torch

from slri.models import IdentityNSDConv, IdentityNSDModel, build_nsd_model


def cycle_edges(nodes: int) -> torch.Tensor:
    pairs = []
    for index in range(nodes):
        target = (index + 1) % nodes
        pairs.extend(((index, target), (target, index)))
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def test_identity_maps_are_exact_and_not_parameterized():
    layer = IdentityNSDConv(stalk_dim=3, in_channels=4, hidden_dim=4)
    maps = layer.restriction_maps(7)
    expected = torch.eye(3).expand(7, -1, -1)
    assert torch.equal(maps, expected)
    assert not hasattr(layer, "map_generator")
    parameter_names = {name for name, _ in layer.named_parameters()}
    assert parameter_names == {"W1", "W2", "alpha"}


def test_identity_diffusion_treats_equal_stalk_coordinates_equally():
    layer = IdentityNSDConv(
        stalk_dim=2,
        in_channels=3,
        hidden_dim=3,
        add_self_loops=False,
    )
    with torch.no_grad():
        layer.W1.copy_(torch.eye(2))
        layer.W2.copy_(torch.eye(3))
        layer.alpha.fill_(0.25)
    row = torch.randn(5, 1, 3)
    stalk = row.expand(-1, 2, -1).clone()
    context = stalk.reshape(5, -1)
    output = layer(context, stalk, cycle_edges(5))
    assert torch.allclose(output[:, 0], output[:, 1])


@pytest.mark.parametrize(
    ("variant", "stalk_dim", "hidden_dim"),
    itertools.product(
        ["general", "orthogonal", "diagonal", "identity"],
        [2, 3, 5],
        [16, 32],
    ),
)
def test_all_24_models_forward_and_backward(variant, stalk_dim, hidden_dim):
    model = build_nsd_model(
        variant=variant,
        in_channels=5,
        out_channels=3,
        stalk_dim=stalk_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
    )
    output = model(torch.randn(6, 5), cycle_edges(6))
    assert output.shape == (6, 3)
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_identity_model_has_no_restriction_parameters():
    model = IdentityNSDModel(
        in_channels=5,
        out_channels=2,
        stalk_dim=3,
        hidden_dim=4,
        num_layers=2,
    )
    assert all(
        not hasattr(layer, "map_generator")
        for layer in model.layers
    )


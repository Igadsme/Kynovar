"""Construct dynamics models that share one training-set normalizer."""

from __future__ import annotations

from kynovar.data.normalize import Normalizer
from kynovar.models.baselines.dynamics import ConstantVelocity, GRUDynamics, LinearDynamics, MLPDynamics
from kynovar.models.gnn.dynamics import DynamicsGNN
from kynovar.models.gnn.interaction import InteractionGNN

MODEL_NAMES = ("constant_velocity", "linear", "mlp", "gru", "gnn", "interaction_gnn", "interaction_gnn_single")


def build_model(name: str, normalizer: Normalizer, hidden_dim: int, gnn_layers: int, sequence_length: int):
    if name == "constant_velocity":
        return ConstantVelocity()
    if name == "linear":
        return LinearDynamics(normalizer)
    if name == "mlp":
        return MLPDynamics(normalizer, hidden_dim=hidden_dim)
    if name == "gru":
        return GRUDynamics(normalizer, hidden_dim=hidden_dim, history=sequence_length)
    if name == "gnn":
        return DynamicsGNN(normalizer, hidden_dim=hidden_dim, layers=gnn_layers)
    if name == "interaction_gnn":
        return InteractionGNN(normalizer, hidden_dim=hidden_dim, layers=gnn_layers, history=max(2, sequence_length))
    if name == "interaction_gnn_single":
        return InteractionGNN(normalizer, hidden_dim=hidden_dim, layers=gnn_layers, history=1)
    raise ValueError(f"Unknown model {name!r}. Known models: {MODEL_NAMES}.")


def trainable_parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

import torch

from src.training.plan_feature_store import ModeSlice, PlanFeatureStore


def _make_slice(mode: str, rows: int, cols: int, *, gate: bool = False, adherence: bool = False) -> ModeSlice:
    features = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    gate_targets = torch.arange(rows, dtype=torch.long) % 3 if gate else None
    adherence_flags = torch.linspace(0.1, 0.9, steps=rows) if adherence else None
    extras = {"encode_flag": torch.ones(rows, 1)} if mode == "overworld" else {}
    return ModeSlice(
        mode=mode,
        plan_features=features,
        gate_targets=gate_targets,
        adherence=adherence_flags,
        extras=extras,
    )


def test_sample_weights_respects_priors():
    battle_slice = _make_slice("battle", rows=4, cols=2, gate=True, adherence=True)
    overworld_slice = _make_slice("overworld", rows=2, cols=2)

    store = PlanFeatureStore(
        slices={"battle": battle_slice, "overworld": overworld_slice},
        metadata={"order": ["battle", "overworld"]},
    )

    weights = store.sample_weights({"battle": 0.7, "overworld": 0.3})

    expected = torch.tensor(
        [0.7 / 4] * 4 + [0.3 / 2] * 2,
        dtype=torch.float32,
    )
    assert torch.allclose(weights, expected)


def test_sample_weights_handles_empty_slices():
    empty_slice = ModeSlice(mode="battle", plan_features=torch.zeros((0, 2)))
    store = PlanFeatureStore(slices={"battle": empty_slice}, metadata={})

    weights = store.sample_weights({"battle": 1.0})

    assert weights.numel() == 0

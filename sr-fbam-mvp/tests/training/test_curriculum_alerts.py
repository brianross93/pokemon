from src.training.train_battle_il import _evaluate_curriculum_alerts


def test_curriculum_alerts_within_bounds():
    curriculum = {
        "gating_heuristics": {
            "mixed_curriculum": {
                "encode_band": [0.2, 0.4],
                "plan_lookup_band": [0.1, 0.3],
                "plan_step_band": [0.05, 0.15],
                "adherence_target": 0.6,
            },
            "alerts": {"skip_spike_threshold": 0.2},
        }
    }
    stats = {
        "encode": 0.3,
        "lookup": 0.2,
        "step": 0.1,
        "skip": 0.1,
        "adherence_mean": 0.7,
        "adherence_positive": 0.9,
    }

    alerts = _evaluate_curriculum_alerts(curriculum, stats)
    assert alerts == {}


def test_curriculum_alerts_raise_when_out_of_band():
    curriculum = {
        "gating_heuristics": {
            "mixed_curriculum": {
                "encode_band": [0.2, 0.4],
                "plan_lookup_band": [0.1, 0.3],
                "plan_step_band": [0.05, 0.15],
                "adherence_target": 0.6,
            },
            "alerts": {"skip_spike_threshold": 0.2},
        }
    }
    stats = {
        "encode": 0.85,
        "lookup": 0.01,
        "step": 0.01,
        "skip": 0.3,
        "adherence_mean": 0.1,
        "adherence_positive": 0.1,
    }

    alerts = _evaluate_curriculum_alerts(curriculum, stats)
    assert "encode_band" in alerts
    assert "plan_lookup_band" in alerts
    assert "plan_step_band" in alerts
    assert "adherence_target" in alerts
    assert "skip_spike" in alerts

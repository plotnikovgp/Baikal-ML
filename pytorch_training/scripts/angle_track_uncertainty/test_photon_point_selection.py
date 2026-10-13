"""Point-only checkpoints are ranked by working-hit line quality."""

from train_photon_residual import point_selection


def metric(pred_line_q68=4.3, pred_line_q95=13.8, gt_line_q68=6.0, angle=4.34):
    return {
        "pred_val": {
            "direction_deg": {"q68": angle},
            "line_m": {"q68": pred_line_q68, "q95": pred_line_q95},
        },
        "gt_val_clean": {
            "direction_deg": {"q68": 3.94},
            "line_m": {"q68": gt_line_q68},
        },
    }


def test_point_selection_rewards_working_line_and_guards_quality():
    baseline = metric()
    score, admitted = point_selection(
        metric(pred_line_q68=4.2, pred_line_q95=13.7, gt_line_q68=6.03), baseline
    )
    assert score == 4.2
    assert admitted
    assert not point_selection(metric(angle=4.35), baseline)[1]
    assert not point_selection(metric(gt_line_q68=6.06), baseline)[1]
    assert not point_selection(metric(pred_line_q95=14.01), baseline)[1]

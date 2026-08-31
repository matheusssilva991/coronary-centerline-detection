"""Aliases mantidos exclusivamente para leitura de resultados históricos."""

from __future__ import annotations


LEGACY_READABLE_COLUMN_NAMES: dict[str, str] = {
    "aorta_recovered_initialization": "aorta_recovered_initialization",
    "aorta_circle_filter_interpolated_count": "aorta_circle_filter_interpolated_count",
    "aorta_circle_filter_fallback_enabled": "aorta_circle_filter_fallback_enabled",
    "aorta_circle_filter_accepted": "aorta_circle_filter_accepted",
    "aorta_circle_filter_rejected": "aorta_circle_filter_rejected",
    "aorta_circle_filter_rejection_reason": "aorta_circle_filter_rejection_reason",
    "aorta_circle_filter_candidate_controller_state": "aorta_circle_filter_candidate_controller_state",
    "aorta_circle_filter_fallback_controller_state": "aorta_circle_filter_fallback_controller_state",
    "aorta_circle_filter_low_coverage_fallback_enabled": "aorta_circle_filter_low_coverage_fallback_enabled",
    "aorta_circle_filter_low_coverage_fallback_attempted": "aorta_circle_filter_low_coverage_fallback_attempted",
    "aorta_circle_filter_low_coverage_fallback_accepted": "aorta_circle_filter_low_coverage_fallback_accepted",
    "aorta_circle_filter_low_coverage_fallback_rejection_reason": "aorta_circle_filter_low_coverage_fallback_rejection_reason",
    "aorta_circle_filter_low_coverage_candidate_area_ratio_p90": "aorta_circle_filter_low_coverage_candidate_area_ratio_p90",
    "aorta_circle_filter_low_coverage_retry_area_ratio_p90": "aorta_circle_filter_low_coverage_retry_area_ratio_p90",
    "aorta_conditional_correction_enabled": "aorta_conditional_correction_enabled",
    "aorta_conditional_correction_state": "aorta_conditional_correction_state",
    "aorta_conditional_correction_attempted": "aorta_conditional_correction_attempted",
    "aorta_conditional_correction_accepted": "aorta_conditional_correction_accepted",
    "aorta_conditional_correction_method": "aorta_conditional_correction_method",
    "aorta_conditional_leak_candidate_mode": "aorta_conditional_leak_candidate_mode",
    "aorta_conditional_correction_rejection_reason": "aorta_conditional_correction_rejection_reason",
    "aorta_conditional_circle_signal_count": "aorta_conditional_circle_signal_count",
    "aorta_conditional_original_circle_count": "aorta_conditional_original_circle_count",
    "aorta_conditional_candidate_circle_count": "aorta_conditional_candidate_circle_count",
    "aorta_conditional_hough_added_count": "aorta_conditional_hough_added_count",
    "aorta_conditional_synthetic_added_count": "aorta_conditional_synthetic_added_count",
    "aorta_conditional_baseline_area_ratio_p90": "aorta_conditional_baseline_area_ratio_p90",
    "aorta_conditional_candidate_area_ratio_p90": "aorta_conditional_candidate_area_ratio_p90",
    "aorta_conditional_baseline_fill_q25": "aorta_conditional_baseline_fill_q25",
    "aorta_conditional_candidate_fill_q25": "aorta_conditional_candidate_fill_q25",
    "aorta_conditional_baseline_volume_fraction": "aorta_conditional_baseline_volume_fraction",
    "aorta_conditional_candidate_volume_fraction": "aorta_conditional_candidate_volume_fraction",
    "aorta_conditional_baseline_slice_area_jump_p95": "aorta_conditional_baseline_slice_area_jump_p95",
    "aorta_conditional_candidate_slice_area_jump_p95": "aorta_conditional_candidate_slice_area_jump_p95",
    "aorta_level_set_refinement_applied": "aorta_level_set_refinement_applied",
    "aorta_level_set_refinement_accepted": "aorta_level_set_refinement_accepted",
    "aorta_level_set_refinement_iterations": "aorta_level_set_refinement_iterations",
    "aorta_level_set_refinement_balloon": "aorta_level_set_refinement_balloon",
    "aorta_level_set_refinement_smoothing": "aorta_level_set_refinement_smoothing",
    "aorta_level_set_refinement_transition_mode": "aorta_level_set_refinement_transition_mode",
    "aorta_level_set_refinement_anomaly_margin_slices": "aorta_level_set_refinement_anomaly_margin_slices",
    "aorta_level_set_refinement_volume_loss_fraction": "aorta_level_set_refinement_volume_loss_fraction",
    "aorta_level_set_refinement_rejection_reason": "aorta_level_set_refinement_rejection_reason",
    "ostia_surface_mode": "ostia_surface_mode",
    "ostia_surface_thickness_mm": "ostia_surface_thickness_mm",
    "ostia_candidate_score_mode": "ostia_candidate_score_mode",
    "ostia_pair_selection_mode": "ostia_pair_selection_mode",
}

LEGACY_READABLE_BOOL_COLUMNS: set[str] = {
    "aorta_level_set_refinement_applied",
    "aorta_level_set_refinement_accepted",
}


__all__ = ["LEGACY_READABLE_BOOL_COLUMNS", "LEGACY_READABLE_COLUMN_NAMES"]

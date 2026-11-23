"""Testa a busca em etapas por listas menores de sigmas da aorta."""

import unittest
from itertools import combinations
from unittest.mock import Mock, call

from utils.project.analysis.aorta_vesselness import (
    AortaVesselnessResult,
    search_reduced_aorta_sigmas,
)

REFERENCE_SIGMAS = (16.0, 17.0, 18.0, 19.0)
REFERENCE_RESULT = AortaVesselnessResult(
    "Máscara produzida", dice=0.9586, volume_ml=110.0, frangi_seconds=250.0
)


class AortaVesselnessSearchTests(unittest.TestCase):
    def test_uses_fastest_eligible_single_sigma_and_reuses_results(self):
        results = {REFERENCE_SIGMAS: REFERENCE_RESULT}
        evaluations = {
            (16.0,): AortaVesselnessResult(
                "Máscara produzida",
                dice=0.9586 - 0.005,
                volume_ml=110.0,
                frangi_seconds=80.0,
            ),
            (17.0,): AortaVesselnessResult(
                "Máscara produzida", dice=0.96, volume_ml=110.0, frangi_seconds=250.0
            ),
            (18.0,): AortaVesselnessResult(
                "Máscara produzida", dice=0.955, volume_ml=110.0, frangi_seconds=75.0
            ),
            (19.0,): AortaVesselnessResult(
                "Máscara produzida", dice=0.95359, volume_ml=110.0, frangi_seconds=60.0
            ),
        }
        evaluate = Mock(side_effect=evaluations.__getitem__)
        tested, recommendation = search_reduced_aorta_sigmas(
            REFERENCE_SIGMAS, results, evaluate, max_dice_loss=0.005
        )
        self.assertEqual(recommendation, (18.0,))
        self.assertEqual(tuple(tested), (REFERENCE_SIGMAS, *evaluations))
        self.assertEqual(
            evaluate.call_args_list, [call(sigmas) for sigmas in evaluations]
        )

        evaluate.reset_mock()
        _, recommendation = search_reduced_aorta_sigmas(
            REFERENCE_SIGMAS, results, evaluate, max_dice_loss=0.005
        )
        self.assertEqual(recommendation, (18.0,))
        evaluate.assert_not_called()

    def test_accepts_tolerance_boundary_after_rejecting_slower_groups(self):
        candidate = (16.0, 17.0, 18.0)
        full = AortaVesselnessResult(
            "Máscara produzida", dice=0.95, volume_ml=110.0, frangi_seconds=250.0
        )
        slower = AortaVesselnessResult(
            "Máscara produzida", dice=0.96, volume_ml=110.0, frangi_seconds=251.0
        )
        faster = AortaVesselnessResult(
            "Máscara produzida", dice=0.945, volume_ml=110.0, frangi_seconds=175.0
        )
        evaluate = Mock(
            side_effect=lambda sigmas: faster if sigmas == candidate else slower
        )
        tested, recommendation = search_reduced_aorta_sigmas(
            REFERENCE_SIGMAS, {REFERENCE_SIGMAS: full}, evaluate, max_dice_loss=0.005
        )
        self.assertEqual(recommendation, candidate)
        self.assertEqual(
            tuple(tested),
            (
                REFERENCE_SIGMAS,
                *(
                    s
                    for size in (1, 2, 3)
                    for s in combinations(REFERENCE_SIGMAS, size)
                ),
            ),
        )

    def test_keeps_reference_when_no_list_is_eligible(self):
        empty = AortaVesselnessResult(
            "Máscara vazia", dice=0.0, volume_ml=0.0, frangi_seconds=60.0
        )
        evaluate = Mock(return_value=empty)
        tested, recommendation = search_reduced_aorta_sigmas(
            REFERENCE_SIGMAS,
            {REFERENCE_SIGMAS: REFERENCE_RESULT},
            evaluate,
            max_dice_loss=0.005,
        )
        self.assertIsNone(recommendation)
        self.assertEqual(len(tested), 15)
        self.assertEqual(evaluate.call_count, 14)

    def test_only_measures_single_sigmas_without_valid_reference_dice(self):
        for dice in (None, float("nan")):
            with self.subTest(dice=dice):
                result = AortaVesselnessResult(
                    "Máscara produzida",
                    dice=dice,
                    volume_ml=110.0,
                    frangi_seconds=100.0,
                )
                evaluate = Mock(return_value=result)
                tested, recommendation = search_reduced_aorta_sigmas(
                    REFERENCE_SIGMAS, {}, evaluate, max_dice_loss=0.005
                )
                self.assertIsNone(recommendation)
                self.assertEqual(
                    tuple(tested),
                    (REFERENCE_SIGMAS, *combinations(REFERENCE_SIGMAS, 1)),
                )
                self.assertEqual(
                    evaluate.call_args_list, [call(sigmas) for sigmas in tested]
                )

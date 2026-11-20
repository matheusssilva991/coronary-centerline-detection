"""Verifica a filtragem de casos com óstios aceitos."""

import unittest

import pandas as pd

from utils.comparison_utils.bad_cases import filter_correct_ostia_cases


class BadCasesAnalysisTests(unittest.TestCase):
    def test_status_column_precedes_flags_even_when_all_statuses_are_missing(self):
        flags = pd.DataFrame({"both_correct": ["y", "true"], "both_tolerable": [0, 0]})
        for statuses, expected in (([None, None], []), ([None, "both_tolerable"], [1])):
            with self.subTest(statuses=statuses):
                frame = flags.assign(ostia_status=statuses)
                self.assertEqual(
                    filter_correct_ostia_cases(frame).index.tolist(), expected
                )
        self.assertEqual(filter_correct_ostia_cases(flags).index.tolist(), [0, 1])

    def test_legacy_status_accepts_an_explicit_empty_or_custom_selection(self):
        frame = pd.DataFrame({"status": ["both_correct", "one_correct"]})
        self.assertTrue(filter_correct_ostia_cases(frame, []).empty)
        selected = filter_correct_ostia_cases(frame, ["one_correct"])
        self.assertEqual(selected.index.tolist(), [1])


if __name__ == "__main__":
    unittest.main()

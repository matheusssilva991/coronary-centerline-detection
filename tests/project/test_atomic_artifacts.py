import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils.project.results import make_json_safe as legacy_make_json_safe
from utils.project.results.io import save_dataframe_atomic
from utils.project.runtime.run_logging import add_run_file_handler
from utils.utils.json_io import load_json_file, make_json_safe, save_json_atomic


class AtomicArtifactsTest(unittest.TestCase):
    def test_json_preserves_serialization_and_public_alias(self):
        payload = {
            "nome": "aorta",
            "valores": np.array([1, 2]),
            "pasta": Path("exame"),
            "tupla": (1, 2),
        }
        self.assertIs(legacy_make_json_safe, make_json_safe)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "numeric" / "metadata.json"
            save_json_atomic(payload, path)
            self.assertEqual(
                path.read_text(),
                json.dumps(make_json_safe(payload), indent=2, ensure_ascii=False),
            )
            self.assertEqual(load_json_file(path)["valores"], [1, 2])
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_json_failure_preserves_old_file_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            path.write_text('{"previous": true}')
            for error_stage in ("serialization", "replacement"):
                with self.subTest(stage=error_stage):
                    if error_stage == "serialization":
                        with self.assertRaises(TypeError):
                            save_json_atomic({"invalid": object()}, path)
                    else:
                        with (
                            patch.object(
                                Path, "replace", side_effect=OSError("interrompido")
                            ),
                            self.assertRaises(OSError),
                        ):
                            save_json_atomic({"new": True}, path)
                    self.assertEqual(load_json_file(path), {"previous": True})
                    self.assertEqual(list(path.parent.iterdir()), [path])

    def test_csv_preserves_order_columns_values_and_cleans_failed_write(self):
        frame = pd.DataFrame(
            {"exam_id": ["B", "A"], "dice": [0.2, 0.8], "note": ["áorta", "a,b"]}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            save_dataframe_atomic(frame, path)
            expected = frame.to_csv(index=False)
            self.assertEqual(path.read_text(), expected)
            pd.testing.assert_frame_equal(pd.read_csv(path), frame)
            with (
                patch.object(
                    pd.DataFrame, "to_csv", side_effect=OSError("disco cheio")
                ),
                self.assertRaises(OSError),
            ):
                save_dataframe_atomic(frame, path)
            self.assertEqual(path.read_text(), expected)
            self.assertEqual(list(path.parent.iterdir()), [path])
            with (
                patch.object(Path, "replace", side_effect=OSError("interrompido")),
                self.assertRaises(OSError),
            ):
                save_dataframe_atomic(frame, path)
            self.assertEqual(path.read_text(), expected)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_each_write_uses_its_own_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            replace = Path.replace
            temporary_paths = []

            def record_replace(source, destination):
                temporary_paths.append(source)
                self.assertEqual(source.parent, path.parent)
                return replace(source, destination)

            with patch.object(Path, "replace", record_replace):
                save_json_atomic({"n": 1}, path)
                save_json_atomic({"n": 2}, path)
            self.assertNotEqual(temporary_paths[0], temporary_paths[1])
            self.assertEqual(load_json_file(path), {"n": 2})

    def test_reader_rejects_non_object_and_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            for invalid in ("[]", "{"):
                path.write_text(invalid)
                with self.assertRaises(ValueError):
                    load_json_file(path)

    def test_logging_does_not_repeat_messages_for_same_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.log"
            formatter = logging.Formatter("%(message)s")
            handler = add_run_file_handler(path, formatter=formatter)
            try:
                second = add_run_file_handler(path, formatter=formatter)
                self.assertIs(handler, second)
                logging.getLogger().warning("uma mensagem")
                handler.flush()
                self.assertEqual(path.read_text().splitlines(), ["uma mensagem"])
            finally:
                logging.getLogger().removeHandler(handler)
                handler.close()

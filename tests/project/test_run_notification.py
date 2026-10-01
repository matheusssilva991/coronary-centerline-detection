"""Testa avisos opcionais de conclusão sem acessar o desktop real."""

import subprocess
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from utils.project.run_notification import notify_run_completion


class RunNotificationTests(TestCase):
    @patch("utils.project.run_notification.subprocess.run")
    def test_complete_run_uses_desktop_and_success_sound(self, run_command):
        run_command.return_value = subprocess.CompletedProcess([], 0, "", "")

        notify_run_completion(
            pipeline="ImageCAS",
            split="train",
            resolution="mid",
            run_dir=Path("/runs/example"),
            status="complete",
        )

        self.assertEqual(run_command.call_count, 2)
        desktop = run_command.call_args_list[0].args[0]
        sound = run_command.call_args_list[1].args[0]
        self.assertEqual(desktop[0], "notify-send")
        self.assertIn("ImageCAS: concluído", desktop)
        self.assertIn("Saída: /runs/example", desktop[-1])
        self.assertEqual(sound, ["canberra-gtk-play", "--id=complete"])

    @patch("utils.project.run_notification.subprocess.run")
    def test_warning_and_failure_use_distinct_sounds(self, run_command):
        run_command.return_value = subprocess.CompletedProcess([], 0, "", "")

        for status, sound_id in (
            ("complete_with_errors", "dialog-warning"),
            ("complete_with_warnings", "dialog-warning"),
            ("incomplete", "dialog-warning"),
            ("failed", "dialog-error"),
            ("interrupted", "dialog-error"),
        ):
            with self.subTest(status=status):
                run_command.reset_mock()
                notify_run_completion(
                    pipeline="MM-WHS",
                    split="test",
                    resolution="high",
                    run_dir=None,
                    status=status,
                )
                self.assertEqual(run_command.call_count, 2)
                self.assertEqual(
                    run_command.call_args_list[1].args[0],
                    ["canberra-gtk-play", f"--id={sound_id}"],
                )

    @patch("utils.project.run_notification.subprocess.run")
    def test_unavailable_desktop_and_sound_do_not_raise(self, run_command):
        run_command.side_effect = FileNotFoundError("comando indisponível")

        with self.assertLogs("utils.project.run_notification", level="WARNING") as logs:
            notify_run_completion(
                pipeline="OrCaScore",
                split="all",
                resolution="mid",
                run_dir=None,
                status="failed",
            )

        self.assertEqual(run_command.call_count, 2)
        self.assertEqual(len(logs.output), 2)

    @patch("utils.project.run_notification.subprocess.run")
    def test_nonzero_exit_and_timeout_are_only_warnings(self, run_command):
        run_command.side_effect = [
            subprocess.CompletedProcess([], 1, "", "sem sessão gráfica"),
            subprocess.TimeoutExpired(["canberra-gtk-play"], 5),
        ]

        with self.assertLogs("utils.project.run_notification", level="WARNING"):
            notify_run_completion(
                pipeline="ImageCAS",
                split="val",
                resolution="mid",
                run_dir=Path("/runs/example"),
                status="complete",
            )

        self.assertEqual(run_command.call_count, 2)

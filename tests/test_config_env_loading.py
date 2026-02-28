from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ConfigEnvLoadingTests(unittest.TestCase):
    def _run_import(self, bot_env_file: str) -> subprocess.CompletedProcess[str]:
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["BOT_ENV_FILE"] = bot_env_file
        return subprocess.run(
            [sys.executable, "-c", "import config; print('ok')"],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
        )

    def test_missing_bot_env_file_fails_fast(self) -> None:
        missing_path = "data/matrix/user_presets/__definitely_missing_env_for_test__.env"
        result = self._run_import(missing_path)
        self.assertNotEqual(result.returncode, 0)
        details = (result.stdout + "\n" + result.stderr).lower()
        self.assertIn("bot_env_file", details)
        self.assertIn("does not exist", details)

    def test_existing_bot_env_file_is_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "bot.env"
            env_path.write_text("UNITTEST_BOT_ENV_FLAG=loaded\n", encoding="utf-8")
            root = Path(__file__).resolve().parents[1]
            env = os.environ.copy()
            env["BOT_ENV_FILE"] = str(env_path)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os, config; print(os.getenv('UNITTEST_BOT_ENV_FLAG', ''))",
                ],
                cwd=str(root),
                env=env,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "loaded")

    def test_plan_diversity_keys_are_loaded_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "bot.env"
            env_path.write_text(
                "\n".join(
                    [
                        "PLAN_MAX_SINGLE_SOURCE_SHARE=0.42",
                        "PLAN_MAX_WATCHLIST_SHARE=0.18",
                        "PLAN_MIN_NON_WATCHLIST_PER_BATCH=4",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            root = Path(__file__).resolve().parents[1]
            env = os.environ.copy()
            env["BOT_ENV_FILE"] = str(env_path)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import config; "
                        "print(f\"{config.PLAN_MAX_SINGLE_SOURCE_SHARE}|"
                        "{config.PLAN_MAX_WATCHLIST_SHARE}|"
                        "{config.PLAN_MIN_NON_WATCHLIST_PER_BATCH}\")"
                    ),
                ],
                cwd=str(root),
                env=env,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "0.42|0.18|4")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class RepoEncodingTests(unittest.TestCase):
    def test_repo_has_no_utf8_bom_prefixes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tool = root / "tools" / "check_utf8_bom.py"
        result = subprocess.run(
            [sys.executable, str(tool)],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            details = (result.stdout + "\n" + result.stderr).strip()
            self.fail(f"UTF-8 BOM audit failed.\n{details}")


if __name__ == "__main__":
    unittest.main()

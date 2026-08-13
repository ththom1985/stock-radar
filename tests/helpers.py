from __future__ import annotations

import shutil
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class ProjectTempMixin:
    def setUp(self):
        super().setUp()
        self.work = ROOT / "tests" / ".runtime" / str(uuid.uuid4())
        self.work.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)
        runtime = self.work.parent
        try:
            runtime.rmdir()
        except OSError:
            pass
        super().tearDown()

#!/usr/bin/env python3
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "pipelines" / "run_vlm_pipeline.py"), run_name="__main__")

"""
Evaluation metrics for Task 2.

Re-uses metrics from Task 1:
- CLAP similarity
- Melody accuracy
- Audiobox Aesthetics (CE, CU, PC, PQ)
"""

import sys
from pathlib import Path

# Add task1 to path (must be done before imports)
task1_path = Path(__file__).parent.parent.parent / "task1"
sys.path.insert(0, str(task1_path))

# ruff: noqa: E402
from task1.metrics.clap_metric import CLAPMetric
from task1.metrics.melody_metric import MelodyMetric
from task1.metrics.aesthetics_metric import AestheticsMetric

__all__ = ["CLAPMetric", "MelodyMetric", "AestheticsMetric"]

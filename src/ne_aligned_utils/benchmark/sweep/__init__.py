"""Benchmark sweep execution and orchestration.

Public names remain available from this package while the focused modules expose
the task, aggregation, and command-line boundaries directly.
"""

from . import core as _core
from .aggregation import *  # noqa: F403
from .config import *  # noqa: F403
from .core import *  # noqa: F403
from .registry import *  # noqa: F403
from .tasks import *  # noqa: F403

# A few tests and diagnostic scripts intentionally exercise these internal contracts.
_compute_task = _core._compute_task
_merge_previous_methods = _core._merge_previous_methods


def __getattr__(name: str):
    return getattr(_core, name)

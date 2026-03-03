"""LLM Assay — an evaluation harness for LLM quality, latency, and cost."""

from assay.config import Config, load_config
from assay.runner import RunResult, run_eval

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Config",
    "load_config",
    "RunResult",
    "run_eval",
]

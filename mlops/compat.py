"""
Shared MLflow compatibility shim.

Import ``mlflow`` from this module instead of directly so that all code
gracefully degrades when MLflow is not installed.
"""
from __future__ import annotations

try:
    import mlflow  # noqa: F401  (re-exported)
except Exception:

    class _NoopRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _NoopMlflow:
        @staticmethod
        def start_run(*args, **kwargs):
            return _NoopRun()

        @staticmethod
        def set_experiment(*args, **kwargs):
            pass

        @staticmethod
        def log_param(*args, **kwargs):
            pass

        @staticmethod
        def log_params(*args, **kwargs):
            pass

        @staticmethod
        def log_metric(*args, **kwargs):
            pass

        @staticmethod
        def log_metrics(*args, **kwargs):
            pass

        @staticmethod
        def set_tag(*args, **kwargs):
            pass

        @staticmethod
        def log_dict(*args, **kwargs):
            pass

        @staticmethod
        def log_artifact(*args, **kwargs):
            pass

        @staticmethod
        def active_run(*args, **kwargs):
            """Always returns None — no active run when MLflow is absent."""
            return None

        # pytorch sub-module stub
        pytorch = None

    mlflow = _NoopMlflow()  # type: ignore[assignment]

__all__ = ["mlflow"]

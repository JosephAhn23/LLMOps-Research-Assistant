"""
Celery-based batch ingestion processor.
"""
from __future__ import annotations

import os

from celery import Celery

from agents.orchestrator import run_pipeline

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("llmops_ingestion", broker=REDIS_URL, backend=REDIS_URL)


@celery_app.task(name="batch.answer_query")
def answer_query_task(query: str) -> dict:
    return run_pipeline(query)

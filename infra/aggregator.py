"""
Re-exports the canonical shard aggregator from ``api.shard_aggregator``.

Kept for backwards-compatible import paths;
new code should import from ``api.shard_aggregator`` directly.
"""
from api.shard_aggregator import app, SearchRequest, search, health  # noqa: F401

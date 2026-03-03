"""
Re-exports the canonical shard server from ``api.shard_server``.

Kept for backwards-compatible import paths;
new code should import from ``api.shard_server`` directly.
"""
from api.shard_server import app, SearchRequest, search, health  # noqa: F401

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)

import asyncio
import json

import pytest

from agentbench.search_cache import CacheMiss, RecordReplaySearch
from tests.helpers import CountingBackend


def test_record_then_replay(tmp_path):
    backend = CountingBackend()
    rec = RecordReplaySearch(tmp_path, "record", backend)
    first = asyncio.run(rec.search("JWST launch date"))
    again = asyncio.run(rec.search("  jwst   LAUNCH date "))  # normalized to the same key
    assert backend.calls == 1 and first == again

    replay = RecordReplaySearch(tmp_path, "replay")
    assert asyncio.run(replay.search("JWST launch date")) == first
    assert replay.stats["hits"] == 1


def test_replay_miss_raises(tmp_path):
    with pytest.raises(CacheMiss):
        asyncio.run(RecordReplaySearch(tmp_path, "replay").search("never recorded"))


def test_fuzzy_replay_matches_drifted_query(tmp_path):
    asyncio.run(RecordReplaySearch(tmp_path, "record", CountingBackend()).search("when did the JWST launch"))
    fuzzy = RecordReplaySearch(tmp_path, "replay", fuzzy_threshold=0.6)
    assert asyncio.run(fuzzy.search("when did JWST launch exactly"))
    assert fuzzy.stats["fuzzy_hits"] == 1


def test_concurrent_identical_queries_hit_backend_once(tmp_path):
    backend = CountingBackend()
    rs = RecordReplaySearch(tmp_path, "record", backend)

    async def go():
        return await asyncio.gather(*(rs.search("same query") for _ in range(10)))

    results = asyncio.run(go())
    assert backend.calls == 1 and all(r == results[0] for r in results)


def test_cache_files_are_complete_json(tmp_path):
    asyncio.run(RecordReplaySearch(tmp_path, "record", CountingBackend()).search("q"))
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["query"] == "q"
    assert not list(tmp_path.glob("*.tmp"))

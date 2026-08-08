"""read_trace_posts extracts the content-bearing posts a finished sim wrote to its trace DBs,
across both platforms and all rounds, skipping no-content actions (likes/follows/interviews).
See app/services/trace_posts.py and the /api/simulation/<id>/posts endpoint."""
import json
import os
import sqlite3

from app.services.trace_posts import read_trace_posts


def _write_trace(path, rows):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE trace (user_id INTEGER, action TEXT, info TEXT, created_at INTEGER)")
    conn.executemany(
        "INSERT INTO trace (user_id, action, info, created_at) VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


def test_reads_content_posts_from_both_platforms_and_skips_no_content(tmp_path):
    _write_trace(str(tmp_path / "twitter_simulation.db"), [
        (1, "create_post", json.dumps({"content": "NVDA up", "post_id": 10}), 5),
        (2, "like_post", json.dumps({"post_id": 10}), 6),            # no content -> skipped
    ])
    _write_trace(str(tmp_path / "reddit_simulation.db"), [
        (3, "create_comment", json.dumps({"content": "INTC weak", "comment_id": 7}), 7),
    ])

    posts = read_trace_posts(str(tmp_path))

    assert len(posts) == 2                                            # the like was dropped
    assert {p["platform"] for p in posts} == {"twitter", "reddit"}
    by_content = {p["content"]: p for p in posts}
    assert by_content["NVDA up"]["agent_id"] == 1
    assert by_content["NVDA up"]["created_at"] == 5                   # per-post timing preserved
    assert by_content["NVDA up"]["action_type"] == "create_post"


def test_missing_dir_and_missing_dbs_return_empty(tmp_path):
    assert read_trace_posts(str(tmp_path / "nope")) == []            # no dir
    assert read_trace_posts(str(tmp_path)) == []                     # dir exists, no DBs

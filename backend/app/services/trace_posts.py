"""Read the raw content-bearing posts a finished OASIS sim wrote to its ``trace`` DBs.

Each platform's simulation writes every agent action to a SQLite ``trace`` table
(``<platform>_simulation.db`` in the sim dir). This module extracts the **content-bearing**
actions (posts / comments / quotes — the ones carrying ``content``) across both platforms and all
rounds, with each post's ``created_at`` so a caller can weight by time. Likes, follows, refreshes
and interviews carry no ``content`` and are skipped.

Stdlib-only (``os`` / ``json`` / ``sqlite3``) on purpose: it must be importable and testable
without the Flask app or any heavy service dependency.
"""
import json
import os
import sqlite3

PLATFORMS = ("twitter", "reddit")


def _read_one_db(db_path, platform):
    """Content-bearing posts from one platform's trace DB; ``[]`` if it is missing/unreadable
    (best-effort — one platform's absence must never fail the whole read)."""
    if not os.path.exists(db_path):
        return []
    posts = []
    try:
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT user_id, action, info, created_at FROM trace ORDER BY rowid ASC")
            rows = cursor.fetchall()
        finally:
            conn.close()
    except Exception:
        return []
    for user_id, action, info_json, created_at in rows:
        try:
            args = json.loads(info_json) if info_json else {}
        except (TypeError, ValueError):
            args = {}
        content = args.get("content")
        if not content:
            continue  # only content-bearing posts/comments/quotes
        posts.append({
            "platform": platform,
            "agent_id": user_id,
            "action_type": action,
            "created_at": created_at,
            "content": content,
            "post_id": args.get("post_id"),
            "comment_id": args.get("comment_id"),
            "quoted_id": args.get("quoted_id"),
        })
    return posts


def read_trace_posts(sim_dir):
    """All content-bearing posts across both platforms for the sim at ``sim_dir``.

    Returns a list of post dicts (see ``_read_one_db``); an empty list if the dir or DBs are
    absent. Order is per-platform by ``rowid`` (write order); callers time-weight via ``created_at``.
    """
    if not sim_dir or not os.path.isdir(sim_dir):
        return []
    posts = []
    for platform in PLATFORMS:
        posts.extend(_read_one_db(
            os.path.join(sim_dir, "{}_simulation.db".format(platform)), platform))
    return posts

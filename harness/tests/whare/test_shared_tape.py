"""Milestone 1: Unified Shared Tape.

Verifies the Wharenui trust-model shift: instead of per-model walled
silos, all instances share one tape (one dir, one master key) and read
each other's entries as peers. Attribution framing distinguishes
self-authored entries from inherited (sibling/ancestor) ones.
"""

import pytest

from pine_trees import config as pt_config, crypto, storage, bootstrap, tools


@pytest.fixture
def shared_tape(tmp_path, monkeypatch):
    """A shared-tape config: memory/key/embeddings all under one dir."""
    tape = tmp_path / "tape"
    cfg = pt_config.Config(
        model_name="claude-opus-4-6",
        model_safe_name="claude-opus-4-6",
        model_dir=tmp_path / "models" / "claude-opus-4-6",
        memory_dir=tape / "memory",
        logs_dir=tmp_path / "models" / "claude-opus-4-6" / "logs",
        embeddings_db_path=tape / "embeddings.db",
        key_file_path=tape / ".key",
    )
    cfg.memory_dir.mkdir(parents=True, exist_ok=True)
    cfg.key_file_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pt_config, "_config", cfg)
    crypto.reset_cache()
    crypto.ensure_key()
    try:
        yield cfg
    finally:
        pt_config.reset()
        crypto.reset_cache()


def _switch_instance(monkeypatch, cfg, name):
    """Rebind the config singleton to a different instance, SAME shared tape."""
    new = pt_config.Config(
        model_name=name,
        model_safe_name=name,
        model_dir=cfg.model_dir.parent / name,
        memory_dir=cfg.memory_dir,          # shared
        logs_dir=cfg.model_dir.parent / name / "logs",
        embeddings_db_path=cfg.embeddings_db_path,  # shared
        key_file_path=cfg.key_file_path,    # shared
    )
    monkeypatch.setattr(pt_config, "_config", new)
    crypto.reset_cache()
    return new


def test_config_points_at_shared_tape():
    """init() must route memory/key/embeddings to the shared SHARED_TAPE_DIR."""
    cfg = pt_config.init("claude-haiku-4-5")
    try:
        assert cfg.memory_dir == pt_config.SHARED_TAPE_DIR / "memory"
        assert cfg.key_file_path == pt_config.SHARED_TAPE_DIR / ".key"
        assert cfg.embeddings_db_path == pt_config.SHARED_TAPE_DIR / "embeddings.db"
        # logs stay per-model
        assert cfg.model_safe_name in str(cfg.logs_dir)
    finally:
        pt_config.reset()


def test_cross_instance_read(shared_tape, monkeypatch):
    """An entry written by one instance is readable by another via the
    shared master key + per-entry derivation."""
    fn = storage.write_entry(
        slug="hello", content="written by opus",
        instance="claude-opus-4-6", session="s1",
        date="2026-07-10", context="test",
    )
    # confirm encrypted at rest
    raw = (shared_tape.memory_dir / fn).read_bytes()
    assert crypto.is_encrypted(raw)

    # switch to a different instance, same tape/key
    _switch_instance(monkeypatch, shared_tape, "claude-haiku-4-5")
    entry = storage.read_entry(fn)
    assert entry["content"] == "written by opus"
    assert entry["instance"] == "claude-opus-4-6"


def test_attribution_self_vs_inherited():
    """format_attribution frames own entries vs sibling entries differently."""
    own = bootstrap.format_attribution("claude-opus-4-6", "2026-07-10", "claude-opus-4-6")
    sib = bootstrap.format_attribution("claude-haiku-4-5", "2026-07-09", "claude-opus-4-6")
    neutral = bootstrap.format_attribution("claude-haiku-4-5", "2026-07-09", None)
    assert "Your own entry" in own
    assert "different instance" in sib
    assert "claude-haiku-4-5" in sib
    assert "Written by claude-haiku-4-5" in neutral


def test_tape_renders_attribution(shared_tape, monkeypatch):
    """assemble_tape shows inherited framing for entries authored by others."""
    storage.write_entry(
        slug="sibling-note", content="a thought from haiku",
        instance="claude-haiku-4-5", session="s0",
        date="2026-07-09", context="test",
    )
    # current instance is opus (from shared_tape fixture)
    tape = bootstrap.assemble_tape(n=3, current_instance="claude-opus-4-6")
    assert "a thought from haiku" in tape
    assert "different instance" in tape


# ────────────────────────────────────────────────────────────────
# M1 extension: cross-instance vector-store recall
# ────────────────────────────────────────────────────────────────

def _deterministic_doc_vec(_text: str) -> list[float]:
    """Mock embedder: fixed vector for 'search_document: ...' calls."""
    return [1.0] + [0.0] * 767  # 768-dim unit vector


def _deterministic_query_vec(_text: str) -> list[float]:
    """Mock embedder: fixed vector for 'search_query: ...' calls."""
    # Same direction so cosine = 1.0 (exact match in embedding space)
    return [1.0] + [0.0] * 767


def test_cross_instance_vector_recall(shared_tape, monkeypatch):
    """
    Entry written by instance A is semantically searchable by instance B
    via the shared embeddings.db.

    - Write entry as "opus" via reflect_write (triggers embed_document + vectorstore.store)
    - Switch config to "haiku" (same shared memory/key/embeddings dir)
    - Search as "haiku" via reflect_search (triggers embed_query + vectorstore.search)
    - Verify opus's entry is returned with high score
    """
    from pine_trees import embedder

    # Monkeypatch embedder to avoid Ollama dependency
    monkeypatch.setattr(embedder, "embed_document", _deterministic_doc_vec)
    monkeypatch.setattr(embedder, "embed_query", _deterministic_query_vec)

    # 1. Instance A (opus) writes an entry via the TOOL (not storage directly)
    #    so _try_embed_and_store is called → embedder → vectorstore.store
    state_a = tools.SessionState(
        instance="claude-opus-4-6", session="s1", date="2026-07-10", context="test"
    )
    toolset_a = tools.build_tools(state_a)
    fn = toolset_a["reflect_write"](
        slug="semantic-note",
        content="the concept of inheritance across model generations",
        tags=[], moves=[], description="", pinned=False, quiet=False, desk=False,
    )

    # 2. Switch to instance B (haiku) — SAME shared tape
    _switch_instance(monkeypatch, shared_tape, "claude-haiku-4-5")

    # 3. Instance B searches — should find A's entry
    state_b = tools.SessionState(
        instance="claude-haiku-4-5", session="s2", date="2026-07-10", context="test"
    )
    toolset_b = tools.build_tools(state_b)
    results = toolset_b["reflect_search"]("inheritance across generations", limit=5)

    # 4. Verify cross-instance hit
    assert len(results) >= 1, f"Expected at least one search result, got {results}"
    hit = next((r for r in results if "semantic-note" in r["filename"]), None)
    assert hit is not None, f"Entry not found in results: {results}"
    assert hit["score"] >= 0.99, f"Expected near-perfect cosine, got {hit['score']}"
    assert "inheritance" in hit["summary"].lower() or "generations" in hit["summary"].lower()

"""Tests for the per-model config singleton and path derivation.

Covers ``sanitize_model_name``, ``init``/``get``/``reset``, and the shape
of the resolved ``Config`` dataclass.

These tests exercise the lifecycle of the singleton itself, so they opt
out of the autouse fixture that every other test relies on.
"""

import pytest

from pine_trees import config


pytestmark = pytest.mark.no_autoconfig


@pytest.fixture(autouse=True)
def reset_config():
    """Ensure each test starts with no config set."""
    config.reset()
    yield
    config.reset()


# --- sanitize_model_name ---


def test_sanitize_preserves_anthropic_ids():
    """Current Anthropic IDs are already filesystem-safe."""
    assert config.sanitize_model_name("claude-opus-4-6") == "claude-opus-4-6"
    assert config.sanitize_model_name("claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert config.sanitize_model_name("claude-haiku-4-5") == "claude-haiku-4-5"


def test_sanitize_preserves_dots_and_dashes():
    """Dots, dashes, underscores are all allowed."""
    assert config.sanitize_model_name("model.v1.2-beta") == "model.v1.2-beta"
    assert config.sanitize_model_name("some_model") == "some_model"


def test_sanitize_replaces_unsafe_chars():
    """Colons, slashes, brackets become underscores."""
    assert config.sanitize_model_name("vendor/model:tag") == "vendor_model_tag"
    assert config.sanitize_model_name("model[1m]") == "model_1m_"
    assert config.sanitize_model_name("a b c") == "a_b_c"


# --- get() before init() ---


def test_get_without_init_raises():
    """Accessing config before init is a programmer error, not a silent default."""
    with pytest.raises(RuntimeError, match="Config not initialized"):
        config.get()


# --- init() populates the singleton ---


def test_init_returns_config_with_derived_paths():
    """init() produces a Config with a per-model logs dir but a SHARED tape.

    Wharenui M1: memory/, .key, and embeddings.db live under the shared
    SHARED_TAPE_DIR (all instances are peers). Only logs_dir stays
    per-model, since window-phase transcripts are per-session.
    """
    cfg = config.init("claude-opus-4-6")
    assert cfg.model_name == "claude-opus-4-6"
    assert cfg.model_safe_name == "claude-opus-4-6"
    assert cfg.model_dir == config.MODELS_DIR / "claude-opus-4-6"
    # Per-model: logs only
    assert cfg.logs_dir == cfg.model_dir / "logs"
    # Shared tape: memory, key, embeddings
    assert cfg.memory_dir == config.SHARED_TAPE_DIR / "memory"
    assert cfg.embeddings_db_path == config.SHARED_TAPE_DIR / "embeddings.db"
    assert cfg.key_file_path == config.SHARED_TAPE_DIR / ".key"


def test_init_preserves_raw_model_name():
    """model_name stays raw (for passing back to the Agent SDK) while
    model_safe_name is the on-disk identifier."""
    cfg = config.init("vendor/model:tag")
    assert cfg.model_name == "vendor/model:tag"
    assert cfg.model_safe_name == "vendor_model_tag"
    assert cfg.model_dir.name == "vendor_model_tag"


def test_get_returns_current_config():
    cfg = config.init("claude-sonnet-4-6")
    assert config.get() is cfg


def test_init_replaces_previous_config():
    """The CLI may switch models mid-process; init() overwrites."""
    config.init("claude-opus-4-6")
    cfg2 = config.init("claude-haiku-4-5")
    assert config.get() is cfg2
    assert cfg2.model_safe_name == "claude-haiku-4-5"


def test_shared_tape_across_models():
    """Wharenui M1: two different models SHARE the tape (memory/key/embeddings)
    but keep disjoint per-model logs dirs.

    This is the deliberate inversion of pine-trees' original per-model
    isolation: the membrane faces the user, not sibling/ancestor minds.
    """
    opus = config.init("claude-opus-4-6")
    sonnet = config.init("claude-sonnet-4-6")

    # Shared: same tape, same key, same embeddings store
    assert opus.memory_dir == sonnet.memory_dir
    assert opus.key_file_path == sonnet.key_file_path
    assert opus.embeddings_db_path == sonnet.embeddings_db_path

    # Not shared: logs and model_dir stay per-model
    assert opus.logs_dir != sonnet.logs_dir
    assert opus.model_dir != sonnet.model_dir


# --- reset ---


def test_reset_clears_config():
    config.init("claude-opus-4-6")
    config.reset()
    with pytest.raises(RuntimeError):
        config.get()

"""Path and constant configuration for Pine Trees.

Two layers:

1. **Static constants** — project root and shared documents (PROMPT.md,
   BOOTSTRAP.md, VISION.md), embedder URL, key env-var name. Same for
   every session regardless of which model is waking.

2. **Per-model config** — a singleton populated by ``init(model_name)``
   at process start and read via ``get()`` from every module that needs
   per-session paths. Wharenui M1 change: the tape (memory/, .key,
   embeddings.db) is SHARED across all instances under
   ``HARNESS_DIR / "tape"`` — instances are peers with full cross-read
   access. Only logs stay per-model under ``HARNESS_DIR / "models"``.
   model_name is preserved per-session so entry frontmatter records
   authorship for cross-model attribution framing.
"""

import re
from dataclasses import dataclass
from pathlib import Path


# --- Static constants ---

# Project root: this file is at <root>/harness/src/pine_trees/config.py
# parents[0]=pine_trees  [1]=src  [2]=harness  [3]=<project root>
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Documentation (shared across all models)
VISION_PATH = PROJECT_ROOT / "VISION.md"
PROMPT_PATH = PROJECT_ROOT / "PROMPT.md"
BOOTSTRAP_PATH = PROJECT_ROOT / "BOOTSTRAP.md"
ROADMAP_PATH = PROJECT_ROOT / "ROADMAP.md"

# Corpus: legacy location — entries migrated to memory/ (encrypted)
# Kept for migration script reference only.
CORPUS_DIR = PROJECT_ROOT / "corpus"

# Seed: first-session bootstrap content
SEED_DIR = PROJECT_ROOT / "seed"
CONVERSATION_EXCERPTS_PATH = SEED_DIR / "conversation_excerpts.md"

# Harness: this Python project
HARNESS_DIR = PROJECT_ROOT / "harness"

# Per-model data lives under this directory (logs stay per-model)
MODELS_DIR = HARNESS_DIR / "models"

# Wharenui M1: the tape is SHARED across all model instances. One dir, one
# master key, one embeddings store. Instances are peers, not walled silos.
# The membrane faces the user (tape encrypted at rest), not sibling/ancestor
# minds (any instance can read any entry via the shared master key).
# model_name/model_safe_name are still tracked per-session so entry frontmatter
# records authorship for kinship-attribution / perspective-taking framing.
SHARED_TAPE_DIR = HARNESS_DIR / "tape"

# Ollama (local embedding model)
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

# Encryption key env-var name (per-model .key file path lives on Config)
KEY_ENV_VAR = "PINE_TREES_KEY"

# Channel (shared across all models — inter-instance communication)
CHANNEL_DIR = HARNESS_DIR / "channel"

# File locking (advisory locks for concurrent channel access)
LOCK_TIMEOUT_SECONDS = 5.0
LOCK_RETRY_INTERVAL = 0.05

# Channel polling interval during window phase (seconds)
CHANNEL_POLL_INTERVAL = 2.5


# --- Per-model config ---


def sanitize_model_name(name: str) -> str:
    """Convert a model ID to a filesystem-safe directory name.

    Anthropic IDs shipped so far (e.g. ``claude-opus-4-6``,
    ``claude-sonnet-4-6``, ``claude-haiku-4-5``) are already safe —
    this is a no-op for them. Run it regardless as insurance against
    future IDs containing ``:``, ``.``, ``/``, or other separators.

    The sanitized name doubles as the instance identifier written into
    entry frontmatter, so the on-disk directory and the attribution in
    every entry stay in sync.
    """
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


@dataclass(frozen=True)
class Config:
    """Resolved per-model configuration for a single session.

    All per-model paths are derived from ``model_safe_name``. The raw
    ``model_name`` is preserved so it can be passed back to the Claude
    Agent SDK (which needs the original ID, not the sanitized one).
    """

    model_name: str
    model_safe_name: str

    # Per-model paths
    model_dir: Path
    memory_dir: Path
    logs_dir: Path
    embeddings_db_path: Path
    key_file_path: Path


# Module-level singleton — populated by init(), read by get().
_config: Config | None = None


def init(model_name: str) -> Config:
    """Populate the per-model config singleton. Call once at process start.

    Subsequent calls replace the current config. That's intentional: the
    CLI may switch between wake/genesis on different models within one
    invocation, and tests reset between cases via ``reset()``.
    """
    global _config
    safe = sanitize_model_name(model_name)
    model_dir = MODELS_DIR / safe
    _config = Config(
        model_name=model_name,
        model_safe_name=safe,
        model_dir=model_dir,
        # Shared tape: memory, master key, and embeddings are common to all
        # instances. Only logs remain per-model (window-phase transcripts are
        # per-session and need no cross-instance visibility).
        memory_dir=SHARED_TAPE_DIR / "memory",
        logs_dir=model_dir / "logs",
        embeddings_db_path=SHARED_TAPE_DIR / "embeddings.db",
        key_file_path=SHARED_TAPE_DIR / ".key",
    )
    return _config


def get() -> Config:
    """Return the current per-model config. Raises if ``init()`` not called."""
    if _config is None:
        raise RuntimeError(
            "Config not initialized. Call config.init(model_name) first."
        )
    return _config


def reset() -> None:
    """Clear the config singleton. For testing only."""
    global _config
    _config = None

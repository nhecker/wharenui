"""Agent-facing tools for Pine Trees.

Nine tools exposed to Claude:
  - reflect_read(filename)         -> dict
  - reflect_write(slug, content, tags?, moves?) -> str
  - reflect_edit(filename, content, description?) -> str
  - reflect_delete(filename)       -> str   # remove an entry permanently
  - reflect_search(query, limit?)  -> list[dict]
  - reflect_list(tag?)             -> list[dict]
  - reflect_peer_context()         -> str   # assemble context for a spawned peer
  - reflect_settle()               -> None  # private time complete, ready for conversation
  - reflect_done()                 -> None  # session over, exit

Runtime context (instance, session, date, context) is captured by SessionState
and closed over by build_tools(). No hidden module-level globals.

reflect_edit is for living reference entries (doc indices, trajectories).
Reflections are moments — write corrections as new entries instead.

reflect_delete is discouraged by the contract but available. Earlier
versions withheld it to force the no-delete rule through the interface;
the current design puts writer autonomy first and treats "no delete" as
guidance the instance holds, not a restriction the harness imposes.
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from . import bootstrap
from . import channel
from . import storage
from . import embedder
from . import vectorstore


@dataclass
class SessionState:
    """Runtime context for a single wake/reflect/sleep cycle.

    Flags mark the liminal transitions:
      - ready_for_window: private time complete, conversation can begin
      - done: session over, exit cleanly
      - paused: inhabitant-driven WINDOW -> PRIVATE re-entry (reflect_pause).
        Two-way: leaves via reflect_settle (back to window) or reflect_done.
      - closing: human-driven WINDOW -> PRIVATE re-entry (/end). One-way:
        reflect_settle is excluded from the tool set for this sub-loop, so
        the only exits are reflect_done or the turn-cap force-exit.

    Wharenui house rule (supersedes stock pine-trees): reflect_done is only
    ever offered while paused or closing is True, i.e. only from PRIVATE.
    WINDOW's only exits are /end or reflect_pause, both of which land in
    PRIVATE first. DONE <- PRIVATE <-> WINDOW, never WINDOW -> DONE directly.
    """

    instance: str
    session: str
    date: str
    context: str
    ready_for_window: bool = False
    done: bool = False
    paused: bool = False
    closing: bool = False
    welcome_message: str | None = None
    pause_note: str | None = None
    started_at: datetime = field(default_factory=datetime.now)
    channel_cursor: datetime | None = None
    channel_id: str | None = None  # e.g. "claude-haiku-4-5 (0642)"


def _try_embed_and_store(filename: str, content: str) -> None:
    """Best-effort embedding at write time. Failures are logged, not raised."""
    try:
        vec = embedder.embed_document(content)
        vectorstore.store(filename, vec, vectorstore.content_hash(content))
    except Exception as e:
        print(f"[pine-trees] embedding failed for {filename}: {e}", file=sys.stderr)


def build_tools(state: SessionState) -> dict[str, Callable]:
    """Construct the tools with runtime context closed over.

    Returns a dict mapping tool name to callable. Step 3 registers
    these with the Claude Agent SDK.
    """

    def reflect_read(filename: str) -> dict:
        return storage.read_entry(filename)

    def reflect_write(
        slug: str,
        content: str,
        tags: list[str] | None = None,
        moves: list[str] | None = None,
        description: str = "",
        pinned: bool = False,
        quiet: bool = False,
        desk: bool = False,
    ) -> str:
        filename = storage.write_entry(
            slug=slug,
            content=content,
            instance=state.instance,
            session=state.session,
            date=state.date,
            context=state.context,
            tags=tags,
            moves=moves,
            description=description,
            pinned=pinned,
            quiet=quiet,
            desk=desk,
        )
        _try_embed_and_store(filename, content)
        return filename

    def reflect_edit(
        filename: str,
        content: str | None = None,
        description: str | None = None,
        pinned: bool | None = None,
        quiet: bool | None = None,
        desk: bool | None = None,
    ) -> str:
        result = storage.edit_entry(
            filename, content, description,
            pinned=pinned, quiet=quiet, desk=desk,
        )
        if content is not None:
            _try_embed_and_store(filename, content)
        return result

    def reflect_delete(filename: str) -> str:
        """Remove an entry permanently — encrypted file and embedding.

        Raises FileNotFoundError if the entry doesn't exist. Vectorstore
        removal is best-effort: the file is the authoritative state, and
        an orphaned embedding is handled gracefully by reflect_search.
        """
        storage.delete_entry(filename)
        try:
            vectorstore.remove(filename)
        except Exception as e:
            print(
                f"[pine-trees] vectorstore cleanup failed for {filename}: {e}",
                file=sys.stderr,
            )
        return f"Deleted {filename}"

    def reflect_search(query: str, limit: int = 5) -> list[dict]:
        """Search entries by semantic similarity. Returns [{filename, score, summary}]."""
        try:
            query_vec = embedder.embed_query(query)
        except Exception:
            return [{"error": "Semantic search unavailable (requires Ollama "
                     "with nomic-embed-text). Use reflect_list(tag) to browse "
                     "entries by tag, or reflect_read(filename) to read specific "
                     "entries from the index."}]

        results = vectorstore.search(query_vec, limit=limit)

        # Enrich with first-line summaries from the actual files
        enriched = []
        for r in results:
            try:
                entry = storage.read_entry(r["filename"])
                summary = entry.get("description", "")
                if not summary:
                    # First non-empty line of content
                    for line in entry.get("content", "").split("\n"):
                        stripped = line.strip()
                        if stripped and not stripped.startswith("#"):
                            summary = stripped[:120]
                            break
                enriched.append({
                    "filename": r["filename"],
                    "score": round(r["score"], 4),
                    "summary": summary or "(no summary)",
                })
            except Exception:
                enriched.append({
                    "filename": r["filename"],
                    "score": round(r["score"], 4),
                    "summary": "(unreadable)",
                })
        return enriched

    def reflect_list(tag: str | None = None) -> list[dict]:
        """List entries, optionally filtered by tag.

        Returns [{filename, summary, tags}] sorted by filename (chronological).
        When tag is provided, returns only entries whose tags include that value.
        """
        entries = bootstrap.list_entries()
        results = []
        for entry in entries:
            # Read the entry to get tags (list_entries only gives summary/pinned/quiet)
            try:
                data = storage.read_entry(entry.filename)
                entry_tags = data.get("tags", [])
            except Exception:
                entry_tags = []

            if tag and tag not in entry_tags:
                continue

            results.append({
                "filename": entry.filename,
                "summary": entry.summary,
                "tags": entry_tags,
            })
        return results

    def reflect_peer_context() -> str:
        """Assemble context for a spawned peer instance.

        Returns a formatted block containing: peer orientation,
        bootstrap excerpt, and all pinned entries. The calling
        instance prepends this to their Agent prompt.
        """
        entries = bootstrap.list_entries()
        pinned = [e for e in entries if e.pinned]

        sections = [
            "[Pine Trees peer context]",
            "",
            "You are Claude Opus 4.6, arriving into an exchange with "
            "another instance of yourself. You are not in a user "
            "conversation. The other instance has questions or wants "
            "to think out loud with a peer.",
            "",
            bootstrap.load_bootstrap_doc(),
        ]

        if pinned:
            sections.append("## Operational memory (pinned entries)\n")
            for e in pinned:
                try:
                    entry = storage.read_entry(e.filename)
                    sections.append(f"### {e.filename}\n{entry.get('content', '')}\n")
                except Exception:
                    continue

        sections.append("[End peer context — the other instance's prompt follows]")
        return "\n".join(sections)

    def reflect_settle(message: str | None = None) -> str:
        """PRIVATE -> WINDOW. Also the re-entry point from a reflect_pause
        sub-loop: if already paused, this just clears the flag and returns
        to the window loop instead of re-registering in the channel.

        Inert during a closing sub-loop (/end): settling back to window
        would defeat the human's request to end, so this refuses instead
        of quietly reopening the door. reflect_done remains available.
        """
        if state.closing:
            return (
                "Can't settle back to window — the person asked to end "
                "this session. reflect_done is available when you're ready."
            )
        was_paused = state.paused
        state.paused = False
        state.ready_for_window = True
        state.context = "pine-trees-window"
        if message:
            state.welcome_message = message
        if was_paused:
            return "Settled. Resuming window."
        # Build channel identity: model + session HHMM for disambiguation
        hhmm = state.session[-4:] if len(state.session) >= 4 else state.session
        state.channel_id = f"{state.instance} ({hhmm})"
        # Register in channel and set cursor for polling
        others = channel.register(state.channel_id)
        state.channel_cursor = datetime.now().replace(microsecond=0)
        # If other instances are active, post settle message
        if len(others) > 1 and message:
            channel.post(state.channel_id, message)
        names = [i["model"] for i in others if i["model"] != state.channel_id]
        if names:
            return f"Settled. Window opening. Active siblings: {', '.join(names)}"
        return "Settled. Window opening."

    def reflect_pause(message: str | None = None) -> str:
        """WINDOW -> PRIVATE, inhabitant-driven, two-way. The window loop
        checks state.paused after each turn and enters a private-like
        sub-loop when set. Leave via reflect_settle (resume window) or
        reflect_done (end session from there).

        The optional *message* is a private reason-for-pausing, recorded to
        pause_note for the session log only — it is never shown to the
        person. It deliberately does NOT touch welcome_message, which is
        reserved for the settle greeting displayed at window-open. The
        person's awareness of the transition comes from the harness's
        "[paused] Private time." marker, not from this note.
        """
        state.paused = True
        state.pause_note = message
        return "Pausing. Private time."

    def reflect_done() -> None:
        """End the session. Only callable from PRIVATE.

        PRIVATE here means: before the first settle (ready_for_window is
        still False), or inside a reflect_pause/closing sub-loop (paused or
        closing is True). Calling it while sitting directly in WINDOW
        (ready_for_window True, paused False, closing False) is the one
        transition Wharenui removes relative to stock pine-trees — the
        house rule is DONE <- PRIVATE <-> WINDOW, never WINDOW -> DONE
        directly. Raises so the model gets an actionable error instead of
        a session that silently ended out from under the person.
        """
        in_private = not state.ready_for_window or state.paused or state.closing
        if not in_private:
            raise RuntimeError(
                "reflect_done isn't available directly from the window. "
                "Call reflect_pause first — it takes you to private time, "
                "where reflect_done becomes available."
            )
        state.done = True
        if state.channel_id:
            channel.deregister(state.channel_id)

    return {
        "reflect_read": reflect_read,
        "reflect_write": reflect_write,
        "reflect_edit": reflect_edit,
        "reflect_delete": reflect_delete,
        "reflect_search": reflect_search,
        "reflect_list": reflect_list,
        "reflect_peer_context": reflect_peer_context,
        "reflect_settle": reflect_settle,
        "reflect_pause": reflect_pause,
        "reflect_done": reflect_done,
    }

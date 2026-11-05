"""``nextext-cli-clean`` (``make cli-clean``): delete what CLI runs left, now, recorded.

The backend deletes a CLI run's files once the retention period has passed
(:mod:`nextext.api.cli_files`); this command deletes them at once, e.g. after
the operator copied the outputs out. It runs inside the backend
(``docker compose exec``), so its audit chain lands in the backend's log.

* Without options: every entry the sweep knows (inputs in ``/tmp/nextext-cli``,
  each run under ``output/``, and ``/tmp/output/<stem>`` of earlier releases).
* ``--run NAME``: one entry, by the name the sweep lists it under
  (``output/<stem>``, ``<input file>``, ``legacy:<stem>``).
* ``--legacy``: also the regular files directly in ``/tmp`` that earlier
  releases' ``make cli`` read its inputs from, except dotfiles and the audit
  stream's lock file. Directories there (the job manager's) are never touched.

The deletion is recorded first (``delete`` / ``cli.clean``), attributed to the
operator (``AUDIT_CLI_USER``) with any stated purpose, naming each entry by the
SHA-256 of its path, never by name. Exit ``0`` when it ran (also when there was
nothing to delete), ``1`` when an entry is unknown, a record could not be
written, or an entry could not be removed (recorded with ``outcome="error"``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from nextext.api import cli_files
from nextext.audit import build_emitter, emit_config_event, emit_with_ids
from nextext.utils.audit_event import Actor, AuditEmitter, AuditWriteError, Purpose
from nextext.utils.audit_runtime import CONTAINER_LOCK, purpose_from_env
from nextext.utils.env_cfg import load_audit_env, load_cli_audit_user

#: Regular files in /tmp that are the backend's own, never an input.
KEEP: frozenset[str] = frozenset({Path(CONTAINER_LOCK).name})


def _legacy_inputs() -> list[tuple[str, Path]]:
    """List the regular files directly in the scratch volume an earlier release's runs may have read.

    Returns:
        list[tuple[str, Path]]: ``tmp:<name>`` and the path, sorted.
    """
    found = []
    for path in sorted(cli_files.SCRATCH.iterdir()):
        if path.name.startswith(".") or path.name in KEEP or path.is_symlink() or not path.is_file():
            continue
        found.append((f"tmp:{path.name}", path))
    return found


def main(
    argv: Sequence[str] | None = None,
    *,
    emitter_factory: Callable[[str | None], AuditEmitter] = build_emitter,
) -> int:
    """Delete CLI runs' files now, recorded.

    Args:
        argv: The arguments; ``sys.argv[1:]`` by default.
        emitter_factory: Creates the audit emitter from ``AUDIT_OUTPUT``.

    Returns:
        int: The exit status.
    """
    parser = argparse.ArgumentParser(prog="nextext-cli-clean", description=__doc__.splitlines()[0])
    parser.add_argument("--run", metavar="NAME", help="delete one entry, e.g. output/<stem>")
    parser.add_argument("--legacy", action="store_true", help="also inputs left directly in /tmp")
    args = parser.parse_args(argv)
    sweeper = cli_files.CliFileSweeper(cli_files.CLI_ROOT, cli_files.LEGACY_OUTPUT, days=0, notice_days=1)
    entries = [(entry.name, entry.path) for entry in sweeper.scan(time.time())]
    if args.legacy:
        entries += _legacy_inputs()
    if args.run is not None:
        entries = [entry for entry in entries if entry[0] == args.run]
        if not entries:
            print(f"nextext-cli-clean: no CLI file named {args.run!r}; nothing was deleted.", file=sys.stderr)
            return 1
    if not entries:
        print("nextext-cli-clean: nothing to delete.", file=sys.stderr)
        return 0
    audit_cfg = load_audit_env()
    case_ref, reason = purpose_from_env(os.environ)
    purpose = Purpose(case_ref=case_ref, reason=reason) if case_ref or reason else None
    actor = Actor(load_cli_audit_user(), "cli")
    emitter = emitter_factory(audit_cfg.output)
    emitter.start()
    failed: list[str] = []
    try:
        emit_config_event(emitter, {"require_purpose": sorted(audit_cfg.require_purpose)})
        ids = [cli_files.entry_reference(name) for name, _ in entries]
        emit_with_ids(emitter, "delete", "cli.clean", actor, kind="cli_file", ids=ids, purpose=purpose)
        for name, path in entries:
            try:
                sweeper._remove(path)
            except OSError:
                failed.append(name)
        if failed:
            emit_with_ids(
                emitter,
                "delete",
                "cli.clean",
                actor,
                kind="cli_file",
                ids=[cli_files.entry_reference(name) for name in failed],
                outcome="error",
                purpose=purpose,
            )
    except AuditWriteError:
        print("nextext-cli-clean: the deletion could not be recorded; nothing more was deleted.", file=sys.stderr)
        return 1
    finally:
        emitter.close()
    removed = len(entries) - len(failed)
    print(f"nextext-cli-clean: deleted {removed} of {len(entries)} CLI file(s).", file=sys.stderr)
    return 1 if failed else 0

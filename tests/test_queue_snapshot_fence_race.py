"""An unreadable ``queue_snapshot.json`` must never kill a whole task.

Measured 2026-10-17 on this host: the supervisor rewrites
``data/state/queue_snapshot.json`` every tick through ``utils.atomic_write_json``
(unique temp sibling + ``os.replace``). Windows denies an open that meets
another thread's replace of the same file, so a reader landing in that window
gets ``PermissionError`` although BOTH versions are intact.

``usage_accounting._check_dispatch_fences`` runs twice per physical send, and it
converted ANY read failure into ``UsageAccountingError`` -- a class
``loop_llm_call`` deliberately re-raises with NO retry, because a monetary rail
is not a provider failure. So a ~30 ms file lock became an unretryable stop:
task ``7202acbcb58dd568`` ran 22 rounds / 28 calls / 3522 s, wrote ZERO files and
never answered its owner. Reproduction under the live writer: 37,314 reads
produced exactly 1 ``PermissionError`` and 0 ``JSONDecodeError``.

Two construction traps, both hit while this fix was banked:
1. ``ua.BudgetExceeded`` SUBCLASSES ``UsageAccountingError`` (MRO:
   BudgetExceeded -> UsageAccountingError -> RuntimeError). In any ad-hoc
   except-chain the subclass must be caught FIRST or the fence verdict is
   swallowed and mislabelled.
2. The transient arm patches ``pathlib.Path.read_text`` scoped to the snapshot
   path -- the ONE seam the old and the new implementation both cross. Patching
   the fix's own helper (``read_text_across_replace``) instead would be a FALSE
   GREEN: the old code never calls it. Content arms write the REAL file and
   patch nothing, so a green means the same thing under either implementation.
"""

import json
import pathlib

import pytest

from ouroboros import usage_accounting as ua
from ouroboros.usage_ledger import UsageAccountingError


def _snapshot_path(root):
    return root / "state" / "queue_snapshot.json"


def _write_snapshot(root, content):
    path = _snapshot_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content),
                    encoding="utf-8")
    return path


def _scope(task_id="task-1", root_task_id="root-1"):
    return ua.UsageScope(task_id=task_id, root_task_id=root_task_id)


@pytest.fixture(autouse=True)
def _no_process_local_fence(monkeypatch):
    # The process-local pause latch is a different owner (budget_pause.py);
    # isolate it so these tests measure the durable snapshot fence only.
    monkeypatch.setattr("ouroboros.budget_pause.dispatch_fenced", lambda task_id: False)


def test_missing_snapshot_is_no_fence_reference_semantics(tmp_path):
    # Pre-existing semantics, pinned as the REFERENCE the unreadable arm now
    # mirrors: no snapshot file means no fence authority at all.
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    ua._check_dispatch_fences(_scope(), tmp_path)  # must not raise


def test_transient_read_failure_is_survived_not_fatal(tmp_path, monkeypatch):
    # THE defect: a ~30 ms sharing violation became an unretryable monetary
    # stop. The fix retries that transient inside read_text_across_replace and,
    # still failing, treats the snapshot as no fence.
    snapshot = _write_snapshot(tmp_path, {"budget_root_fences": []})
    original = pathlib.Path.read_text
    calls = {"n": 0}
    exc = PermissionError(13, "simulated antivirus sharing violation")

    def _read(self, encoding=None, errors=None):
        if pathlib.Path(self) == pathlib.Path(snapshot):
            calls["n"] += 1
            raise exc
        return original(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(pathlib.Path, "read_text", _read)
    ua._check_dispatch_fences(_scope(), tmp_path)  # must NOT raise
    # The bounded retry lives in read_text_across_replace (10 attempts), not
    # here: at least one retry happened and the bound held.
    assert 2 <= calls["n"] <= 10, calls["n"]


def test_unreadable_snapshot_is_disclosed(tmp_path, monkeypatch, caplog):
    # Leniency must be LOUD: an unavailable money authority is disclosed, never
    # silently swallowed.
    snapshot = _write_snapshot(tmp_path, {"budget_root_fences": []})
    original = pathlib.Path.read_text

    def _read(self, encoding=None, errors=None):
        if pathlib.Path(self) == pathlib.Path(snapshot):
            raise PermissionError(13, "transient")
        return original(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(pathlib.Path, "read_text", _read)
    with caplog.at_level("WARNING"):
        ua._check_dispatch_fences(_scope(), tmp_path)
    assert "unreadable" in caplog.text.lower()


def test_malformed_json_is_not_fatal(tmp_path):
    # Real bytes on disk; the old code converted this into UsageAccountingError.
    # The fix treats unparseable exactly like missing: no authority, no fence.
    _write_snapshot(tmp_path, "{not json")
    ua._check_dispatch_fences(_scope(), tmp_path)  # must NOT raise


def test_structural_invalid_authority_stays_loud(tmp_path):
    # A snapshot that PARSES but is structurally wrong stays a hard error:
    # leniency must never silence a genuinely broken money authority.
    _write_snapshot(tmp_path, {"budget_root_fences": "not-a-list"})
    with pytest.raises(UsageAccountingError):
        ua._check_dispatch_fences(_scope(), tmp_path)


def test_structural_invalid_fence_row_stays_loud(tmp_path):
    _write_snapshot(tmp_path, {"budget_root_fences": [1, 2]})
    with pytest.raises(UsageAccountingError):
        ua._check_dispatch_fences(_scope(), tmp_path)


def test_active_root_fence_still_blocks_dispatch(tmp_path):
    # The control this module exists for must survive the leniency.
    _write_snapshot(tmp_path, {"budget_root_fences": [
        {"root_task_id": "root-1", "cause": "budget", "status": "active", "fence_id": "f1"},
    ]})
    with pytest.raises(ua.BudgetExceeded):
        ua._check_dispatch_fences(_scope(), tmp_path)


def test_owner_pause_row_never_blocks_a_money_send(tmp_path):
    # An owner Pause gates launches itself (owner_pause.py): never a money stop.
    _write_snapshot(tmp_path, {"budget_root_fences": [
        {"root_task_id": "root-1", "cause": "owner_pause", "status": "active", "fence_id": "f1"},
    ]})
    ua._check_dispatch_fences(_scope(), tmp_path)


def test_fence_for_another_root_does_not_stop_this_one(tmp_path):
    _write_snapshot(tmp_path, {"budget_root_fences": [
        {"root_task_id": "root-OTHER", "cause": "budget", "status": "active", "fence_id": "f9"},
    ]})
    ua._check_dispatch_fences(_scope(), tmp_path)


def test_member_selected_against_the_fence_is_admitted(tmp_path):
    # Owner Q9 (#1196): one member explicitly selected against THIS fence
    # generation is admitted; every unselected member stays fenced.
    _write_snapshot(tmp_path, {
        "budget_root_fences": [
            {"root_task_id": "root-1", "cause": "budget", "status": "active", "fence_id": "f1"},
        ],
        "pending": [
            {"task": {"id": "task-1", "_budget_pause_hold": {"selected": True, "fence_id": "f1"}}},
        ],
    })
    ua._check_dispatch_fences(_scope(), tmp_path)


def test_stale_selection_against_another_generation_stays_fenced(tmp_path):
    # The same selection bound to a DIFFERENT fence generation must not admit.
    _write_snapshot(tmp_path, {
        "budget_root_fences": [
            {"root_task_id": "root-1", "cause": "budget", "status": "active", "fence_id": "f2"},
        ],
        "pending": [
            {"task": {"id": "task-1", "_budget_pause_hold": {"selected": True, "fence_id": "f1"}}},
        ],
    })
    with pytest.raises(ua.BudgetExceeded):
        ua._check_dispatch_fences(_scope(), tmp_path)


@pytest.mark.parametrize("payload", ["null", "[1, 2]", '"text"', "42"])
def test_parseable_non_object_snapshot_stays_loud(tmp_path, payload):
    # Valid JSON that is not an object carries no fence authority and must NOT
    # be read leniently: leniency is for an UNREADABLE file, never for a
    # readable one whose shape we cannot interpret.
    _write_snapshot(tmp_path, payload)
    with pytest.raises(UsageAccountingError):
        ua._check_dispatch_fences(_scope(), tmp_path)
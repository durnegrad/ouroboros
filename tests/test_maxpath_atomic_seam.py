"""MAX_PATH seam: the shared atomic temp sibling must never exceed the host limit.

With long paths disabled system-wide an ordinary path fails at a total of 260
characters (measured on this host: 259 creates, 260 does not), and the classic
Windows error names neither the cause nor the length. The atomic temp sibling is
~40 characters longer than the name it protects, so a target that fits can still
be unwritable — that is the seam these tests pin.

The refusal half of the contract — an ordinary path with no extended-length
form — is what this file pins, deterministically via ``_no_extended``; the
degradation half (an extended-capable host keeps the full handle and issues
the over-limit I/O through that form) is pinned in
``tests/test_path_ceiling_degradation.py``.

Determinism: the seam reads one cached ceiling (``utils.path_length_ceiling``),
so these tests drive that value directly instead of depending on the running
host's limit. One Windows-only case additionally proves the real limit by
construction.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
import types

import pytest

from ouroboros import utils
from ouroboros.utils import sweep_stale_temp_files, write_bytes_atomic


def _pin_ceiling(monkeypatch, ceiling: int) -> None:
    """Drive the seam against an exact ceiling, cross-platform and deterministic."""
    monkeypatch.setattr(utils, "_path_ceiling_cache", ceiling)


def _wide_ids(monkeypatch) -> None:
    """Impersonate the widest pid/tid digit widths without touching the real modules."""
    monkeypatch.setattr(utils, "os", types.SimpleNamespace(getpid=lambda: 999999999))
    monkeypatch.setattr(
        utils, "threading", types.SimpleNamespace(get_ident=lambda: 99999999999999999999)
    )


def _no_extended(monkeypatch) -> None:
    """Drive the composers as a host with NO extended-length form.

    The honest refusal when nothing fits is the POSIX / long-paths-disabled
    Windows behaviour; pinning it keeps these cases deterministic instead of
    depending on the running host's facility. BOTH import seams are pinned:
    ``utils`` owns the guard and the temp composer, while ``artifacts``
    re-imports ``long_path_io_available`` for ``choose_source_name`` and the
    ``path_fits`` pre-check, so patching one module alone leaves the other
    half of the seam live.
    """
    monkeypatch.setattr(utils, "long_path_io_available", lambda: False)
    from ouroboros import artifacts as artifacts_module
    monkeypatch.setattr(artifacts_module, "long_path_io_available", lambda: False)


def _padded_dir(root: pathlib.Path, total_len: int) -> pathlib.Path:
    """A real directory whose string length is exactly ``total_len``."""
    filler = total_len - len(str(root)) - 1
    assert filler > 0, f"{total_len} is too short for {root}"
    path = root / ("d" * filler)
    path.mkdir(parents=True)
    return path


def _cleanup_dir(path: pathlib.Path) -> None:
    """Remove a padded test directory and whatever landed inside it."""
    import shutil
    shutil.rmtree(path, ignore_errors=True)


def test_short_names_keep_the_classic_temp_signature(monkeypatch):
    """A target that fits keeps the historic name, byte for byte.

    Two existing readers match ``.{name}.tmp.`` and the stale-temp sweep keys on
    the signature after the final ``.tmp.`` — clamping must be invisible here.
    """
    _pin_ceiling(monkeypatch, 260)
    name = utils._atomic_temp_name("state.json", parent_chars=10, ceiling=260)
    assert name.startswith(".state.json.tmp.")
    assert name.split(".tmp.", 1)[1].count(".") == 2  # pid.tid.uuid
    assert len(name.rsplit(".", 1)[1]) == 8  # 8-hex uuid


def test_temp_sibling_fits_when_the_target_fits(monkeypatch):
    """The TEMP sibling — not only the final name — must stay under the ceiling.

    Claim 1. The numerics are deliberately outside the composed name's control:
    the seam must clamp the stem rather than emit a path the OS will reject.
    """
    _pin_ceiling(monkeypatch, 240)
    _no_extended(monkeypatch)
    _wide_ids(monkeypatch)
    for parent_chars, name_len in ((100, 60), (140, 60), (170, 50), (180, 40), (190, 5)):
        name = utils._atomic_temp_name("x" * name_len, parent_chars=parent_chars, ceiling=240)
        temp_total = parent_chars + 1 + len(name)
        assert temp_total < 240, f"temp path {temp_total} crosses the 240 ceiling"
        assert name.startswith(".")
        assert ".tmp." in name
        # The reaper keys on the segment after the final ".tmp." — hex/dots only.
        tail = name.rsplit(".tmp.", 1)[1]
        assert tail and set(tail) <= set("0123456789abcdef.")


def test_temp_composer_refuses_with_measured_numbers_when_nothing_fits(monkeypatch):
    """When even the narrowest grammar cannot fit, the refusal names the facts."""
    _pin_ceiling(monkeypatch, 240)
    _no_extended(monkeypatch)
    _wide_ids(monkeypatch)
    with pytest.raises(OSError) as excinfo:
        utils._atomic_temp_name("x" * 20, parent_chars=195, ceiling=240)
    message = str(excinfo.value)
    assert "240" in message
    assert "195" in message


def test_a_target_that_cannot_fit_is_refused_with_measured_numbers(monkeypatch):
    """The guard replaces an opaque OS error with the length and the ceiling."""
    _pin_ceiling(monkeypatch, 120)
    _no_extended(monkeypatch)
    root = pathlib.Path(".").resolve() / "pytest_maxpath_tmp"
    parent = _padded_dir(root, 110)
    try:
        target = parent / ("t" * 20)  # 110 + 1 + 20 = 131 >= 120
        with pytest.raises(OSError) as excinfo:
            write_bytes_atomic(target, b"payload")
        message = str(excinfo.value)
        assert str(len(str(target))) in message
        assert "120" in message
        assert not target.exists()
    finally:
        _cleanup_dir(parent)
    _cleanup_dir(root)


def test_atomic_round_trip_just_under_the_atomic_bound(monkeypatch):
    """A real overwrite lands near the ceiling — construction, not arithmetic.

    The atomic contract needs BOTH the target and its temp sibling to fit, so the
    writable bound sits about one tail-length below the OS ceiling; the write is
    performed and read back rather than argued about.
    """
    root = pathlib.Path(".").resolve() / "pytest_maxpath_tmp"
    parent = _padded_dir(root, 165)
    try:
        target = parent / ("t" * 40)  # 165 + 1 + 40 = 206
        ceiling = 251  # within this host's real limit; leaves the temp sibling room
        _pin_ceiling(monkeypatch, ceiling)
        _no_extended(monkeypatch)
        write_bytes_atomic(target, b"near-limit payload")
        assert target.read_bytes() == b"near-limit payload"
        assert not list(parent.glob(".*.tmp.*")), "temp sibling left behind"

        # A target that still fits gets a CLAMPED temp sibling and the write lands.
        near = parent / ("t" * 84)  # 165 + 1 + 84 = 250 < 251
        assert len(str(near)) < ceiling
        write_bytes_atomic(near, b"clamped temp")
        assert near.read_bytes() == b"clamped temp"
        assert not list(parent.glob(".*.tmp.*")), "temp sibling left behind"

        # A target that itself cannot fit is refused by the guard, naming both
        # the measured length and the ceiling — never an opaque OS error.
        over = parent / ("t" * 85)  # 165 + 1 + 85 = 251 >= ceiling
        assert len(str(over)) >= ceiling
        with pytest.raises(OSError) as excinfo:
            write_bytes_atomic(over, b"nope")
        assert str(len(str(over))) in str(excinfo.value)
        assert str(ceiling) in str(excinfo.value)
        assert not over.exists()
    finally:
        _cleanup_dir(parent)
    _cleanup_dir(root)


def test_sweep_still_reclaims_the_clamped_temp_names(monkeypatch, tmp_path):
    """Claim 3: the stale-temp sweep must still reap a clamped temp.

    A hard kill orphans a temp whose name was shortened to fit, so the trailing
    signature after the final ``.tmp.`` has to stay hexadecimal/dots even when
    the uuid is shortened and pid/tid are wide.
    """
    _pin_ceiling(monkeypatch, 160)
    _wide_ids(monkeypatch)
    name = utils._atomic_temp_name("w" * 90, parent_chars=len(str(tmp_path)), ceiling=160)
    temp = tmp_path / name
    temp.write_text("orphan", encoding="utf-8")
    keep = tmp_path / ".config.tmp.backup"
    keep.write_text("keep", encoding="utf-8")
    aged = time.time() - 7200
    os.utime(temp, (aged, aged))
    os.utime(keep, (aged, aged))

    removed = sweep_stale_temp_files(tmp_path, min_age_sec=3600)
    assert removed == 1
    assert not temp.exists(), "clamped temp name was not reclaimed"
    assert keep.exists(), "a non-atomic dotfile must never be deleted"


def test_sweep_still_reclaims_the_classic_temp_names(tmp_path):
    """The sweep keeps working for names the seam did not have to shorten."""
    temp = tmp_path / ".state.json.tmp.12345.140234567890176.abcd1234"
    temp.write_text("orphan", encoding="utf-8")
    aged = time.time() - 7200
    os.utime(temp, (aged, aged))
    assert sweep_stale_temp_files(tmp_path, min_age_sec=3600) == 1
    assert not temp.exists()


def test_source_handle_name_fits_and_is_its_own_fixed_point(monkeypatch, tmp_path):
    """A long source id is shortened to fit and survives a re-store unchanged.

    ``source_retention._source`` reads ``source_id`` back OUT of the stored
    filename and stores it again, so a shortened stem that did not sanitize to
    itself would compose a second path on the next retention pass.
    """
    from ouroboros.artifacts import store_actor_source_bytes, task_artifact_dir_path

    artifact_dir = task_artifact_dir_path(tmp_path, "task-1", create=True)
    source_id = "plan-review-wave-0001-" + "a" * 64
    tail = len(f"-{'b' * 64}.json")
    ceiling = len(str(artifact_dir)) + len("source_handles/context_checkpoints") + 2 + 20 + tail + 1
    assert ceiling < 260, "fixture must stay within this host's real limit"
    _pin_ceiling(monkeypatch, ceiling)
    _no_extended(monkeypatch)

    ref = store_actor_source_bytes(
        tmp_path, "task-1", category="context_checkpoints",
        source_id=source_id, data=b'{"wave": 1}', extension="json",
    )
    stored = artifact_dir / ref["path"]
    assert len(str(stored)) < ceiling
    assert stored.read_bytes() == b'{"wave": 1}'
    # Both 64-hex identities stay intact: the content digest is in the name.
    assert ref["path"].endswith(f"-{ref['sha256']}.json")
    assert len(ref["sha256"]) == 64

    derived = pathlib.PurePosixPath(ref["path"]).name.rsplit(f"-{ref['sha256']}.", 1)[0]
    again = store_actor_source_bytes(
        tmp_path, "task-1", category="context_checkpoints",
        source_id=derived, data=b'{"wave": 1}', extension="json",
    )
    assert again["path"] == ref["path"], "shortened name is not a fixed point"


def test_the_wave_fingerprint_keeps_its_identity_under_clamping(monkeypatch, tmp_path):
    """The plan-review wave fingerprint survives a clamped name.

    The fingerprint is the requirement's identity; only the readable
    ``plan-review-wave-NNNN`` prefix may yield.
    """
    from ouroboros.artifacts import store_actor_source_bytes, task_artifact_dir_path

    artifact_dir = task_artifact_dir_path(tmp_path, "task-2", create=True)
    fingerprint = "f" * 64
    tail = len(f"-{'0' * 64}.json")
    ceiling = len(str(artifact_dir)) + len("source_handles/context_checkpoints") + 2 + 20 + tail + 1
    assert ceiling < 260, "fixture must stay within this host's real limit"
    _pin_ceiling(monkeypatch, ceiling)
    _no_extended(monkeypatch)

    ref = store_actor_source_bytes(
        tmp_path, "task-2", category="context_checkpoints",
        source_id=f"plan-review-wave-0007-{fingerprint}", data=b"{}", extension="json",
    )
    name = pathlib.PurePosixPath(ref["path"]).name
    assert len(str(artifact_dir / ref["path"])) < ceiling
    # The wave reader matches on this prefix, so the digest suffix plus a nonempty
    # stem must still satisfy it; the digest itself is never truncated.
    assert name.rsplit(f"-{ref['sha256']}.", 1)[0]
    assert ref["path"].endswith(f"-{ref['sha256']}.json")


def test_a_wave_handle_refuses_rather_than_cut_its_reader_prefix(monkeypatch, tmp_path):
    """A length fit must never cut the prefix the plan-review reader classifies on.

    ``review_source_closure`` decides a stored handle is a plan-review wave by
    ``source_id.startswith("plan-review-wave-")``, so a fit that removed that
    prefix would silently reclassify the source. At a ceiling that cannot hold it,
    the composer refuses instead.
    """
    from ouroboros.artifacts import store_actor_source_bytes, task_artifact_dir_path

    source_id = "plan-review-wave-0002-" + "c" * 64
    artifact_dir = task_artifact_dir_path(tmp_path, "task-3", create=True)
    tail = len(f"-{'0' * 64}.json")
    # Room for only part of the prefix (12 characters of "plan-review-wave-").
    ceiling = (len(str(artifact_dir)) + len("source_handles/context_checkpoints")
               + 2 + 12 + tail + 1)
    assert ceiling < 260, "fixture must stay within this host's real limit"
    _pin_ceiling(monkeypatch, ceiling)
    _no_extended(monkeypatch)
    with pytest.raises(OSError) as excinfo:
        store_actor_source_bytes(tmp_path, "task-3", category="context_checkpoints",
                                 source_id=source_id, data=b"{}", extension="json")
    assert "plan-review-wave-" in str(excinfo.value)


@pytest.mark.skipif(not sys.platform.startswith("win"),
                    reason="the classic MAX_PATH boundary is a Windows behaviour")
def test_real_measured_limit_round_trip_on_this_host(monkeypatch, tmp_path):
    """Prove the ceiling by construction: a real write beside the measured limit.

    The seam measures this host instead of assuming 260, so this test uses the
    measured number and performs a genuine atomic overwrite under it.
    """
    ceiling = utils.path_length_ceiling()
    assert ceiling >= 100
    _no_extended(monkeypatch)

    depth = tmp_path
    while len(str(depth)) < ceiling - 60:
        remaining = ceiling - 60 - len(str(depth)) - 1
        chunk = min(80, remaining)
        if chunk <= 0:
            break
        depth = depth / ("p" * chunk)
    depth.mkdir(parents=True, exist_ok=True)

    # The atomic contract needs room for the target AND its ~40-char temp sibling.
    name_len = ceiling - 55 - len(str(depth)) - 1
    assert 1 <= name_len <= 250, f"unexpected name budget {name_len}"
    target = depth / ("t" * name_len)
    assert len(str(target)) < ceiling

    write_bytes_atomic(target, b"at the measured limit")
    assert target.read_bytes() == b"at the measured limit"
    assert not list(depth.glob(".*.tmp.*"))

    # Past the atomic bound the seam refuses with the measured numbers.
    over = depth / ("t" * (name_len + 60))
    with pytest.raises(OSError) as excinfo:
        write_bytes_atomic(over, b"nope")
    assert str(ceiling) in str(excinfo.value)
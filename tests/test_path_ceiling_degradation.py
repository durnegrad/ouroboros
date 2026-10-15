"""Deep-root degradation: the v7.6.8 path guards degrade instead of refusing.

Measured on this class of host: an ordinary Windows path past its classic
ceiling fails with a bare FileNotFoundError, while the SAME logical path opens
through the extended-length form. The composers therefore keep the ordinary
spelling as identity and issue the I/O of an over-long path through that form,
instead of losing the handle (the ``review_source_closure_unavailable`` class)
or refusing. POSIX has no extended form: its honest refusal stays and is pinned
in ``test_maxpath_atomic_seam.py``.
"""
from __future__ import annotations

import pathlib
import types

import pytest

from ouroboros import artifacts as artifacts_module
from ouroboros import utils
from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes, task_artifact_dir_path


def _deep_dir(root: pathlib.Path, target_len: int) -> pathlib.Path:
    """A real directory whose string length is at least ``target_len``.

    Each level is created through the extended form: past the classic ceiling a
    plain ``mkdir(parents=True)`` fails mid-chain, which is the failure class
    under test. On POSIX the extended form is the identity, so the same helper
    builds deep directories natively.
    """
    path = pathlib.Path(root)
    while len(str(path)) < target_len:
        step = max(1, min(60, target_len - len(str(path)) - 1))
        path = path / ("d" * step)
        utils.extended_io_path(path).mkdir(exist_ok=True)
    return path


def test_deep_root_round_trip_keeps_the_full_handle(tmp_path):
    deep = _deep_dir(tmp_path, 210)
    source_id = "plan-review-wave-0001-" + "a" * 64
    data = b'{"wave": 1}'

    ref = store_actor_source_bytes(
        deep, "task-deep", category="context_checkpoints",
        source_id=source_id, data=data, extension="json",
    )
    stored = task_artifact_dir_path(deep, "task-deep") / ref["path"]
    assert len(str(stored)) >= 260, "fixture must actually cross the classic bound"
    # The plan-review reader classifies on this prefix; the digest is identity.
    assert pathlib.PurePosixPath(ref["path"]).name.startswith("plan-review-wave-")
    assert ref["path"].endswith(f"-{ref['sha256']}.json")

    assert read_actor_source_bytes(deep, "task-deep", ref) == data

    # Fixed point: retention reads the source id back out of the stored name and
    # stores it again; a name that re-composed differently would split.
    derived = pathlib.PurePosixPath(ref["path"]).name.rsplit(f"-{ref['sha256']}.", 1)[0]
    again = store_actor_source_bytes(
        deep, "task-deep", category="context_checkpoints",
        source_id=derived, data=data, extension="json",
    )
    assert again["path"] == ref["path"], "degraded name is not a fixed point"
    assert not list(utils.extended_io_path(stored.parent).glob(".*.tmp.*")), "temp sibling left behind"


def test_choose_source_name_degrades_to_a_component_fit(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts_module, "long_path_io_available", lambda: True)
    monkeypatch.setattr(utils, "_path_ceiling_cache", 260)
    deep = _deep_dir(tmp_path, 170)

    # A deep root leaves no budget at all: the whole-path ceiling no longer
    # bounds the stem, so the full id survives (the write reaches it extended).
    full_id = "plan-review-wave-0003-" + "b" * 64
    name = artifacts_module.choose_source_name(
        full_id, "0" * 64, "json", deep, primary_category="context_checkpoints")
    assert name == full_id

    # A stem past the per-component bound still fits, keeping the prefix the
    # plan-review reader classifies on — and it is its own fixed point.
    long_id = "plan-review-wave-0003-" + "b" * 400
    fitted = artifacts_module.choose_source_name(
        long_id, "0" * 64, "json", deep, primary_category="context_checkpoints")
    assert fitted.startswith("plan-review-wave-")
    assert len(fitted) + len(f"-{'0' * 64}.json") <= artifacts_module._MAX_SOURCE_NAME_CHARS
    again = artifacts_module.choose_source_name(
        fitted, "0" * 64, "json", deep, primary_category="context_checkpoints")
    assert again == fitted, "component fit is not a fixed point"


def test_choose_source_name_still_refuses_without_extended_form(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts_module, "long_path_io_available", lambda: False)
    monkeypatch.setattr(utils, "_path_ceiling_cache", 260)
    deep = _deep_dir(tmp_path, 170)
    with pytest.raises(OSError):
        artifacts_module.choose_source_name(
            "plan-review-wave-0004-" + "c" * 64, "0" * 64, "json", deep,
            primary_category="context_checkpoints")


def test_atomic_temp_name_degrades_instead_of_refusing(monkeypatch):
    monkeypatch.setattr(utils, "long_path_io_available", lambda: True)
    monkeypatch.setattr(utils, "_path_ceiling_cache", 240)
    monkeypatch.setattr(utils, "os", types.SimpleNamespace(getpid=lambda: 999999999))
    monkeypatch.setattr(
        utils, "threading", types.SimpleNamespace(get_ident=lambda: 99999999999999999999))
    name = utils._atomic_temp_name("x" * 20, parent_chars=195, ceiling=240)
    assert name.startswith(".x.tmp.")
    # The reaper keys on the segment after the final ".tmp." — hex/dots only.
    tail = name.rsplit(".tmp.", 1)[1]
    assert tail and set(tail) <= set("0123456789abcdef.")


def test_preflight_env_pins_a_short_test_temp_root(tmp_path):
    from ouroboros.preflight_runner import _preflight_env

    env = _preflight_env(tmp_path, tmp_path / "repo", create=False)
    assert env["OUROBOROS_TEST_TEMP_ROOT"] == str(tmp_path)

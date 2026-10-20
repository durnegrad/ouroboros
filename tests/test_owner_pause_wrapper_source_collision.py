"""A tool whose own argument is named ``source`` must reach the wrapper intact.

The registry dispatches every tool as ``run_tool_handler(ctx, entry.handler,
ctx, **handler_args)`` (``ouroboros/tools/registry_core.py``). The three
Pause-admission wrappers in ``ouroboros/owner_pause.py`` therefore receive the
tool's keyword arguments AND their own first parameter ``source``; without a
positional-only marker an argument named ``source`` binds twice and the call
dies with ``TypeError: got multiple values for argument 'source'``.

That broke two registered tools on the live build (2026-10-20): the pre-commit
``preflight_review`` gate and ``promote_chat_to_task``. The gate matters most:
it is the review the constitution requires before every commit, and it failed
with a host ``TypeError`` before any review ran.

The marker costs no lines and changes no behaviour: the wrapper still receives
its own ``source`` positionally and the handler still receives its argument.
"""
from __future__ import annotations

import inspect

import pytest

from ouroboros import owner_pause


# The exact call shape ``registry_core`` uses: the wrapper's own source, the
# handler, its positional context, then the tool's keyword arguments.
_WRAPPERS = ("run_tool_handler", "run_operation", "submit_async_operation")


@pytest.mark.parametrize("name", _WRAPPERS)
def test_wrapper_binds_a_tool_argument_named_source(name):
    wrapper = getattr(owner_pause, name)
    signature = inspect.signature(wrapper)
    # Must not raise: ``source`` positionally belongs to the wrapper, the
    # keyword belongs to the tool.
    signature.bind(object(), object(), "positional", source="worktree")


@pytest.mark.parametrize("name", _WRAPPERS)
def test_wrapper_still_requires_its_own_source_positionally(name):
    """The marker must not make the wrapper's own source optional or keyword-only."""
    wrapper = getattr(owner_pause, name)
    params = list(inspect.signature(wrapper).parameters.values())
    assert params[0].name == "source", f"{name}: first parameter must stay 'source'"
    assert params[0].kind is inspect.Parameter.POSITIONAL_ONLY, (
        f"{name}: 'source' must be positional-only, or a tool argument with that "
        "name collides with it"
    )
    assert params[1].name == "function", f"{name}: second parameter must stay 'function'"


def test_preparation_stays_sole_caller_of_its_own_source():
    """``submit_async_preparation`` declares ``source`` AFTER *args on purpose."""
    params = inspect.signature(owner_pause.submit_async_preparation).parameters
    assert params["source"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["source"].default is None


def test_run_tool_handler_delivers_a_source_argument_to_the_handler():
    """End to end through the real wrapper, not only through its signature."""
    seen = {}

    def handler(ctx, *, source=""):
        seen["ctx"] = ctx
        seen["source"] = source
        return "ok"

    class Ctx:
        pass

    ctx = Ctx()
    result = owner_pause.run_tool_handler(ctx, handler, ctx, source="worktree")
    assert result == "ok"
    assert seen["ctx"] is ctx
    assert seen["source"] == "worktree"

"""The two-tier money rule: only PROVABLE spend may refuse a dispatch (v7.6.12).

Owner decision 2026-10-15: an amount the provider never stood behind — a figure
this install derived from a published tariff catalog — must WARN, never halt a
task. The decision only holds if such an amount actually lands in the
`estimated_usd` bucket, because the hard admission basis sums exactly
`confirmed + reserved + unresolved`.

That is one fact with two ends, and they are asserted separately here:

* the ADAPTER computes `cost_final` from `cost_estimated` (`llm_openai_compatible.py`),
  so the route says "this is an estimate";
* the EXTRACTOR must carry that same distinction, because a settled row reaches
  `estimated_usd` only when `cost_final` is falsy. Deriving `final` from the mere
  presence of a cost silently promoted every estimate into `confirmed_usd`, where
  it kept the power to refuse — i.e. it made the owner's rule a no-op.

Without the second assertion the feature is untestable from the outside: the
parameter would exist, the docstring would promise the behaviour, and estimates
would still stop tasks.
"""

from __future__ import annotations

import pytest

from ouroboros import _usage_money as money
from ouroboros._usage_response import usage_from_response


class _Response:
    """The minimum a provider response needs to be for the extractor."""

    def __init__(self, usage: dict) -> None:
        self.usage = usage


def _extract(usage: dict):
    return usage_from_response(_Response(usage))


@pytest.mark.parametrize("estimated,expected_final", [(True, False), (False, True)])
def test_extractor_derives_final_from_cost_estimated_not_from_presence(estimated, expected_final):
    """Both arms: an estimate is NOT final, a provider-reported cost IS."""
    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "cost": 0.00045}
    if estimated:
        usage.update(cost_estimated=True, cost_basis="catalog_estimate")
    normalized, cost, final = _extract(usage)
    assert cost == pytest.approx(0.00045)
    assert final is expected_final, (
        "an estimate must not be reported as a settled charge; a provider-reported "
        "cost must stay final"
    )
    assert bool(normalized.get("cost_estimated")) is estimated


def test_extractor_keeps_a_costless_response_unknown_in_both_arms():
    """Cost unknown stays unknown: no cost, no finality, and never a fabricated $0."""
    for usage in ({"prompt_tokens": 10, "completion_tokens": 2},
                  {"prompt_tokens": 10, "completion_tokens": 2, "cost_estimated": True}):
        _normalized, cost, final = _extract(usage)
        assert cost is None and final is False


def test_the_two_buckets_a_settled_row_can_land_in_are_distinct():
    """The bucket split the hard basis depends on, read off the real reducer."""
    settled = {"kind": "attempt", "state": "settled", "cost_usd": "1.0"}
    final_cash = money.cash_contribution({**settled, "cost_final": True})
    estimated_cash = money.cash_contribution({**settled, "cost_final": False})
    buckets = {key: index for index, key in enumerate(money.CASH_KEYS)}
    assert final_cash[buckets["confirmed_usd"]] == 1
    assert final_cash[buckets["estimated_usd"]] == 0
    assert estimated_cash[buckets["confirmed_usd"]] == 0
    assert estimated_cash[buckets["estimated_usd"]] == 1


@pytest.mark.parametrize("basis,may_refuse", [("hard", False), ("full", True)])
def test_only_the_hard_basis_excludes_an_estimate_from_refusal(basis, may_refuse):
    """The owner's rule in one expression: an estimate refuses only under `full`."""
    total = money.cash_contribution({"kind": "attempt", "state": "settled",
                                     "cost_usd": "1.0", "cost_final": False})
    assert money.exceeds_limit(total, 0.5, basis=basis) is may_refuse


def test_a_provider_confirmed_amount_refuses_under_both_bases():
    """The control: provable money keeps its power to stop a task."""
    total = money.cash_contribution({"kind": "attempt", "state": "settled",
                                     "cost_usd": "1.0", "cost_final": True})
    for basis in ("hard", "full"):
        assert money.exceeds_limit(total, 0.5, basis=basis) is True
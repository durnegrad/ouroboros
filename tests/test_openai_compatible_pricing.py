"""The openai-compatible gate's published tariff catalog (p1, v7.6.12).

A new-api reseller gate reports NO cost in its chat responses, so an estimate from
its published catalog is the only price this route can ever have — and it must
stay visibly an ALTERNATIVE (an estimate), never a settlement. These tests pin the
traps that would silently turn the estimate back into `unknown` (which is the bug
the feature exists to fix) or into a fabricated `$0`:

* the 37.5 default coefficient and free/per-image rows stay ABSENT;
* the cache tiers equal the input tier, because `_cost_from_pricing` returns
  unknown for any cache-hit call whose cached rate is `None` and this install is
  cache-heavy;
* a dashed configured id finds its dotted catalog row through an alias that can
  never shadow a real tariff;
* a provider-reported cost still WINS over the catalog.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ouroboros.llm_pricing import fetch_openai_compatible_pricing
from ouroboros.pricing import estimate_cost_optional, get_pricing

_BASE_URL = "https://gate.example.test/v1"


def _stub_gate(*, quota_per_unit=500000, rows=(), pricing_body=None):
    """Serve /api/status and /api/pricing for one patched ``requests.get``."""

    def _get(url, **_kwargs):
        response = MagicMock()
        response.raise_for_status.return_value = None
        if url.endswith("/api/status"):
            response.json.return_value = {"data": {"quota_per_unit": quota_per_unit}}
        else:
            response.json.return_value = (
                pricing_body if pricing_body is not None else {"data": list(rows)}
            )
        return response

    return _get


def _row(name, ratio, completion_ratio, *, quota_type=0, **extra):
    payload = {"model_name": name, "model_ratio": ratio,
               "completion_ratio": completion_ratio, "quota_type": quota_type}
    payload.update(extra)
    return payload


@pytest.fixture(autouse=True)
def _reset_pricing_cache():
    import ouroboros.pricing as pricing

    pricing._cached_pricing.clear()
    pricing._pricing_fetched_at.clear()
    pricing._pricing_retry_after.clear()
    pricing._pricing_fetch_in_progress.clear()
    yield
    pricing._cached_pricing.clear()
    pricing._pricing_fetched_at.clear()
    pricing._pricing_retry_after.clear()
    pricing._pricing_fetch_in_progress.clear()


def test_base_price_is_derived_from_the_gate_constant_not_hardcoded():
    # quota_per_unit 500000 => BASE_PRICE 2.0 USD. A different constant must move
    # the tariff, proving the number is READ rather than assumed.
    rows = [_row("vendor/alpha", 1.0, 3.0)]
    with patch("requests.get", side_effect=_stub_gate(quota_per_unit=250000, rows=rows)):
        catalog = fetch_openai_compatible_pricing(base_url=_BASE_URL)
    assert catalog["openai-compatible/vendor/alpha"] == (4.0, 4.0, 4.0, 12.0)


def test_cache_tiers_equal_the_input_tier_so_a_cache_hit_is_still_priced(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://gate.example.test/v1")
    rows = [_row("vendor/alpha", 0.5, 2.0)]
    with patch("requests.get", side_effect=_stub_gate(rows=rows)):
        catalog = fetch_openai_compatible_pricing(base_url=_BASE_URL)
        # The real consumer path: get_pricing -> _fetch_live_rows(provider) -> this catalog.
        cost = estimate_cost_optional(
            "openai-compatible/vendor/alpha", 1_000, 0,
            cache_usage={"cached_tokens": 1_000}, provider="openai-compatible",
        )
    input_price, cached, written, output = catalog["openai-compatible/vendor/alpha"]
    assert (input_price, cached, written, output) == (1.0, 1.0, 1.0, 2.0)
    # The trap this guards: a cache-hit call must not fall to unknown. If the
    # cached tier were None, `_cost_from_pricing` returns None for this call.
    assert cost == pytest.approx(0.001)


def test_default_sentinel_free_and_per_image_rows_stay_absent():
    rows = [
        _row("vendor/real", 2.0, 1.0),
        _row("vendor/sentinel", 37.5, 1.0),      # the panel's default coefficient
        _row("vendor/free", 0, 1.0),             # a free row is not a zero tariff
        _row("vendor/image", 1.0, 1.0, quota_type=1),   # fixed per-image price
        _row("vendor/no-completion", 1.0, 0),
    ]
    with patch("requests.get", side_effect=_stub_gate(rows=rows)):
        catalog = fetch_openai_compatible_pricing(base_url=_BASE_URL)
    assert set(catalog) == {"openai-compatible/vendor/real"}
    # And a sentinel-priced model reads unknown, never $0.
    assert estimate_cost_optional(
        "openai-compatible/vendor/sentinel", 1_000, 100,
        provider="openai-compatible", allow_live_fetch=False,
    ) is None


def test_missing_or_unusable_status_returns_no_catalog():
    with patch("requests.get", side_effect=_stub_gate(quota_per_unit=0, rows=[_row("a/b", 1.0, 1.0)])):
        assert fetch_openai_compatible_pricing(base_url=_BASE_URL) == {}
    assert fetch_openai_compatible_pricing(base_url="") == {}
    with patch("requests.get", side_effect=_stub_gate(pricing_body={"data": "not-a-list"})):
        assert fetch_openai_compatible_pricing(base_url=_BASE_URL) == {}


def test_a_transport_failure_is_unknown_not_an_exception():
    import requests

    with patch("requests.get", side_effect=requests.RequestException("gate down")):
        assert fetch_openai_compatible_pricing(base_url=_BASE_URL) == {}


def test_dashed_configured_id_reaches_its_dotted_catalog_row(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://gate.example.test/v1")
    rows = [_row("deepseek-v4.1-flash", 0.15, 4.0)]
    with patch("requests.get", side_effect=_stub_gate(rows=rows)):
        catalog = fetch_openai_compatible_pricing(base_url=_BASE_URL)
        dashed_cost = estimate_cost_optional(
            "openai-compatible/deepseek-v4-1-flash", 1_000, 0,
            provider="openai-compatible",
        )
    assert catalog["openai-compatible/deepseek-v4.1-flash"] == catalog["openai-compatible/deepseek-v4-1-flash"]
    # A configured dashed route is served the SAME tariff as its dotted row.
    assert dashed_cost == pytest.approx(catalog["openai-compatible/deepseek-v4.1-flash"][0] / 1_000)


def test_an_alias_never_shadows_a_real_dashed_row():
    # A gate that genuinely publishes BOTH spellings must keep their own tariffs:
    # the alias is added only where the dashed spelling is not already a row.
    rows = [_row("vendor-v1.2-flash", 1.0, 1.0), _row("vendor-v1-2-flash", 7.0, 1.0)]
    with patch("requests.get", side_effect=_stub_gate(rows=rows)):
        catalog = fetch_openai_compatible_pricing(base_url=_BASE_URL)
    assert catalog["openai-compatible/vendor-v1.2-flash"][0] == 2.0
    assert catalog["openai-compatible/vendor-v1-2-flash"][0] == 14.0


def test_the_provider_route_reads_the_catalog_through_get_pricing(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://gate.example.test/v1")
    rows = [_row("deepseek-v4.1-flash", 0.15, 4.0)]
    with patch("requests.get", side_effect=_stub_gate(rows=rows)):
        catalog = get_pricing(provider="openai-compatible")
    assert "openai-compatible/deepseek-v4.1-flash" in catalog


def test_a_configured_route_without_a_base_url_has_no_catalog(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)
    with patch("requests.get") as request:
        assert get_pricing(provider="openai-compatible") == {}
    request.assert_not_called()


def test_the_catalog_does_not_leak_across_providers(monkeypatch):
    # The gate's tariff is keyed under its own provider prefix; the SAME model id
    # routed through another provider must not borrow it.
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://gate.example.test/v1")
    rows = [_row("vendor/alpha", 1.0, 1.0)]
    with patch("requests.get", side_effect=_stub_gate(rows=rows)):
        assert estimate_cost_optional(
            "openai-compatible/vendor/alpha", 1_000, 0, provider="openai-compatible",
        ) is not None
        with patch("ouroboros.llm.fetch_openrouter_pricing", return_value={}) as or_fetch:
            assert estimate_cost_optional(
                "openai-compatible/vendor/alpha", 1_000, 0, provider="openrouter",
            ) is None
    or_fetch.assert_called_once_with(timeout_sec=5.0)
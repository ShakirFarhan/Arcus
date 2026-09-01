import json
import time
from pathlib import Path

from platformdirs import user_cache_dir

from arcus.adapters.arc_adapter import ArcAdapter, ArcModel

# model catalogs don't change often enough to justify a network call on
# every single invocation, a few hours of staleness is a fine tradeoff
# for not paying that latency on every `arcus "..."` run.
_CACHE_TTL_SECONDS = 6 * 60 * 60


def _cache_path() -> Path:
    cache_dir = Path(user_cache_dir("arcus"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "model_catalog.json"


def _read_cache() -> set[str] | None:
    path = _cache_path()
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    if time.time() - data.get("checked_at", 0) > _CACHE_TTL_SECONDS:
        return None

    return set(data.get("models", []))


def _write_cache(model_ids: set[str]) -> None:
    path = _cache_path()
    try:
        path.write_text(json.dumps({"checked_at": time.time(), "models": sorted(model_ids)}))
    except OSError:
        # not being able to write the cache just means the next call
        # checks again, not worth surfacing to the user over it
        pass


def _live_model_ids(adapter: ArcAdapter) -> set[str] | None:
    """The live catalog, from cache if it's fresh enough, otherwise a
    real check against ARC. Returns None (rather than an empty set) when
    the catalog couldn't be determined at all, so callers can tell "ARC
    has nothing live" apart from "couldn't find out right now" and fall
    back to trusting their own hardcoded list in the latter case.
    """
    live_ids = _read_cache()
    if live_ids is not None:
        return live_ids

    try:
        live_ids = set(adapter.list_models())
    except Exception:
        return None

    _write_cache(live_ids)
    return live_ids


def known_arms(adapter: ArcAdapter) -> list[str]:
    """Cross-checks the models this build knows how to route to against
    what ARC is actually serving right now, and drops anything that's
    been renamed or retired instead of leaving it in the bandit's arm
    list to fail on every attempt. ARC runs its own model catalog and
    can change it without notice, so the hardcoded ArcModel list is a
    snapshot, not something the router should ever fully trust on its
    own.

    Falls back to the full hardcoded list whenever the live catalog
    can't be checked (offline, ARC's own endpoint is down, an empty or
    unrecognizable response) rather than leave the bandit with nothing
    to route to over what's likely a transient problem.
    """
    return filter_to_live(adapter, [m.value for m in ArcModel])


def filter_to_live(adapter: ArcAdapter, candidates: list[str]) -> list[str]:
    """Same live cross-check as known_arms, generalized to any candidate
    list, not just the core ArcModel arms. Used for the special-purpose
    model variants (web search's legacy-tool-calling models, for
    instance) that live outside the normal routing table but are just
    as capable of being renamed or retired out from under us.
    """
    live_ids = _live_model_ids(adapter)
    if live_ids is None:
        return candidates

    known = [candidate for candidate in candidates if candidate in live_ids]
    return known or candidates

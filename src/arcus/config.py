import os
import tomllib
from pathlib import Path
from typing import Literal

from platformdirs import user_config_dir
from pydantic_settings import BaseSettings

BanditAlgorithm = Literal["epsilon_greedy", "ucb1", "thompson"]


class ArcusConfig(BaseSettings):
    arc_api_key: str
    # thompson sampling is the default, it's the one that needs the least
    # hand-tuning (no epsilon to pick) and adapts the fastest early on.
    # the algorithm is swappable here rather than hardcoded, so a user
    # can pick a different one without touching code.
    bandit_algorithm: BanditAlgorithm = "thompson"
    # off by default: routing code/math/long-document questions across
    # ARC's -thinking-* model variants too, not just the base four, is
    # built and tested against a fake adapter, but hasn't been run
    # against a real ARC key yet, see cli.py's _REASONING_VARIANTS.
    # flip this on with `arcus config set enable_reasoning_variants true`
    # once that's been confirmed live.
    enable_reasoning_variants: bool = False


def config_path() -> Path:
    # computed inside a function rather than a module constant so tests
    # can point it somewhere disposable the same way storage/db.py's data
    # dir already gets monkeypatched.
    return Path(user_config_dir("arcus")) / "config.toml"


def load_config(path: Path | None = None) -> ArcusConfig:
    path = path or config_path()
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return ArcusConfig(**data)


def save_config(config: ArcusConfig, path: Path | None = None) -> None:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # still just a handful of fields, not worth pulling in a TOML-writing
    # dependency to serialize a few key/value lines. TOML booleans are
    # bare lowercase true/false, not Python's True/False, hence the str.lower().
    path.write_text(
        f'arc_api_key = "{config.arc_api_key}"\n'
        f'bandit_algorithm = "{config.bandit_algorithm}"\n'
        f"enable_reasoning_variants = {str(config.enable_reasoning_variants).lower()}\n"
    )
    # the API key lives on disk in plain text, chmod 600 so it's at least
    # not readable by other users on the same machine. skipped on
    # Windows, where chmod only toggles the read-only attribute and
    # doesn't map onto per-user access control the way it does on POSIX,
    # the per-user profile directory Windows resolves config_path() into
    # already isn't readable by other accounts by default.
    if os.name != "nt":
        path.chmod(0o600)

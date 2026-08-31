import pytest

from arcus.config import ArcusConfig, config_path, load_config, save_config


def test_config_path_uses_platformdirs(monkeypatch, tmp_path):
    monkeypatch.setattr("arcus.config.user_config_dir", lambda name: str(tmp_path))
    assert config_path() == tmp_path / "config.toml"


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    save_config(ArcusConfig(arc_api_key="sk-test-123"), path=path)

    loaded = load_config(path=path)
    assert loaded.arc_api_key == "sk-test-123"


def test_bandit_algorithm_defaults_to_thompson(tmp_path):
    path = tmp_path / "config.toml"
    save_config(ArcusConfig(arc_api_key="sk-test-123"), path=path)

    loaded = load_config(path=path)
    assert loaded.bandit_algorithm == "thompson"


def test_bandit_algorithm_roundtrips_a_non_default_choice(tmp_path):
    path = tmp_path / "config.toml"
    save_config(ArcusConfig(arc_api_key="sk-test-123", bandit_algorithm="ucb1"), path=path)

    loaded = load_config(path=path)
    assert loaded.bandit_algorithm == "ucb1"


def test_save_config_sets_restrictive_permissions(tmp_path):
    path = tmp_path / "config.toml"
    save_config(ArcusConfig(arc_api_key="sk-test-123"), path=path)

    mode = path.stat().st_mode & 0o777
    assert mode == 0o600


def test_save_config_skips_chmod_on_windows(monkeypatch, tmp_path):
    monkeypatch.setattr("arcus.config.os.name", "nt")

    def _boom(self, mode):
        raise AssertionError("chmod shouldn't be called on Windows")

    monkeypatch.setattr("pathlib.Path.chmod", _boom)

    path = tmp_path / "config.toml"
    save_config(ArcusConfig(arc_api_key="sk-test-123"), path=path)

    assert path.read_text().startswith('arc_api_key')


def test_save_config_creates_parent_dir(tmp_path):
    path = tmp_path / "nested" / "config.toml"
    save_config(ArcusConfig(arc_api_key="sk-test-123"), path=path)

    assert path.exists()


def test_load_config_missing_key_raises(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('something_else = "value"\n')

    with pytest.raises(Exception):
        load_config(path=path)

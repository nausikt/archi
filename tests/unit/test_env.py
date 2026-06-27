import os

from src.utils.env import read_or_create_persistent_secret


def test_returns_configured_env_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("MY_TEST_SECRET_FILE", raising=False)
    monkeypatch.setenv("MY_TEST_SECRET", "configured-value")
    assert read_or_create_persistent_secret("MY_TEST_SECRET", str(tmp_path)) == "configured-value"
    # nothing is persisted when a value is already configured
    assert not os.path.exists(os.path.join(str(tmp_path), ".my_test_secret"))


def test_reads_previously_persisted_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("MY_TEST_SECRET", raising=False)
    monkeypatch.delenv("MY_TEST_SECRET_FILE", raising=False)
    (tmp_path / ".my_test_secret").write_text("persisted-key")
    assert read_or_create_persistent_secret("MY_TEST_SECRET", str(tmp_path)) == "persisted-key"


def test_generates_then_stable_across_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("MY_TEST_SECRET", raising=False)
    monkeypatch.delenv("MY_TEST_SECRET_FILE", raising=False)
    # first boot: no env, no file -> generate + persist a 64-hex-char key
    first = read_or_create_persistent_secret("MY_TEST_SECRET", str(tmp_path))
    assert len(first) == 64
    assert os.path.exists(os.path.join(str(tmp_path), ".my_test_secret"))
    # second boot (same dir): SAME key, so signed sessions survive a restart
    second = read_or_create_persistent_secret("MY_TEST_SECRET", str(tmp_path))
    assert second == first

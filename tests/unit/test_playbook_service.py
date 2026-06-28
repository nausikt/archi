"""Unit tests for PlaybookService — mocked psycopg2, no real DB required."""

import pytest

from src.utils.playbook_service import (
    MAX_ENABLED_PUBLIC_PER_USER,
    PlaybookNotFoundError,
    PlaybookService,
    PlaybookValidationError,
)


# ---------------------------------------------------------------------------
# Minimal fake helpers (no real DB)
# ---------------------------------------------------------------------------

class _FakeCursor:
    """A cursor stub that pops from a pre-loaded fetchone_values list and counts INSERTs."""

    def __init__(self, fetchone_values=None):
        self._fetchone_values = list(fetchone_values or [])
        self.executed_inserts = 0

    def execute(self, sql, params=None):
        if sql.strip().upper().startswith("INSERT"):
            self.executed_inserts += 1

    def fetchone(self):
        if self._fetchone_values:
            return self._fetchone_values.pop(0)
        return None

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _FakeConn:
    """A connection stub that returns a single shared _FakeCursor and records commits/rollbacks."""

    def __init__(self, cursor):
        self._cursor = cursor
        self.rollback_called = False

    def cursor(self, **kwargs):
        return self._cursor

    def commit(self):
        pass

    def rollback(self):
        self.rollback_called = True


# ---------------------------------------------------------------------------
# Visibility guard
# ---------------------------------------------------------------------------

def test_enable_playbook_rejects_foreign_private(monkeypatch):
    svc = PlaybookService(pg_config={"dummy": True})
    fake_cursor = _FakeCursor(fetchone_values=[("private", "owner-x")])  # visibility, owner
    monkeypatch.setattr(svc, "_get_connection", lambda: _FakeConn(fake_cursor))
    monkeypatch.setattr(svc, "_release_connection", lambda conn: None)
    with pytest.raises(PlaybookValidationError):
        svc.enable_playbook("user-B", 42)
    assert fake_cursor.executed_inserts == 0  # never reached the INSERT


def test_enable_playbook_rolls_back_on_foreign_private(monkeypatch):
    """Connection must be rolled back before release on the foreign-private guard path."""
    svc = PlaybookService(pg_config={"dummy": True})
    fake_cursor = _FakeCursor(fetchone_values=[("private", "owner-x")])
    fake_conn = _FakeConn(fake_cursor)
    monkeypatch.setattr(svc, "_get_connection", lambda: fake_conn)
    monkeypatch.setattr(svc, "_release_connection", lambda conn: None)
    with pytest.raises(PlaybookValidationError):
        svc.enable_playbook("user-B", 42)
    assert fake_conn.rollback_called, "rollback() must be called before releasing on a guard raise"
    assert fake_cursor.executed_inserts == 0


def test_enable_playbook_allows_public(monkeypatch):
    """A public playbook owned by someone else should reach the INSERT without raising."""
    svc = PlaybookService(pg_config={"dummy": True})
    # First fetchone: (visibility, owner_id); second fetchone: COUNT result
    fake_cursor = _FakeCursor(fetchone_values=[("public", "owner-x"), (0,)])
    fake_conn = _FakeConn(fake_cursor)
    monkeypatch.setattr(svc, "_get_connection", lambda: fake_conn)
    monkeypatch.setattr(svc, "_release_connection", lambda conn: None)
    svc.enable_playbook("user-B", 42)  # should not raise
    assert fake_cursor.executed_inserts == 1
    assert not fake_conn.rollback_called


def test_enable_playbook_enforces_cap(monkeypatch):
    """When the user is at the cap, enable_playbook raises and does NOT INSERT."""
    svc = PlaybookService(pg_config={"dummy": True})
    fake_cursor = _FakeCursor(
        fetchone_values=[("public", "owner-x"), (MAX_ENABLED_PUBLIC_PER_USER,)]
    )
    fake_conn = _FakeConn(fake_cursor)
    monkeypatch.setattr(svc, "_get_connection", lambda: fake_conn)
    monkeypatch.setattr(svc, "_release_connection", lambda conn: None)
    with pytest.raises(PlaybookValidationError, match="limit reached"):
        svc.enable_playbook("user-B", 42)
    assert fake_cursor.executed_inserts == 0
    assert fake_conn.rollback_called, "rollback() must be called on the cap-exceeded path"

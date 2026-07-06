"""Regression tests for the ChatWrapper <-> PlaybookService seam.

The Blueprint refactor (383d1e5b) moved the playbook side-table SQL onto
PlaybookService and left the chat flow calling ``self._playbook_svc()`` — but
the accessor lived only on FlaskAppWrapper. Every chat-flow call raised
AttributeError, silently swallowed by the surrounding best-effort excepts:
turn recording, the regenerate re-apply lookup and A/B turn recording all
no-oped (chips vanished on reload). No suite caught it because every layer
mocked or bypassed this exact seam.

These tests exercise the REAL accessor on a REAL ChatWrapper instance with
only the *service* mocked (never the accessor itself), so a missing or broken
accessor fails loudly instead of degrading silently.
"""
import pytest

flask = pytest.importorskip("flask", reason="chat app import chain needs flask")

from unittest.mock import MagicMock, patch

from src.interfaces.chat_app.app import ChatWrapper


def _wrapper() -> ChatWrapper:
    """A ChatWrapper without the heavy __init__ (config/DB); only what the
    tested methods touch is set."""
    wrapper = ChatWrapper.__new__(ChatWrapper)
    wrapper.pg_config = {"host": "unused-in-tests"}
    return wrapper


def _context(playbook_name="pb-name", playbook_id=7):
    ctx = MagicMock()
    ctx.playbook_name = playbook_name
    ctx.playbook_id = playbook_id
    ctx.provider_used = "prov"
    ctx.model_used = "model"
    return ctx


def _factory_with(svc):
    factory = MagicMock()
    factory.playbook_service = svc
    return factory


def test_chat_wrapper_playbook_svc_uses_pooled_factory():
    svc = MagicMock()
    with patch(
        "src.utils.postgres_service_factory.PostgresServiceFactory.get_instance",
        return_value=_factory_with(svc),
    ):
        assert _wrapper()._playbook_svc() is svc


def test_insert_conversation_records_playbook_turn_through_real_accessor():
    """The user turn of a /name invocation must land in the side table via the
    real accessor — only the service is mocked, so a ChatWrapper without a
    working _playbook_svc fails here instead of warn-and-continuing."""
    wrapper = _wrapper()
    svc = MagicMock()
    fake_cursor = MagicMock()
    fake_cursor.fetchall.return_value = [(101,), (102,)]
    fake_conn = MagicMock()
    fake_conn.cursor.return_value = fake_cursor

    with patch("src.interfaces.chat_app.app.psycopg2") as fake_pg, patch(
        "src.utils.postgres_service_factory.PostgresServiceFactory.get_instance",
        return_value=_factory_with(svc),
    ):
        fake_pg.connect.return_value = fake_conn
        message_ids = wrapper.insert_conversation(
            1,
            ("User", "run it", "2026-01-01T00:00:00Z"),
            ("archi", "done", "2026-01-01T00:00:30Z"),
            "",
            "",
            _context(),
        )

    assert message_ids == [101, 102]
    svc.record_playbook_turn.assert_called_once_with(101, "pb-name", 7)


def test_insert_conversation_plain_turn_does_not_touch_the_service():
    """No playbook on the turn -> no side-table write at all."""
    wrapper = _wrapper()
    svc = MagicMock()
    fake_cursor = MagicMock()
    fake_cursor.fetchall.return_value = [(201,), (202,)]
    fake_conn = MagicMock()
    fake_conn.cursor.return_value = fake_cursor

    with patch("src.interfaces.chat_app.app.psycopg2") as fake_pg, patch(
        "src.utils.postgres_service_factory.PostgresServiceFactory.get_instance",
        return_value=_factory_with(svc),
    ):
        fake_pg.connect.return_value = fake_conn
        wrapper.insert_conversation(
            1,
            ("User", "plain question", "2026-01-01T00:00:00Z"),
            ("archi", "plain answer", "2026-01-01T00:00:30Z"),
            "",
            "",
            _context(playbook_name=None, playbook_id=None),
        )

    svc.record_playbook_turn.assert_not_called()


def test_insert_conversation_survives_a_failing_side_table_write():
    """The side-table write stays best-effort: a service error must not break
    the conversation insert itself (a failed migration degrades, not 500s)."""
    wrapper = _wrapper()
    svc = MagicMock()
    svc.record_playbook_turn.side_effect = RuntimeError("side table missing")
    fake_cursor = MagicMock()
    fake_cursor.fetchall.return_value = [(301,), (302,)]
    fake_conn = MagicMock()
    fake_conn.cursor.return_value = fake_cursor

    with patch("src.interfaces.chat_app.app.psycopg2") as fake_pg, patch(
        "src.utils.postgres_service_factory.PostgresServiceFactory.get_instance",
        return_value=_factory_with(svc),
    ):
        fake_pg.connect.return_value = fake_conn
        message_ids = wrapper.insert_conversation(
            1,
            ("User", "run it", "2026-01-01T00:00:00Z"),
            ("archi", "done", "2026-01-01T00:00:30Z"),
            "",
            "",
            _context(),
        )

    assert message_ids == [301, 302]
    svc.record_playbook_turn.assert_called_once()


# ── M1: conversation loads degrade gracefully when the side table is missing ──

def test_convo_history_query_falls_back_when_side_table_missing():
    """A failed migration (no conversation_playbook_turns) must not 500 every
    conversation load: the helper rolls back the aborted transaction and
    retries with the no-playbooks variant."""
    import psycopg2

    from src.interfaces.chat_app import app as chat_app

    cursor = MagicMock()
    cursor.execute.side_effect = [psycopg2.errors.UndefinedTable("no cpt"), None]
    cursor.fetchall.return_value = [("User", "hi", 1, None, 0, None, None)]

    rows = chat_app._query_convo_history_rows(cursor, 42)

    assert rows == [("User", "hi", 1, None, 0, None, None)]
    cursor.connection.rollback.assert_called_once()
    fallback_sql = cursor.execute.call_args_list[1].args[0]
    assert "conversation_playbook_turns" not in fallback_sql
    assert "NULL AS playbook_name" in fallback_sql


def test_convo_history_query_single_roundtrip_when_table_exists():
    from src.interfaces.chat_app import app as chat_app

    cursor = MagicMock()
    cursor.fetchall.return_value = []

    chat_app._query_convo_history_rows(cursor, 42)

    assert cursor.execute.call_count == 1
    cursor.connection.rollback.assert_not_called()


def test_load_conversation_closes_connection_on_error():
    """M1's second half: an unexpected error mid-load must not leak the
    connection — it is closed in a finally, not only on the success path."""
    from src.interfaces.chat_app.app import FlaskAppWrapper

    wrapper = FlaskAppWrapper.__new__(FlaskAppWrapper)
    wrapper.pg_config = {"host": "unused-in-tests"}
    wrapper.chat = MagicMock()

    fake_conn = MagicMock()
    fake_conn.closed = False
    fake_cursor = MagicMock()
    fake_cursor.execute.side_effect = RuntimeError("boom mid-query")
    fake_conn.cursor.return_value = fake_cursor

    app = flask.Flask(__name__)
    with app.test_request_context(
        json={"conversation_id": 1, "client_id": "c1"}
    ), patch("src.interfaces.chat_app.app.psycopg2") as fake_pg:
        fake_pg.connect.return_value = fake_conn
        _resp, code = wrapper.load_conversation()

    assert code == 500
    fake_conn.close.assert_called_once()

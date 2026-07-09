"""
TokenService - Stores SSO access/refresh tokens in the existing Postgres
``sessions`` table provisioned by ``archi/src/cli/templates/init.sql``.

Why this exists
---------------
Cookie-based Flask sessions cannot reliably hold a CERN SSO access_token plus
refresh_token (combined size frequently exceeds the ~4 KB browser cookie
limit, especially for users with many roles in the JWT). They also cannot
coordinate refresh-token rotation across concurrent in-flight requests,
because each request gets its own snapshot of the cookie session.

This service moves the bulky/sensitive token material into Postgres:

- The cookie keeps only an opaque ``sso_sid`` (32-byte hex), so total cookie
  payload stays well under the 4 KB browser limit regardless of role count.
- ``access_token`` and ``refresh_token`` are stored encrypted in the
  ``sessions.data`` JSONB column using pgcrypto's ``pgp_sym_encrypt``
  (same key/pattern as ``UserService``'s BYOK API key storage).
- Refresh-token rotation is coalesced via a Postgres advisory lock keyed by
  hashtext(user_id), so concurrent chat turns never race two refresh calls
  against Keycloak (which would invalidate the rotated token).

The ``sessions`` table is already created by ``init.sql`` (lines 80-86) and
unused by anything else in the codebase as of this patch.
"""

from __future__ import annotations

import secrets
import time
from typing import Any, Dict, Optional

import psycopg2
import psycopg2.extras

from src.utils.env import read_secret
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Hard guard: never store anything we cannot decrypt.
_NO_KEY_MSG = (
    "BYOK_ENCRYPTION_KEY is not set; SSO token storage is disabled. "
    "Set BYOK_ENCRYPTION_KEY in the chat container environment to enable "
    "MCP user-token forwarding."
)


class TokenService:
    """Postgres-backed store for per-user SSO access/refresh tokens.

    All public methods accept either an ``sso_sid`` (the opaque cookie key)
    or a ``user_id`` (the SSO ``sub`` claim) where appropriate. The cookie
    must carry ``sso_sid`` to retrieve the token row.
    """

    # Each successful refresh slides the SSO session window this far into the
    # future, so an actively-chatting user is never forced to re-login. Idle
    # users still expire once they stop triggering refreshes for this long
    # (and upstream Keycloak refresh-token lifetime is the hard ceiling).
    SESSION_SLIDING_SECONDS = 12 * 3600

    def __init__(
        self,
        pg_config: Optional[Dict[str, Any]] = None,
        *,
        connection_pool=None,
        encryption_key: Optional[str] = None,
    ) -> None:
        self._pool = connection_pool
        self._pg_config = pg_config
        self._encryption_key = encryption_key or read_secret(
            "BYOK_ENCRYPTION_KEY", default=""
        )
        if not self._encryption_key:
            logger.warning(_NO_KEY_MSG)

    # ------------------------------------------------------------------
    # Connection plumbing (mirrors UserService for consistency)
    # ------------------------------------------------------------------

    def _get_connection(self) -> psycopg2.extensions.connection:
        if self._pool:
            return self._pool.get_connection()
        if self._pg_config:
            return psycopg2.connect(**self._pg_config)
        raise ValueError("TokenService requires a connection_pool or pg_config")

    def _release_connection(self, conn) -> None:
        if self._pool:
            self._pool.release_connection(conn)
        else:
            conn.close()

    @property
    def enabled(self) -> bool:
        """Whether token storage/retrieval is functional in this deployment."""
        return bool(self._encryption_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def mint_sso_sid() -> str:
        """Generate a new opaque session id for the cookie."""
        return secrets.token_hex(32)

    def store_sso_token(
        self,
        *,
        sso_sid: str,
        user_id: str,
        access_token: str,
        refresh_token: Optional[str],
        access_expires_at: int,
        session_expires_at: int,
    ) -> bool:
        """Persist an encrypted token bundle for a freshly-completed SSO login.

        Returns True on success, False if storage is disabled or fails.
        Failures are logged but never raised: callers should treat "no
        stored token" as "MCP user-scoped tools unavailable" and degrade
        gracefully.
        """
        if not self.enabled:
            return False
        if not sso_sid or not user_id or not access_token:
            logger.warning(
                "store_sso_token called with incomplete data "
                "(sid=%s user=%s has_at=%s)",
                bool(sso_sid), bool(user_id), bool(access_token),
            )
            return False

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions (id, user_id, data, expires_at)
                    VALUES (
                        %s,
                        %s,
                        jsonb_build_object(
                            'access_token',      encode(pgp_sym_encrypt(%s, %s), 'base64'),
                            'refresh_token',     CASE WHEN %s::text IS NULL THEN NULL
                                                      ELSE encode(pgp_sym_encrypt(%s, %s), 'base64') END,
                            'access_expires_at', %s::bigint,
                            'refreshed_at',      EXTRACT(EPOCH FROM NOW())::bigint
                        ),
                        to_timestamp(%s)
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        data       = EXCLUDED.data,
                        expires_at = EXCLUDED.expires_at
                    """,
                    (
                        sso_sid,
                        user_id,
                        access_token, self._encryption_key,
                        refresh_token, refresh_token, self._encryption_key,
                        int(access_expires_at),
                        int(session_expires_at),
                    ),
                )
                conn.commit()
            logger.info(
                "stored sso token sid=%s user=%s access_ttl=%ds session_ttl=%ds",
                sso_sid[:8] + "...",
                user_id,
                int(access_expires_at) - int(time.time()),
                int(session_expires_at) - int(time.time()),
            )
            return True
        except Exception as exc:
            logger.warning("store_sso_token failed for user=%s: %s", user_id, exc)
            return False
        finally:
            self._release_connection(conn)

    def revoke_session(self, *, sso_sid: str) -> None:
        """Delete the token row for this session (called on logout)."""
        if not sso_sid:
            return
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM sessions WHERE id = %s", (sso_sid,))
                conn.commit()
        except Exception as exc:
            logger.warning("revoke_session failed for sid=%s...: %s", sso_sid[:8], exc)
        finally:
            self._release_connection(conn)

    def get_access_token(
        self,
        *,
        sso_sid: str,
        early_refresh_seconds: int = 60,
        refresher=None,
    ) -> Optional[str]:
        """Return a currently-valid access token for ``sso_sid`` or None.

        If the access token is within ``early_refresh_seconds`` of expiry, a
        refresh is attempted using ``refresher`` (a callable that takes the
        decrypted refresh_token and returns a new authlib-style token dict
        with at least ``access_token`` and ``expires_in`` keys, optionally
        ``refresh_token``).

        Refresh is serialized per-user via a Postgres advisory lock so two
        concurrent in-flight chat turns never both POST a refresh and break
        each other under refresh-token rotation.
        """
        if not self.enabled or not sso_sid:
            return None

        rec = self._read_token_row(sso_sid)
        if rec is None:
            return None

        now = int(time.time())
        if rec["access_expires_at"] - early_refresh_seconds > now:
            return rec["access_token"]

        if not rec.get("refresh_token") or refresher is None:
            logger.info(
                "sso access token expired sid=%s user=%s and no refresher available",
                sso_sid[:8] + "...", rec["user_id"],
            )
            return None

        return self._refresh_under_lock(
            sso_sid=sso_sid,
            user_id=rec["user_id"],
            current_refresh_token=rec["refresh_token"],
            refresher=refresher,
            early_refresh_seconds=early_refresh_seconds,
        )
    
    def session_alive(self, sso_sid: str) -> bool:
        """Cheap check: is the SSO token-session window still open?

        Unlike get_access_token this does no decryption and no refresh — it
        only answers "does a non-expired row exist?". Used by the web layer to
        tie the Flask login lifetime to the token-session lifetime without
        incurring a refresh on every request.
        """
        if not self.enabled or not sso_sid:
            return False
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM sessions WHERE id = %s AND expires_at > NOW()",
                    (sso_sid,),
                )
                return cur.fetchone() is not None
        except Exception as exc:
            logger.warning(
                "session_alive check failed for sid=%s...: %s", sso_sid[:8], exc
            )
            return True  # fail-open: don't mass-logout on a transient DB blip
        finally:
            self._release_connection(conn)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_token_row(self, sso_sid: str) -> Optional[Dict[str, Any]]:
        """Fetch + decrypt the token row. Returns None if missing/expired."""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        user_id,
                        (data->>'access_expires_at')::bigint AS access_expires_at,
                        pgp_sym_decrypt(decode(data->>'access_token', 'base64'), %s)
                            AS access_token,
                        CASE WHEN data->>'refresh_token' IS NULL THEN NULL
                             ELSE pgp_sym_decrypt(
                                 decode(data->>'refresh_token', 'base64'), %s)
                        END AS refresh_token,
                        EXTRACT(EPOCH FROM expires_at)::bigint AS session_expires_at
                    FROM sessions
                    WHERE id = %s
                      AND expires_at > NOW()
                    """,
                    (self._encryption_key, self._encryption_key, sso_sid),
                )
                row = cur.fetchone()
                if row is None:
                    return None

                def _to_str(v):
                    if isinstance(v, (bytes, memoryview)):
                        return bytes(v).decode("utf-8")
                    return v

                return {
                    "user_id": row["user_id"],
                    "access_token": _to_str(row["access_token"]),
                    "refresh_token": _to_str(row["refresh_token"]),
                    "access_expires_at": int(row["access_expires_at"] or 0),
                    "session_expires_at": int(row["session_expires_at"] or 0),
                }
        except Exception as exc:
            logger.warning("_read_token_row failed for sid=%s...: %s", sso_sid[:8], exc)
            return None
        finally:
            self._release_connection(conn)

    def _refresh_under_lock(
        self,
        *,
        sso_sid: str,
        user_id: str,
        current_refresh_token: str,
        refresher,
        early_refresh_seconds: int,
    ) -> Optional[str]:
        """Refresh the access token under a per-user advisory lock.

        The lock guarantees that two concurrent callers serialize: the loser
        re-reads the row after the winner has written, and observes the new
        access_token without making its own (now-invalid) refresh call.
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Transaction-scoped lock: released automatically on commit/rollback.
                cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    (f"sso_refresh:{user_id}",),
                )

                # Re-read inside the lock; another thread may have just refreshed.
                cur.execute(
                    """
                    SELECT
                        (data->>'access_expires_at')::bigint AS access_expires_at,
                        pgp_sym_decrypt(decode(data->>'access_token', 'base64'), %s)
                            AS access_token,
                        CASE WHEN data->>'refresh_token' IS NULL THEN NULL
                             ELSE pgp_sym_decrypt(
                                 decode(data->>'refresh_token', 'base64'), %s)
                        END AS refresh_token,
                        EXTRACT(EPOCH FROM expires_at)::bigint AS session_expires_at
                    FROM sessions
                    WHERE id = %s
                      AND expires_at > NOW()
                    FOR UPDATE
                    """,
                    (self._encryption_key, self._encryption_key, sso_sid),
                )
                row = cur.fetchone()
                if row is None:
                    conn.commit()
                    return None

                now = int(time.time())
                if int(row["access_expires_at"] or 0) - early_refresh_seconds > now:
                    # Another thread refreshed while we waited on the lock.
                    conn.commit()
                    at = row["access_token"]
                    if isinstance(at, (bytes, memoryview)):
                        at = bytes(at).decode("utf-8")
                    return at

                # We are the elected refresher; talk to Keycloak.
                rt = row["refresh_token"]
                if isinstance(rt, (bytes, memoryview)):
                    rt = bytes(rt).decode("utf-8")
                if not rt:
                    conn.commit()
                    return None

                try:
                    new_tok = refresher(rt)
                except Exception as exc:
                    logger.warning(
                        "refresh_token exchange failed for user=%s: %s",
                        user_id, exc,
                    )
                    conn.commit()
                    return None

                if not new_tok or not new_tok.get("access_token"):
                    logger.info(
                        "refresh returned no access_token for user=%s; "
                        "dropping session row",
                        user_id,
                    )
                    cur.execute("DELETE FROM sessions WHERE id = %s", (sso_sid,))
                    conn.commit()
                    return None

                new_access = new_tok["access_token"]
                new_refresh = new_tok.get("refresh_token", rt)
                new_access_exp = int(
                    new_tok.get("expires_at")
                    or (now + int(new_tok.get("expires_in", 0)))
                )

                ## Slide the SSO session window forward on each successful
                # refresh so an actively-chatting user is never forced to
                # re-login. Idle users still expire once they stop triggering
                # refreshes for SESSION_SLIDING_SECONDS.
                new_session_exp = now + self.SESSION_SLIDING_SECONDS
                cur.execute(
                    """
                    UPDATE sessions
                    SET data = jsonb_set(
                        jsonb_set(
                            jsonb_set(
                                jsonb_set(data,
                                    '{access_token}',
                                    to_jsonb(encode(pgp_sym_encrypt(%s, %s), 'base64'))
                                ),
                                '{refresh_token}',
                                to_jsonb(encode(pgp_sym_encrypt(%s, %s), 'base64'))
                            ),
                            '{access_expires_at}', to_jsonb(%s::bigint)
                        ),
                        '{refreshed_at}', to_jsonb(EXTRACT(EPOCH FROM NOW())::bigint)
                    ),
                    expires_at = to_timestamp(%s)
                    WHERE id = %s
                    """,
                    (
                        new_access, self._encryption_key,
                        new_refresh, self._encryption_key,
                        new_access_exp,
                        new_session_exp,
                        sso_sid,
                    ),
                )
                conn.commit()
                logger.info(
                    "refreshed sso access token user=%s new_ttl=%ds session_ttl=%ds",
                    user_id, new_access_exp - now, new_session_exp - now,
                )
                return new_access
        except Exception as exc:
            logger.warning(
                "_refresh_under_lock failed for sid=%s... user=%s: %s",
                sso_sid[:8], user_id, exc,
            )
            try:
                conn.rollback()
            except Exception:
                pass
            return None
        finally:
            self._release_connection(conn)

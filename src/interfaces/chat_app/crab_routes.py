"""CRAB task prefetch + autocomplete endpoints (Flask Blueprint).

Self-contained CRAB-specific chat-app surface, relocated out of the old
``app.py`` overlay patch so the shared chat application no longer needs a
CRAB-specific fork. Registered by ``FlaskAppWrapper`` via
:func:`register_crab_routes`, mirroring ``playbook_routes.register_playbooks``
and ``service_alerts.register_service_alerts``.

The endpoints are inherently safe to register on any deployment: when the
active pipeline is not the CRAB agent (i.e. it does not expose
``prefetch_user_task_summaries``), or the caller has no SSO token, they
collapse to an empty result instead of erroring.

Endpoints
---------
``GET /api/crab/tasks/prefetch``
    Pre-fetch the logged-in user's recent CRAB task summaries (skips the LLM;
    the agent does a direct ``list_tasks`` MCP call with the user's CERN SSO
    token). Cached per ``sso_sid`` so the hint endpoint stays cheap.

``GET /api/crab/tasks/hint``
    Autocomplete CRAB task IDs from the per-session prefetch cache for the
    chat input's ``@<query>`` mention picker.
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from flask import Blueprint, jsonify, request, session

from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_TTL_SECONDS = 60
_DEFAULT_DAYS = 30


class CrabTaskCache:
    """Tiny per-session cache of prefetched CRAB task summaries.

    Keyed by the opaque ``sso_sid``; entries expire after ``ttl_seconds``.
    A lock keeps dict operations atomic across Flask worker threads. The
    cache only stores the *default-window* result so the hint endpoint has a
    stable view; non-default windows bypass it entirely.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        default_days: int = _DEFAULT_DAYS,
    ) -> None:
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self.ttl_seconds = ttl_seconds
        self.default_days = default_days

    def get(self, sso_sid: str) -> Optional[List[Dict[str, Any]]]:
        """Return cached tasks for ``sso_sid`` if still fresh, else None."""
        if not sso_sid:
            return None
        with self._lock:
            entry = self._entries.get(sso_sid)
            if not entry:
                return None
            if (time.time() - entry["fetched_at"]) > self.ttl_seconds:
                return None
            return list(entry["tasks"])

    def set(self, sso_sid: str, tasks: List[Dict[str, Any]]) -> None:
        """Replace the cached tasks for ``sso_sid``."""
        if not sso_sid:
            return
        with self._lock:
            self._entries[sso_sid] = {
                "tasks": list(tasks or []),
                "fetched_at": time.time(),
            }

    def invalidate(self, sso_sid: Optional[str]) -> None:
        """Drop the cache entry for ``sso_sid``."""
        if not sso_sid:
            return
        with self._lock:
            self._entries.pop(sso_sid, None)


def register_crab_routes(
    app,
    *,
    auth_enabled: bool,
    require_auth: Callable,
    get_pipeline: Callable[[], Any],
    token_service_getter: Callable[[], Any],
    refresher_factory: Callable[[], Any],
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    default_days: int = _DEFAULT_DAYS,
) -> CrabTaskCache:
    """Register the CRAB task blueprint with a Flask app.

    Parameters
    ----------
    app : Flask
        The Flask application instance.
    auth_enabled : bool
        Whether authentication is enabled; when true all blueprint routes go
        through the same auth gate as the rest of the app via ``before_request``.
    require_auth : callable
        The ``require_auth`` decorator from ``FlaskAppWrapper``.
    get_pipeline : callable
        Returns the active pipeline (or ``None``). The CRAB agent exposes
        ``prefetch_user_task_summaries``; other pipelines don't, in which case
        the endpoints return empty results.
    token_service_getter : callable
        Returns the ``TokenService`` (or ``None``) used to fetch the caller's
        encrypted CERN SSO access token from Postgres.
    refresher_factory : callable
        ``FlaskAppWrapper._refresh_sso_token_callable`` — builds the OAuth
        refresher passed to ``TokenService.get_access_token``.
    ttl_seconds, default_days : int
        Cache tuning.

    Returns
    -------
    CrabTaskCache
        The cache instance (exposed for callers that want to warm/invalidate
        it, e.g. on login/logout — optional; the endpoints self-warm on a cold
        cache).
    """
    cache = CrabTaskCache(ttl_seconds=ttl_seconds, default_days=default_days)
    crab_bp = Blueprint("crab", __name__)

    def _access_token(sso_sid: str) -> Optional[str]:
        """Fetch a currently-valid SSO access token for ``sso_sid``.

        Reads the encrypted token row via TokenService, refreshing
        transparently when near expiry. Returns None when forwarding is
        unavailable (BYOK disabled / no token / anonymous).
        """
        svc = token_service_getter()
        if svc is None:
            return None
        return svc.get_access_token(sso_sid=sso_sid, refresher=refresher_factory())

    def _resolve_prefetch() -> Optional[Callable]:
        pipeline = get_pipeline()
        prefetch = getattr(pipeline, "prefetch_user_task_summaries", None)
        return prefetch if callable(prefetch) else None

    def _fetch_tasks(
        sso_sid: str,
        *,
        days: int,
        username: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run the agent's MCP prefetch for ``sso_sid``; never raises.

        Returns the freshly-fetched list, or ``[]`` on any failure path.
        """
        prefetch = _resolve_prefetch()
        if prefetch is None:
            return []
        access_token = _access_token(sso_sid)
        if not access_token:
            return []
        if not username:
            try:
                user = session.get("user") or {}
                username = (user.get("username") or "").strip()
            except RuntimeError:
                username = ""

        def _token_getter() -> Optional[str]:
            return _access_token(sso_sid)

        try:
            return (
                prefetch(
                    access_token,
                    username,
                    days=days,
                    token_getter=_token_getter,
                )
                or []
            )
        except Exception as exc:
            logger.warning(
                "CRAB tasks prefetch failed for sso_sid=%s: %s",
                (sso_sid[:8] + "…") if sso_sid else "?",
                exc,
                exc_info=True,
            )
            return []

    def _refresh_cache(sso_sid: str) -> List[Dict[str, Any]]:
        """Fetch the default-window tasks and store them in the cache."""
        tasks = _fetch_tasks(sso_sid, days=cache.default_days)
        cache.set(sso_sid, tasks)
        return tasks

    @crab_bp.route("/api/crab/tasks/prefetch", methods=["GET"])
    def crab_tasks_prefetch():
        """Pre-fetch CRAB task summaries for the logged-in user.

        Reads from the per-session cache (self-warming on cold/expired
        entries); refreshes from MCP when stale or when ``?refresh=1``. A
        non-default ``?days=N`` bypasses the cache so the hint endpoint keeps
        its stable default-window view. Always returns 200; ``tasks`` is
        ``[]`` when the active pipeline is not the CRAB agent or the user has
        no SSO token.
        """
        sso_sid = session.get("sso_sid")
        if not sso_sid:
            return jsonify({"tasks": []}), 200

        try:
            req_days = int(request.args.get("days", cache.default_days))
        except (TypeError, ValueError):
            req_days = cache.default_days
        if req_days <= 0:
            req_days = cache.default_days

        force = request.args.get("refresh") == "1"
        is_default_window = req_days == cache.default_days

        if is_default_window and not force:
            cached = cache.get(sso_sid)
            if cached is not None:
                return jsonify({"tasks": cached}), 200

        if is_default_window:
            tasks = _refresh_cache(sso_sid)
        else:
            tasks = _fetch_tasks(sso_sid, days=req_days)

        return jsonify({"tasks": tasks}), 200

    @crab_bp.route("/api/crab/tasks/hint", methods=["GET"])
    def crab_tasks_hint():
        """Autocomplete CRAB task IDs from the per-session prefetch cache.

        Cache-only by default (each keystroke is a fast in-process lookup, not
        an MCP round-trip); refreshes only on a cold/expired cache. Query
        semantics: a leading ``@`` is stripped; empty ``q`` returns the most
        recent N; matching is case-insensitive on ``crab_id`` with prefix
        matches ranked above substring matches.
        """
        sso_sid = session.get("sso_sid")
        if not sso_sid:
            return jsonify({"hints": [], "query": "", "from_cache": False}), 200

        raw_q = (request.args.get("q") or "").strip()
        if raw_q.startswith("@"):
            raw_q = raw_q[1:].lstrip()
        q = raw_q.lower()

        try:
            limit = int(request.args.get("limit", 10))
        except (TypeError, ValueError):
            limit = 10
        if limit <= 0 or limit > 50:
            limit = 10

        cached = cache.get(sso_sid)
        from_cache = cached is not None
        tasks = cached if from_cache else _refresh_cache(sso_sid)

        if not tasks:
            return (
                jsonify({"hints": [], "query": raw_q, "from_cache": from_cache}),
                200,
            )

        if not q:
            hints = tasks[:limit]
        else:
            scored: List[tuple] = []
            for idx, t in enumerate(tasks):
                cid_lower = (t.get("crab_id") or "").lower()
                if not cid_lower:
                    continue
                if cid_lower.startswith(q):
                    score = 1000
                elif q in cid_lower:
                    score = 500 - cid_lower.index(q)
                else:
                    continue
                # Secondary key: original cache index (most-recent first).
                scored.append((-score, idx, t))
            scored.sort()
            hints = [t for _, _, t in scored[:limit]]

        return (
            jsonify({"hints": hints, "query": raw_q, "from_cache": from_cache}),
            200,
        )

    if auth_enabled:
        @crab_bp.before_request
        def _check_auth():
            """Apply the same auth gate used by the rest of the app."""
            sentinel = object()

            @require_auth
            def _probe():
                return sentinel

            result = _probe()
            if result is not sentinel:
                return result  # redirect / 401 from require_auth

    app.register_blueprint(crab_bp)
    logger.info("Registered CRAB tasks blueprint at /api/crab/tasks/*")
    return cache

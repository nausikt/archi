from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable as CallableABC
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple

import httpx

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

from src.utils.logging import get_logger
from src.archi.pipelines.agents.base_react import BaseReActAgent
from src.archi.pipelines.agents.utils.mcp_utils import AsyncLoopThread
from src.archi.pipelines.agents.playbook_mixin import SupportsPlaybooks
from src.data_manager.vectorstore.retrievers import HybridRetriever
from src.archi.pipelines.agents.tools import (
    create_document_fetch_tool,
    create_file_search_tool,
    create_metadata_search_tool,
    create_metadata_schema_tool,
    create_retriever_tool,
    RemoteCatalogClient,
)

logger = get_logger(__name__)


# Per-server YAML flag that opts an MCP server into per-user CERN SSO token
# forwarding. Servers without this flag (or with it set to false) keep the
# original singleton-per-process behavior and never see the user's token.
_FORWARD_TOKEN_FLAG = "forward_sso_token"

# Per-server YAML key that declares the audience the MCP server expects in
# the JWT's ``aud`` claim. Today every CRAB-side MCP server lives behind
# the same Keycloak client_id as this chat app, so the default is fine and
# the value is just stashed on the agent. Reserved as the hook point for
# future RFC 8693 token exchange when a downstream server lives in a
# different SSO audience and we need to mint a new token per-server.
_AUDIENCE_FLAG = "sso_aud"
DEFAULT_SSO_AUDIENCE = "archi-crab"


class CMSCRABAgent(SupportsPlaybooks, BaseReActAgent):
    """Bare-minimum agent for CMS CRAB operations.

    Functions like ``CMSCompOpsAgent`` but trims optional integrations
    (MONIT/HTCondor OpenSearch) to keep the surface area small. Provides:

    - catalog file/metadata search and fetch tools
    - optional hybrid (BM25 + semantic) vectorstore retriever tool
    - MCP tools (split into "static" and "per-user / SSO-forwarding")

    MCP server splitting
    --------------------
    Servers in ``mcp_servers`` whose YAML carries ``forward_sso_token: true``
    are *not* loaded at startup. Instead they are loaded per chat turn by
    :meth:`build_user_scoped_mcp_tools` with the calling user's CERN SSO
    access token attached as ``Authorization: Bearer ...`` on listing and on
    each tool call (via an MCP adapter interceptor and a ``token_getter`` that
    re-resolves the token from storage). All other servers
    keep the original singleton-per-process behavior via
    :meth:`_build_mcp_tools`.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config, *args, **kwargs)

        self.catalog_service = RemoteCatalogClient.from_deployment_config(self.config)
        self._vector_retrievers = None
        self._vector_tools = None
        self.enable_vector_tools = "search_vectorstore_hybrid" in self.selected_tool_names

        # Pre-split MCP server config into static vs user-scoped buckets,
        # and remember each user-scoped server's expected SSO audience.
        static, user_scoped, user_scoped_aud = self._split_mcp_servers(
            self.config.get("mcp_servers") or {}
        )
        self._static_mcp_servers: Dict[str, Dict[str, Any]] = static
        self._user_scoped_mcp_servers: Dict[str, Dict[str, Any]] = user_scoped
        self._user_scoped_audiences: Dict[str, str] = user_scoped_aud
        if user_scoped:
            logger.info(
                "user-scoped MCP servers configured: %s",
                list(user_scoped.keys()),
            )

        # Retain the per-turn MultiServerMCPClient for user-scoped servers so
        # streamable-http tool invocations reuse the same authenticated client
        # after the adapter's post-tools/list DELETE (otherwise tools/call can
        # hit the server without Authorization).
        self._user_scoped_mcp_client: Any = None

        # Snapshot of the most recently built user-scoped MCP tool callables.
        # We re-inject these on every `refresh_agent(...)` call below so that
        # the upstream `_prepare_agent_inputs` re-refresh (which only knows
        # about ``self._vector_tools``) does not drop them on the floor.
        self._user_scoped_tools_cache: List[Callable] = []

        self.rebuild_static_tools()
        self.rebuild_static_middleware()
        self.refresh_agent()

    # ------------------------------------------------------------------
    # Tool registry (unchanged from previous CMS-CRAB minimal agent)
    # ------------------------------------------------------------------

    def get_tool_registry(self) -> Dict[str, Callable[[], Any]]:
        return {name: entry["builder"] for name, entry in self._tool_definitions().items()}

    def get_tool_descriptions(self) -> Dict[str, str]:
        return {name: entry["description"] for name, entry in self._tool_definitions().items()}

    def _tool_definitions(self) -> Dict[str, Dict[str, Any]]:
        defs = {
            "search_local_files": {
                "builder": self._build_file_search_tool,
                "description": (
                    "Grep-like search over file contents. Provide a distinctive phrase or regex; optionally use "
                    "regex=true, case_sensitive=true, and context (before/after). Returns matching lines with hashes; "
                    "use fetch_catalog_document for full text."
                ),
            },
            "search_metadata_index": {
                "builder": self._build_metadata_search_tool,
                "description": (
                    "Query the files' metadata catalog (ticket IDs, source URLs, resource types, etc.). "
                    "Supports key:value filters and OR (e.g., source_type:git OR url:https://... ticket_id:CMS-123). "
                    "Returns matching files with metadata; use fetch_catalog_document to pull full text."
                ),
            },
            "list_metadata_schema": {
                "builder": self._build_metadata_schema_tool,
                "description": (
                    "List metadata schema hints: supported keys, distinct source_type values, and suffixes. "
                    "Use this to learn which key:value filters are available before searching."
                ),
            },
            "fetch_catalog_document": {
                "builder": self._build_fetch_tool,
                "description": (
                    "Fetch full document text by resource hash after a search hit. "
                    "Use this sparingly to pull only the most relevant files."
                ),
            },
            "search_vectorstore_hybrid": {
                "builder": self._build_vector_tool_placeholder,
                "description": (
                    "Hybrid search over the knowledge base that combines lexical (BM25) and semantic (vector) matching.\n"
                    "Input must be a plain text query string."
                ),
            },
            "mcp": {
                "builder": self._build_mcp_tools,
                "description": "Access tools served via configured MCP servers.",
            },
        }
        # Playbook authoring tools (save/update/delete) from SupportsPlaybooks.
        defs.update(super()._tool_definitions())
        return defs

    def _build_file_search_tool(self) -> Callable:
        description = self._tool_definitions()["search_local_files"]["description"]
        return create_file_search_tool(
            self.catalog_service,
            description=description,
            store_docs=self._store_documents,
            store_tool_input=getattr(self, "_store_tool_input", None),
        )

    def _build_metadata_search_tool(self) -> Callable:
        description = self._tool_definitions()["search_metadata_index"]["description"]
        return create_metadata_search_tool(
            self.catalog_service,
            description=description,
            store_docs=self._store_documents,
            store_tool_input=getattr(self, "_store_tool_input", None),
        )

    def _build_metadata_schema_tool(self) -> Callable:
        description = self._tool_definitions()["list_metadata_schema"]["description"]
        return create_metadata_schema_tool(
            self.catalog_service,
            description=description,
        )

    def _build_fetch_tool(self) -> Callable:
        description = self._tool_definitions()["fetch_catalog_document"]["description"]
        return create_document_fetch_tool(
            self.catalog_service,
            description=description,
            store_tool_input=getattr(self, "_store_tool_input", None),
        )

    def _build_vector_tool_placeholder(self) -> List[Callable]:
        return []

    # ------------------------------------------------------------------
    # MCP tool building (split into static + per-user)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_mcp_servers(
        mcp_servers: Dict[str, Any],
    ) -> Tuple[
        Dict[str, Dict[str, Any]],
        Dict[str, Dict[str, Any]],
        Dict[str, str],
    ]:
        """Partition mcp_servers into static, user-scoped, and per-server aud.

        Both flags (``forward_sso_token`` and ``sso_aud``) are stripped from
        the config that gets handed to ``MultiServerMCPClient``, since they
        are not part of the upstream schema. ``sso_aud`` defaults to
        ``DEFAULT_SSO_AUDIENCE`` and is currently only stashed on the agent
        for the future audience-exchange path.
        """
        static: Dict[str, Dict[str, Any]] = {}
        user_scoped: Dict[str, Dict[str, Any]] = {}
        user_scoped_aud: Dict[str, str] = {}
        for name, server_cfg in (mcp_servers or {}).items():
            cfg = dict(server_cfg or {})
            forward = bool(cfg.pop(_FORWARD_TOKEN_FLAG, False))
            audience = cfg.pop(_AUDIENCE_FLAG, None) or DEFAULT_SSO_AUDIENCE
            if forward:
                user_scoped[name] = cfg
                user_scoped_aud[name] = str(audience)
            else:
                static[name] = cfg
        return static, user_scoped, user_scoped_aud

    def _build_mcp_tools(self) -> List[Callable]:
        """Process-wide MCP tools (static servers only).

        Overrides :meth:`BaseReActAgent._build_mcp_tools` so we can pass the
        pre-split *static* server set instead of the raw YAML. Mirrors the
        upstream sync-wrapper trick to keep the long-lived MCP sessions on
        the shared :class:`AsyncLoopThread` loop.
        """
        if not self._static_mcp_servers:
            logger.info("No static MCP servers configured for CMSCRABAgent.")
            return []

        try:
            self._async_runner = AsyncLoopThread.get_instance()
            client, mcp_tools = self._async_runner.run(
                _initialize_mcp_client(self._static_mcp_servers)
            )
            if client is None:
                return []
            self.mcp_client = client
            return [self._make_synchronous(t) for t in (mcp_tools or [])]
        except Exception as exc:
            logger.error("Failed to load static MCP tools: %s", exc, exc_info=True)
            return []

    def _dispose_user_scoped_mcp_client(self) -> None:
        """Close and drop the per-turn user-scoped MCP client, if any."""
        # Always invalidate the cached tool list — it is bound to the client
        # we are about to dispose, so leaving stale entries around would let
        # ``refresh_agent`` re-attach already-closed sessions.
        self._user_scoped_tools_cache = []
        client = getattr(self, "_user_scoped_mcp_client", None)
        if client is None:
            return
        self._user_scoped_mcp_client = None
        try:
            runner = self._async_runner if hasattr(self, "_async_runner") else AsyncLoopThread.get_instance()
            _dispose_mcp_client_instance(client, runner)
        except Exception as exc:
            logger.warning("Failed to dispose user-scoped MCP client: %s", exc)

    def refresh_agent(
        self,
        *,
        static_tools: Optional[Sequence[Callable]] = None,
        extra_tools: Optional[Sequence[Callable]] = None,
        middleware: Optional[Sequence[Callable]] = None,
        force: bool = False,
    ):
        """Refresh the LangGraph agent, preserving user-scoped MCP tools.

        Upstream :meth:`BaseReActAgent._prepare_agent_inputs` calls
        :meth:`refresh_agent` once per chat turn with ``extra_tools`` set to
        only :attr:`_vector_tools`. Without this override, that second
        refresh would silently drop the per-turn user-scoped MCP tools that
        the chat-app already attached via the first
        :meth:`refresh_agent(extra_tools=user_scoped_tools)` call. We
        re-merge ``self._user_scoped_tools_cache`` (populated by
        :meth:`build_user_scoped_mcp_tools`) on every refresh so the
        authenticated user keeps their CRAB tools for the entire turn.

        Dedup is by Python object identity, so the chat-app's explicit
        ``extra_tools=user_scoped_tools`` argument and our cache do not
        double-up when both reference the same tool callables.
        """
        cached = list(getattr(self, "_user_scoped_tools_cache", []) or [])
        incoming = list(extra_tools or [])
        seen = {id(t) for t in incoming}
        merged_extra = incoming + [t for t in cached if id(t) not in seen]
        return super().refresh_agent(
            static_tools=static_tools,
            extra_tools=merged_extra or None,
            middleware=middleware,
            force=force,
        )

    def build_user_scoped_mcp_tools(
        self,
        access_token: Optional[str],
        token_getter: Optional[Callable[[], Optional[str]]] = None,
    ) -> List[Callable]:
        """Build the list of user-scoped MCP tools for the current chat turn.

        Returns an empty list when:
        - no user-scoped MCP servers are configured, or
        - no access token is available (anonymous user, expired session, etc.)

        Each call constructs a *fresh* :class:`MultiServerMCPClient` whose
        per-server ``headers`` carry the user's bearer token. The client is
        retained on ``self._user_scoped_mcp_client`` for the rest of the chat
        turn so that after ``tools/list`` the adapter's streamable-http DELETE
        does not orphan ``tools/call`` — subsequent invocations still see the
        same authenticated :class:`MultiServerMCPClient` instance.

        ``token_getter`` is invoked on the MCP background thread for every
        tool call; it must return a valid access token without using Flask
        ``session`` (capture ``sso_sid`` in a closure from the HTTP worker).
        When omitted, the initial ``access_token`` string is reused for every
        call.

        The tool callables themselves are still short-lived: they are passed to
        ``refresh_agent(extra_tools=...)`` and discarded when the agent is
        rebuilt; the client is explicitly disposed on the next
        :meth:`build_user_scoped_mcp_tools` call or when the token disappears.
        """
        if not self._user_scoped_mcp_servers or not access_token:
            self._dispose_user_scoped_mcp_client()
            return []

        try:
            self._dispose_user_scoped_mcp_client()
            runner = AsyncLoopThread.get_instance()
            effective_getter = token_getter or (lambda: access_token)
            servers_with_auth = _attach_bearer(
                self._user_scoped_mcp_servers,
                access_token,
                token_getter=effective_getter,
            )
            client, tools = runner.run(
                _initialize_mcp_client(
                    servers_with_auth,
                    tool_interceptors=[
                        _make_bearer_interceptor(
                            effective_getter,
                            access_token_fallback=access_token,
                        )
                    ],
                )
            )
            if not tools:
                _dispose_mcp_client_instance(client, runner)
                return []
            self._user_scoped_mcp_client = client
            sync_tools = [self._make_synchronous(t) for t in tools]
            # Snapshot for refresh_agent() so the upstream
            # _prepare_agent_inputs re-refresh cannot strip these tools out
            # by passing only ``self._vector_tools`` as extra_tools.
            self._user_scoped_tools_cache = list(sync_tools)
            return sync_tools
        except Exception as exc:
            logger.warning(
                "Failed to build user-scoped MCP tools: %s", exc, exc_info=True
            )
            return []

    def prefetch_user_task_summaries(
        self,
        access_token: Optional[str],
        on_behalf_of_username: Optional[str] = None,
        days: int = 30,
        token_getter: Optional[Callable[[], Optional[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Run ``list_tasks`` headlessly and return UI-ready row summaries.

        Used by the chat sidebar pre-fetch endpoint so the frontend can show
        each authenticated CRAB user's recent tasks (with green / red /
        bold-grey status spans) without having to spin up the LLM agent.

        Returns a list of
        ``{"crab_id", "status", "color", "summary"}`` dicts. The list is
        always returnable — every failure path (no token, no servers, MCP
        error, impersonation rejected by CRABServer) collapses to ``[]`` so
        the frontend boot path can stay simple.

        ``on_behalf_of_username`` is currently informational only: the
        upstream ``crab-mcp-server`` derives the CRAB username from the
        forwarded JWT (``list_tasks`` accepts only ``days``). We still log
        it for traceability and refuse to call out for an empty / missing
        token, but we do not pass it to the MCP tool.

        This deliberately runs on its **own** ephemeral
        :class:`MultiServerMCPClient` and does not touch
        ``self._user_scoped_mcp_client`` so it can race safely against an
        in-flight chat turn.
        """
        if not (self._user_scoped_mcp_servers and access_token):
            return []

        runner = AsyncLoopThread.get_instance()
        effective_getter = token_getter or (lambda: access_token)
        servers_with_auth = _attach_bearer(
            self._user_scoped_mcp_servers,
            access_token,
            token_getter=effective_getter,
        )
        client: Any = None
        try:
            client, tools = runner.run(
                _initialize_mcp_client(
                    servers_with_auth,
                    tool_interceptors=[
                        _make_bearer_interceptor(
                            effective_getter,
                            access_token_fallback=access_token,
                        )
                    ],
                )
            )
            if not tools:
                return []
            list_tool = next(
                (t for t in tools if getattr(t, "name", "") == "list_tasks"),
                None,
            )
            if list_tool is None:
                return []
            # ``list_tasks`` only takes ``days`` upstream — the username
            # comes from the JWT the bearer-auth httpx flow already
            # attached. Passing ``on_behalf_of_username`` here would raise
            # ``unexpected_keyword_argument`` from the MCP-side schema.
            raw = runner.run(list_tool.coroutine(days=days))
            # lcma>=0.2 StructuredTool with response_format="content_and_artifact"
            # returns (content_blocks, MCPToolArtifact) when invoked via
            # ``coroutine``. Prefer the structured artifact (already-parsed
            # rows from FastMCP), fall back to the text block payload, then
            # to the raw response so `_coerce_task_rows` can do its best.
            payload: Any = raw
            if isinstance(raw, tuple) and len(raw) == 2:
                content, artifact = raw
                if isinstance(artifact, dict) and isinstance(
                    artifact.get("structured_content"), dict
                ):
                    payload = artifact["structured_content"]
                elif isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict) and first.get("type") == "text":
                        payload = first.get("text", "")
                    else:
                        payload = first
                else:
                    payload = content
            summaries: List[Dict[str, Any]] = []
            for row in _coerce_task_rows(payload):
                crab_id = (
                    row.get("tm_taskname")
                    or row.get("name")
                    or row.get("workflow")
                    or row.get("crab_id")
                    or row.get("task_name")
                    or row.get("workflow_name")
                )
                if not crab_id:
                    continue
                status, color = _normalize_crab_status(
                    row.get("tm_task_status")
                    or row.get("task_status")
                    or row.get("status"),
                    row.get("DAGstatus") or row.get("dag_status"),
                )
                summaries.append(
                    {
                        "crab_id": str(crab_id),
                        "status": status,
                        "color": color,
                        "summary": _short_task_summary(row, status),
                    }
                )
            return summaries
        except Exception as exc:
            logger.warning(
                "CRAB prefetch (list_tasks) failed for %s: %s",
                on_behalf_of_username,
                exc,
                exc_info=True,
            )
            return []
        finally:
            _dispose_mcp_client_instance(client, runner)

    def _make_synchronous(self, async_tool):
        """Wrap an async MCP tool so it runs on the shared AsyncLoopThread.

        Identical to the wrapper in :meth:`BaseReActAgent._build_mcp_tools`,
        reproduced here so we don't depend on it being exposed.
        """
        runner = self._async_runner if hasattr(self, "_async_runner") else AsyncLoopThread.get_instance()

        def sync_wrapper(*args, **kwargs):
            if runner.in_loop_thread():
                raise RuntimeError(
                    "sync_wrapper called from MCP loop thread; would deadlock"
                )
            return runner.run(async_tool.coroutine(*args, **kwargs))

        async_tool.func = sync_wrapper
        return async_tool

    # ------------------------------------------------------------------
    # Vector retrievers (unchanged)
    # ------------------------------------------------------------------

    def _update_vector_retrievers(self, vectorstore: Any) -> None:
        """Instantiate or refresh the vectorstore retriever tool using hybrid retrieval."""
        if not self.enable_vector_tools:
            self._vector_retrievers = None
            self._vector_tools = None
            return

        retrievers_cfg = self.dm_config.get("retrievers", {})
        hybrid_cfg = retrievers_cfg.get("hybrid_retriever", {})

        k = hybrid_cfg.get("num_documents_to_retrieve", 5)
        bm25_weight = hybrid_cfg.get("bm25_weight", 0.6)
        semantic_weight = hybrid_cfg.get("semantic_weight", 0.4)

        hybrid_retriever = HybridRetriever(
            vectorstore=vectorstore,
            k=k,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
        )

        hybrid_description = self._tool_definitions()["search_vectorstore_hybrid"]["description"]

        self._vector_retrievers = [hybrid_retriever]
        self._vector_tools = [
            create_retriever_tool(
                hybrid_retriever,
                name="search_vectorstore_hybrid",
                description=hybrid_description,
                store_docs=self._store_documents,
                store_tool_input=getattr(self, "_store_tool_input", None),
            )
        ]


# ---------------------------------------------------------------------------
# Module-level helpers (kept out of the class so the same machinery serves
# both the static path and the per-turn user-scoped path).
# ---------------------------------------------------------------------------

def _dispose_mcp_client_instance(client: Any, runner: Any) -> None:
    """Best-effort async close for a MultiServerMCPClient (or similar)."""
    if client is None or runner is None:
        return

    async def _aclose() -> None:
        aclose = getattr(client, "aclose", None)
        if callable(aclose):
            await aclose()
            return
        close = getattr(client, "close", None)
        if callable(close):
            out = close()
            if asyncio.iscoroutine(out):
                await out

    try:
        runner.run(_aclose())
    except Exception as exc:
        logger.warning("MCP client close failed: %s", exc)


def _make_bearer_interceptor(
    token_getter: Callable[[], Optional[str]],
    *,
    access_token_fallback: Optional[str] = None,
):
    """Build an MCP tool interceptor that merges a fresh Bearer on each call.

    ``token_getter`` is invoked on the MCP async loop thread; it must not rely
    on Flask request context (capture ``sso_sid`` in a closure from the HTTP
    worker thread instead). ``access_token_fallback`` is the turn's initial
    token so ``Authorization`` is still set if the getter briefly returns nothing.
    """

    async def _inject_mcp_bearer_headers(
        request: MCPToolCallRequest,
        handler: CallableABC[[MCPToolCallRequest], Awaitable[Any]],
    ) -> Any:
        token = token_getter() or access_token_fallback
        logger.info(
            "MCP bearer interceptor fired tool=%s server=%s token_present=%s",
            getattr(request, "name", "?"),
            getattr(request, "server_name", "?"),
            bool(token),
        )
        if not token:
            return await handler(request)
        merged = {**(request.headers or {}), "Authorization": f"Bearer {token}"}
        return await handler(request.override(headers=merged))

    return _inject_mcp_bearer_headers


class _BearerHttpxAuth(httpx.Auth):
    """``httpx.Auth`` that stamps ``Authorization: Bearer <token>`` on every request.

    The interceptor in :func:`_make_bearer_interceptor` is not reliably
    invoked across lcma versions for ``tools/call`` (in our deployment it
    never fires — the merge log never appears), so bearer injection cannot
    depend on it. ``connection["headers"]`` only seeds the httpx client's
    default headers and is also fragile across the streamable-HTTP
    transport's per-request header construction.

    Attaching an ``httpx.Auth`` runs inside httpx's own request pipeline
    for *every* HTTP request the streamable-HTTP transport sends —
    initialize, tools/list, tools/call, the GET-stream subscription, and
    the DELETE that terminates the session — so the bearer is impossible
    to lose between successive POSTs on a per-call session.

    ``token_getter`` is consulted on every request (so SSO refreshes show
    up); ``fallback`` is the turn's initial token, used if the getter
    briefly returns nothing on the MCP loop thread.
    """

    requires_request_body = False
    requires_response_body = False

    def __init__(
        self,
        token_getter: Callable[[], Optional[str]],
        *,
        fallback: Optional[str] = None,
    ) -> None:
        self._token_getter = token_getter
        self._fallback = fallback

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        token: Optional[str] = None
        try:
            token = self._token_getter()
        except Exception as exc:  # noqa: BLE001 — never let token errors break a request
            logger.warning("MCP bearer token_getter raised: %s", exc)
        if not token:
            token = self._fallback
        if token:
            request.headers["Authorization"] = f"Bearer {token}"
        # One-shot diagnostic: prints once per outbound MCP HTTP request so we
        # can tell from chat-app logs whether httpx actually drove our Auth and
        # whether a token was on hand at the time. Never logs the token value.
        logger.info(
            "MCP httpx auth: %s %s token_present=%s auth_header_set=%s",
            request.method,
            request.url,
            bool(token),
            "Authorization" in request.headers,
        )
        yield request


def _attach_bearer(
    servers: Dict[str, Dict[str, Any]],
    access_token: str,
    *,
    token_getter: Optional[Callable[[], Optional[str]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return a deep-enough copy of ``servers`` with bearer auth attached.

    Two layers, defence in depth:

    1. ``connection["headers"]["Authorization"]`` — seeds the streamable-HTTP
       transport's request headers; covers the path where lcma's documented
       header merge actually fires.
    2. ``connection["auth"] = _BearerHttpxAuth(...)`` — runs inside httpx's
       per-request auth pipeline, so the bearer lands on every single
       request the underlying client sends, regardless of whether the lcma
       interceptor or header-merge runs.

    ``token_getter`` is what (2) calls on every request; when omitted it
    just returns ``access_token`` (fine for short-lived one-shot clients
    like the prefetch endpoint).
    """
    effective_getter: Callable[[], Optional[str]] = (
        token_getter if token_getter is not None else (lambda: access_token)
    )
    bearer_auth = _BearerHttpxAuth(effective_getter, fallback=access_token)

    out: Dict[str, Dict[str, Any]] = {}
    for name, cfg in servers.items():
        new_cfg = dict(cfg)
        headers = dict(new_cfg.get("headers") or {})
        headers["Authorization"] = f"Bearer {access_token}"
        new_cfg["headers"] = headers
        new_cfg["auth"] = bearer_auth
        out[name] = new_cfg
    return out


# ---------------------------------------------------------------------------
# CRAB status normalization (mirrors files/agents/cms-crab.md "Status
# normalization"). Kept in sync with PublisherMaster.getTaskStatusFromSched()
# / collapseDAGStatus(): DAGstatus is authoritative when present because the
# DB ``task_status`` is updated asynchronously by the TaskWorker and lags
# behind the schedd.
# ---------------------------------------------------------------------------

# DAG code → analyst-facing state word.
_DAG_CODE_TO_STATUS: Dict[int, str] = {
    0: "PENDING",
    1: "SUBMITTED",
    2: "SUBMITTED",
    3: "SUBMITTED",
    4: "SUBMITTED",
    5: "COMPLETED",
    6: "FAILED",
}

# Most-active wins when collapsing multi-subdag payloads.
_STATUS_PRIORITY: Dict[str, int] = {
    "PENDING": 0,
    "SUBMITTED": 1,
    "FAILED": 2,
    "COMPLETED": 3,
}

# CSS color hint for the chat span and the prefetch JSON payload.
_STATUS_COLOR: Dict[str, str] = {
    "PENDING": "grey",
    "SUBMITTED": "green",
    "RUNNING": "green",
    "COMPLETED": "grey",
    "FAILED": "red",
    "KILLED": "red",
}


def _normalize_crab_status(
    task_status: Optional[Any],
    dag_status: Optional[Any],
) -> Tuple[str, str]:
    """Return ``(status, color)`` per the agent's status-normalization rules.

    ``DAGstatus`` is preferred when present; for multi-subdag payloads the
    most-active state wins (PENDING → SUBMITTED → FAILED → COMPLETED). When
    no DAG status is available, fall back to ``task_status`` and apply the
    same color map.
    """
    status: Optional[str] = None
    if dag_status is not None:
        codes: List[Any] = (
            list(dag_status)
            if isinstance(dag_status, (list, tuple, set))
            else [dag_status]
        )
        translated: List[str] = []
        for code in codes:
            try:
                translated.append(_DAG_CODE_TO_STATUS[int(code)])
            except (TypeError, ValueError, KeyError):
                continue
        if translated:
            translated.sort(key=lambda s: _STATUS_PRIORITY.get(s, 99))
            status = translated[0]

    if status is None and task_status:
        candidate = str(task_status).strip().upper()
        if candidate in _STATUS_COLOR:
            status = candidate
        elif candidate in {"NEW", "QUEUED", "WAITING", "HOLDING"}:
            status = "PENDING"

    if status is None:
        status = "PENDING"
    return status, _STATUS_COLOR.get(status, "grey")


def _coerce_task_rows(payload: Any) -> List[Dict[str, Any]]:
    """Best-effort: turn whatever ``list_tasks`` returned into dict rows.

    The MCP server may serialize as a JSON string, a top-level list, or a
    dict wrapping the rows under ``tasks``/``result``/``items``/``data``.
    """
    if payload is None:
        return []
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = payload.decode("utf-8", errors="replace")
        except Exception:
            return []
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return []
    if isinstance(payload, dict):
        for key in ("tasks", "result", "items", "data", "rows"):
            inner = payload.get(key)
            if isinstance(inner, list):
                payload = inner
                break
        else:
            return [payload]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def _short_task_summary(row: Dict[str, Any], status: str) -> str:
    """One-line analyst-facing summary derived from a ``list_tasks`` row."""
    parts: List[str] = []
    done = (
        row.get("jobs_done")
        or row.get("nJobsDone")
        or row.get("nb_done")
    )
    total = (
        row.get("jobs_total")
        or row.get("nJobs")
        or row.get("nb_jobs")
        or row.get("total_jobs")
    )
    if done is not None and total is not None:
        parts.append(f"{done}/{total} jobs")
    submitted_at = (
        row.get("submitted_at")
        or row.get("submission_time")
        or row.get("start_time")
    )
    if submitted_at:
        parts.append(f"submitted {submitted_at}")
    # Empty when the row carries no extra info — the colored status pill on
    # the frontend already conveys the state, so a "submitted" summary right
    # next to a SUBMITTED pill is pure visual noise.
    return ", ".join(parts)


async def _initialize_mcp_client(
    mcp_servers: Dict[str, Any],
    *,
    tool_interceptors: Optional[List[Any]] = None,
):
    """Instantiate a MultiServerMCPClient and pull tools from each named server.

    Local copy of the upstream helper that takes an explicit ``mcp_servers``
    dict instead of reading from the global config, so we can supply either
    the static subset or a per-turn user-scoped subset with auth headers.

    ``tool_interceptors`` is forwarded for user-scoped clients so each
    ``tools/call`` can attach a freshly resolved Bearer token (see
    :func:`_make_bearer_interceptor`).
    """
    if not mcp_servers:
        return None, []
    logger.info("Configuring MCP client with servers: %s", list(mcp_servers.keys()))
    kwargs: Dict[str, Any] = {}
    if tool_interceptors:
        kwargs["tool_interceptors"] = tool_interceptors
    client = MultiServerMCPClient(mcp_servers, **kwargs)
    all_tools: List[Any] = []
    failed: Dict[str, str] = {}
    for name in mcp_servers:
        try:
            tools = await client.get_tools(server_name=name)
            for tool in tools:
                logger.info(
                    "Loaded tool from MCP server '%s': %s - %s",
                    name, tool.name, tool.description,
                )
            all_tools.extend(tools)
        except BaseException as exc:
            if type(exc).__name__ in ("ExceptionGroup", "BaseExceptionGroup"):
                sub = getattr(exc, "exceptions", ())
                msg = "; ".join(f"{type(s).__name__}: {s}" for s in sub)
                logger.error(
                    "Failed to fetch tools from MCP server '%s': %s",
                    name,
                    msg,
                    exc_info=True,
                )
                failed[name] = msg
            elif isinstance(exc, Exception):
                logger.error(
                    "Failed to fetch tools from MCP server '%s': %s", name, exc
                )
                failed[name] = str(exc)
            else:
                raise
    if failed:
        logger.warning("MCP servers failed to initialize: %s", list(failed.keys()))
    return client, all_tools

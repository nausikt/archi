"""
Unit tests for PostgreSQL services.

Tests cover:
- ConnectionPool
- UserService  
- ConfigService
- DocumentSelectionService
- ConversationService
- PostgresServiceFactory
"""
import os
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

# Import services
from src.utils.connection_pool import ConnectionPool, ConnectionPoolError, ConnectionTimeoutError
from src.utils.user_service import UserService, User
from src.utils.config_service import ConfigService, StaticConfig, DynamicConfig, ConfigValidationError
from src.utils.document_selection_service import DocumentSelectionService, DocumentSelection
from src.utils.conversation_service import ConversationService, Message, ABComparison
from src.utils.postgres_service_factory import PostgresServiceFactory, create_services
from src.utils.playbook_service import (
    PlaybookService, Playbook,
    PlaybookValidationError, PlaybookConflictError, PlaybookNotFoundError,
    resolve_playbook_owner,
)
from psycopg2 import errors as pg_errors


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_connection():
    """Create a mock psycopg2 connection."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cursor


@pytest.fixture
def mock_pool(mock_connection):
    """Create a mock connection pool."""
    conn, cursor = mock_connection
    pool = MagicMock(spec=ConnectionPool)
    pool.get_connection.return_value = conn
    pool.get_connection_direct.return_value = conn
    pool.release_connection = MagicMock()
    return pool


# =============================================================================
# ConnectionPool Tests
# =============================================================================

class TestConnectionPool:
    """Tests for ConnectionPool."""
    
    def test_init_requires_params_or_dsn(self):
        """Test that ConnectionPool requires connection info."""
        with pytest.raises(ValueError, match="Either pg_config or connection_params must be provided"):
            ConnectionPool()
    
    @patch('psycopg2.pool.ThreadedConnectionPool')
    def test_init_with_params(self, mock_tcp):
        """Test initialization with connection params."""
        params = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test',
            'user': 'user',
            'password': 'pass',
        }
        pool = ConnectionPool(connection_params=params)
        
        mock_tcp.assert_called_once()
        assert pool._pool is not None
    
    @patch('psycopg2.pool.ThreadedConnectionPool')
    def test_singleton_pattern(self, mock_tcp):
        """Test singleton pattern."""
        params = {'host': 'localhost', 'database': 'test', 'user': 'user', 'password': 'pass'}
        
        # Reset singleton
        ConnectionPool._instance = None
        
        pool1 = ConnectionPool.get_instance(connection_params=params)
        pool2 = ConnectionPool.get_instance()
        
        assert pool1 is pool2


# =============================================================================
# UserService Tests
# =============================================================================

class TestUserService:
    """Tests for UserService."""
    
    def test_get_or_create_user_creates_new(self, mock_pool, mock_connection):
        """Test creating a new user."""
        conn, cursor = mock_connection
        # First call (get_user check) returns None, second call (INSERT) returns dict
        cursor.fetchone.side_effect = [
            None,  # User doesn't exist on initial check
            {  # INSERT RETURNING result
                "id": "user123",
                "display_name": None,
                "email": None,
                "auth_provider": "anonymous",
                "theme": "system",
                "preferred_model": None,
                "preferred_temperature": None,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }
        ]
        
        service = UserService(connection_pool=mock_pool, encryption_key="test-key")
        user = service.get_or_create_user("user123", auth_provider="anonymous")
        
        assert user.id == "user123"
        assert user.auth_provider == "anonymous"
    
    def test_get_or_create_user_returns_existing(self, mock_pool, mock_connection):
        """Test returning existing user."""
        conn, cursor = mock_connection
        # Simulate existing user as dict
        cursor.fetchone.return_value = {
            "id": "user123",
            "display_name": "Test User",
            "email": "test@example.com",
            "auth_provider": "basic",
            "theme": "dark",
            "preferred_model": "gpt-4",
            "preferred_temperature": 0.7,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        
        service = UserService(connection_pool=mock_pool, encryption_key="test-key")
        user = service.get_or_create_user("user123")
        
        assert user.id == "user123"
        assert user.display_name == "Test User"
        assert user.theme == "dark"
    
    def test_update_preferences(self, mock_pool, mock_connection):
        """Test updating user preferences."""
        conn, cursor = mock_connection
        cursor.fetchone.return_value = {
            "id": "user123",
            "display_name": "Test User",
            "email": None,
            "auth_provider": "anonymous",
            "theme": "light",
            "preferred_model": "gpt-4o",
            "preferred_temperature": 0.5,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        
        service = UserService(connection_pool=mock_pool, encryption_key="test-key")
        user = service.update_preferences(
            user_id="user123",
            theme="light",
        )
        
        assert user.theme == "light"


# =============================================================================
# ConfigService Tests
# =============================================================================

class TestConfigService:
    """Tests for ConfigService."""
    
    def test_get_static_config(self, mock_pool, mock_connection):
        """Test getting static config."""
        conn, cursor = mock_connection
        cursor.fetchone.return_value = {
            "deployment_name": "test-deployment",
            "config_version": "2.0.0",
            "data_path": "/data",
            "prompts_path": "/prompts",
            "embedding_model": "text-embedding-ada-002",
            "embedding_dimensions": 1536,
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "distance_metric": "cosine",
            "available_pipelines": ["QAPipeline"],
            "available_models": ["gpt-4"],
            "available_providers": ["openai"],
            "auth_enabled": False,
            "session_lifetime_days": 30,
            "created_at": datetime.now(),
        }
        
        service = ConfigService(connection_pool=mock_pool)
        config = service.get_static_config()
        
        assert config.deployment_name == "test-deployment"
        assert config.embedding_dimensions == 1536
        assert "QAPipeline" in config.available_pipelines
    
    def test_get_static_config_caching(self, mock_pool, mock_connection):
        """Test that static config is cached."""
        conn, cursor = mock_connection
        cursor.fetchone.return_value = {
            "deployment_name": "test",
            "config_version": "2.0.0",
            "data_path": "/data",
            "prompts_path": "/prompts",
            "embedding_model": "model",
            "embedding_dimensions": 384,
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "distance_metric": "cosine",
            "available_pipelines": [],
            "available_models": [],
            "available_providers": [],
            "auth_enabled": False,
            "session_lifetime_days": 30,
            "created_at": datetime.now(),
        }
        
        service = ConfigService(connection_pool=mock_pool)
        
        # First call
        config1 = service.get_static_config()
        calls_after_first = mock_pool.get_connection_direct.call_count
        
        # Second call should use cache
        config2 = service.get_static_config()
        calls_after_second = mock_pool.get_connection_direct.call_count
        
        assert config1 is config2
        # Second call should not make additional DB calls (cached)
        assert calls_after_second == calls_after_first
    
    def test_update_dynamic_config_validation(self, mock_pool, mock_connection):
        """Test dynamic config validation."""
        conn, cursor = mock_connection
        
        service = ConfigService(connection_pool=mock_pool)
        
        with pytest.raises(ConfigValidationError, match="temperature"):
            service.update_dynamic_config(temperature=5.0)
        
        with pytest.raises(ConfigValidationError, match="max_tokens"):
            service.update_dynamic_config(max_tokens=-1)


# =============================================================================
# DocumentSelectionService Tests
# =============================================================================

class TestDocumentSelectionService:
    """Tests for DocumentSelectionService."""
    
    def test_get_enabled_document_ids(self, mock_pool, mock_connection):
        """Test getting enabled document IDs."""
        conn, cursor = mock_connection
        cursor.fetchall.return_value = [(1,), (2,), (5,)]
        
        service = DocumentSelectionService(connection_pool=mock_pool)
        doc_ids = service.get_enabled_document_ids(
            user_id="user123",
            conversation_id="conv42",
        )
        
        # Returns a set of IDs
        assert doc_ids == {1, 2, 5}
    
    def test_set_user_default(self, mock_pool, mock_connection):
        """Test setting user default."""
        conn, cursor = mock_connection
        
        service = DocumentSelectionService(connection_pool=mock_pool)
        service.set_user_default(
            user_id="user123",
            document_id=10,
            enabled=False,
        )
        
        # Verify UPSERT was called
        conn.commit.assert_called()
    
    def test_3tier_precedence_query(self, mock_pool, mock_connection):
        """Test that the 3-tier precedence is in the query."""
        conn, cursor = mock_connection
        cursor.fetchall.return_value = []
        
        service = DocumentSelectionService(connection_pool=mock_pool)
        service.get_enabled_document_ids("user", 1)
        
        # Check that the query includes COALESCE for precedence
        call_args = cursor.execute.call_args
        query = call_args[0][0]
        assert "COALESCE" in query


# =============================================================================
# ConversationService Tests
# =============================================================================

class TestConversationService:
    """Tests for ConversationService."""
    
    def test_insert_message(self, mock_pool, mock_connection):
        """Test inserting a message."""
        conn, cursor = mock_connection
        
        # Mock execute_values return
        with patch('src.utils.conversation_service.execute_values') as mock_exec:
            mock_exec.return_value = None
            cursor.fetchall.return_value = [(1,)]
            mock_exec.return_value = [(1,)]
            
            service = ConversationService(connection_pool=mock_pool)
            
            msg = Message(
                conversation_id="conv123",
                sender="user",
                content="Hello",
                model_used="gpt-4",
                pipeline_used="QAPipeline",
            )
            
            # The service calls execute_values which returns IDs
            with patch.object(service, 'insert_messages', return_value=[1]):
                msg_id = service.insert_message(msg)
                assert msg_id == 1
    
    def test_create_ab_comparison(self, mock_pool, mock_connection):
        """Test creating A/B comparison."""
        conn, cursor = mock_connection
        cursor.fetchone.return_value = (42,)  # comparison_id
        
        service = ConversationService(connection_pool=mock_pool)
        comparison_id = service.create_ab_comparison(
            conversation_id="conv123",
            user_prompt_mid=1,
            response_a_mid=2,
            response_b_mid=3,
            model_a="gpt-4",
            pipeline_a="QAPipeline",
            model_b="claude-3",
            pipeline_b="QAPipeline",
        )
        
        assert comparison_id == 42
    
    def test_record_ab_preference_validation(self, mock_pool, mock_connection):
        """Test preference validation."""
        service = ConversationService(connection_pool=mock_pool)
        
        with pytest.raises(ValueError, match="Invalid preference"):
            service.record_ab_preference(1, "invalid")


# =============================================================================
# PostgresServiceFactory Tests
# =============================================================================

class TestPostgresServiceFactory:
    """Tests for PostgresServiceFactory."""
    
    @patch('src.utils.postgres_service_factory.ConnectionPool')
    def test_from_config(self, mock_pool_class):
        """Test factory creation from config."""
        factory = PostgresServiceFactory.from_config(
            connection_params={
                'host': 'localhost',
                'database': 'test',
                'user': 'user',
                'password': 'pass',
            }
        )
        
        assert factory is not None
        mock_pool_class.assert_called_once()
    
    @patch('src.utils.postgres_service_factory.ConnectionPool')
    def test_lazy_service_initialization(self, mock_pool_class):
        """Test that services are lazy-initialized."""
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool
        
        factory = PostgresServiceFactory.from_config(
            connection_params={'host': 'localhost', 'database': 'test', 'user': 'u', 'password': 'p'}
        )
        
        # Services should not be created yet
        assert factory._user_service is None
        assert factory._config_service is None
        
        # Access services
        _ = factory.user_service
        _ = factory.config_service
        
        # Now they should exist
        assert factory._user_service is not None
        assert factory._config_service is not None
    
    @patch('src.utils.postgres_service_factory.ConnectionPool')
    def test_context_manager(self, mock_pool_class):
        """Test context manager cleanup."""
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool
        
        with PostgresServiceFactory.from_config(
            connection_params={'host': 'localhost', 'database': 'test', 'user': 'u', 'password': 'p'}
        ) as factory:
            _ = factory.user_service
        
        # Pool should be closed
        mock_pool.close.assert_called_once()
    
    def test_from_yaml_config_deprecated(self):
        """from_yaml_config should still parse postgres settings for ingest."""
        config = {
            'database': {
                'postgres': {
                    'host': 'db.example.com',
                    'port': 5433,
                    'database': 'archi',
                    'user': 'app',
                    'password': 'secret',
                    'pool': {
                        'min_connections': 2,
                        'max_connections': 10,
                    }
                }
            }
        }

        with patch('src.utils.postgres_service_factory.ConnectionPool') as mock_pool_class:
            factory = PostgresServiceFactory.from_yaml_config(config)

            # Verify connection params were extracted correctly
            call_kwargs = mock_pool_class.call_args[1]
            assert call_kwargs['connection_params']['host'] == 'db.example.com'
            assert call_kwargs['connection_params']['port'] == 5433
            assert call_kwargs['min_conn'] == 2
            assert call_kwargs['max_conn'] == 10

    def test_playbook_service_lazy_init(self, mock_pool):
        factory = PostgresServiceFactory(connection_pool=mock_pool)
        assert factory._playbook_service is None
        svc = factory.playbook_service
        assert isinstance(svc, PlaybookService)
        assert factory._playbook_service is svc  # cached


# =============================================================================
# Integration-style Tests (with mocked DB)
# =============================================================================

class TestServiceIntegration:
    """Integration tests for service interactions."""
    
    @patch('src.utils.postgres_service_factory.ConnectionPool')
    def test_user_document_selection_flow(self, mock_pool_class):
        """Test user setting document defaults."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        mock_pool.get_connection.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool_class.return_value = mock_pool
        
        factory = PostgresServiceFactory.from_config(
            connection_params={'host': 'localhost', 'database': 'test', 'user': 'u', 'password': 'p'}
        )
        
        # Set user document default
        factory.document_selection_service.set_user_default(
            user_id="user123",
            document_id=10,
            enabled=False,
        )
        
        # Verify commit was called
        mock_conn.commit.assert_called()


# =============================================================================
# Message/ABComparison Dataclass Tests
# =============================================================================

class TestDataclasses:
    """Tests for dataclass structures."""
    
    def test_message_defaults(self):
        """Test Message default values."""
        msg = Message()
        
        assert msg.message_id is None
        assert msg.conversation_id == ""
        assert msg.sender == ""
        assert msg.archi_service == "chat"
    
    def test_ab_comparison_defaults(self):
        """Test ABComparison default values."""
        ab = ABComparison()
        
        assert ab.comparison_id is None
        assert ab.is_config_a_first is True
        assert ab.preference is None
    
    def test_document_selection_repr(self):
        """Test DocumentSelection representation."""
        ds = DocumentSelection(
            document_id=1,
            resource_hash="abc123",
            display_name="Test Doc",
            source_type="file",
            user_default=False,
            conversation_override=True,
        )
        
        assert ds.document_id == 1
        assert ds.enabled is True  # conversation_override takes precedence


class TestPlaybookService:
    def test_playbook_dataclass_defaults(self):
        playbook = Playbook(id=1, name="rucio-triage", description="d", body="b", owner_id="c1")
        assert playbook.created_at is None
        assert playbook.updated_at is None

    def test_validate_rejects_bad_name(self):
        service = PlaybookService(connection_pool=MagicMock())
        with pytest.raises(PlaybookValidationError, match="lowercase"):
            service.create_playbook("c1", "Bad Name!", "desc", "body")

    def test_validate_rejects_consecutive_hyphens(self):
        # Agent Skills spec: no leading/trailing/consecutive hyphens.
        service = PlaybookService(connection_pool=MagicMock())
        for bad in ("a--b", "-ab", "ab-"):
            with pytest.raises(PlaybookValidationError, match="hyphens"):
                service.create_playbook("c1", bad, "desc", "body")

    def test_validate_rejects_empty_description(self):
        service = PlaybookService(connection_pool=MagicMock())
        with pytest.raises(PlaybookValidationError, match="description"):
            service.create_playbook("c1", "ok-name", "  ", "body")

    def test_validate_rejects_oversized_body(self):
        service = PlaybookService(connection_pool=MagicMock())
        with pytest.raises(PlaybookValidationError, match="exceeds"):
            service.create_playbook("c1", "ok-name", "desc", "x" * 16385)

    def test_validate_rejects_oversized_description(self):
        # the Agent Skills spec limit is 1024 chars
        service = PlaybookService(connection_pool=MagicMock())
        with pytest.raises(PlaybookValidationError, match="description exceeds"):
            service.create_playbook("c1", "ok-name", "d" * 1025, "body")

    def test_validate_accepts_spec_sized_description(self):
        # 1024 chars is valid under the spec; only the DB call should be reached.
        pool = MagicMock()
        service = PlaybookService(connection_pool=pool)
        try:
            service.create_playbook("c1", "ok-name", "d" * 1024, "body")
        except PlaybookValidationError as exc:  # pragma: no cover - regression guard
            pytest.fail(f"1024-char description rejected: {exc}")
        except Exception:
            pass  # mocked-DB fallout is fine; validation passed

    def test_validate_rejects_multiline_description(self):
        # public descriptions render into OTHER users' system prompts: a newline could
        # forge extra listing lines there
        service = PlaybookService(connection_pool=MagicMock())
        for bad in ("line1\nline2", "tab\there", "bell\x07"):
            with pytest.raises(PlaybookValidationError, match="single line"):
                service.create_playbook("c1", "ok-name", bad, "body")

    def test_list_playbooks_without_bodies_skips_body_column(self):
        pool = MagicMock()
        conn = pool.get_connection_direct.return_value
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = []
        PlaybookService(connection_pool=pool).list_playbooks("c1", with_bodies=False)
        sql = cursor.execute.call_args[0][0]
        assert "'' AS body" in sql

    def test_validate_rejects_bad_visibility(self):
        service = PlaybookService(connection_pool=MagicMock())
        with pytest.raises(PlaybookValidationError, match="visibility"):
            service.create_playbook("c1", "ok-name", "desc", "body", visibility="everyone")

    def test_create_playbook_forwards_visibility(self, mock_pool, mock_connection):
        conn, cursor = mock_connection
        cursor.fetchone.side_effect = [{"n": 0}, {
            "id": 7, "name": "shared-run", "description": "d", "body": "b",
            "owner_id": "c1", "visibility": "public",
            "created_at": None, "updated_at": None,
        }]
        service = PlaybookService(connection_pool=mock_pool)
        playbook = service.create_playbook("c1", "shared-run", "d", "b", visibility="public")
        assert playbook.visibility == "public"
        sql, params = cursor.execute.call_args[0]  # last execute = the INSERT
        assert "visibility" in sql and "public" in params

    def test_list_playbooks_includes_public_rows(self, mock_pool, mock_connection):
        conn, cursor = mock_connection
        cursor.fetchall.return_value = []
        PlaybookService(connection_pool=mock_pool).list_playbooks("c1")
        sql, params = cursor.execute.call_args[0]
        # own OR public filter, plus the own-first ordering param
        assert "visibility = 'public'" in sql
        assert params.count("c1") == 2

    def test_get_playbook_by_name_public_lookup_prefers_own(self, mock_pool, mock_connection):
        conn, cursor = mock_connection
        cursor.fetchone.return_value = {
            "id": 1, "name": "a", "description": "d", "body": "b",
            "owner_id": "c1", "visibility": "public", "created_at": None, "updated_at": None,
        }
        PlaybookService(connection_pool=mock_pool).get_playbook_by_name("c1", "a", include_public=True)
        sql, params = cursor.execute.call_args[0]
        assert "visibility = 'public'" in sql
        assert "ORDER BY (owner_id = %s) DESC" in sql

    def test_create_playbook_returns_playbook(self, mock_pool, mock_connection):
        conn, cursor = mock_connection
        # first fetchone serves the per-owner count check, second the INSERT .. RETURNING row
        cursor.fetchone.side_effect = [{"n": 0}, {
            "id": 7, "name": "rucio-triage", "description": "triage transfers",
            "body": "step 1...", "owner_id": "c1",
            "created_at": datetime.now(), "updated_at": datetime.now(),
        }]
        service = PlaybookService(connection_pool=mock_pool)
        playbook = service.create_playbook("c1", "rucio-triage", "triage transfers", "step 1...")
        assert playbook.id == 7
        assert playbook.name == "rucio-triage"
        conn.commit.assert_called()

    def test_create_playbook_duplicate_raises_conflict(self, mock_pool, mock_connection):
        conn, cursor = mock_connection
        cursor.fetchone.return_value = {"n": 0}
        # count query succeeds; the INSERT itself hits the unique index
        cursor.execute.side_effect = [None, pg_errors.UniqueViolation()]
        service = PlaybookService(connection_pool=mock_pool)
        with pytest.raises(PlaybookConflictError, match="already exists"):
            service.create_playbook("c1", "dupe-name", "desc", "body")
        conn.rollback.assert_called()

    def test_create_playbook_rejects_at_owner_cap(self, mock_pool, mock_connection):
        conn, cursor = mock_connection
        cursor.fetchone.return_value = {"n": 100}
        service = PlaybookService(connection_pool=mock_pool)
        with pytest.raises(PlaybookValidationError, match="limit reached"):
            service.create_playbook("c1", "one-too-many", "desc", "body")
        # only the count query ran — the INSERT was never attempted
        assert cursor.execute.call_count == 1
        conn.commit.assert_not_called()

    def test_list_playbooks_returns_list(self, mock_pool, mock_connection):
        conn, cursor = mock_connection
        cursor.fetchall.return_value = [
            {"id": 1, "name": "a", "description": "da", "body": "ba",
             "owner_id": "c1", "created_at": None, "updated_at": None},
            {"id": 2, "name": "b", "description": "db", "body": "bb",
             "owner_id": "c1", "created_at": None, "updated_at": None},
        ]
        service = PlaybookService(connection_pool=mock_pool)
        playbooks = service.list_playbooks("c1")
        assert [s.name for s in playbooks] == ["a", "b"]

    def test_get_playbook_found(self, mock_pool, mock_connection):
        conn, cursor = mock_connection
        cursor.fetchone.return_value = {
            "id": 3, "name": "c", "description": "dc", "body": "bc",
            "owner_id": "c1", "created_at": None, "updated_at": None,
        }
        service = PlaybookService(connection_pool=mock_pool)
        assert service.get_playbook("c1", 3).name == "c"

    def test_get_playbook_not_found_raises(self, mock_pool, mock_connection):
        conn, cursor = mock_connection
        cursor.fetchone.return_value = None
        service = PlaybookService(connection_pool=mock_pool)
        with pytest.raises(PlaybookNotFoundError):
            service.get_playbook("c1", 999)

    def test_get_playbook_by_name_not_found_raises(self, mock_pool, mock_connection):
        conn, cursor = mock_connection
        cursor.fetchone.return_value = None
        service = PlaybookService(connection_pool=mock_pool)
        with pytest.raises(PlaybookNotFoundError):
            service.get_playbook_by_name("c1", "nope")

    def test_update_playbook_commits(self, mock_pool, mock_connection):
        conn, cursor = mock_connection
        existing = {"id": 4, "name": "old", "description": "d", "body": "b",
                    "owner_id": "c1", "created_at": None, "updated_at": None}
        updated = {**existing, "name": "new"}
        cursor.fetchone.side_effect = [existing, updated]  # get_playbook, then UPDATE RETURNING
        service = PlaybookService(connection_pool=mock_pool)
        result = service.update_playbook("c1", 4, name="new")
        assert result.name == "new"
        conn.commit.assert_called()

    def test_update_playbook_not_found_raises(self, mock_pool, mock_connection):
        conn, cursor = mock_connection
        cursor.fetchone.return_value = None  # get_playbook finds nothing
        service = PlaybookService(connection_pool=mock_pool)
        with pytest.raises(PlaybookNotFoundError):
            service.update_playbook("c1", 4, name="new")

    def test_delete_playbook_removes_row(self, mock_pool, mock_connection):
        conn, cursor = mock_connection
        cursor.rowcount = 1
        service = PlaybookService(connection_pool=mock_pool)
        service.delete_playbook("c1", 4)
        conn.commit.assert_called()

    # ── IDOR invariant: every single-owner query is owner-scoped in SQL + params ──

    def test_read_queries_are_owner_scoped(self, mock_pool, mock_connection):
        conn, cursor = mock_connection
        service = PlaybookService(connection_pool=mock_pool)
        row = {"id": 1, "name": "a", "description": "d", "body": "b",
               "owner_id": "c1", "created_at": None, "updated_at": None}
        cursor.fetchall.return_value = []
        service.list_playbooks("c1")
        sql, params = cursor.execute.call_args[0]
        assert "owner_id = %s" in sql and "c1" in params
        cursor.fetchone.return_value = row
        service.get_playbook("c1", 1)
        sql, params = cursor.execute.call_args[0]
        assert "owner_id = %s" in sql and "c1" in params and 1 in params
        service.get_playbook_by_name("c1", "a")
        sql, params = cursor.execute.call_args[0]
        assert "owner_id = %s" in sql and "c1" in params

    def test_delete_is_owner_scoped(self, mock_pool, mock_connection):
        conn, cursor = mock_connection
        cursor.rowcount = 1
        PlaybookService(connection_pool=mock_pool).delete_playbook("c1", 7)
        sql, params = cursor.execute.call_args[0]
        assert "owner_id = %s" in sql and "c1" in params and 7 in params

    def test_update_is_owner_scoped(self, mock_pool, mock_connection):
        conn, cursor = mock_connection
        existing = {"id": 4, "name": "old", "description": "d", "body": "b",
                    "owner_id": "c1", "created_at": None, "updated_at": None}
        cursor.fetchone.side_effect = [existing, {**existing, "name": "new"}]
        PlaybookService(connection_pool=mock_pool).update_playbook("c1", 4, name="new")
        sql, params = cursor.execute.call_args[0]  # last execute = the UPDATE
        assert "owner_id = %s" in sql and "c1" in params and 4 in params

    def test_delete_playbook_not_found_raises(self, mock_pool, mock_connection):
        conn, cursor = mock_connection
        cursor.rowcount = 0
        service = PlaybookService(connection_pool=mock_pool)
        with pytest.raises(PlaybookNotFoundError):
            service.delete_playbook("c1", 999)

    def test_update_playbook_conflict_raises(self, mock_pool, mock_connection):
        conn, cursor = mock_connection
        existing = {"id": 4, "name": "old", "description": "d", "body": "b",
                    "owner_id": "c1", "created_at": None, "updated_at": None}
        cursor.fetchone.side_effect = [existing]            # get_playbook succeeds
        cursor.execute.side_effect = [None, pg_errors.UniqueViolation()]  # SELECT ok, UPDATE conflicts
        service = PlaybookService(connection_pool=mock_pool)
        with pytest.raises(PlaybookConflictError, match="already exists"):
            service.update_playbook("c1", 4, name="other-existing")
        conn.rollback.assert_called()

    def test_playbook_invocation_text_uses_command_block(self):
        from src.utils.playbook_service import playbook_invocation_text
        out = playbook_invocation_text("do the thing", "deploy-checklist", "PLAYBOOK BODY")
        # Claude Code slash-command expansion: command tags carry the invocation,
        # the body follows; without $ARGUMENTS the text is appended as ARGUMENTS:.
        assert out.startswith("<command-message>deploy-checklist is running…</command-message>")
        assert "<command-name>/deploy-checklist</command-name>" in out
        assert "<command-args>do the thing</command-args>" in out
        assert "PLAYBOOK BODY" in out
        assert "ARGUMENTS: do the thing" in out

    def test_playbook_invocation_text_substitutes_arguments(self):
        from src.utils.playbook_service import playbook_invocation_text
        out = playbook_invocation_text("T2_US_MIT", "site-check", "inspect $ARGUMENTS closely")
        assert "inspect T2_US_MIT closely" in out
        assert "ARGUMENTS:" not in out  # placeholder consumed the args

    def test_playbook_invocation_text_fences_foreign_body(self):
        from src.utils.playbook_service import playbook_invocation_text
        out = playbook_invocation_text("go", "theirs", "BODY", foreign=True)
        assert "Public playbook shared by another user" in out

    def test_playbook_invocation_text_empty_body_unchanged(self):
        from src.utils.playbook_service import playbook_invocation_text
        assert playbook_invocation_text("hi", "x", "") == "hi"

    def test_render_and_parse_playbook_md_round_trip(self):
        from src.utils.playbook_service import render_playbook_md, parse_playbook_md
        md = render_playbook_md("rucio-triage", "Triage stuck transfers. Use when…", "Step 1\nStep 2", "public")
        parsed = parse_playbook_md(md)
        assert parsed == {
            "name": "rucio-triage",
            "description": "Triage stuck transfers. Use when…",
            "body": "Step 1\nStep 2",
            "visibility": "public",
        }

    def test_parse_playbook_md_defaults_and_fallbacks(self):
        from src.utils.playbook_service import parse_playbook_md
        md = "---\ndescription: d\n---\n\nBODY\n"
        parsed = parse_playbook_md(md, fallback_name="from-folder")
        assert parsed["name"] == "from-folder"
        assert parsed["visibility"] == "private"
        # unknown frontmatter keys are tolerated (spec allows extras)
        md2 = "---\nname: a\ndescription: d\nlicense: MIT\n---\nB"
        assert parse_playbook_md(md2)["name"] == "a"

    def test_parse_playbook_md_rejects_missing_frontmatter(self):
        from src.utils.playbook_service import parse_playbook_md
        with pytest.raises(PlaybookValidationError, match="frontmatter"):
            parse_playbook_md("just a body, no frontmatter")

    def test_parse_playbook_md_ignores_indented_fence(self):
        # an indented '---' is YAML content (block-scalar continuation), not a fence
        from src.utils.playbook_service import parse_playbook_md
        md = "---\nname: a\ndescription: |\n  part one\n  ---\n  part two\n---\nBODY"
        parsed = parse_playbook_md(md)
        assert parsed["name"] == "a"
        assert "part two" in parsed["description"]
        assert parsed["body"] == "BODY"

    def test_pending_playbook_contextvar_roundtrip(self):
        from src.archi.pipelines.agents.tools.playbook_tools import (
            set_pending_playbook, get_pending_playbook, clear_pending_playbook,
        )
        clear_pending_playbook()
        assert get_pending_playbook() is None
        set_pending_playbook("deploy-checklist", "BODY")
        assert get_pending_playbook() == {"name": "deploy-checklist", "body": "BODY", "foreign": False}
        set_pending_playbook("public-runbook", "B", foreign=True)
        assert get_pending_playbook()["foreign"] is True
        clear_pending_playbook()
        assert get_pending_playbook() is None

    def test_get_connection_uses_direct_accessor_with_pool(self):
        # ConnectionPool.get_connection() is a @contextmanager; PlaybookService manages
        # the conn manually, so it must use the raw accessor get_connection_direct().
        pool = MagicMock()
        svc = PlaybookService(connection_pool=pool)
        svc._get_connection()
        pool.get_connection_direct.assert_called_once()
        pool.get_connection.assert_not_called()


# =============================================================================
# TestResolvePlaybookOwner Tests
# =============================================================================

class TestResolvePlaybookOwner:
    """Tests for resolve_playbook_owner — session-identity guard for playbook IDOR mitigation."""

    def test_authed_logged_in_email_returns_email(self):
        """When auth is on and user is logged in with email, return the email."""
        owner, err = resolve_playbook_owner(
            auth_enabled=True,
            logged_in=True,
            session_user={"email": "alice@example.com", "name": "Alice"},
            request_client_id="some-uuid-from-frontend",
        )
        assert owner == "alice@example.com"
        assert err is None

    def test_authed_logged_in_different_client_id_ignored_not_rejected(self):
        """When auth on and logged in with email, a different request client_id is IGNORED.

        This is the IDOR fix: the frontend legitimately sends a UUID client_id that
        never equals the SSO email — rejecting it would break authed requests.
        The server-verified identity wins and the supplied client_id is silently ignored.
        """
        owner, err = resolve_playbook_owner(
            auth_enabled=True,
            logged_in=True,
            session_user={"email": "alice@example.com"},
            request_client_id="attacker-or-unrelated-uuid",
        )
        # Must return the session email, not the request client_id, and no error
        assert owner == "alice@example.com"
        assert err is None
        assert owner != "attacker-or-unrelated-uuid"

    def test_authed_logged_in_no_email_falls_back_to_sub(self):
        """When logged in but no email, use sub as verified identity."""
        owner, err = resolve_playbook_owner(
            auth_enabled=True,
            logged_in=True,
            session_user={"sub": "sub|12345"},
            request_client_id="frontend-uuid",
        )
        assert owner == "sub|12345"
        assert err is None

    def test_authed_logged_in_no_email_or_sub_falls_back_to_name(self):
        """When logged in but no email/sub, use name as verified identity."""
        owner, err = resolve_playbook_owner(
            auth_enabled=True,
            logged_in=True,
            session_user={"name": "Bob"},
            request_client_id="frontend-uuid",
        )
        assert owner == "Bob"
        assert err is None

    def test_authed_logged_in_empty_session_user_fails_closed(self):
        """Logged in but session_user has no usable identity -> fail closed, do NOT trust client_id."""
        owner, err = resolve_playbook_owner(
            auth_enabled=True,
            logged_in=True,
            session_user={},
            request_client_id="frontend-uuid",
        )
        assert owner is None   # IDOR not re-opened
        assert err             # error returned

    def test_authed_not_logged_in_uses_request_client_id(self):
        """Auth enabled but user not logged in → anonymous, use request client_id."""
        owner, err = resolve_playbook_owner(
            auth_enabled=True,
            logged_in=False,
            session_user=None,
            request_client_id="anon-uuid",
        )
        assert owner == "anon-uuid"
        assert err is None

    def test_auth_disabled_uses_request_client_id(self):
        """Auth disabled (anonymous deployment) → always use request client_id."""
        owner, err = resolve_playbook_owner(
            auth_enabled=False,
            logged_in=False,
            session_user=None,
            request_client_id="anon-uuid",
        )
        assert owner == "anon-uuid"
        assert err is None

    def test_anon_no_client_id_returns_error(self):
        """Anonymous + no client_id → rejectable error."""
        owner, err = resolve_playbook_owner(
            auth_enabled=False,
            logged_in=False,
            session_user=None,
            request_client_id=None,
        )
        assert owner is None
        assert err == "client_id is required"

    def test_authed_not_logged_in_no_client_id_returns_error(self):
        """Auth enabled, not logged in, no client_id supplied → rejectable error."""
        owner, err = resolve_playbook_owner(
            auth_enabled=True,
            logged_in=False,
            session_user=None,
            request_client_id=None,
        )
        assert owner is None
        assert err == "client_id is required"

    def test_authed_logged_in_oidc_id_shape_returns_id(self):
        # OIDC callback stores the subject claim under 'id' (not 'sub'), no email.
        owner, err = resolve_playbook_owner(True, True, {"id": "sub|12345", "email": "", "name": ""}, "attacker-uuid")
        assert owner == "sub|12345"
        assert err is None

    def test_legacy_team_visibility_normalizes_to_public(self):
        # 'team' was the original name for public visibility; old exports and
        # API clients may still send it
        from src.utils.playbook_service import _normalize_visibility, parse_playbook_md
        assert _normalize_visibility("team") == "public"
        assert _normalize_visibility("public") == "public"
        assert _normalize_visibility("private") == "private"
        md = "---\nname: a\ndescription: d\nmetadata:\n  visibility: team\n---\nB"
        assert parse_playbook_md(md)["visibility"] == "public"

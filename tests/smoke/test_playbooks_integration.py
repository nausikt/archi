import uuid

import pytest

from src.utils.playbook_service import PlaybookService, PlaybookNotFoundError

PG_CONFIG = {
    "host": "localhost", "port": 5439,
    "database": "archi", "user": "archi", "password": "testpassword123",
}


@pytest.mark.integration
def test_playbook_roundtrip_against_live_db():
    svc = PlaybookService(pg_config=PG_CONFIG)
    run_id = uuid.uuid4().hex[:8]
    playbook_name = f"itest-playbook-{run_id}"
    created = svc.create_playbook("itest-owner", playbook_name, "when testing", "the body")
    assert created.id
    # list is owner-scoped
    assert playbook_name in [s.name for s in svc.list_playbooks("itest-owner")]
    # the load path
    assert svc.get_playbook_by_name("itest-owner", playbook_name).body == "the body"
    # isolation: a different owner never sees a private playbook
    other_owner = f"other-{uuid.uuid4().hex[:8]}"
    assert playbook_name not in [s.name for s in svc.list_playbooks(other_owner)]
    # cleanup
    svc.delete_playbook("itest-owner", created.id)
    with pytest.raises(PlaybookNotFoundError):
        svc.get_playbook_by_name("itest-owner", playbook_name)


@pytest.mark.integration
def test_public_playbook_visible_to_other_owner_and_read_only():
    svc = PlaybookService(pg_config=PG_CONFIG)
    run_id = uuid.uuid4().hex[:8]
    name = f"itest-public-{run_id}"
    owner_a, owner_b = f"itest-a-{run_id}", f"itest-b-{run_id}"
    created = svc.create_playbook(owner_a, name, "public test", "the public body", visibility="public")
    try:
        # another owner sees and can load it...
        assert name in [s.name for s in svc.list_playbooks(owner_b)]
        assert svc.get_playbook_by_name(owner_b, name, include_public=True).body == "the public body"
        # ...but cannot modify or delete it
        with pytest.raises(PlaybookNotFoundError):
            svc.update_playbook(owner_b, created.id, description="hijack")
        with pytest.raises(PlaybookNotFoundError):
            svc.delete_playbook(owner_b, created.id)
    finally:
        svc.delete_playbook(owner_a, created.id)

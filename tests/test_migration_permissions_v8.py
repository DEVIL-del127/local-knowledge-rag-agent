import copy

import pytest

from agent.migration_permissions import MigrationPermissionError, validate_windows_acl


USER = "S-1-5-21-1-2-3-1001"


def private_acl():
    return {"owner_sid": USER, "dacl_present": True, "reparse_point": False,
            "entries": [{"sid": USER, "type": "allow", "rights": 2032127}]}


def test_private_acl_accepts_user_system_administrators():
    acl = private_acl()
    for sid in ("S-1-5-18", "S-1-5-32-544"):
        acl["entries"].append({"sid": sid, "type": "allow", "rights": 2032127})
    before = copy.deepcopy(acl)
    validate_windows_acl(acl, current_user_sid=USER)
    assert acl == before


@pytest.mark.parametrize("sid", ["S-1-1-0", "S-1-5-11", "S-1-5-32-545", "S-1-5-21-1-2-3-1002"])
def test_other_accounts_are_rejected_even_with_read_only_rights(sid):
    acl = private_acl()
    acl["entries"].append({"sid": sid, "type": "allow", "rights": 1})
    with pytest.raises(MigrationPermissionError, match="^migration_acl_not_private$"):
        validate_windows_acl(acl, current_user_sid=USER)


@pytest.mark.parametrize("field,value", [("owner_sid", "unknown"), ("dacl_present", False),
                                        ("reparse_point", True), ("entries", [])])
def test_unverifiable_acl_is_rejected(field, value):
    acl = private_acl()
    acl[field] = value
    with pytest.raises(MigrationPermissionError):
        validate_windows_acl(acl, current_user_sid=USER)

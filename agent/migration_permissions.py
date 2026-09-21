"""Fail-closed admission for a separately collected Windows ACL inventory.

This validates inventory, not the filesystem. The maintenance runner must
collect fresh ACLs for the source, destination, keys and trusted bridge before
effects, and prevent directory replacement during migration.
"""
from __future__ import annotations

import re
from collections.abc import Mapping


class MigrationPermissionError(RuntimeError):
    pass


def validate_windows_acl(inventory: Mapping, *, current_user_sid: str) -> None:
    """Only the current user, SYSTEM and local administrators may have access.

    Numeric SIDs avoid localized names. Deny entries never cancel unknown allow
    entries in this conservative check. No ACEs are removed or changed here.
    """
    failure = "migration_acl_not_private"
    if not isinstance(current_user_sid, str) or not re.fullmatch(r"S-1-5-21-\d+-\d+-\d+-\d+", current_user_sid):
        raise MigrationPermissionError(failure)
    allowed = {current_user_sid, "S-1-5-18", "S-1-5-32-544"}
    if not isinstance(inventory, Mapping) or set(inventory) != {"owner_sid", "dacl_present", "reparse_point", "entries"}:
        raise MigrationPermissionError(failure)
    if inventory["owner_sid"] not in allowed or inventory["dacl_present"] is not True or inventory["reparse_point"] is not False:
        raise MigrationPermissionError(failure)
    entries = inventory["entries"]
    if not isinstance(entries, list) or not entries:
        raise MigrationPermissionError(failure)
    has_user_access = False
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"sid", "type", "rights"}:
            raise MigrationPermissionError(failure)
        if entry["type"] not in {"allow", "deny"} or type(entry["rights"]) is not int or entry["rights"] < 0:
            raise MigrationPermissionError(failure)
        if entry["type"] == "allow" and entry["rights"]:
            if entry["sid"] not in allowed:
                raise MigrationPermissionError(failure)
            has_user_access |= entry["sid"] == current_user_sid
    if not has_user_access:
        raise MigrationPermissionError(failure)

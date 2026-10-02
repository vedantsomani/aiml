"""Merge semantics of F1 live-timing deltas.

The feed sends a full object once, then partial updates:
  * dict update into dict   -> merge key by key (recursively)
  * dict update into list   -> keys are list indices ("0", "1", ...)
  * list update             -> replaces the target
  * {"_deleted": [keys]}    -> remove those keys
Event payloads are never mutated: values are copied on insert, so replaying the
same log twice always gives the same state.
"""

from __future__ import annotations

from typing import Any


def clone(value: Any) -> Any:
    """Fast deep copy for JSON-like data."""
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone(v) for v in value]
    return value


def _merge_into_list(target: list, update: dict) -> list:
    for key, value in update.items():
        if key == "_deleted":
            for idx in sorted((int(i) for i in value), reverse=True):
                if 0 <= idx < len(target):
                    target.pop(idx)
            continue
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue  # not an index; ignore rather than corrupt the list
        while len(target) <= idx:
            target.append({})
        current = target[idx]
        if isinstance(value, dict) and isinstance(current, (dict, list)):
            target[idx] = deep_merge(current, value)
        else:
            target[idx] = clone(value)
    return target


def deep_merge(target: Any, update: Any) -> Any:
    """Merge ``update`` into ``target`` and return the merged object.

    ``target`` is modified in place when it is a container; ``update`` never is.
    """
    if isinstance(update, dict):
        if isinstance(target, list):
            return _merge_into_list(target, update)
        if not isinstance(target, dict):
            target = {}
        for key, value in update.items():
            if key == "_deleted":
                for k in value:
                    target.pop(str(k), None)
                continue
            current = target.get(key)
            if isinstance(value, dict) and isinstance(current, (dict, list)):
                target[key] = deep_merge(current, value)
            else:
                target[key] = clone(value)
        return target
    # lists and scalars replace
    return clone(update)

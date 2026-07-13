"""Common base classes and utilities for FANET state models.

All state classes inherit from `StateBase` which provides:
- `to_dict()`: convert the dataclass instance into a plain dict (JSON-serializable)
- `from_dict()`: construct an instance from a dict (typically from JSON / simulator output)
- `to_json()`: serialize directly to a JSON string
- `summary()`: short string representation for logging

These helpers are mainly used for:
1. Converting simulation outputs (ns-3 / OMNeT++ / custom) into typed state objects.
2. Feeding structured state into LLM agents (the dict / JSON form is prompt-friendly).
"""

from __future__ import annotations

import json
import typing
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Dict, List, Type, TypeVar, get_args, get_origin

T = TypeVar("T", bound="StateBase")


@dataclass
class StateBase:
    """Base class providing dict / JSON conversion utilities."""

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert this dataclass (and nested dataclasses) to a dict."""
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Construct an instance from a dict, ignoring unknown keys.

        Nested dataclass fields and `list[DataclassClass]` fields are decoded
        recursively.  Unknown dict keys are silently dropped.
        """
        if data is None:
            raise ValueError(f"from_dict: input data is None for {cls.__name__}")
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} is not a dataclass")
        if not isinstance(data, dict):
            raise TypeError(
                f"from_dict({cls.__name__}): expected dict, got {type(data).__name__}"
            )

        # Resolve real types (handles `from __future__ import annotations`)
        try:
            hints = typing.get_type_hints(cls)
        except Exception:
            hints = {}

        valid_keys = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key not in valid_keys:
                continue
            field_type = hints.get(key, None)
            kwargs[key] = _decode_value(value, field_type)
        return cls(**kwargs)  # type: ignore[call-arg]

    def to_json(self, indent: int | None = None) -> str:
        """Serialize the state to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary(self) -> str:
        """Return a one-line summary for logging."""
        d = self.to_dict()
        parts = [f"{k}={d[k]}" for k in list(d.keys())[:6]]
        return f"{type(self).__name__}({', '.join(parts)}...)"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


def _decode_value(value: Any, type_hint: Any) -> Any:
    """Best-effort decoding of a nested value based on a resolved type hint."""
    if value is None:
        return None

    # list[T] / List[T] / Tuple[T, ...]  -> decode each element
    origin = get_origin(type_hint)
    if origin in (list, List, tuple):
        args = get_args(type_hint)
        if args and isinstance(value, list):
            inner = args[0]
            return [_decode_value(v, inner) for v in value]
        return value

    # dict[K, V]  -> recurse on values
    if origin is dict:
        args = get_args(type_hint)
        if isinstance(value, dict) and len(args) == 2:
            return {k: _decode_value(v, args[1]) for k, v in value.items()}
        return value

    # Dataclass subclass  -> recursive from_dict
    if isinstance(type_hint, type) and is_dataclass(type_hint):
        if isinstance(value, dict):
            return type_hint.from_dict(value)
        if isinstance(value, type_hint):  # already decoded
            return value

    # Anything else (primitives, strings, enums) -> return as-is
    return value

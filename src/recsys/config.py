"""Versioned YAML configuration, loaded into typed dataclasses with strict key checking."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping."""
    with open(path) as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def build(cls: type[T], values: dict[str, Any] | None) -> T:
    """Instantiate dataclass ``cls`` from ``values``, rejecting unknown keys (typos fail loudly)."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    values = values or {}
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def to_dict(obj: Any) -> dict[str, Any]:
    """Dataclass to plain dictionary (for saving resolved configs)."""
    return dataclasses.asdict(obj)

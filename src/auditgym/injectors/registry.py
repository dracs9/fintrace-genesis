"""Discover explicitly registered plugin classes without a mutable global registry."""

from collections.abc import Callable, Iterable
from typing import TypeVar

from .base import Injector

T = TypeVar("T", bound=type[Injector])


def register(name: str) -> Callable[[T], T]:
    """Bind the configured issue type to a plugin class at definition time."""
    def decorate(cls: T) -> T:
        cls.type = name
        return cls
    return decorate


def registry(classes: Iterable[type[Injector]]) -> dict[str, Injector]:
    """Construct a deterministic registry, rejecting duplicate names."""
    result = {}
    for cls in sorted(classes, key=lambda item: item.type):
        if cls.type in result:
            raise ValueError("duplicate injector type")
        result[cls.type] = cls()
    return result

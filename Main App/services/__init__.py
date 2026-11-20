"""Core services package for Apna AI Gym Coach."""

from . import auth
from . import coaching
from . import config
from . import persistence
from . import state
from . import tracking
from . import ui

__all__ = [
    "auth",
    "coaching",
    "config",
    "persistence",
    "state",
    "tracking",
    "ui",
    "vision",
]


def __getattr__(name: str):
    if name == "vision":
        from . import vision
        return vision
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

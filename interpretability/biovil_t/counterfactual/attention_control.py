from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Callable


_ACTIVE_ATTENTION_SUPPRESSOR = None


def get_active_attention_suppressor() -> Optional[Callable]:
    return _ACTIVE_ATTENTION_SUPPRESSOR


def set_active_attention_suppressor(suppressor: Optional[Callable]) -> None:
    global _ACTIVE_ATTENTION_SUPPRESSOR
    _ACTIVE_ATTENTION_SUPPRESSOR = suppressor


@contextmanager
def use_attention_suppressor(suppressor: Optional[Callable]):
    global _ACTIVE_ATTENTION_SUPPRESSOR
    old = _ACTIVE_ATTENTION_SUPPRESSOR
    _ACTIVE_ATTENTION_SUPPRESSOR = suppressor
    try:
        yield
    finally:
        _ACTIVE_ATTENTION_SUPPRESSOR = old
"""The pit wall's voice: radio messages, briefs and answers, from our own small language model.

Only `slm`, `train` and the model part of `api` need torch; importing this package does not.
"""

from __future__ import annotations

__all__ = ["say", "brief", "answer"]


def __getattr__(name):  # lazy: `from pitsense.voice import say`
    if name in __all__:
        from . import api

        return getattr(api, name)
    raise AttributeError(name)

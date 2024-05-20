from typing import Any, Dict

_ATTENTIONS: Dict[str, Any] = {}


def register_attention_fn(fn):
    _ATTENTIONS[fn.__name__.lower()] = fn
    return fn


from fanan.modules.attentions.self_attention import *  # noqa: E402, F403


def get_attention_fn(name: str):
    assert name in _ATTENTIONS, f"attention fn {name=} is not supported. Available attentions: {_ATTENTIONS.keys()}"
    return _ATTENTIONS[name.lower()]

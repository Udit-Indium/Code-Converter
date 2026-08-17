from __future__ import annotations
import functools
import logging
import os
import sys
from typing import Any, Callable

from .rate_limit import (
    LIMITER,
    RateLimiter,
    call_with_rate_limit,
    call_with_rate_limit_async,
)
logger = logging.getLogger(__name__)
DEFAULT_MAX_TOKENS = int(os.environ.get("RATE_LIMIT_DEFAULT_MAX_TOKENS", "4096"))
ANNOUNCE = os.environ.get("RATE_LIMIT_ANNOUNCE", "").strip().lower() in {"1", "true", "yes"}

_MARKER = "__rate_limited__"
STATS: dict[str, int] = {"completion": 0, "acompletion": 0, "passthrough": 0, "rebound": 0}

_installed = False
_originals: dict[str, Callable[..., Any]] = {}


def _accountable(args: tuple, kwargs: dict) -> bool:
    """True if this call has the keyword shape the limiter can account for.

    Anything else is forwarded unmetered rather than guessed at: one unmetered
    request is a smaller problem than a budget corrupted by mis-accounting for
    every request that follows.
    """
    return not args and bool(kwargs.get("model")) and kwargs.get("messages") is not None


def _split(kwargs: dict) -> tuple[str, Any, int, dict]:
    """Pull the limiter's arguments out, leaving the rest to forward."""
    rest = dict(kwargs)
    model = rest.pop("model")
    messages = rest.pop("messages")
    max_tokens = rest.pop("max_tokens", None) or DEFAULT_MAX_TOKENS
    return model, messages, int(max_tokens), rest


def install(limiter: RateLimiter | None = None) -> bool:
    """Patch `litellm.completion` and `litellm.acompletion`. Idempotent.

    Returns True if this call installed the patch, False if it was already in
    place.
    """
    global _installed
    import litellm

    if getattr(litellm.completion, _MARKER, False) or getattr(
        litellm.acompletion, _MARKER, False
    ):
        _installed = True
        return False

    original_completion = litellm.completion
    original_acompletion = litellm.acompletion
    _originals["completion"] = original_completion
    _originals["acompletion"] = original_acompletion

    active = limiter or LIMITER

    @functools.wraps(original_completion)
    def completion(*args: Any, **kwargs: Any) -> Any:
        if not _accountable(args, kwargs):
            STATS["passthrough"] += 1
            return original_completion(*args, **kwargs)
        STATS["completion"] += 1
        if ANNOUNCE:
            print(f"RATE LIMITER ACTIVE (sync) -> {kwargs.get('model')}", flush=True)
        model, messages, max_tokens, rest = _split(kwargs)
        return call_with_rate_limit(
            original_completion,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            limiter=active,
            **rest,
        )

    @functools.wraps(original_acompletion)
    async def acompletion(*args: Any, **kwargs: Any) -> Any:
        if not _accountable(args, kwargs):
            STATS["passthrough"] += 1
            return await original_acompletion(*args, **kwargs)
        STATS["acompletion"] += 1
        if ANNOUNCE:
            print(f"RATE LIMITER ACTIVE (async) -> {kwargs.get('model')}", flush=True)
        model, messages, max_tokens, rest = _split(kwargs)
        return await call_with_rate_limit_async(
            original_acompletion,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            limiter=active,
            **rest,
        )

    setattr(completion, _MARKER, True)
    setattr(acompletion, _MARKER, True)

    litellm.completion = completion
    litellm.acompletion = acompletion
    STATS["rebound"] = (
        _rebind_imported(original_completion, completion)
        + _rebind_imported(original_acompletion, acompletion)
    )

    _installed = True
    logger.info("LiteLLM rate limiting installed")
    return True


def _rebind_imported(original: Callable[..., Any], replacement: Callable[..., Any]) -> int:
    """Repoint modules that did `from litellm import completion`.

    Such a module holds the original function object directly, so rebinding the
    attribute on `litellm` never reaches it, and the call would slip past the
    limiter entirely.

    Scope, stated plainly: this rewrites EVERY module-level name that IS the
    original function object — not only names created by an import. A module
    that deliberately stashed the original (say, `_fallback = litellm.acompletion`)
    gets the limited version instead. That is broader than the name suggests,
    and it is the deliberate trade: missing a real import binding would leave a
    silent hole in the enforcement, which is the failure this module exists to
    prevent. It cannot cause double-charging — the replacement calls the true
    original, which is never itself patched.

    `litellm`'s own modules are skipped, so its internals keep their direct
    references.
    """
    rebound = 0
    for name, module in list(sys.modules.items()):
        if module is None or name.startswith("litellm"):
            continue
        try:
            entries = list(vars(module).items())
        except Exception:
            continue
        for attr, value in entries:
            if value is original:
                try:
                    setattr(module, attr, replacement)
                    rebound += 1
                except Exception:
                    logger.warning("could not rebind %s.%s", name, attr)
    return rebound


def uninstall() -> bool:
    """Restore the original LiteLLM functions. Mainly for tests."""
    global _installed
    if not _originals:
        return False
    import litellm

    litellm.completion = _originals["completion"]
    litellm.acompletion = _originals["acompletion"]
    _rebind_imported_back()
    _originals.clear()
    _installed = False
    return True


def _rebind_imported_back() -> None:
    """Undo `_rebind_imported` using the marker attribute."""
    for name, module in list(sys.modules.items()):
        if module is None or name.startswith("litellm"):
            continue
        try:
            entries = list(vars(module).items())
        except Exception:
            continue
        for attr, value in entries:
            if getattr(value, _MARKER, False):
                original = _originals.get(getattr(value, "__name__", ""))
                if original is not None:
                    try:
                        setattr(module, attr, original)
                    except Exception:
                        pass


def status() -> dict[str, Any]:
    """Whether the patch is live, and how many calls it has seen.

    `completion`/`acompletion` staying at zero during a real run means ADK is
    not reaching LiteLLM through these functions — patch a different layer.
    `passthrough` climbing means calls arrived in a shape the limiter could not
    account for and went through unmetered.
    """
    try:
        import litellm

        live = bool(getattr(litellm.completion, _MARKER, False)) and bool(
            getattr(litellm.acompletion, _MARKER, False)
        )
    except ImportError:
        live = False

    return {
        "patched": live and _installed,
        "announce": ANNOUNCE,
        "default_max_tokens": DEFAULT_MAX_TOKENS,
        **STATS,
        "throttled_seconds": LIMITER.report(),
    }

try:
    install()
except ImportError:
    logger.warning("litellm is not installed; rate limiting NOT active")
except Exception as exc:
    logger.error("could not install LiteLLM rate limiting (%s); calls are UNTHROTTLED", exc)

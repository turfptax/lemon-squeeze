"""Shared utilities: hashing, token counting, normalization."""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

from lemon_squeeze.config import settings

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_prompt(text: str) -> str:
    """Normalize for hashing — collapse whitespace, strip ends. Display text is preserved separately."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def hash_prompt(text: str) -> str:
    return hashlib.sha256(normalize_prompt(text).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding(settings.default_token_encoding)
    except Exception:
        return None


def count_tokens(text: str) -> int | None:
    enc = _encoder()
    if enc is None:
        return None
    return len(enc.encode(text, disallowed_special=()))


def split_provider_family(name: str) -> tuple[str, str | None]:
    """Parse a slash-qualified model name into (provider, family).

    Examples:
      `anthropic/claude-sonnet-4-6` → ("anthropic", "claude")
      `lm_studio/llama-3.1-8b`      → ("lm_studio", "llama")
      `llama-3.1-8b-instruct`       → ("unknown", "llama")   ← family from bare name
      `gpt`                          → ("unknown", "gpt")

    Best-effort — used by ingest, CLI register, and provider discovery to
    derive metadata when the user didn't supply it explicitly. LM Studio
    routinely returns bare model IDs (no provider slash), so we still
    extract a family from those.
    """
    if "/" in name:
        provider, rest = name.split("/", 1)
        family_src = rest
    else:
        provider, family_src = "unknown", name
    family = family_src.split("-", 1)[0] if "-" in family_src else family_src
    return provider, (family or None)

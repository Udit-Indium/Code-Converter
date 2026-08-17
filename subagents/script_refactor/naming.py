from __future__ import annotations
import json
import keyword
import re
from typing import Protocol
from .categories import verb_for
from .models import BlockSummary
PROMPT_TEMPLATE = """You are naming Python functions.

Given the following summary:

{summary}

Return ONLY a concise snake_case function name.
Do not include explanations or markdown."""

MAX_NAME_LENGTH = 40

_IDENTIFIER_SAFE = re.compile(r"[^0-9a-zA-Z_]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def sanitise(raw: str) -> str:
    """Coerce a model reply into a valid snake_case identifier.

    Handles what models actually return: markdown fences, backticks, a
    trailing `()`, a leading `def `, an explanatory second line, camelCase.
    Returns `""` if nothing usable survives, which tells the caller to fall
    back rather than emit a broken name.
    """
    text = raw.strip()
    if not text:
        return ""
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    text = text.splitlines()[0].strip() if text.splitlines() else ""
    text = text.strip("`'\" \t")

    if text.startswith("def "):
        text = text[4:]
    text = text.split("(")[0].strip()

    text = _CAMEL_BOUNDARY.sub("_", text)
    text = _IDENTIFIER_SAFE.sub("_", text).strip("_").lower()
    text = re.sub(r"__+", "_", text)

    if not text:
        return ""
    if text[0].isdigit():
        text = f"step_{text}"
    if keyword.iskeyword(text) or keyword.issoftkeyword(text):
        text = f"{text}_step"
    if len(text) > MAX_NAME_LENGTH:
        text = text[:MAX_NAME_LENGTH].rstrip("_")
    return text if text.isidentifier() else ""


class FunctionNamer(Protocol):
    """Anything that can name a block from its summary.

    `category` is passed for the benefit of namers that use it (the
    deterministic one builds its verb from it); an LLM namer ignores it, since
    requirement 8 fixes the prompt to the summary alone.
    """

    def name_for(self, summary: BlockSummary, category: str = "") -> str:
        """Return a snake_case name, or `""` to defer to the fallback."""
        ...


class DeterministicNamer:
    """Names blocks from the summary alone, with no model.

    Used as the fallback for every LLM failure, and as the primary namer when
    the caller opts out of LLM use. The names are plainer than a model's
    (`load_sales_df` rather than `ingest_regional_sales`) but they are stable,
    free, and always valid — which makes them the right default for tests.
    """

    def __init__(self, category_lookup: dict[int, str] | None = None) -> None:
        self._categories = category_lookup or {}

    def name_for(self, summary: BlockSummary, category: str = "") -> str:
        """Build `<verb>_<subject>` from the summary."""
        verb = verb_for(category) if category else "process"
        subject = ""
        for candidate in (*summary.outputs, *summary.modifies, *summary.inputs):
            if not candidate.startswith("_"):
                subject = candidate
                break

        if not subject and summary.operations:
            subject = summary.operations[0]

        name = f"{verb}_{subject}" if subject else verb
        return sanitise(name) or "process_data"


class LLMFunctionNamer:
    """Asks a model for a name, given only the summary.

    The LiteLLM import is deferred to first use so that importing this package
    — and running its tests — needs no LLM dependency installed at all.

    Args:
        model: LiteLLM model id, e.g. `databricks/databricks-claude-sonnet-4-5`.
        max_tokens: hard ceiling on the reply. A snake_case name is a handful of
            tokens; anything longer is a model ignoring the instruction, and
            truncating it costs nothing.
        timeout: seconds to wait before falling back.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 24,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._cache: dict[str, str] = {}
        self.failures: list[str] = []

    def _complete(self, prompt: str) -> str:
        """One completion call. Isolated so tests can patch it.

        Routed through the shared limiter. Naming is the pipeline's highest
        REQUEST rate — one call per block, ~66 of them back to back on a large
        notebook — even though its token volume is trivial, so it is the call
        site most likely to trip a requests-per-minute quota.
        """
        import litellm

        from ..rate_limit import call_with_rate_limit

        response = call_with_rate_limit(
            litellm.completion,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0,
            timeout=self.timeout,
        )
        return response["choices"][0]["message"]["content"] or ""

    def name_for(self, summary: BlockSummary, category: str = "") -> str:
        """Return a model-generated name, or `""` if the call was unusable."""
        payload = json.dumps(summary.to_dict(), indent=4, sort_keys=True)
        if payload in self._cache:
            return self._cache[payload]

        prompt = PROMPT_TEMPLATE.format(summary=payload)
        try:
            reply = self._complete(prompt)
        except Exception as exc:
            self.failures.append(f"{type(exc).__name__}: {exc}")
            return ""

        name = sanitise(reply)
        if not name:
            self.failures.append(f"unusable reply: {reply!r}")
            return ""

        self._cache[payload] = name
        return name


def deduplicate(names: list[str]) -> list[str]:
    """Make every name unique by suffixing repeats.

    Two blocks can legitimately summarise identically — two cleaning passes over
    different frames — and the model will name them the same. Later definitions
    would silently shadow earlier ones, so repeats become `clean_df_2`,
    `clean_df_3`, and so on.
    """
    seen: dict[str, int] = {}
    result: list[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            result.append(name)
            continue
        seen[name] += 1
        candidate = f"{name}_{seen[name]}"
        while candidate in seen:
            seen[name] += 1
            candidate = f"{name}_{seen[name]}"
        seen[candidate] = 1
        result.append(candidate)
    return result

"""Pipeline sub-agents.

Two side effects on import, both deliberate and both order-sensitive.

**1. `.env` is loaded explicitly, by path.** The agent modules each call a bare
`load_dotenv()`, which searches upward from the CALLER's directory for a file
named `.env` — a search that never finds this project's file. Verified:
`find_dotenv()` returns nothing even when called from inside this package, so
every one of those calls was a no-op and `DATABRICKS_HOST`, `DATABRICKS_API_KEY`
and the rate-limit quotas were silently missing.

Both plausible locations are tried, because `.env.example` ships at the repo
root while the code originally read only `subagents/.env` — so copying the
example where it sits produced a file nothing loaded. Whichever exists wins;
if both do, the more specific `subagents/.env` takes precedence, and a real
environment variable beats either (`override=False`), which is what CI and
container deployments expect.

**2. The LiteLLM rate limiter is installed.** It lives here rather than in the
top-level entrypoint because every agent module is `subagents.something`, so
Python imports THIS file before any of them can be reached, whichever entrypoint
is used. The patch must be in place before an agent is constructed.

The `.env` load comes first: the limiter reads its quotas from the environment,
so loading afterwards would leave the buckets on their defaults.
"""

from pathlib import Path

#: Checked in order; the first value found for a given key wins.
ENV_CANDIDATES = (
    Path(__file__).with_name(".env"),           # subagents/.env
    Path(__file__).resolve().parent.parent / ".env",  # <repo root>/.env
)

ENV_REPORT: dict[str, object] = {
    "dotenv_installed": False,
    "files_found": [str(p) for p in ENV_CANDIDATES if p.is_file()],
    "files_missing": [str(p) for p in ENV_CANDIDATES if not p.is_file()],
    "loaded": False,
}

try:
    from dotenv import load_dotenv

    ENV_REPORT["dotenv_installed"] = True
    for _candidate in ENV_CANDIDATES:
        if _candidate.is_file():
            load_dotenv(_candidate, override=False)
            ENV_REPORT["loaded"] = True
except ImportError:
    # Real environment variables still apply, so this is not fatal — but it
    # does mean no .env is read at all, which is indistinguishable from an
    # empty file unless it is reported.
    import warnings

    warnings.warn(
        "python-dotenv is not installed, so no .env file was loaded. "
        "Run: pip install python-dotenv",
        RuntimeWarning,
        stacklevel=2,
    )


def env_status() -> dict[str, object]:
    """Report whether the environment actually loaded, and from where.

    Run this first when a credential or quota looks unset:

        python -c "import subagents, json; print(json.dumps(subagents.env_status(), indent=2))"
    """
    import os

    required = ("DATABRICKS_HOST", "DATABRICKS_API_KEY", "USER_ID")
    quotas = (
        "DATABRICKS_ITPM_DATABRICKS_CLAUDE_SONNET_4_6",
        "DATABRICKS_OTPM_DATABRICKS_CLAUDE_SONNET_4_6",
        "DATABRICKS_ITPM_DATABRICKS_CLAUDE_OPUS_4_7",
        "DATABRICKS_OTPM_DATABRICKS_CLAUDE_OPUS_4_7",
    )
    return {
        **ENV_REPORT,
        # Presence only — never echo a credential.
        "required_set": {k: bool(os.environ.get(k)) for k in required},
        "quotas_set": {k: os.environ.get(k) for k in quotas},
    }

from . import litellm_patch  # noqa: E402,F401  (imported for its side effect)

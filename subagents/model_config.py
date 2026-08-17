"""Which Databricks serving endpoint each agent uses.

Only TWO endpoints are entitled in this workspace, so every assignment below
resolves to one of them. Anything else fails on the first call.

Two reasons the assignment is per-agent rather than one constant:

  * **Capability.** The fixers and the test writer do the hardest reasoning —
    diagnosing why a converted function misbehaves, or writing an assertion that
    reflects real behaviour. The parser barely reasons at all.
  * **Rate limits.** The endpoint rejects with "Exceeded workspace input tokens
    per minute rate limit for databricks-claude-sonnet-4-6" — it names the
    model, so the quota is per-model. With only two endpoints there are only two
    buckets, so the split below deliberately keeps the single heaviest consumer
    (the converter, ~123k input tokens per run) on a different bucket from the
    next two heaviest (the code fixer and the parity test writer).

Measured input-token load for a 69-function source, one clean run:

    converter          ~123k    (9 batches x ~13.7k, mostly the conventions blob)
    code fixer          ~85k    (conventions re-sent every fixer turn)
    semantic + fixer    ~45k
    parity test writer  ~41k
    case-fact checker   ~14k
    code parser        <1k

    SONNET bucket:  converter + semantic       = ~168k
    OPUS bucket:    fixer + parity + case-fact = ~140k

Every value must be an endpoint name exactly as it appears under Serving in the
workspace. A name that is not served fails at the first call.
"""

# --- the only two entitled endpoints ---------------------------------------
# VERIFY the Opus name against Serving in your workspace: Opus endpoint naming
# varies by region and entitlement, and the previous value here was
# "databricks-claude-opus-4-6", which is not entitled.
OPUS = "databricks-claude-opus-4-7"
SONNET = "databricks-claude-sonnet-4-6"

# Kept so old imports do not break; both now resolve to an entitled endpoint.
SONNET_4_6 = SONNET

# --- per agent -------------------------------------------------------------
# Hardest reasoning: repairing code against a failing test, and writing tests
# that assert real behaviour rather than restating the function name.
CODE_FIXER_MODEL = OPUS
PARITY_MODEL = OPUS

# Bulk conversion, function by function. The heaviest consumer by far, so it
# gets a bucket that the fixers are not also competing for.
CONVERTER_MODEL = SONNET

# Semantic stage: compares two pipelines' output and repairs the difference.
# Shares the converter's bucket, but runs after it has finished.
SEMANTIC_MODEL = SONNET
SEMANTIC_FIXER_MODEL = SONNET

# Restructuring a flat script into functions. Only ever asked for a function
# NAME from a small structured summary — see the note below about disabling
# these calls entirely, which is the recommended setting under a tight quota.
REFACTOR_MODEL = OPUS

# Barely reasons: calls two tools with a path. Was SONNET_4_5, which is not
# entitled. Parked on the lighter bucket; the load is under 1k tokens.
CODE_PARSER_MODEL = OPUS

# Compares two name lists. Its verdict is advisory — check_fact_status
# recomputes it deterministically by AST — so capability barely matters here.
# Parked with the parser to keep the converter's bucket clear.
CASE_FACT_MODEL = OPUS

---
name: py2snow-skill
description: >
  Convert a Python data-transformation script (pandas ETL) into an equivalent
  distributed **PySpark** module. Triggers: "convert this to PySpark", "py to
  spark", "py2snow", "translate this pandas pipeline to Spark", "migrate this
  script to Databricks". The skill is GENERIC — never hardcode a specific
  pipeline, column, or function.
compatibility: Python 3.9+; PySpark 3.5+ (4.x ok)
---

# Python → PySpark Converter (py2snow)

Convert an arbitrary pandas/Python ETL script into a **distributed PySpark**
module. The output is a SINGLE file, `<stem>_spark.py`, assembled batch by batch
by `add_converted_functions_tool` — never several files. All the conventions to
follow when converting are in `references/`.

— **read all reference files before converting and follow them exactly.**

The skill is generic: it must work for **any** input set. Never hardcode columns,
functions, or a pipeline — drive everything off the inputs.

## How this skill works — there is NO converter program

**YOU (the LLM) write the PySpark code yourself**, All the constant has to be same as
as mentioned in source script or you can also refer "constants" in ast parsed content,
you need to do the conversion function by function, class by class and save
it with **add_converted_functions_tool**. There is **no deterministic converter, no
`run_skill_script` entrypoint, and no script that takes the inputs and emits the
result.** Do **NOT** try to "run the skill" by calling `run_skill_script` (or any
tool) with `source_py_content` / `parsed_json_content` arguments — that
tool/those parameters do not exist for conversion, and you must never pass file
paths in their place.

Make sure, the converted script has all the availale functions, classes and correct
constant with correct values present in source script or repersent by
**constants** reported by `read_source_index_tool`.

## Important Files (Input)

You do not need to ask the user for any input files. There are two sources, and
they are reached differently:

1. **The parsed AST inventory** — the **contract of WHAT to convert**:
   `functions[]`, `classes[]`, `constants`, and per-function `parameters`,
   `returns`/output, `columns`, `joins`, `aggregations`, `filters`,
   `window_functions`, `transformation_types`, plus `pyspark.pattern`/`risk`/
   `warnings`, `call_graph`, and `summary`. Its **`call_graph` is the contract of
   the FLOW** — who calls whom, in what order — and is what your orchestrator
   must reproduce.

   It is **NOT in your prompt and NOT in agent state** — it is on disk, and the
   tools surface the parts you need: **read_source_index_tool()** gives the
   function names, signatures, per-function risk and the constant names.
2. **The original Python source** — on disk, NOT in state and NOT in your prompt.
   It is the **source of truth for behaviour**. Read it in pieces with
   **read_source_functions_tool(function_names=[...])**, and get the names from
   **read_source_index_tool()**. Ask for the functions you are converting this
   batch, not the whole file.

> **NEVER invent a filename** (e.g. `input_script.py`) and never try to open the
> inventory or the source yourself. Both are reached ONLY with the two tools
> named above.

## Workflow (do this every time)

**STEP 1 — Analyse the source script.** Use the already-provided source `.py`
content. Understand the overall pipeline: the constants, functions, classes, 
what it ingests, the transformations it performs, and what it produces.

**STEP 2 — Analyse the ast parsed JSON: build the function inventory.** From the parsed
JSON enumerate EVERY constants along with it's value, top-level `functions[]`, 
EVERY `classes[]`, and EVERY method, plus nested helpers seen in the source. For 
each, note its **parameters**, its **output/return**, and the **operations** it performs (`pyspark.pattern` + construct lists). This inventory is your checklist — you are not done until every item is converted.

**STEP 3 — Recover the flow.** From the inventory's `call_graph`, recover the
orchestrator's execution order (the calls in line order, repeated calls kept) and
the data hand-off between steps — which function's return feeds which parameter.
This is the order your converted orchestrator must reproduce. Where the call
graph is ambiguous about the hand-off, read the original orchestrator function
with **read_source_functions_tool** and follow it exactly. Confirm the inventory
and the call graph reconcile.

**STEP 4 — Read the conventions.** Read **all** files in `references/`:
- `pyspark_etl_conventions.md` — HOW to convert: function counting, flow
  recovery, pandas→Spark mapping, and divergence traps.
- `SEMS_Compliance.md` — docstrings, comments, naming, logging, error handling,
  no secrets.
- `pyspark_visualisations_conventions.md` — plotting/charting (apply only if the
  source plots).

**STEP 5 — Present the pre-conversion preview to the user, then proceed.** Before
writing any code, show the user:
- **how many functions/classes/methods will be converted, what are all constants avalable** (the inventory count),
  with the list;
Then continue — do not block waiting for approval.

**STEP 6 — Generate `<stem>_spark.py`.**
- Make sure, the constant values in the source file and the converted file are same. you can refer all 
the constants from ast parsed json and also refer the source python code to check the constant, functions, 
classes and main method in the file.
- One converted definition per inventory entry — **exhaust the list**. Keep the
  source's function / parameter / column / key / literal names VERBATIM.
- For each function apply the pandas→Spark mapping in ETL conventions §4 driven by
  its `pyspark.pattern` + construct lists; handle every divergence trap in §5
  (`merge_asof`→window+`F.last`, time-rolling→`rangeBetween`, `explode` vs
  `explode_outer`, null ordering, dedup-keep, `qcut`→`percentile_approx`, …).
- Carry `constants` (e.g. `CONFIG` etc) and exception classes (e.g. `PipelineError`)
  over verbatim.
- Write the orchestrator to follow the source's flow exactly, threading
  intermediate frames between steps as the original did.
- Apply SEMS throughout: docstring on every function naming its source function,
  inline comments mapping source tags, `logging` not `print`, same exceptions, no
  hardcoded secrets.
- Use the DataFrame API only (no row loops, no driver collects, no plain
  `F.udf`); expose `get_spark()` with a UTC session time zone so the module runs
  locally and on Databricks.
Save it with **add_converted_functions_tool(functions_code=...)**, passing ONLY the
batch you converted this turn. The tool owns the output path and reassembles the
file deterministically — you never name a file and never resend earlier batches.

**STEP 7 — Run it.** Once the whole inventory is converted, call
**execute_pyspark_script_tool()** as a whole-file check and fix what it reports.
Do not call it after every batch — it submits a Databricks run each time.

**STEP 8 — Self-verify & return.** Confirm: converted callable count ==
inventory count (with nested reconciliation), no item stubbed/missing; the
orchestrator's call sequence == the source's call order; high-risk
functions spot-checked against the traps; SEMS checklist passes. Then return the
saved path, the count/flow confirmation, and an explicit list of any manual
intervention, divergence, or approximation.

## ACT, DON'T NARRATE

Do STEP 1→8 in the run — analyse the source, build the inventory from the parsed
JSON, recover the flow from the call graph, read the conventions, show the
preview, **author and save** the PySpark module, **execute it and loop until it
is syntax-error-free**, self-verify, and return the result. Never reply with
"next I will…" and stop. Only pause if a tool genuinely errors.

## SEMS clean-code rules

`references/SEMS_Compliance.md` carries the full standard (docstrings, comments,
naming, logging, error handling, secrets, modularity, SonarQube rules). The rules
below are the ones it does **not** cover and are equally binding — the converted
file is graded on them.

**Declarations and variables**

- Declare each name at the **smallest reasonable scope** — inside the branch that
  uses it, not at the top of the function.
- **No shadowing.** An inner name never redefines an outer one, and nothing ever
  redefines a builtin (`id`, `type`, `list`, `dict`, `filter`, `sum`, `input`,
  `max`, `min`). `sum` is the common trap, sitting next to `F.sum`.
- **One meaning per variable** for its whole scope. Do not reuse `df` for the raw
  frame, then the filtered frame, then the aggregate — chain, or use new names.
- **Name by meaning, not representation**: `amount_usd`, not `float_col_3`.
- **Scalar singular, collection plural**: `customer_id` vs `customer_ids`.
- **Never mutate a parameter.** Build and return a new object. Spark transforms
  are immutable already, so the trap is Python-side — mutating a passed `dict`,
  `list`, or `CONFIG` in place.
- **Explicit visibility.** Non-public helpers are prefixed `_`. Module constants
  are read-only — never rebind `CONFIG`.
- **One logical effect per statement.**
- **An intentionally empty block is commented** with why:
  `except KeyError:  # optional column, absent is valid`.

**Function and class design**

- **The name reflects everything the function does.** A name that hides a side
  effect is a defect — rename, or split the function.
- **Boolean functions read as a question**: `is_valid_currency`, `has_complaints`.
  Same for boolean columns — unless the source already named it otherwise, in
  which case the source name wins (SEMS §3).
- **A class must represent more than one function's worth of behaviour.** Never
  introduce a class to hold a single method, and never invent a class the source
  does not have. Custom exceptions such as `PipelineError` are fine.
- **Subclasses are substitutable for their base (LSP)**, or say so in the docstring.

**Comments stay true**
 
- **A comment must never contradict its code.** Change the line, change the comment.
- **No maintenance history in source** — no changelogs, `# modified by …`, dates,
  or ticket numbers. That lives in version control.
- **Consistent placement**: directly above the block, same indentation.

**Clean code before performance**

- Prefer the simple, readable form. Optimise only when a requirement demands it.
- **Justify every optimisation at the site** — what changed and why. A
  `broadcast`, `repartition`, `cache`, or `persist` with no comment is a defect.

✅ `df = df.join(F.broadcast(stores), "store_id", "left")  # ~20 rows; avoids a shuffle`
❌ `df = df.repartition(200).cache()` with no explanation.

## Non-negotiable rules (summary)

- **You author the code; there is no converter program** — never call
  `run_skill_script` (or any tool) to "run the conversion", and never pass
  `source_py_content`/`parsed_json_content` args or file paths. Write the PySpark
  yourself and save it with **add_converted_functions_tool**.
- **Inputs already provided** — the inventory and the source are BOTH reached
  with `read_source_index_tool` / `read_source_functions_tool`; neither is in
  your prompt or in agent state. Never invent a filename.
- **Constants need to added carefully** - all the constants along with there values
  needs to be added in the converted pyspark file carefully.
- **Convert every function & class** — count == parsed-JSON inventory; nested
  helpers kept inside parents; no placeholders, no early stop, no silent drops.
- **Show the preview** — function count + output path, before converting.
- **Flow == the source's call order** — same functions, same order, repeated
  calls kept, frames threaded between steps as the original did.
- **Distributed DataFrame API** — no row loops, no driver collects, no plain `F.udf`.
- **Behaviour preserved** — translation, not rewrite; no new logic.
- **Executed clean** — run **execute_pyspark_script_tool()** once the whole file
  is assembled, and loop on fixes until it runs without error. There is no local
  executor; the tool submits the file to Databricks.
- **Columns/keys/literals verbatim** from parsed JSON + source — never invented.
- **Conventions enforced** — read all of `references/`; apply ETL mapping + traps,
  visualisation handling, SEMS, and the SEMS clean-code rules above (scope,
  shadowing, no parameter mutation, naming, truthful comments, justified tuning).
- **Proper SparkSession** — `get_spark()`, UTC tz, runnable locally and on cluster.
- **Generic** — never assume the reference pipeline; drive everything off the inputs.

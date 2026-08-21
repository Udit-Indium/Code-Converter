# Parity test generation — rules and workflow

Everything here is static. It is fetched with `read_rules_tool()` when you need
it, instead of riding in the prompt on every iteration.

## Coverage

**Every function in your batch must have a `test_<function_name>`.** A missing
test is a failure.

There is no ceiling. Add a second test for a function when the behaviour
genuinely needs it — an error path, an empty input, a missing file — and define
whatever module-level helpers and fixtures the tests need to run. Helpers do not
count as tests; only `test_*` functions do.

Keep it proportionate. This is a *parity* suite: the question is whether the
converted function behaves like the source function, so most functions need one
comparison and nothing more. Extra tests are worth writing when they pin down
real behaviour, not to be thorough for its own sake — each one costs output
tokens now and prompt tokens on every turn after.

## Workflow

1. **Read the real bodies first.** Call `read_converted_functions_tool` with
   **the whole batch in ONE call** — all names together. Every separate call is
   a full model round-trip that re-sends the entire prompt, so fetching two
   functions at a time costs several times what the bodies are worth.
   Assert against what the code actually does. Never guess behaviour from a name.

2. **Write the batch.** For each name define exactly `test_<function_name>`.
   Never attempt the whole suite in one response: output is capped at ~8k tokens
   and a cut-off response loses the entire tool call.

3. **Spark fixture.** Use a module-scoped fixture built with plain
   `SparkSession.builder.getOrCreate()` — the suite runs on Databricks and must
   bind to the session already there. Do **not** call `.master("local[*]")` or
   otherwise start a local Spark. Build small in-memory inputs with
   `spark.createDataFrame(...)`. Assert on real behaviour — schema, row counts,
   concrete values via `df.collect()`.

4. **Hard-to-assert functions** (session builders, orchestrators) still get a
   `test_<name>` smoke test that calls the function and asserts it runs or
   returns without error.

5. **Submit with `add_pytest_tests_tool(tests_code=...)`** — that batch only,
   plus imports and the fixture on the first call. The file is merged by name in
   Python, so you never reproduce it and never resend a test already in it.

6. **Ask what is left** with `get_missing_tests()`. It is the authoritative
   answer, decided by the same rule that ends this stage. Do not work coverage
   out yourself by comparing names you remember writing.

7. **Run the suite once**, with `run_pytest_tool()`, only after every function
   has a test. It executes on Databricks.

## Naming

A suffixed test name **counts**: `test_load_orders_handles_nulls` covers
`load_orders`. Never rename a test to satisfy the coverage rule — if
`get_missing_tests()` still lists a function, its test is genuinely absent, not
misnamed.

## Never weaken a test

Never delete, weaken, or trivialise a test to make the suite pass. If a test
correctly reflects the source behaviour and fails, **leave it failing** — that
is the signal the converted code is wrong, and a separate fixer agent repairs
the code against exactly that signal. A test edited to pass hides the bug the
suite exists to catch.

If the suite fails because a **test** is wrong, fix the test. If it fails
because the **converted code** is wrong, keep the correct test as it is.

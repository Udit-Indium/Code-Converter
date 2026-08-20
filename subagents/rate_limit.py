from __future__ import annotations
import asyncio
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

DEFAULT_INPUT_TOKENS_PER_MINUTE = 40_000
DEFAULT_OUTPUT_TOKENS_PER_MINUTE = 8_000
DEFAULT_REQUESTS_PER_MINUTE = 30
# Burst allowance, as a fraction of one minute's quota. NOT free: a bucket
# starts full, so over a window T it admits `capacity + rate * T`. At 1.0 the
# first minute spends twice the quota — measured at 36,000 output tokens
# against a 20,000 OTPM limit — which a server enforcing a strict per-minute
# window rejects even though the long-run average is correct.
# Lower this before touching the quotas if 429s persist.
DEFAULT_BURST_FRACTION = float(os.environ.get("RATE_LIMIT_BURST_FRACTION", "0.25"))
MAX_WAIT_SECONDS = 300.0
MAX_RETRY_AFTER_SECONDS = 120.0
DEFAULT_MAX_ATTEMPTS = 4

_TOKEN_EPSILON = 1e-6

#: Smallest wait the bucket will ask for, so every loop iteration advances the
#: clock by something real.
_MIN_DELAY_SECONDS = 1e-3


def estimate_tokens(text: str) -> int:
    """Rough input-token estimate for `text`.

    Deliberately crude (~4 chars per token) and deliberately NOT a tokenizer
    call: this runs before every request, and an exact count would cost a round
    trip to refine a budget that is itself a conservative guess.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_message_tokens(messages: Iterable[Any]) -> int:
    """Estimate the prompt tokens of a LiteLLM `messages` list.

    Handles both content shapes: a bare string, and the list-of-blocks form
    used for structured content.
    """
    total = 0
    for message in messages or ():
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, Sequence):
            for block in content:
                if isinstance(block, dict):
                    total += estimate_tokens(block.get("text") or "")
        total += 4
    return total


def _env_key(model: str) -> str:
    """Environment-variable suffix for an endpoint name."""
    return re.sub(r"[^0-9A-Za-z]+", "_", model).strip("_").upper()


def _configured(prefix: str, model: str, fallback: int) -> int:
    """Read a per-endpoint quota from the environment."""
    raw = os.environ.get(f"{prefix}_{_env_key(model)}")
    if raw is None:
        return fallback
    try:
        value = int(raw)
    except ValueError:
        return fallback
    return value if value > 0 else fallback


@dataclass
class TokenBucket:
    """A refilling bucket with burst capacity.

    Capacity is what allows a burst: the bucket starts full, so the first
    `capacity` units cost no wait. Once drained, acquisitions are paced by
    `rate_per_minute`.

    The clock and sleep function are injected so behaviour can be exercised
    against a virtual clock — a limiter tested against the wall clock is a test
    that either takes minutes or proves nothing.
    """

    rate_per_minute: float
    capacity: float
    now: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    asleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    _tokens: float = field(default=0.0, init=False)
    _updated: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.capacity)
        self._updated = self.now()

    @property
    def _per_second(self) -> float:
        return self.rate_per_minute / 60.0

    def _refill_locked(self) -> None:
        """Add tokens accrued since the last check. Caller holds the lock."""
        current = self.now()
        elapsed = max(0.0, current - self._updated)
        self._updated = current
        self._tokens = min(self.capacity, self._tokens + elapsed * self._per_second)

    def _clamp(self, amount: float) -> float:
        """An amount larger than the bucket can never be satisfied, so clamp it
        rather than deadlock — one oversized request should be slow, not fatal.
        """
        return min(max(0.0, float(amount)), float(self.capacity))

    def _step(self, amount: float, waited: float) -> tuple[bool, float]:
        """One acquisition attempt.

        Returns `(acquired, delay)`. The lock is released before the caller
        waits, which is what lets the sync and async paths share this policy —
        neither ever sleeps while holding it.
        """
        with self._lock:
            self._refill_locked()
            # The epsilon is load-bearing, not decoration. Refilling accumulates
            # float error, so a request for exactly `capacity` can sit forever a
            # fraction of a token short: the shortfall shrinks toward zero, the
            # computed delay shrinks with it, and the loop spins without the
            # clock meaningfully advancing. An exact-capacity request is the
            # common case here — any prompt at or above the bucket size is
            # clamped to precisely this value.
            if self._tokens + _TOKEN_EPSILON >= amount:
                self._tokens = max(0.0, self._tokens - amount)
                return True, 0.0
            shortfall = amount - self._tokens
            per_second = self._per_second

        if per_second <= 0:
            raise RuntimeError(f"bucket for rate {self.rate_per_minute} cannot refill")

        delay = min(shortfall / per_second, MAX_WAIT_SECONDS - waited)
        if delay <= 0:
            raise RuntimeError(
                f"rate limiter waited {waited:.0f}s for {amount:.0f} units and gave up; "
                "the configured quota is too small for this request"
            )
        # Floor the wait so every iteration makes real progress, belt-and-braces
        # against the same livelock.
        return False, max(delay, _MIN_DELAY_SECONDS)

    def acquire(self, amount: float = 1.0) -> float:
        """Block until `amount` is available; return the seconds spent waiting."""
        amount = self._clamp(amount)
        if amount == 0.0:
            return 0.0
        waited = 0.0
        while True:
            acquired, delay = self._step(amount, waited)
            if acquired:
                return waited
            self.sleep(delay)
            waited += delay

    async def acquire_async(self, amount: float = 1.0) -> float:
        """Await until `amount` is available; return the seconds spent waiting.

        The async twin of `acquire`, and the one ADK must use. The sync version
        blocks the calling thread — inside a coroutine that is the event loop
        itself, so throttling one agent would stall every other concurrent
        request rather than just its own.
        """
        amount = self._clamp(amount)
        if amount == 0.0:
            return 0.0
        waited = 0.0
        while True:
            acquired, delay = self._step(amount, waited)
            if acquired:
                return waited
            await self.asleep(delay)
            waited += delay

    def debit(self, amount: float) -> None:
        """Charge tokens after the fact, even past zero.

        Used when a response spent MORE than was reserved — possible whenever
        the burst capacity is smaller than a single request's real output, since
        `acquire` clamps the reservation to capacity. Going negative is the
        point: the deficit is carried, and the next acquisition waits for it to
        refill, so the long-run rate stays honest instead of silently drifting
        over quota.
        """
        if amount <= 0:
            return
        with self._lock:
            self._refill_locked()
            self._tokens -= float(amount)

    def release(self, amount: float) -> None:
        """Hand unused capacity back, never exceeding the bucket's ceiling."""
        if amount <= 0:
            return
        with self._lock:
            self._refill_locked()
            self._tokens = min(self.capacity, self._tokens + float(amount))


@dataclass
class Reservation:
    """Budget held for one in-flight request.

    Created by `EndpointLimiter.reserve`, closed by `settle`. Until it settles,
    the full `prompt_tokens + max_tokens` is charged against the bucket, so a
    concurrent caller cannot spend the same headroom.
    """

    limiter: "EndpointLimiter"
    prompt_tokens: int
    max_tokens: int
    #: What the output bucket was actually charged. Differs from `max_tokens`
    #: when the ceiling exceeds the bucket's capacity: `acquire` clamps to
    #: capacity, so refunding against `max_tokens` would hand back MORE than was
    #: taken and leak quota on every call.
    output_charged: int = 0
    waited_seconds: float = 0.0
    _settled: bool = field(default=False, init=False)

    @property
    def reserved(self) -> int:
        """Total tokens held: the estimated prompt plus the output ceiling."""
        return self.prompt_tokens + self.max_tokens

    def settle(self, usage: Any = None) -> int:
        """Release the reserved budget the response did not use.

        Args:
            usage: the response's usage object or dict, if available. Only the
                OUTPUT budget is reconciled — the prompt estimate stays charged
                even when the reported prompt is smaller, because releasing on
                an under-estimate would let a systematically low estimator
                inflate the effective quota.

        Returns:
            Tokens released. Zero when the response used its full ceiling, or
            when usage is unavailable and the conservative reservation stands.
        """
        if self._settled:
            return 0
        self._settled = True

        completion = _usage_field(usage, "completion_tokens", "output_tokens")
        if completion is None:
            # No usage reported: keep the pessimistic reservation. Refunding a
            # guess would defeat the point of reserving the ceiling.
            return 0

        charged = self.output_charged or self.max_tokens
        unused = charged - int(completion)
        if unused < 0:
            # The response outran the reservation. Carry the difference as a
            # debit so the next call pays for it, rather than dropping it and
            # letting the bucket believe it spent less than it did.
            self.limiter.output_tokens.debit(-unused)
            return 0
        if unused == 0:
            return 0
        self.limiter.output_tokens.release(unused)
        return unused

    def __enter__(self) -> "Reservation":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        # Releases the full output ceiling if the caller never settled — an
        # exception must not leave the budget charged until it refills.
        if not self._settled:
            self._settled = True
            self.limiter.output_tokens.release(self.output_charged or self.max_tokens)


def _usage_field(usage: Any, *names: str) -> int | None:
    """Read the first present field from a usage object or dict."""
    if usage is None:
        return None
    for name in names:
        value = None
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if isinstance(value, (int, float)):
            return int(value)
    return None


class EndpointLimiter:
    """Input, output and request limits for a single serving endpoint.

    Three buckets, because Databricks meters three things independently:

        input_tokens   ITPM — charged the estimated prompt, never refunded
        output_tokens  OTPM — reserved at the `max_tokens` ceiling, refunded
                       down to actual usage once the response lands
        requests       RPM — derived from QPH when RPM is not published

    Only the output bucket uses the reserve/release cycle, because it is the
    only one whose true cost is unknown at request time. The prompt is measured
    before the call, so it is simply charged.
    """

    def __init__(
        self,
        model: str,
        input_tokens_per_minute: int | None = None,
        output_tokens_per_minute: int | None = None,
        requests_per_minute: int | None = None,
        burst_fraction: float = DEFAULT_BURST_FRACTION,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        asleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.model = model
        itpm = input_tokens_per_minute or _configured(
            "DATABRICKS_ITPM", model, DEFAULT_INPUT_TOKENS_PER_MINUTE
        )
        otpm = output_tokens_per_minute or _configured(
            "DATABRICKS_OTPM", model, DEFAULT_OUTPUT_TOKENS_PER_MINUTE
        )
        rpm = requests_per_minute or _configured(
            "DATABRICKS_RPM", model, DEFAULT_REQUESTS_PER_MINUTE
        )
        self.input_tokens = TokenBucket(
            itpm, itpm * burst_fraction, now=now, sleep=sleep, asleep=asleep
        )
        self.output_tokens = TokenBucket(
            otpm, otpm * burst_fraction, now=now, sleep=sleep, asleep=asleep
        )
        self.requests = TokenBucket(
            rpm, max(1.0, rpm * burst_fraction), now=now, sleep=sleep, asleep=asleep
        )
        #: Cumulative seconds spent throttled on this endpoint, for reporting.
        self.waited_seconds = 0.0

    def _output_charge(self, max_tokens: int) -> int:
        """What the output bucket can actually be charged for this ceiling.

        `acquire` clamps anything above capacity, so this mirrors that clamp and
        the reservation remembers it. A ceiling above capacity also means the
        burst allowance is too small for the workload — every such call drains
        the bucket completely and then paces on refill, which is correct but
        worth saying out loud once.
        """
        capacity = self.output_tokens.capacity
        if max_tokens > capacity:
            logger.warning(
                "RateLimiter | %s | max_tokens=%d exceeds the output burst "
                "capacity of %.0f; raise RATE_LIMIT_BURST_FRACTION or lower "
                "max_tokens, otherwise every call waits on refill",
                self.model, max_tokens, capacity,
            )
            return int(capacity)
        return int(max_tokens)

    def _input_charge(self, prompt_tokens: int) -> int:
        """Mirror of `_output_charge` for the input bucket.

        `acquire` clamps silently, so a prompt larger than the burst capacity is
        charged as capacity and the rest is spent unaccounted — the limiter then
        believes it can afford several times the real rate and the 429 arrives
        from the server instead. Output has warned about this since the start;
        input did the same clamp with no warning at all, which is how a prompt
        four times the burst allowance went unnoticed.

        A prompt above the FULL per-minute quota is worse than a tuning problem:
        no pacing can make one request fit in a minute's budget, so that case is
        called out separately.
        """
        capacity = self.input_tokens.capacity
        if prompt_tokens > self.input_tokens.rate:
            logger.error(
                "RateLimiter | %s | prompt of %d tokens EXCEEDS the whole ITPM "
                "quota of %.0f. No pacing can fit this request into a minute's "
                "budget — shrink the prompt (history/compaction) or raise "
                "DATABRICKS_ITPM_* to the endpoint's real quota. 429s are "
                "expected until then.",
                self.model, prompt_tokens, self.input_tokens.rate,
            )
        elif prompt_tokens > capacity:
            logger.warning(
                "RateLimiter | %s | prompt of %d tokens exceeds the input burst "
                "capacity of %.0f; the charge is clamped, so the limiter is "
                "under-counting by %d per call. Raise RATE_LIMIT_BURST_FRACTION "
                "or shrink the prompt.",
                self.model, prompt_tokens, capacity, prompt_tokens - int(capacity),
            )
        return int(prompt_tokens)

    def reserve(self, prompt_tokens: int, max_tokens: int) -> Reservation:
        """Acquire budget for one request, blocking until it is available.

        Ordered cheapest-first — requests, then input, then output — so a call
        that will ultimately wait on the tight output bucket is not also
        sitting on a request slot and input headroom while it waits.
        """
        charged = self._output_charge(max_tokens)
        self._input_charge(prompt_tokens)
        waited = self.requests.acquire(1.0)
        waited += self.input_tokens.acquire(prompt_tokens)
        waited += self.output_tokens.acquire(charged)
        self.waited_seconds += waited
        return Reservation(
            limiter=self,
            prompt_tokens=int(prompt_tokens),
            max_tokens=int(max_tokens),
            output_charged=charged,
            waited_seconds=waited,
        )

    async def reserve_async(self, prompt_tokens: int, max_tokens: int) -> Reservation:
        """Async twin of `reserve`, for callers on an event loop."""
        charged = self._output_charge(max_tokens)
        self._input_charge(prompt_tokens)
        waited = await self.requests.acquire_async(1.0)
        waited += await self.input_tokens.acquire_async(prompt_tokens)
        waited += await self.output_tokens.acquire_async(charged)
        self.waited_seconds += waited
        return Reservation(
            limiter=self,
            prompt_tokens=int(prompt_tokens),
            max_tokens=int(max_tokens),
            output_charged=charged,
            waited_seconds=waited,
        )


class RateLimiter:
    """The pipeline's limiters, one per endpoint.

    Shared deliberately. The quota is per-model across the whole workspace, so
    the converter and the fixers competing for `databricks-claude-sonnet-4-6`
    must draw from the SAME bucket — separate limiters would each stay under the
    limit while their sum went over it, which is the failure this exists to
    prevent.
    """

    def __init__(
        self,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        asleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._endpoints: dict[str, EndpointLimiter] = {}
        self._lock = threading.Lock()
        self._now = now
        self._sleep = sleep
        self._asleep = asleep

    def for_model(self, model: str) -> EndpointLimiter:
        """Limiter for `model`, created on first use.

        The provider prefix is stripped, so "databricks/databricks-claude-
        sonnet-4-6" and the bare endpoint name share one bucket rather than
        quietly getting one each.
        """
        endpoint = str(model).split("/")[-1]
        with self._lock:
            limiter = self._endpoints.get(endpoint)
            if limiter is None:
                limiter = EndpointLimiter(
                    endpoint, now=self._now, sleep=self._sleep, asleep=self._asleep
                )
                self._endpoints[endpoint] = limiter
            return limiter

    def reserve(self, model: str, prompt_tokens: int, max_tokens: int) -> Reservation:
        """Block until `model` has room, then hold the budget."""
        return self.for_model(model).reserve(prompt_tokens, max_tokens)

    async def reserve_async(
        self, model: str, prompt_tokens: int, max_tokens: int
    ) -> Reservation:
        """Await until `model` has room, then hold the budget."""
        return await self.for_model(model).reserve_async(prompt_tokens, max_tokens)

    def report(self) -> dict[str, float]:
        """Seconds spent throttled per endpoint — for logging after a run."""
        with self._lock:
            return {
                name: limiter.waited_seconds
                for name, limiter in self._endpoints.items()
                if limiter.waited_seconds
            }


#: The process-wide limiter. Import this; do not construct another, or the
#: buckets stop being shared and the quota is double-spent.
LIMITER = RateLimiter()


def is_rate_limit_error(exc: BaseException) -> bool:
    """True if `exc` looks like a 429 from the serving endpoint."""
    if getattr(exc, "status_code", None) == 429:
        return True
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


def retry_after_seconds(exc: BaseException, default: float = 20.0) -> float:
    """Seconds to wait before retrying a rate-limited request.

    Providers surface this inconsistently — a response header, a number in the
    message — so this checks the shapes that actually turn up and falls back to
    `default` rather than guessing zero, which would hammer an endpoint that
    just asked to be left alone.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None and hasattr(headers, "get"):
        for key in ("retry-after", "Retry-After", "x-ratelimit-reset"):
            value = headers.get(key)
            if value:
                try:
                    return min(float(value), MAX_RETRY_AFTER_SECONDS)
                except (TypeError, ValueError):
                    pass

    match = re.search(r"retry[- ]after[:\s]+(\d+(?:\.\d+)?)", str(exc), re.I)
    if match:
        return min(float(match.group(1)), MAX_RETRY_AFTER_SECONDS)
    return default


def _log_reservation(model: str, reservation: "Reservation") -> None:
    """One line per admitted request.

    `waited` is the whole point of the line: zero means the burst buffer
    absorbed the call, anything above zero is the limiter actively pacing the
    pipeline. A run whose waits climb steadily is one whose configured quota is
    below what the workload wants.
    """
    logger.info(
        "RateLimiter | %s | prompt=%d reserved=%d waited=%.2fs",
        model,
        reservation.prompt_tokens,
        reservation.reserved,
        reservation.waited_seconds,
    )


def _log_settlement(model: str, reservation: "Reservation", released: int) -> None:
    """Debug-level record of what the response handed back."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    bucket = reservation.limiter.output_tokens
    logger.debug(
        "RateLimiter | %s | released=%d remaining=%.0f/%.0f",
        model, released, bucket._tokens, bucket.capacity,
    )


def call_with_rate_limit(
    complete: Callable[..., Any],
    *,
    model: str,
    messages: Sequence[Any],
    max_tokens: int,
    limiter: RateLimiter | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> Any:
    """Run one LiteLLM completion under the local rate limit.

    Implements the full sequence: estimate the prompt, reserve
    prompt + max_tokens, acquire from the bucket, call, retry a 429 after its
    Retry-After, and release the unused output budget.

    Args:
        complete: the completion callable, e.g. `litellm.completion`.
        model: endpoint name, with or without a provider prefix.
        messages: the LiteLLM messages list; used for the prompt estimate.
        max_tokens: the request's output ceiling — reserved in full, refunded
            down to actual usage once the response lands.
        limiter: override the process-wide limiter (tests, isolation).
        max_attempts: total tries, including the first.
        sleep: injected for testability.
        **kwargs: forwarded to `complete` unchanged.

    Returns:
        Whatever `complete` returns.

    Raises:
        The last exception if every attempt was rate limited.
    """
    limiter = limiter or LIMITER
    prompt_tokens = estimate_message_tokens(messages)
    last_error: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        # Re-reserved on every attempt: a retry is a new request against the
        # quota, and the previous reservation was already settled below.
        reservation = limiter.reserve(model, prompt_tokens, max_tokens)
        _log_reservation(model, reservation)
        try:
            response = complete(
                model=model, messages=messages, max_tokens=max_tokens, **kwargs
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            # Release the whole output ceiling: nothing was generated that we
            # know of, and holding it would throttle the retry we are about to
            # make.
            reservation.limiter.output_tokens.release(reservation.output_charged or reservation.max_tokens)
            reservation._settled = True
            if not is_rate_limit_error(exc) or attempt == max_attempts:
                raise
            last_error = exc
            delay = retry_after_seconds(exc)
            logger.warning(
                "RateLimiter | %s | RATE LIMITED (attempt %d/%d); retrying in %.0fs — "
                "the configured quota is above what the endpoint actually allows",
                model, attempt, max_attempts, delay,
            )
            sleep(delay)
            continue

        _log_settlement(model, reservation, reservation.settle(_response_usage(response)))
        return response

    assert last_error is not None
    raise last_error


def _response_usage(response: Any) -> Any:
    """Pull the usage record off a LiteLLM response, whatever shape it is."""
    if response is None:
        return None
    if isinstance(response, dict):
        return response.get("usage")
    return getattr(response, "usage", None)


def _is_stream(response: Any) -> bool:
    """True if the response is a streamed iterator rather than a finished reply.

    A streamed call returns immediately with an iterator; the tokens are only
    produced as it is consumed, so the usage record does not exist yet.
    """
    if response is None or isinstance(response, (dict, str, bytes)):
        return False
    return hasattr(response, "__aiter__") or hasattr(response, "__anext__")



async def acall_with_rate_limit(
    acomplete: Callable[..., Awaitable[Any]],
    *,
    model: str,
    messages: Sequence[Any],
    max_tokens: int,
    limiter: RateLimiter | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    asleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    **kwargs: Any,
) -> Any:
    """Async twin of `call_with_rate_limit`, for `acompletion`.

    Same sequence — estimate, reserve prompt + max_tokens, acquire, call, retry
    a 429 after its Retry-After, release the unused output budget — but every
    wait yields to the event loop instead of blocking it.
    """
    limiter = limiter or LIMITER
    prompt_tokens = estimate_message_tokens(messages)
    last_error: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        reservation = await limiter.reserve_async(model, prompt_tokens, max_tokens)
        _log_reservation(model, reservation)
        try:
            response = await acomplete(
                model=model, messages=messages, max_tokens=max_tokens, **kwargs
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            # Nothing was generated that we know of; holding the output ceiling
            # would only throttle the retry we are about to make.
            reservation.limiter.output_tokens.release(reservation.output_charged or reservation.max_tokens)
            reservation._settled = True
            if not is_rate_limit_error(exc) or attempt == max_attempts:
                raise
            last_error = exc
            delay = retry_after_seconds(exc)
            logger.warning(
                "RateLimiter | %s | RATE LIMITED (attempt %d/%d); retrying in %.0fs — "
                "the configured quota is above what the endpoint actually allows",
                model, attempt, max_attempts, delay,
            )
            await asleep(delay)
            continue

        if _is_stream(response):
            # Streamed: the tokens do not exist yet, so there is nothing to
            # reconcile against. Keep the pessimistic reservation and return
            # litellm's own object UNTOUCHED.
            #
            # An earlier version returned a settling proxy here so the unused
            # ceiling could be refunded when the stream finished. That proxy is
            # not a litellm `CustomStreamWrapper`, and ADK feeds this value
            # straight into its Pydantic-typed response path — a foreign object
            # there fails serialization. Over-reserving costs throughput;
            # substituting the type breaks the caller, so the reservation stands.
            logger.debug(
                "RateLimiter | %s | streamed response: holding the full %d-token "
                "output reservation (no usage to reconcile against)",
                model, reservation.output_charged or reservation.max_tokens,
            )
            return response

        _log_settlement(model, reservation, reservation.settle(_response_usage(response)))
        return response

    assert last_error is not None
    raise last_error

#: Alias matching the naming used by `litellm_patch`.
call_with_rate_limit_async = acall_with_rate_limit

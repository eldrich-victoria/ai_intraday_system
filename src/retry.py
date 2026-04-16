# -*- coding: utf-8 -*-
"""Reusable retry decorator with exponential backoff for external API calls."""

import functools
import logging
import time
from typing import Any, Callable, Optional, Sequence, Type

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    exceptions: Optional[Sequence[Type[BaseException]]] = None,
) -> Callable:
    """
    Decorator that retries a function on failure with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts after initial failure.
        base_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay cap in seconds.
        backoff_factor: Multiplier for delay on each subsequent retry.
        exceptions: Tuple of exception types to catch. Defaults to (Exception,).

    Returns:
        Decorated function with retry logic.
    """
    caught: tuple = tuple(exceptions) if exceptions else (Exception,)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            last_exc: Optional[BaseException] = None
            for attempt in range(1, max_retries + 2):  # 1 initial + max_retries
                try:
                    return func(*args, **kwargs)
                except caught as exc:
                    last_exc = exc
                    if attempt > max_retries:
                        logger.error(
                            "All {} retries exhausted for {}: {}".format(
                                max_retries, func.__name__, exc
                            )
                        )
                        raise
                    logger.warning(
                        "Attempt {}/{} for {} failed ({}). "
                        "Retrying in {:.1f}s...".format(
                            attempt,
                            max_retries + 1,
                            func.__name__,
                            exc,
                            delay,
                        )
                    )
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
            # Should not reach here, but raise last exception if it does.
            if last_exc is not None:
                raise last_exc  # pragma: no cover

        return wrapper

    return decorator

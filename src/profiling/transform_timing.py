import time
from typing import Any, Iterable, List


class TimedTransform:
    """Wrap a transform so each stage logs its elapsed runtime."""

    def __init__(self, transform: Any, *, name: str | None = None) -> None:
        self.transform = transform
        self.name = name or type(transform).__name__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return self.transform(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            print(f"[transform-timing] {self.name}: {elapsed:.6f}s", flush=True)


class StepTimer:
    """Context manager for per-step training-time logging."""

    def __init__(self, name: str, *, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled
        self.start = 0.0

    def __enter__(self) -> "StepTimer":
        if self.enabled:
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.start
        print(f"[step-timing] {self.name}: {elapsed:.6f}s", flush=True)


def timed_transform_pipeline(transform_steps: Iterable[Any], *, enabled: bool = False) -> List[Any]:
    """Attach runtime logging wrappers only when profiling is explicitly enabled."""
    if not enabled:
        return list(transform_steps)

    timed: List[Any] = []
    for step in transform_steps:
        name = getattr(step, "__class__", type(step)).__name__
        timed.append(TimedTransform(step, name=name))
    return timed

from contextlib import contextmanager
from functools import wraps
from time import perf_counter
from typing import Any


class RunTimings:
    def __init__(self) -> None:
        """Initialize an empty timing accumulator."""
        self.entries = {}

    @contextmanager
    def section(self, group: str, name: str, queue=None) -> None:
        """Measure one section and optionally synchronize an OpenCL queue."""
        if queue is not None:
            queue.finish()

        start = perf_counter()
        try:
            yield
        finally:
            try:
                if queue is not None:
                    queue.finish()
            finally:
                elapsed = perf_counter() - start
                entry = self.entries.setdefault((group, name), [0, 0.0])
                entry[0] += 1
                entry[1] += elapsed

    def report(self, total: float, status: str) -> None:
        """Print grouped call counts, durations, shares, and mean runtimes."""
        print(f"Timing report ({status}) - total run: {total:.6f} s")
        width = max(62, max((len(name) for _, name in self.entries), default=0))

        groups = {key[0] for key in self.entries}
        for group in sorted(groups, key=lambda name: (name != "solver", name)):
            print(f"[{group}]")
            print(
                f"{'Section / method':{width}} "
                f"{'Calls':>8} {'Total (s)':>12} {'% run':>9} {'Avg (ms)':>12}"
            )
            entries = (
                (name, values)
                for (entry_group, name), values in self.entries.items()
                if entry_group == group
            )
            for name, (calls, elapsed) in sorted(
                entries,
                key=lambda item: item[1][1],
                reverse=True,
            ):
                percent = 100.0 * elapsed / total if total > 0.0 else 0.0
                average_ms = 1000.0 * elapsed / calls if calls else 0.0
                print(
                    f"{name:{width}} {calls:8d} {elapsed:12.6f} "
                    f"{percent:8.2f}% {average_ms:12.6f}"
                )


def profiled_run(function: Any) -> Any:
    """Decorate a solver function with timing collection and reporting."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("timings") is not None:
            return function(*args, **kwargs)

        timings = RunTimings()
        kwargs["timings"] = timings
        start = perf_counter()
        status = "failed / partial"
        try:
            result = function(*args, **kwargs)
            status = "finished / clean stop"
            return result
        finally:
            timings.report(perf_counter() - start, status)

    return wrapped

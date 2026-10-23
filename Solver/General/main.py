"""Continuum Liquid solver entry point and runtime message channel."""

import threading
from time import perf_counter
from typing import Any, Callable


MessageCallback = Callable[[dict[str, Any]], None]

_message_callback: MessageCallback | None = None
_message_callback_lock = threading.Lock()


def set_message_callback(callback: MessageCallback | None) -> None:
    """Set the callback used for solver runtime messages."""
    global _message_callback
    with _message_callback_lock:
        _message_callback = callback


def clear_message_callback() -> None:
    """Remove the current solver runtime callback."""
    global _message_callback
    with _message_callback_lock:
        _message_callback = None


def emit_message(message: dict[str, Any]) -> None:
    """Forward a runtime message to the active caller."""
    with _message_callback_lock:
        callback = _message_callback
    if callback is not None:
        callback(message)


def main(config: dict[str, Any]) -> None:
    """Select the configured backend and execute the solver."""
    total_start_time = perf_counter()
    try:
        simulation_config = config.get("simulation") or {}
        settings = simulation_config.get("settings") or {}
        solver_backend = str(settings.get("solver_backend", "CPU")).strip().upper()

        if solver_backend in {"CPU", "GPU"}:
            from ..Kernel import solver as solver_kernel_module

            return solver_kernel_module.solver(config)

        raise ValueError(f"Unsupported solver backend: {solver_backend}")
    finally:
        total_runtime = perf_counter() - total_start_time
        print(f"Bake runtime: {total_runtime:.3f} s")
        print("################################################################")

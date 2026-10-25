from __future__ import annotations

import json
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any

import numpy as np
import pyopencl as cl

CACHE_FILENAME = "particle_cache.json"
FRAME_DIRECTORY = "particles"
FRAME_PATTERN = "particles_{frame:06d}.bin"
WRITER_COUNT = 2
PINNED_MEMORY_SLOTS = WRITER_COUNT + 1


def create_cache(directory, fps, start_frame, end_frame, particle_count=None):
    directory = Path(directory)
    (directory / FRAME_DIRECTORY).mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": 2,
        "fps": int(fps),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "variable_particle_count": True,
        "dtype": "float32",
        "components": 3,
        "frame_directory": FRAME_DIRECTORY,
        "frame_pattern": FRAME_PATTERN,
    }
    (directory / CACHE_FILENAME).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return directory


def write_particles(cache_directory, particles, frame):
    positions = np.ascontiguousarray(particles, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Particle positions must have shape (N, 3)")
    filepath = (
        Path(cache_directory) / FRAME_DIRECTORY / FRAME_PATTERN.format(frame=int(frame))
    )
    positions.tofile(filepath)


def _pinned_array(
    queue: cl.CommandQueue, capacity: int
) -> tuple[np.ndarray, cl.Buffer]:
    capacity = max(1, int(capacity))
    host_buffer = cl.Buffer(
        queue.context,
        cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR,
        size=capacity * 3 * np.dtype(np.float32).itemsize,
    )
    host_array, _ = cl.enqueue_map_buffer(
        queue,
        host_buffer,
        cl.map_flags.READ | cl.map_flags.WRITE,
        0,
        (capacity, 3),
        np.float32,
        is_blocking=True,
    )
    return host_array, host_buffer


def _ensure_capacity(
    slot: dict[str, Any], host_queue: cl.CommandQueue, required: int
) -> None:
    required = max(1, int(required))
    if required <= slot["capacity"]:
        return

    slot["particles"], host_buffer = _pinned_array(host_queue, required)
    # Keep superseded mapped buffers alive until shutdown. Particle counts tend
    # to grow, so this is uncommon and avoids unmapping while a view exists.
    slot["host_buffers"].append(host_buffer)
    slot["capacity"] = required


def _raise_writer_error(state: dict[str, Any]) -> None:
    with state["error_lock"]:
        if state["errors"]:
            raise RuntimeError("Asynchronous particle writer failed") from state[
                "errors"
            ][0]


def _writer_loop(state: dict[str, Any]) -> None:
    while True:
        slot = state["write_queue"].get()
        try:
            if slot is None:
                return
            write_particles(
                state["cache_directory"],
                slot["particles"][: slot["particle_count"]],
                slot["frame"],
            )
        except BaseException as exc:
            with state["error_lock"]:
                state["errors"].append(exc)
        finally:
            if slot is not None:
                state["free_slots"].put(slot)
            state["write_queue"].task_done()


def create_output(context: cl.Context, cache_directory) -> dict[str, Any]:
    state = {
        "cache_directory": Path(cache_directory),
        "host_queue": cl.CommandQueue(context),
        "free_slots": Queue(PINNED_MEMORY_SLOTS),
        "write_queue": Queue(PINNED_MEMORY_SLOTS),
        "threads": [],
        "errors": [],
        "error_lock": Lock(),
        "closed": False,
    }

    for _ in range(PINNED_MEMORY_SLOTS):
        state["free_slots"].put(
            {
                "particles": None,
                "host_buffers": [],
                "capacity": 0,
                "particle_count": 0,
                "frame": 0,
            }
        )

    for index in range(WRITER_COUNT):
        thread = Thread(
            target=_writer_loop,
            args=(state,),
            name=f"ContinuumParticleWriter-{index + 1}",
            daemon=True,
        )
        thread.start()
        state["threads"].append(thread)

    return state


def _acquire_slot(state: dict[str, Any], particle_count: int, frame: int):
    _raise_writer_error(state)
    if state["closed"]:
        raise RuntimeError("Particle output is already closed")

    slot = state["free_slots"].get()
    _ensure_capacity(slot, state["host_queue"], particle_count)
    slot["particle_count"] = int(particle_count)
    slot["frame"] = int(frame)
    return slot


def submit_host(state: dict[str, Any], particles, frame: int) -> None:
    positions = np.ascontiguousarray(particles, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Particle positions must have shape (N, 3)")

    slot = _acquire_slot(state, len(positions), frame)
    if len(positions):
        np.copyto(slot["particles"][: len(positions)], positions)
    state["write_queue"].put(slot)


def submit_output(
    state: dict[str, Any],
    *,
    queue: cl.CommandQueue,
    particles: cl.Buffer,
    particle_count: int,
    frame: int,
) -> None:
    slot = _acquire_slot(state, particle_count, frame)
    if particle_count:
        event = cl.enqueue_copy(
            queue,
            slot["particles"][:particle_count],
            particles,
            is_blocking=False,
        )
        event.wait()
    state["write_queue"].put(slot)


def close_output(state: dict[str, Any]) -> None:
    if state["closed"]:
        return

    state["closed"] = True
    state["write_queue"].join()
    for _ in state["threads"]:
        state["write_queue"].put(None)
    for thread in state["threads"]:
        thread.join()
    _raise_writer_error(state)

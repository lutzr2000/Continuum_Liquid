import json
from pathlib import Path

import numpy as np


CACHE_FILENAME = "particle_cache.json"
FRAME_DIRECTORY = "particles"
FRAME_PATTERN = "particles_{frame:06d}.bin"


def create_cache(directory, fps, start_frame, end_frame, particle_count):
    directory = Path(directory)
    (directory / FRAME_DIRECTORY).mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": 1,
        "fps": int(fps),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "particle_count": int(particle_count),
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
        Path(cache_directory)
        / FRAME_DIRECTORY
        / FRAME_PATTERN.format(frame=int(frame))
    )
    positions.tofile(filepath)

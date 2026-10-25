import json
from pathlib import Path

import numpy as np


CACHE_FILENAME = "particle_cache.json"
FRAME_DIRECTORY = "particles"
FRAME_PATTERN = "particles_{frame:06d}.bin"
DIVERGENCE_DIRECTORY = "pressure_convergence"
DIVERGENCE_PLOT_PATTERN = "divergence_frame_{frame:06d}.png"
DIVERGENCE_DATA_PATTERN = "divergence_frame_{frame:06d}.csv"


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


def write_divergence_plot(cache_directory, frame, series):
    """Write per-W-cycle RMS divergence curves for all substeps of a frame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(cache_directory) / DIVERGENCE_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    frame = int(frame)

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    data_columns = []
    for substep, values in enumerate(series, start=1):
        values = np.asarray(values, dtype=np.float64)
        cycles = np.arange(values.size, dtype=np.int32)
        axis.plot(cycles, values, marker="o", markersize=3, label=f"Substep {substep}")
        data_columns.append((cycles, values))

    axis.set_title(f"Pressure projection convergence — frame {frame}")
    axis.set_xlabel("Multigrid W-cycle iteration")
    axis.set_ylabel("RMS divergence [1/s]")
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.3)
    if len(series) > 1:
        axis.legend()
    figure.savefig(directory / DIVERGENCE_PLOT_PATTERN.format(frame=frame), dpi=160)
    plt.close(figure)

    max_rows = max((len(values) for _, values in data_columns), default=0)
    header = ["w_cycle"] + [f"substep_{index}" for index in range(1, len(data_columns) + 1)]
    rows = []
    for row in range(max_rows):
        values = [str(row)]
        for _, column in data_columns:
            values.append(f"{column[row]:.9e}" if row < len(column) else "")
        rows.append(",".join(values))
    (directory / DIVERGENCE_DATA_PATTERN.format(frame=frame)).write_text(
        ",".join(header) + "\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )

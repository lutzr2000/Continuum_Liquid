from pathlib import Path

import numpy as np

FIELD_DTYPE = np.float32


def load_static_masks(simulation, grid_shape):
    bake_path = Path(simulation["output_path"])
    sources = []
    for entry in simulation.get("sources") or []:
        mask_path = bake_path / entry["mask_file"]
        with np.load(mask_path) as archive:
            mask = np.asarray(archive["mask"], dtype=np.bool_)
        if mask.shape != tuple(grid_shape):
            raise ValueError(f"Source mask has invalid shape: {mask_path}")
        sources.append({**entry, "mask": mask})

    obstacle_path = simulation.get("obstacle_mask_file")
    if obstacle_path:
        with np.load(bake_path / obstacle_path) as archive:
            obstacle_mask = np.asarray(archive["mask"], dtype=np.bool_)
    else:
        obstacle_mask = np.zeros(grid_shape, dtype=np.bool_)
    if obstacle_mask.shape != tuple(grid_shape):
        raise ValueError("Obstacle mask has invalid shape")
    return sources, obstacle_mask


def particle_cells(positions, delta, grid_shape):
    if len(positions) == 0:
        return np.empty((0, 3), dtype=np.int32)
    cells = np.floor(np.asarray(positions) / delta).astype(np.int32)
    return np.clip(cells, 0, np.asarray(grid_shape, dtype=np.int32) - 1)


def mask_membership(positions, mask, delta):
    cells = particle_cells(positions, delta, mask.shape)
    return mask[cells[:, 0], cells[:, 1], cells[:, 2]] if len(cells) else np.zeros(0, dtype=np.bool_)


def remove_masked(particles, velocities, mask, delta):
    keep = ~mask_membership(particles, mask, delta)
    return particles[keep], velocities[keep]


def fill_source(particles, velocities, source, obstacle_mask, delta, rng):
    target = max(1, int(source.get("particle_density", 8)))
    source_mask = np.asarray(source["mask"], dtype=np.bool_) & ~obstacle_mask
    cells = np.argwhere(source_mask)
    if len(cells) == 0:
        return particles, velocities

    counts = np.zeros(source_mask.shape, dtype=np.int32)
    particle_cell_indices = particle_cells(particles, delta, source_mask.shape)
    if len(particle_cell_indices):
        flat_indices = np.ravel_multi_index(
            particle_cell_indices.T,
            source_mask.shape,
        )
        counts.flat[:] = np.bincount(
            flat_indices,
            minlength=source_mask.size,
        ).astype(np.int32, copy=False)
    missing = np.maximum(0, target - counts[tuple(cells.T)])
    spawn_cells = np.repeat(cells, missing, axis=0)
    if len(spawn_cells) == 0:
        return particles, velocities

    spawned = (spawn_cells + rng.random((len(spawn_cells), 3))) * delta
    source_velocity = np.asarray(source.get("velocity", (0.0, 0.0, 0.0)), dtype=FIELD_DTYPE)
    spawned_velocity = np.broadcast_to(source_velocity, spawned.shape).copy()
    return (
        np.ascontiguousarray(np.concatenate((particles, spawned)), dtype=FIELD_DTYPE),
        np.ascontiguousarray(np.concatenate((velocities, spawned_velocity)), dtype=FIELD_DTYPE),
    )


def apply_sources(particles, velocities, sources, obstacle_mask, delta, rng, initial=False):
    for source in sources:
        if source.get("type") == "OUTFLOW":
            particles, velocities = remove_masked(
                particles, velocities, source["mask"], delta
            )
    for source in sources:
        source_type = source.get("type")
        if source_type == "INFLOW" or (initial and source_type == "GEOMETRY"):
            particles, velocities = fill_source(
                particles, velocities, source, obstacle_mask, delta, rng
            )
    return particles, velocities


def resolve_obstacle_collisions(previous, particles, velocities, obstacle_mask, delta):
    colliding = mask_membership(particles, obstacle_mask, delta)
    particles[colliding] = previous[colliding]
    velocities[colliding] = 0.0
    return particles, velocities

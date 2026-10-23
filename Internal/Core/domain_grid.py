def grid_shape(domain_node):
    """Return the cell counts used by the liquid solver for a domain node."""
    resolution = float(domain_node.resolution)
    if resolution <= 0.0:
        raise ValueError("resolution must be greater than zero")

    return tuple(
        int(float(getattr(domain_node, axis)) / resolution)
        for axis in ("lx", "ly", "lz")
    )


def dimensions(domain_node):
    """Return the actual simulated domain dimensions in Blender units."""
    resolution = float(domain_node.resolution)
    return tuple(count * resolution for count in grid_shape(domain_node))

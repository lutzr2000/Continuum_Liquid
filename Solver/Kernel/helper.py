import os
import warnings
from pathlib import Path

import numpy as np
import pyopencl as cl

from . import kernel_config

mf = cl.mem_flags
FIELD_DTYPE = kernel_config.FIELD_DTYPE

OPENCL_CACHE_DIR = (
    Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
    / "ContinuumLiquid"
    / "OpenCL"
)

PROGRAM_CACHE = {}


def load_program(context, path):
    path = path.resolve()
    cache_key = (int(context.int_ptr), path)

    cached_kernels = PROGRAM_CACHE.get(cache_key)
    if cached_kernels is not None:
        return cached_kernels

    OPENCL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cl.CompilerWarning)
        program = cl.Program(
            context,
            path.read_text(encoding="utf-8"),
        ).build(
            options=[
                f"-DTILE_SIZE={kernel_config.TILE_SIZE}",
                f"-DREDUCTION_THREADS_PER_BLOCK={kernel_config.REDUCTION_THREADS_PER_BLOCK}",
                f'-I "{path.parent}"',
            ],
            cache_dir=str(OPENCL_CACHE_DIR),
        )

    kernels = {kernel.function_name: kernel for kernel in program.all_kernels()}
    PROGRAM_CACHE[cache_key] = kernels
    return kernels


def select_opencl_device(settings):
    """Resolve the first UI-selected OpenCL device from exported indices."""
    platforms = cl.get_platforms()
    if not platforms:
        raise RuntimeError("No OpenCL platform found")

    backend = str(settings.get("solver_backend", "GPU")).upper()
    device_type = {
        "CPU": cl.device_type.CPU,
        "GPU": cl.device_type.GPU,
    }.get(backend)
    selected_devices = settings.get("compute_devices") or ()
    if not selected_devices:
        raise RuntimeError(f"No OpenCL {backend} device selected")
    selected = selected_devices[0]
    platform_index = int(selected["platform_index"])
    device_index = int(selected["device_index"])
    platform = platforms[platform_index]
    return platform.get_devices(device_type=device_type)[device_index]


def to_device(context, array):
    array = np.ascontiguousarray(array)
    return cl.Buffer(
        context,
        mf.READ_WRITE | mf.COPY_HOST_PTR,
        hostbuf=array,
    )


def device_array(context, shape, dtype=FIELD_DTYPE):
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    return cl.Buffer(context, mf.READ_WRITE, size=max(size, 1))


def fill_device(queue, buffer, value, dtype=FIELD_DTYPE):
    cl.enqueue_fill_buffer(
        queue,
        buffer,
        np.asarray(value, dtype=dtype),
        0,
        buffer.size,
    )

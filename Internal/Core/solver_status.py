MAX_SELECTABLE_DEVICES = 8


def detect_opencl_devices():
    """Return JSON-friendly OpenCL CPU and GPU device descriptions."""
    devices = {"CPU": [], "GPU": []}
    try:
        import pyopencl as cl

        type_by_backend = {"CPU": cl.device_type.CPU, "GPU": cl.device_type.GPU}
        for platform_index, platform in enumerate(cl.get_platforms()):
            for backend, device_type in type_by_backend.items():
                try:
                    platform_devices = platform.get_devices(device_type=device_type)
                except Exception:
                    continue
                for device_index, device in enumerate(platform_devices):
                    if len(devices[backend]) >= MAX_SELECTABLE_DEVICES:
                        break
                    devices[backend].append(
                        {
                            "platform_index": platform_index,
                            "device_index": device_index,
                            "platform_name": str(platform.name).strip(),
                            "device_name": str(device.name).strip(),
                        }
                    )
    except Exception:
        pass
    return devices


def refresh_opencl_devices():
    global opencl_devices, cpu_available, gpu_available
    opencl_devices = detect_opencl_devices()
    cpu_available = bool(opencl_devices["CPU"])
    gpu_available = bool(opencl_devices["GPU"])
    return opencl_devices


opencl_devices = {"CPU": [], "GPU": []}
cpu_available = False
gpu_available = False
refresh_opencl_devices()

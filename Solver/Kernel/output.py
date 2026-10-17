import numpy as np
from pxr import Usd, UsdGeom, Vt

from . import kernel_config

FIELD_DTYPE = kernel_config.FIELD_DTYPE


def create_usd(filepath, fps, start_frame, end_frame):
    stage = Usd.Stage.CreateNew(str(filepath))

    stage.SetStartTimeCode(start_frame)
    stage.SetEndTimeCode(end_frame)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)

    points = UsdGeom.Points.Define(stage, "/WaterParticles")
    stage.SetDefaultPrim(points.GetPrim())

    return stage, points


def write_particles(points, particles, frame):
    positions = np.ascontiguousarray(particles, dtype=FIELD_DTYPE)

    usd_positions = Vt.Vec3fArray.FromNumpy(positions)

    points.GetPointsAttr().Set(usd_positions, Usd.TimeCode(frame))


def save_usd(stage):
    stage.GetRootLayer().Save()

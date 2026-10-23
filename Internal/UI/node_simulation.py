import bpy
from bpy.props import EnumProperty, IntProperty

from . import node_base, sockets


def _update_start_frame(self, _context):
    self._clamp_frame_range()


def _update_end_frame(self, _context):
    self._clamp_frame_range()


class ContinuumLiquidSimulationNode(node_base.ContinuumLiquidBaseNode):
    bl_idname = "CONTINUUM_LIQUID_SIMULATION_NODE"
    bl_label = "Simulation"
    bl_icon = "TIME"
    bl_width_default = 260.0
    bl_width_min = 240.0
    bl_width_max = 420.0
    property_groups = (
        ("Time", ("start_frame", "end_frame", "substeps")),
        ("Solver", ("iterations",)),
    )

    solver_backend: EnumProperty(
        name="Solver",
        items=(
            ("GPU", "GPU", "Use the GPU solver"),
            ("CPU", "CPU", "Use the CPU solver"),
        ),
        default="CPU",
        options=set(),
    )  # type: ignore
    start_frame: IntProperty(
        name="Start Frame",
        default=1,
        min=0,
        description="Starting frame of the simulation",
        options=set(),
        update=_update_start_frame,
    )  # type: ignore
    end_frame: IntProperty(
        name="End Frame",
        default=250,
        min=2,
        description="End frame of the simulation",
        options=set(),
        update=_update_end_frame,
    )  # type: ignore
    substeps: IntProperty(
        name="Substeps",
        default=1,
        min=0,
        soft_max=10,
        description="Additional solver steps per output frame",
        options=set(),
    )  # type: ignore
    iterations: IntProperty(
        name="Iterations",
        default=1,
        min=1,
        max=10,
        soft_min=1,
        soft_max=10,
        description="Number of solver iterations",
        options=set(),
    )  # type: ignore

    def _ensure_input_socket(self, name, multi_input=False):
        if name == "Reference Frame":
            socket_type = sockets.ContinuumLiquidReferenceFrameSocket.bl_idname
        elif name == "Forces":
            socket_type = sockets.ContinuumLiquidForceSocket.bl_idname
        else:
            socket_type = sockets.ContinuumLiquidLinkSocket.bl_idname
        return self._ensure_socket(
            self.inputs,
            socket_type,
            name,
            multi_input=multi_input,
        )

    def _sync_node(self):
        reference_frame = self._ensure_input_socket("Reference Frame")
        reference_frame_index = list(self.inputs).index(reference_frame)
        if reference_frame_index != 0:
            self.inputs.move(reference_frame_index, 0)
        self._ensure_input_socket("Domain")
        self._ensure_input_socket("Physics")
        self._ensure_input_socket("Obstacles")
        self._ensure_input_socket("Source", multi_input=True)
        self._ensure_input_socket("Forces", multi_input=True)
        self._ensure_named_output(
            sockets.ContinuumLiquidResultSocket.bl_idname,
            "Result",
        )

    def init(self, context):
        self._apply_node_style()
        scene = getattr(context, "scene", None) or getattr(bpy.context, "scene", None)
        if scene is not None:
            self.start_frame = int(getattr(scene, "frame_start", self.start_frame))
            self.end_frame = int(getattr(scene, "frame_end", self.end_frame))
        self._clamp_frame_range()
        self._sync_node()

    def _clamp_frame_range(self):
        minimum_end_frame = max(1, int(self.start_frame) + 1)
        if int(self.end_frame) < minimum_end_frame:
            self.end_frame = minimum_end_frame
        maximum_start_frame = max(0, int(self.end_frame) - 1)
        if int(self.start_frame) > maximum_start_frame:
            self.start_frame = maximum_start_frame

    def draw_buttons(self, _context, layout):
        solver_row = layout.row(align=True)
        solver_row.prop_enum(self, "solver_backend", "CPU")
        solver_row.prop_enum(self, "solver_backend", "GPU")

        for title, property_names in self.property_groups:
            self._draw_group(layout, title, property_names)

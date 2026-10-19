import bpy

from .node_tree import ContinuumLiquidNodeTree


class ContinuumLiquidBaseNode(bpy.types.Node):
    @classmethod
    def poll(cls, node_tree):
        return node_tree.bl_idname == ContinuumLiquidNodeTree.bl_idname

    def _sync_node(self):
        pass

    def _apply_node_style(self):
        self.use_custom_color = False

    def init(self, _context):
        self._sync_node()

    def copy(self, _node):
        self._sync_node()

    def update(self):
        self._sync_node()

    def _draw_group(self, layout, title, property_names):
        box = layout.box()
        box.label(text=title)
        column = box.column(align=True)
        for property_name in property_names:
            column.prop(self, property_name)

    def _ensure_socket(self, collection, socket_type, name, multi_input=False):
        socket = collection.get(name)
        if socket is None:
            socket = collection.new(socket_type, name, use_multi_input=multi_input)
        if multi_input and hasattr(socket, "link_limit"):
            socket.link_limit = 0
        return socket

    def _ensure_named_output(self, socket_type, name):
        return self._ensure_socket(self.outputs, socket_type, name)

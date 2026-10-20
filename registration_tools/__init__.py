from .constants import get_gpu_available, set_use_gpu
from .functional import (
    register_pair,
    register_stack,
    apply_registration,
    registration_to_vectors,
    registration_to_field,
    compose_registrations,
    save_registration,
    load_registration,
)


def napari_experimental_provide_function():
    """
    Functions offered as nodes to Napari_Flow (napari_flow_editor), which reads this hook
    through the napari.manifest entry point. See registration_tools.nodes.
    """
    from .nodes import NODES

    return list(NODES)

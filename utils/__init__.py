from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict

__all__ = [
    "DICOMProcessor",
    "MeshProcessor",
    "preprocess_mesh_single",
    "resource_path",
]

_MODULE_MAP: Dict[str, str] = {
    "DICOMProcessor": ".dicom_processor",
    "MeshProcessor": ".mesh_processor",
    "preprocess_mesh_single": ".mesh_processor",
    "resource_path": ".helper",
}

if TYPE_CHECKING:  # pragma: no cover
    from .dicom_processor import DICOMProcessor
    from .mesh_processor import MeshProcessor, preprocess_mesh_single
    from .helper import resource_path


def __getattr__(name: str) -> Any:
    module_name = _MODULE_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, __name__)
    attr = getattr(module, name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted(__all__)

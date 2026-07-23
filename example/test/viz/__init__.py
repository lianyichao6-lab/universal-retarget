"""Visualization helpers copied and adapted for ``example/test/tuning_tool.py``."""

from .config_watcher import ConfigWatcher
from .param_map import PARAM_FINGER_MAP, get_param_description
from .skeleton_drawer import SkeletonDrawer


def __getattr__(name):
    if name == "TuningViewer":
        from .tuning_viewer import TuningViewer

        return TuningViewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TuningViewer",
    "SkeletonDrawer",
    "ConfigWatcher",
    "PARAM_FINGER_MAP",
    "get_param_description",
]

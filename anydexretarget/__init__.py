"""AnyDexRetarget - Hand Pose Retargeting Module.

Provides hand pose retargeting from MediaPipe format to dexterous robot hand joint angles.

Main classes:
- Retargeter: High-level unified interface (recommended)
- BaseOptimizer: Low-level optimizer access

Example:
    from anydexretarget import Retargeter

    retargeter = Retargeter.from_yaml("config/mediapipe/mediapipe_shadow_hand.yaml", hand_side="right")
    qpos = retargeter.retarget(raw_keypoints)  # (21, 3) -> (22,)
"""

__all__ = [
    "Retargeter",
    "BaseOptimizer",
    "LPFilter",
    "apply_mediapipe_transformations",
    "CanonicalHandFrame",
]


def __getattr__(name):
    """Load heavyweight retargeting dependencies only when requested."""
    if name == "Retargeter":
        from .retarget import Retargeter
        return Retargeter
    if name in {"BaseOptimizer", "LPFilter"}:
        from .optimizer import BaseOptimizer, LPFilter
        return {"BaseOptimizer": BaseOptimizer, "LPFilter": LPFilter}[name]
    if name == "apply_mediapipe_transformations":
        from .mediapipe import apply_mediapipe_transformations
        return apply_mediapipe_transformations
    if name == "CanonicalHandFrame":
        from .hand_frame import CanonicalHandFrame
        return CanonicalHandFrame
    raise AttributeError(name)

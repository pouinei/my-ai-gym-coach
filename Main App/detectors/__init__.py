"""Exercise-specific computer vision form detectors for Apna AI Gym Coach."""

__all__ = [
    "BicepsCurlDetector",
    "LungesDetector",
    "PushUpDetector",
    "ShoulderPressDetector",
    "SquatDetector",
]


def __getattr__(name: str):
    if name == "BicepsCurlDetector":
        from .biceps_curl import BicepsCurlDetector
        return BicepsCurlDetector
    elif name == "LungesDetector":
        from .lunges import LungesDetector
        return LungesDetector
    elif name == "PushUpDetector":
        from .pushup import PushUpDetector
        return PushUpDetector
    elif name == "ShoulderPressDetector":
        from .shoulder_press import ShoulderPressDetector
        return ShoulderPressDetector
    elif name == "SquatDetector":
        from .squat import SquatDetector
        return SquatDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

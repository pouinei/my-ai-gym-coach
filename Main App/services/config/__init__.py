"""Configuration constants and WebRTC settings for Apna AI Gym Coach."""

from .workout_config import (
    EXERCISE_OPTIONS,
    DETECTOR_CONFIG,
    PROMPT,
    METRICS_FIELDS,
    POSE_CONNECTIONS,
)
from .webrtc_config import (
    get_rtc_configuration,
    DEFAULT_MEDIA_CONSTRAINTS,
)

__all__ = [
    "EXERCISE_OPTIONS",
    "DETECTOR_CONFIG",
    "PROMPT",
    "METRICS_FIELDS",
    "POSE_CONNECTIONS",
    "get_rtc_configuration",
    "DEFAULT_MEDIA_CONSTRAINTS",
]

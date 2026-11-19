import math
from abc import ABC, abstractmethod


class BaseExercise(ABC):
    def __init__(self):
        self.reps = 0
        self.valid_reps = 0
        self.invalid_reps = 0
        self.stage = None
        self.form_status = "Good"  # "Good", "Needs Correction", "Unable to Assess"
        self.form_issue = "None - Form is clean"
        self.form_correction = "Maintain this form!"

    def calculate_angle(self, a, b, c):
        ax, ay = a[0] - b[0], a[1] - b[1]
        cx, cy = c[0] - b[0], c[1] - b[1]

        dot = ax * cx + ay * cy

        mag_a = math.sqrt(ax ** 2 + ay ** 2)
        mag_c = math.sqrt(cx ** 2 + cy ** 2)

        if mag_a * mag_c == 0:
            return 0.0

        cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_c)))

        return math.degrees(math.acos(cos_angle))

    def distance(self, p1, p2) -> float:
        """Euclidean distance in normalized 2D space."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def get_point(self, landmarks, idx):
        p = landmarks[idx]
        return (p.x, p.y)

    def is_visible(self, landmarks, idx, min_vis=0.65) -> bool:
        if idx >= len(landmarks):
            return False
        return getattr(landmarks[idx], "visibility", 0.0) >= min_vis

    def are_visible(self, landmarks, indices, min_vis=0.65) -> bool:
        return all(self.is_visible(landmarks, idx, min_vis) for idx in indices)

    @abstractmethod
    def process(self, landmarks, current_time=None) -> dict:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

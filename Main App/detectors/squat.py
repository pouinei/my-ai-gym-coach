import time
from core.base_exercise import BaseExercise
from core.rep_state_machine import RepetitionStateMachine, Direction
from services.config.workout_config import DETECTOR_CONFIG


class SquatDetector(BaseExercise):
    LEFT_HIP = 23
    LEFT_KNEE = 25
    LEFT_ANKLE = 27
    RIGHT_HIP = 24
    RIGHT_KNEE = 26
    RIGHT_ANKLE = 28
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    def __init__(self):
        super().__init__()
        cfg = DETECTOR_CONFIG.get("Squats", {})
        self.state_machine = RepetitionStateMachine(
            start_threshold=cfg.get("start_threshold", 160.0),
            inflection_threshold=cfg.get("inflection_threshold", 100.0),
            direction=Direction.START_HIGH_INFLECT_LOW,
            hysteresis=cfg.get("hysteresis", 15.0),
            min_rep_duration=cfg.get("min_rep_duration", 0.8),
            consecutive_frames=cfg.get("consecutive_frames", 2),
        )
        self.min_visibility = cfg.get("min_visibility", 0.65)
        self._active_side = None

    def reset(self) -> None:
        self.state_machine.reset()
        self.reps = 0
        self.stage = None
        self._active_side = None

    def _select_tracking_side(self, landmarks) -> str:
        left_vis = (
            getattr(landmarks[self.LEFT_HIP], "visibility", 0.0)
            + getattr(landmarks[self.LEFT_KNEE], "visibility", 0.0)
            + getattr(landmarks[self.LEFT_ANKLE], "visibility", 0.0)
        ) / 3.0

        right_vis = (
            getattr(landmarks[self.RIGHT_HIP], "visibility", 0.0)
            + getattr(landmarks[self.RIGHT_KNEE], "visibility", 0.0)
            + getattr(landmarks[self.RIGHT_ANKLE], "visibility", 0.0)
        ) / 3.0

        if self._active_side is None:
            self._active_side = "left" if left_vis >= right_vis else "right"
        else:
            if self._active_side == "left" and right_vis > (left_vis + 0.15):
                self._active_side = "right"
            elif self._active_side == "right" and left_vis > (right_vis + 0.15):
                self._active_side = "left"

        return self._active_side

    def process(self, landmarks, current_time=None) -> dict:
        if current_time is None:
            current_time = time.time()

        side = self._select_tracking_side(landmarks)

        if side == "left":
            hip_idx = self.LEFT_HIP
            knee_idx = self.LEFT_KNEE
            ankle_idx = self.LEFT_ANKLE
            shoulder_idx = self.LEFT_SHOULDER
        else:
            hip_idx = self.RIGHT_HIP
            knee_idx = self.RIGHT_KNEE
            ankle_idx = self.RIGHT_ANKLE
            shoulder_idx = self.RIGHT_SHOULDER

        key_landmarks_visible = self.are_visible(
            landmarks, [hip_idx, knee_idx, ankle_idx], min_vis=self.min_visibility
        )

        knee_angle = self.calculate_angle(
            self.get_point(landmarks, hip_idx),
            self.get_point(landmarks, knee_idx),
            self.get_point(landmarks, ankle_idx),
        )

        back_angle = 180.0
        if self.are_visible(landmarks, [shoulder_idx, hip_idx, knee_idx], min_vis=self.min_visibility):
            back_angle = self.calculate_angle(
                self.get_point(landmarks, shoulder_idx),
                self.get_point(landmarks, hip_idx),
                self.get_point(landmarks, knee_idx),
            )

        rep_incremented = False
        if key_landmarks_visible:
            rep_incremented = self.state_machine.update(knee_angle, current_time)
            self.reps = self.state_machine.reps
            self.stage = self.state_machine.stage_name

        if knee_angle <= self.state_machine.inflection_threshold:
            depth_status = "GOOD DEPTH"
        elif knee_angle <= 135:
            depth_status = "HALF SQUAT"
        elif knee_angle >= (self.state_machine.start_threshold - 10):
            depth_status = "STANDING"
        else:
            depth_status = "DESCENDING"

        posture_feedback = "Good Form"
        if back_angle < 120:
            posture_feedback = "Keep chest up and back straight"
        elif knee_angle > 100 and self.stage == "inflected":
            posture_feedback = "Squat a little deeper"

        return {
            "reps": self.reps,
            "rep_incremented": rep_incremented,
            "knee_angle": int(knee_angle),
            "back_angle": int(back_angle),
            "depth_status": depth_status,
            "posture_feedback": posture_feedback,
        }

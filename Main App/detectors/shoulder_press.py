import time
from core.base_exercise import BaseExercise
from core.rep_state_machine import RepetitionStateMachine, Direction
from services.config.workout_config import DETECTOR_CONFIG


class ShoulderPressDetector(BaseExercise):
    LEFT_SHOULDER = 11
    LEFT_ELBOW = 13
    LEFT_WRIST = 15
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW = 14
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26

    def __init__(self):
        super().__init__()
        cfg = DETECTOR_CONFIG.get("Shoulder Press", {})
        self.state_machine = RepetitionStateMachine(
            start_threshold=cfg.get("start_threshold", 95.0),
            inflection_threshold=cfg.get("inflection_threshold", 155.0),
            direction=Direction.START_LOW_INFLECT_HIGH,
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
            getattr(landmarks[self.LEFT_SHOULDER], "visibility", 0.0)
            + getattr(landmarks[self.LEFT_ELBOW], "visibility", 0.0)
            + getattr(landmarks[self.LEFT_WRIST], "visibility", 0.0)
        ) / 3.0

        right_vis = (
            getattr(landmarks[self.RIGHT_SHOULDER], "visibility", 0.0)
            + getattr(landmarks[self.RIGHT_ELBOW], "visibility", 0.0)
            + getattr(landmarks[self.RIGHT_WRIST], "visibility", 0.0)
        ) / 3.0

        if self._active_side is None:
            self._active_side = "left" if left_vis >= right_vis else "right"
        else:
            # Apply deadband hysteresis to prevent arm flickering between frames
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
            shoulder_idx = self.LEFT_SHOULDER
            elbow_idx = self.LEFT_ELBOW
            wrist_idx = self.LEFT_WRIST
            hip_idx = self.LEFT_HIP
            knee_idx = self.LEFT_KNEE
        else:
            shoulder_idx = self.RIGHT_SHOULDER
            elbow_idx = self.RIGHT_ELBOW
            wrist_idx = self.RIGHT_WRIST
            hip_idx = self.RIGHT_HIP
            knee_idx = self.RIGHT_KNEE

        key_landmarks_visible = self.are_visible(
            landmarks, [shoulder_idx, elbow_idx, wrist_idx], min_vis=self.min_visibility
        )

        elbow_angle = self.calculate_angle(
            self.get_point(landmarks, shoulder_idx),
            self.get_point(landmarks, elbow_idx),
            self.get_point(landmarks, wrist_idx),
        )

        rep_incremented = False
        if key_landmarks_visible:
            rep_incremented = self.state_machine.update(elbow_angle, current_time)
            self.reps = self.state_machine.reps
            self.stage = self.state_machine.stage_name

        if elbow_angle >= self.state_machine.inflection_threshold:
            extension_status = "FULL EXTENSION"
        elif elbow_angle >= 130:
            extension_status = "NEARLY EXTENDED"
        elif elbow_angle >= self.state_machine.start_threshold:
            extension_status = "PRESSING"
        else:
            extension_status = "START POSITION"

        back_angle = 180.0
        if self.are_visible(landmarks, [shoulder_idx, hip_idx, knee_idx], min_vis=self.min_visibility):
            back_angle = self.calculate_angle(
                self.get_point(landmarks, shoulder_idx),
                self.get_point(landmarks, hip_idx),
                self.get_point(landmarks, knee_idx),
            )

        if back_angle >= 160:
            back_arch_status = "Neutral"
        elif back_angle >= 140:
            back_arch_status = "Slight Arch"
        else:
            back_arch_status = "Excessive Arch"

        posture_feedback = "Good Form"
        if back_arch_status == "Excessive Arch":
            posture_feedback = "Keep core tight; avoid leaning back"
        elif elbow_angle < 60:
            posture_feedback = "Don't lower elbows too far below shoulders"

        return {
            "reps": self.reps,
            "rep_incremented": rep_incremented,
            "elbow_angle": int(elbow_angle),
            "extension_status": extension_status,
            "back_arch_status": back_arch_status,
            "posture_feedback": posture_feedback,
        }

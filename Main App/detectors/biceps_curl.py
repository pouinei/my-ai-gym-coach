import math
import time
from core.base_exercise import BaseExercise
from core.rep_state_machine import RepetitionStateMachine, Direction
from services.config.workout_config import DETECTOR_CONFIG


class BicepsCurlDetector(BaseExercise):
    LEFT_SHOULDER = 11
    LEFT_ELBOW = 13
    LEFT_WRIST = 15
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW = 14
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24

    def __init__(self):
        super().__init__()
        cfg = DETECTOR_CONFIG.get("Biceps Curls (Dumbbell)", {})
        self.state_machine = RepetitionStateMachine(
            start_threshold=cfg.get("start_threshold", 150.0),
            inflection_threshold=cfg.get("inflection_threshold", 50.0),
            direction=Direction.START_HIGH_INFLECT_LOW,
            hysteresis=cfg.get("hysteresis", 15.0),
            min_rep_duration=cfg.get("min_rep_duration", 0.7),
            consecutive_frames=cfg.get("consecutive_frames", 2),
        )
        self.min_visibility = cfg.get("min_visibility", 0.65)
        self.elbow_drift_tolerance = cfg.get("elbow_drift_tolerance", 0.06)
        self.swing_threshold = cfg.get("swing_threshold", 15.0)
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
            if self._active_side == "left" and right_vis > (left_vis + 0.15):
                self._active_side = "right"
            elif self._active_side == "right" and left_vis > (right_vis + 0.15):
                self._active_side = "left"

        return self._active_side

    def _safe_angle(self, dx, dy):
        return math.degrees(math.atan2(abs(dx), abs(dy))) if dy != 0 else 0.0

    def process(self, landmarks, current_time=None) -> dict:
        if current_time is None:
            current_time = time.time()

        side = self._select_tracking_side(landmarks)

        if side == "left":
            shoulder_idx = self.LEFT_SHOULDER
            elbow_idx = self.LEFT_ELBOW
            wrist_idx = self.LEFT_WRIST
        else:
            shoulder_idx = self.RIGHT_SHOULDER
            elbow_idx = self.RIGHT_ELBOW
            wrist_idx = self.RIGHT_WRIST

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

        shoulder_x = landmarks[shoulder_idx].x
        elbow_x = landmarks[elbow_idx].x
        elbow_drift = abs(elbow_x - shoulder_x)

        if elbow_drift <= self.elbow_drift_tolerance:
            shoulder_status = "STABLE"
        else:
            shoulder_status = "ELBOW DRIFTING"

        swing_status = "NO SWING"
        if self.are_visible(landmarks, [self.LEFT_SHOULDER, self.RIGHT_SHOULDER, self.LEFT_HIP, self.RIGHT_HIP], min_vis=self.min_visibility):
            shoulder_mid_x = (landmarks[self.LEFT_SHOULDER].x + landmarks[self.RIGHT_SHOULDER].x) / 2.0
            shoulder_mid_y = (landmarks[self.LEFT_SHOULDER].y + landmarks[self.RIGHT_SHOULDER].y) / 2.0
            hip_mid_x = (landmarks[self.LEFT_HIP].x + landmarks[self.RIGHT_HIP].x) / 2.0
            hip_mid_y = (landmarks[self.LEFT_HIP].y + landmarks[self.RIGHT_HIP].y) / 2.0

            dx = shoulder_mid_x - hip_mid_x
            dy = shoulder_mid_y - hip_mid_y
            torso_angle = self._safe_angle(dx, dy)

            if torso_angle <= self.swing_threshold:
                swing_status = "NO SWING"
            else:
                swing_status = "SWINGING"

        posture_feedback = "Good Form"
        if swing_status == "SWINGING":
            posture_feedback = "Avoid swinging torso; isolate the bicep"
        elif shoulder_status == "ELBOW DRIFTING":
            posture_feedback = "Keep elbows pinned to your sides"

        return {
            "reps": self.reps,
            "rep_incremented": rep_incremented,
            "elbow_angle": int(elbow_angle),
            "shoulder_status": shoulder_status,
            "swing_status": swing_status,
            "posture_feedback": posture_feedback,
        }

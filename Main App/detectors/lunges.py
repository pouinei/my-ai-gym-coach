import time
from core.base_exercise import BaseExercise
from core.rep_state_machine import RepetitionStateMachine, Direction
from services.config.workout_config import DETECTOR_CONFIG


class LungesDetector(BaseExercise):
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
        cfg = DETECTOR_CONFIG.get("Lunges", {})
        self.state_machine = RepetitionStateMachine(
            start_threshold=cfg.get("start_threshold", 155.0),
            inflection_threshold=cfg.get("inflection_threshold", 100.0),
            direction=Direction.START_HIGH_INFLECT_LOW,
            hysteresis=cfg.get("hysteresis", 15.0),
            min_rep_duration=cfg.get("min_rep_duration", 0.8),
            consecutive_frames=cfg.get("consecutive_frames", 2),
        )
        self.min_visibility = cfg.get("min_visibility", 0.65)
        self.balance_tolerance = cfg.get("balance_tolerance", 0.10)
        self._active_front_leg = None

    def reset(self) -> None:
        self.state_machine.reset()
        self.reps = 0
        self.stage = None
        self._active_front_leg = None

    def process(self, landmarks, current_time=None) -> dict:
        if current_time is None:
            current_time = time.time()

        left_knee_angle = self.calculate_angle(
            self.get_point(landmarks, self.LEFT_HIP),
            self.get_point(landmarks, self.LEFT_KNEE),
            self.get_point(landmarks, self.LEFT_ANKLE),
        )

        right_knee_angle = self.calculate_angle(
            self.get_point(landmarks, self.RIGHT_HIP),
            self.get_point(landmarks, self.RIGHT_KNEE),
            self.get_point(landmarks, self.RIGHT_ANKLE),
        )

        # Stable front leg selection with hysteresis
        if self._active_front_leg is None:
            self._active_front_leg = "left" if left_knee_angle <= right_knee_angle else "right"
        else:
            if self._active_front_leg == "left" and right_knee_angle < (left_knee_angle - 15.0):
                self._active_front_leg = "right"
            elif self._active_front_leg == "right" and left_knee_angle < (right_knee_angle - 15.0):
                self._active_front_leg = "left"

        if self._active_front_leg == "left":
            front_knee_angle = left_knee_angle
            front_hip_idx = self.LEFT_HIP
            front_knee_idx = self.LEFT_KNEE
            front_ankle_idx = self.LEFT_ANKLE
            shoulder_idx_for_torso = self.LEFT_SHOULDER
        else:
            front_knee_angle = right_knee_angle
            front_hip_idx = self.RIGHT_HIP
            front_knee_idx = self.RIGHT_KNEE
            front_ankle_idx = self.RIGHT_ANKLE
            shoulder_idx_for_torso = self.RIGHT_SHOULDER

        key_landmarks_visible = self.are_visible(
            landmarks, [front_hip_idx, front_knee_idx, front_ankle_idx], min_vis=self.min_visibility
        )

        rep_incremented = False
        if key_landmarks_visible:
            rep_incremented = self.state_machine.update(front_knee_angle, current_time)
            self.reps = self.state_machine.reps
            self.stage = self.state_machine.stage_name

        torso_angle = 180.0
        if self.are_visible(landmarks, [shoulder_idx_for_torso, front_hip_idx, front_knee_idx], min_vis=self.min_visibility):
            torso_angle = self.calculate_angle(
                self.get_point(landmarks, shoulder_idx_for_torso),
                self.get_point(landmarks, front_hip_idx),
                self.get_point(landmarks, front_knee_idx),
            )

        balance_status = "BALANCED"
        if self.are_visible(landmarks, [self.LEFT_SHOULDER, self.RIGHT_SHOULDER, self.LEFT_HIP, self.RIGHT_HIP], min_vis=self.min_visibility):
            shoulder_mid_x = (landmarks[self.LEFT_SHOULDER].x + landmarks[self.RIGHT_SHOULDER].x) / 2.0
            hip_mid_x = (landmarks[self.LEFT_HIP].x + landmarks[self.RIGHT_HIP].x) / 2.0
            lateral_offset = abs(shoulder_mid_x - hip_mid_x)

            if lateral_offset <= self.balance_tolerance:
                balance_status = "BALANCED"
            else:
                balance_status = "OFF BALANCE"

        posture_feedback = "Good Form"
        if balance_status == "OFF BALANCE":
            posture_feedback = "Maintain balance and keep torso upright"
        elif front_knee_angle < 75:
            posture_feedback = "Don't let front knee push too far forward"

        return {
            "reps": self.reps,
            "rep_incremented": rep_incremented,
            "front_knee_angle": int(front_knee_angle),
            "torso_angle": int(torso_angle),
            "balance_status": balance_status,
            "posture_feedback": posture_feedback,
        }

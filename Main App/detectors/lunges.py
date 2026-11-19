import time
from core.base_exercise import BaseExercise
from core.rep_state_machine import RepetitionStateMachine, Direction, Stage
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

        # Per-rep tracking
        self._rep_start_time = 0.0
        self._min_knee_angle_in_rep = 180.0
        self._max_lateral_offset_in_rep = 0.0
        self._knee_past_toes_in_rep = False

    def reset(self) -> None:
        self.state_machine.reset()
        self.reps = 0
        self.valid_reps = 0
        self.invalid_reps = 0
        self.stage = None
        self._active_front_leg = None
        self._rep_start_time = 0.0
        self._min_knee_angle_in_rep = 180.0
        self._max_lateral_offset_in_rep = 0.0
        self._knee_past_toes_in_rep = False
        self.form_status = "Good"
        self.form_issue = "None - Form is clean"
        self.form_correction = "Maintain this form!"

    def process(self, landmarks, current_time=None) -> dict:
        if current_time is None:
            current_time = time.time()

        left_visible = self.are_visible(landmarks, [self.LEFT_HIP, self.LEFT_KNEE, self.LEFT_ANKLE], min_vis=self.min_visibility)
        right_visible = self.are_visible(landmarks, [self.RIGHT_HIP, self.RIGHT_KNEE, self.RIGHT_ANKLE], min_vis=self.min_visibility)

        if not left_visible and not right_visible:
            self.form_status = "Unable to Assess"
            self.form_issue = "Leg landmarks occluded or outside camera view"
            self.form_correction = "Adjust camera so your front and back legs are clearly visible."
            return {
                "reps": self.valid_reps,
                "valid_reps": self.valid_reps,
                "invalid_reps": self.invalid_reps,
                "rep_incremented": False,
                "rep_invalid": False,
                "front_knee_angle": 0,
                "torso_angle": 180,
                "balance_status": "BALANCED",
                "form_status": self.form_status,
                "form_issue": self.form_issue,
                "form_correction": self.form_correction,
                "posture_feedback": self.form_issue,
            }

        left_knee_angle = self.calculate_angle(
            self.get_point(landmarks, self.LEFT_HIP),
            self.get_point(landmarks, self.LEFT_KNEE),
            self.get_point(landmarks, self.LEFT_ANKLE),
        ) if left_visible else 180.0

        right_knee_angle = self.calculate_angle(
            self.get_point(landmarks, self.RIGHT_HIP),
            self.get_point(landmarks, self.RIGHT_KNEE),
            self.get_point(landmarks, self.RIGHT_ANKLE),
        ) if right_visible else 180.0

        # Stable front leg selection with hysteresis
        if left_visible and right_visible:
            if self._active_front_leg is None:
                self._active_front_leg = "left" if left_knee_angle <= right_knee_angle else "right"
            else:
                if self._active_front_leg == "left" and right_knee_angle < (left_knee_angle - 15.0):
                    self._active_front_leg = "right"
                elif self._active_front_leg == "right" and left_knee_angle < (right_knee_angle - 15.0):
                    self._active_front_leg = "left"
        elif left_visible:
            self._active_front_leg = "left"
        elif right_visible:
            self._active_front_leg = "right"

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

        torso_angle = 180.0
        if self.are_visible(landmarks, [shoulder_idx_for_torso, front_hip_idx, front_knee_idx], min_vis=self.min_visibility):
            torso_angle = self.calculate_angle(
                self.get_point(landmarks, shoulder_idx_for_torso),
                self.get_point(landmarks, front_hip_idx),
                self.get_point(landmarks, front_knee_idx),
            )

        balance_status = "BALANCED"
        lateral_offset = 0.0
        torso_visible = self.are_visible(landmarks, [self.LEFT_SHOULDER, self.RIGHT_SHOULDER, self.LEFT_HIP, self.RIGHT_HIP], min_vis=self.min_visibility)
        if torso_visible:
            shoulder_mid_x = (landmarks[self.LEFT_SHOULDER].x + landmarks[self.RIGHT_SHOULDER].x) / 2.0
            hip_mid_x = (landmarks[self.LEFT_HIP].x + landmarks[self.RIGHT_HIP].x) / 2.0
            lateral_offset = abs(shoulder_mid_x - hip_mid_x)

            if lateral_offset <= self.balance_tolerance:
                balance_status = "BALANCED"
            else:
                balance_status = "OFF BALANCE"

        knee_past_toes = front_knee_angle < 75.0

        # Track rep progression
        if self.state_machine.stage == Stage.STARTING:
            if self._rep_start_time == 0.0:
                self._rep_start_time = current_time
                self._min_knee_angle_in_rep = front_knee_angle
                self._max_lateral_offset_in_rep = lateral_offset
                self._knee_past_toes_in_rep = knee_past_toes
        else:
            self._min_knee_angle_in_rep = min(self._min_knee_angle_in_rep, front_knee_angle)
            self._max_lateral_offset_in_rep = max(self._max_lateral_offset_in_rep, lateral_offset)
            if knee_past_toes:
                self._knee_past_toes_in_rep = True

        cycle_completed = self.state_machine.update(front_knee_angle, current_time)
        self.stage = self.state_machine.stage_name

        rep_incremented = False
        rep_invalid = False

        if cycle_completed:
            flaws = []
            if self._max_lateral_offset_in_rep > self.balance_tolerance:
                flaws.append((
                    "Losing balance / torso sway",
                    "Your balance is unstable. Take a controlled step forward, keep your upper body upright, and lower yourself slowly while maintaining comfortable knee alignment."
                ))
            if self._knee_past_toes_in_rep:
                flaws.append((
                    "Front knee pushing too far past toes",
                    "Your front knee is tracking too far forward over your toes. Take a slightly larger stride forward so your front knee stays directly above your ankle."
                ))
            if self._min_knee_angle_in_rep > 105.0:
                flaws.append((
                    "Insufficient lunge depth",
                    "Your lunge is too shallow. Lower your hips until your front thigh is parallel to the ground and your back knee hovers just above the floor."
                ))
            if front_knee_angle < 150.0:
                flaws.append((
                    "Incomplete return to standing",
                    "You didn't return to a full standing posture. Press firmly through your front heel to stand all the way back up."
                ))

            if flaws:
                self.invalid_reps += 1
                self.state_machine.reps = self.valid_reps
                self.reps = self.valid_reps
                rep_incremented = False
                rep_invalid = True
                self.form_status = "Needs Correction"
                self.form_issue = flaws[0][0]
                self.form_correction = flaws[0][1]
            else:
                self.valid_reps += 1
                self.reps = self.valid_reps
                rep_incremented = True
                rep_invalid = False
                self.form_status = "Good"
                self.form_issue = "None - Form is clean"
                self.form_correction = "Great adjustment! Your form is clean, balanced, and under control. Keep up this exact technique."

            self._rep_start_time = current_time
            self._min_knee_angle_in_rep = front_knee_angle
            self._max_lateral_offset_in_rep = lateral_offset
            self._knee_past_toes_in_rep = False

        else:
            # Real-time form coaching
            if balance_status == "OFF BALANCE":
                self.form_status = "Needs Correction"
                self.form_issue = "Losing balance / torso sway"
                self.form_correction = "Your balance is unstable. Take a controlled step forward, keep your upper body upright, and lower yourself slowly while maintaining comfortable knee alignment."
            elif knee_past_toes:
                self.form_status = "Needs Correction"
                self.form_issue = "Front knee pushing too far past toes"
                self.form_correction = "Your front knee is tracking too far forward over your toes. Take a slightly larger stride forward so your front knee stays directly above your ankle."
            elif front_knee_angle > 105.0 and self.stage == "inflected":
                self.form_status = "Needs Correction"
                self.form_issue = "Insufficient lunge depth"
                self.form_correction = "Your lunge is too shallow. Lower your hips until your front thigh is parallel to the ground and your back knee hovers just above the floor."
            else:
                self.form_status = "Good"
                self.form_issue = "None - Form is clean"
                self.form_correction = "Great adjustment! Your form is clean, balanced, and under control. Keep up this exact technique."

        return {
            "reps": self.valid_reps,
            "valid_reps": self.valid_reps,
            "invalid_reps": self.invalid_reps,
            "rep_incremented": rep_incremented,
            "rep_invalid": rep_invalid,
            "front_knee_angle": int(front_knee_angle),
            "torso_angle": int(torso_angle),
            "balance_status": balance_status,
            "form_status": self.form_status,
            "form_issue": self.form_issue,
            "form_correction": self.form_correction,
            "posture_feedback": self.form_correction if self.form_status == "Needs Correction" else "Good Form",
        }

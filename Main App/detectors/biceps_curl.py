import math
import time
from core.base_exercise import BaseExercise
from core.rep_state_machine import RepetitionStateMachine, Direction, Stage
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
        self.elbow_drift_tolerance = cfg.get("elbow_drift_tolerance", 0.07)
        self.swing_threshold = cfg.get("swing_threshold", 14.0)
        self._active_side = None

        # Per-rep tracking
        self._rep_start_time = 0.0
        self._min_elbow_angle_in_rep = 180.0
        self._max_elbow_drift_in_rep = 0.0
        self._max_torso_angle_in_rep = 0.0

    def reset(self) -> None:
        self.state_machine.reset()
        self.reps = 0
        self.valid_reps = 0
        self.invalid_reps = 0
        self.stage = None
        self._active_side = None
        self._rep_start_time = 0.0
        self._min_elbow_angle_in_rep = 180.0
        self._max_elbow_drift_in_rep = 0.0
        self._max_torso_angle_in_rep = 0.0
        self.form_status = "Good"
        self.form_issue = "None - Form is clean"
        self.form_correction = "Maintain this form!"

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

        if not key_landmarks_visible:
            self.form_status = "Unable to Assess"
            self.form_issue = "Arm landmarks occluded or outside camera view"
            self.form_correction = "Step back into camera frame so your arm and torso are clearly visible."
            return {
                "reps": self.valid_reps,
                "valid_reps": self.valid_reps,
                "invalid_reps": self.invalid_reps,
                "rep_incremented": False,
                "rep_invalid": False,
                "elbow_angle": 0,
                "shoulder_status": "STABLE",
                "swing_status": "NO SWING",
                "form_status": self.form_status,
                "form_issue": self.form_issue,
                "form_correction": self.form_correction,
                "posture_feedback": self.form_issue,
            }

        elbow_angle = self.calculate_angle(
            self.get_point(landmarks, shoulder_idx),
            self.get_point(landmarks, elbow_idx),
            self.get_point(landmarks, wrist_idx),
        )

        shoulder_x = landmarks[shoulder_idx].x
        elbow_x = landmarks[elbow_idx].x
        elbow_drift = abs(elbow_x - shoulder_x)

        if elbow_drift <= self.elbow_drift_tolerance:
            shoulder_status = "STABLE"
        else:
            shoulder_status = "ELBOW DRIFTING"

        swing_status = "NO SWING"
        torso_angle = 0.0
        torso_visible = self.are_visible(landmarks, [self.LEFT_SHOULDER, self.RIGHT_SHOULDER, self.LEFT_HIP, self.RIGHT_HIP], min_vis=self.min_visibility)

        if torso_visible:
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

        # Track rep progression
        if self.state_machine.stage == Stage.STARTING:
            if self._rep_start_time == 0.0:
                self._rep_start_time = current_time
                self._min_elbow_angle_in_rep = elbow_angle
                self._max_elbow_drift_in_rep = elbow_drift
                self._max_torso_angle_in_rep = torso_angle
        else:
            self._min_elbow_angle_in_rep = min(self._min_elbow_angle_in_rep, elbow_angle)
            self._max_elbow_drift_in_rep = max(self._max_elbow_drift_in_rep, elbow_drift)
            self._max_torso_angle_in_rep = max(self._max_torso_angle_in_rep, torso_angle)

        cycle_completed = self.state_machine.update(elbow_angle, current_time)
        self.stage = self.state_machine.stage_name

        rep_incremented = False
        rep_invalid = False

        if cycle_completed:
            flaws = []
            if torso_visible and self._max_torso_angle_in_rep > self.swing_threshold:
                flaws.append((
                    "Swinging the torso using momentum",
                    "You are swinging your torso to lift the weights. Stand tall, brace your core, and isolate your biceps by moving only your forearms."
                ))
            if self._max_elbow_drift_in_rep > self.elbow_drift_tolerance:
                flaws.append((
                    "Excessive elbow drift",
                    "You are moving your elbows away from your body. Keep your elbows close to your sides, avoid swinging your torso, and curl the weight slowly."
                ))
            if self._min_elbow_angle_in_rep > 65.0:
                flaws.append((
                    "Incomplete curl range",
                    "You are not curling the weight high enough. Lift the weight all the way up toward your shoulder to achieve a full bicep contraction."
                ))
            if elbow_angle < 145.0:
                flaws.append((
                    "Incomplete lowering",
                    "You are not lowering the weight completely. Extend your arms all the way down until your elbows are straight before curling again."
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
            self._min_elbow_angle_in_rep = elbow_angle
            self._max_elbow_drift_in_rep = elbow_drift
            self._max_torso_angle_in_rep = torso_angle

        else:
            # Real-time form coaching
            if swing_status == "SWINGING":
                self.form_status = "Needs Correction"
                self.form_issue = "Swinging the torso using momentum"
                self.form_correction = "You are swinging your torso to lift the weights. Stand tall, brace your core, and isolate your biceps by moving only your forearms."
            elif shoulder_status == "ELBOW DRIFTING":
                self.form_status = "Needs Correction"
                self.form_issue = "Excessive elbow drift"
                self.form_correction = "You are moving your elbows away from your body. Keep your elbows close to your sides, avoid swinging your torso, and curl the weight slowly."
            elif elbow_angle > 65.0 and self.stage == "inflected":
                self.form_status = "Needs Correction"
                self.form_issue = "Incomplete curl range"
                self.form_correction = "You are not curling the weight high enough. Lift the weight all the way up toward your shoulder to achieve a full bicep contraction."
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
            "elbow_angle": int(elbow_angle),
            "shoulder_status": shoulder_status,
            "swing_status": swing_status,
            "form_status": self.form_status,
            "form_issue": self.form_issue,
            "form_correction": self.form_correction,
            "posture_feedback": self.form_correction if self.form_status == "Needs Correction" else "Good Form",
        }

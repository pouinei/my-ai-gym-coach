import time
from core.base_exercise import BaseExercise
from core.rep_state_machine import RepetitionStateMachine, Direction, Stage
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

        # Per-rep tracking for form validation
        self._rep_start_time = 0.0
        self._max_elbow_angle_in_rep = 0.0
        self._min_back_angle_in_rep = 180.0
        self._max_arm_asymmetry_in_rep = 0.0

    def reset(self) -> None:
        self.state_machine.reset()
        self.reps = 0
        self.valid_reps = 0
        self.invalid_reps = 0
        self.stage = None
        self._active_side = None
        self._rep_start_time = 0.0
        self._max_elbow_angle_in_rep = 0.0
        self._min_back_angle_in_rep = 180.0
        self._max_arm_asymmetry_in_rep = 0.0
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

        if not key_landmarks_visible:
            self.form_status = "Unable to Assess"
            self.form_issue = "Key arm landmarks occluded or outside camera view"
            self.form_correction = "Step back into camera frame so your arms and shoulders are visible."
            return {
                "reps": self.valid_reps,
                "valid_reps": self.valid_reps,
                "invalid_reps": self.invalid_reps,
                "rep_incremented": False,
                "rep_invalid": False,
                "elbow_angle": 0,
                "extension_status": "START POSITION",
                "back_arch_status": "Neutral",
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

        # Both arms asymmetry check if both visible
        both_arms_visible = self.are_visible(
            landmarks,
            [self.LEFT_SHOULDER, self.LEFT_ELBOW, self.LEFT_WRIST, self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST],
            min_vis=0.55,
        )
        arm_asymmetry = 0.0
        if both_arms_visible:
            l_angle = self.calculate_angle(
                self.get_point(landmarks, self.LEFT_SHOULDER),
                self.get_point(landmarks, self.LEFT_ELBOW),
                self.get_point(landmarks, self.LEFT_WRIST),
            )
            r_angle = self.calculate_angle(
                self.get_point(landmarks, self.RIGHT_SHOULDER),
                self.get_point(landmarks, self.RIGHT_ELBOW),
                self.get_point(landmarks, self.RIGHT_WRIST),
            )
            arm_asymmetry = abs(l_angle - r_angle)

        # Lower back arch calculation
        back_angle = 180.0
        back_visible = self.are_visible(landmarks, [shoulder_idx, hip_idx, knee_idx], min_vis=self.min_visibility)
        if back_visible:
            back_angle = self.calculate_angle(
                self.get_point(landmarks, shoulder_idx),
                self.get_point(landmarks, hip_idx),
                self.get_point(landmarks, knee_idx),
            )

        if back_angle >= 160:
            back_arch_status = "Neutral"
        elif back_angle >= 145:
            back_arch_status = "Slight Arch"
        else:
            back_arch_status = "Excessive Arch"

        # Track rep progression
        if self.state_machine.stage == Stage.STARTING:
            if self._rep_start_time == 0.0:
                self._rep_start_time = current_time
                self._max_elbow_angle_in_rep = elbow_angle
                self._min_back_angle_in_rep = back_angle
                self._max_arm_asymmetry_in_rep = arm_asymmetry
        else:
            self._max_elbow_angle_in_rep = max(self._max_elbow_angle_in_rep, elbow_angle)
            self._min_back_angle_in_rep = min(self._min_back_angle_in_rep, back_angle)
            self._max_arm_asymmetry_in_rep = max(self._max_arm_asymmetry_in_rep, arm_asymmetry)

        # Update state machine
        cycle_completed = self.state_machine.update(elbow_angle, current_time)
        self.stage = self.state_machine.stage_name

        rep_incremented = False
        rep_invalid = False

        if cycle_completed:
            rep_duration = current_time - self._rep_start_time if self._rep_start_time > 0 else 1.0

            # Evaluate form violations for rep validation
            flaws = []
            if self._min_back_angle_in_rep < 145.0:
                flaws.append((
                    "Excessive lower-back arch",
                    "You are arching your lower back during the press. This puts unnecessary stress on your lumbar spine. Brace your core tightly, keep your spine neutral, and press straight overhead without leaning back."
                ))
            if self._max_elbow_angle_in_rep < 150.0:
                flaws.append((
                    "Incomplete elbow extension overhead",
                    "Your arms are not extending fully overhead. To complete the full movement, press the weights all the way up until your elbows are fully locked out overhead."
                ))
            if self._max_arm_asymmetry_in_rep > 25.0:
                flaws.append((
                    "Uneven arm movement",
                    "Your arms are pressing unevenly. This can create an imbalance in your shoulder movement. Push both arms upward at the same time and keep them at the same height. Keep your core stable and control the movement. Try again."
                ))
            if rep_duration < 0.8:
                flaws.append((
                    "Movement was too fast",
                    "You are pressing too quickly. Control the movement with a steady two-second press up and a controlled lowering down."
                ))

            if flaws:
                # Reject rep
                self.invalid_reps += 1
                self.state_machine.reps = self.valid_reps
                self.reps = self.valid_reps
                rep_incremented = False
                rep_invalid = True
                self.form_status = "Needs Correction"
                self.form_issue = flaws[0][0]
                self.form_correction = flaws[0][1]
            else:
                # Valid rep
                self.valid_reps += 1
                self.reps = self.valid_reps
                rep_incremented = True
                rep_invalid = False
                self.form_status = "Good"
                self.form_issue = "None - Form is clean"
                self.form_correction = "Great adjustment! Your form is clean, balanced, and under control. Keep up this exact technique."

            # Reset tracking for next rep
            self._rep_start_time = current_time
            self._max_elbow_angle_in_rep = elbow_angle
            self._min_back_angle_in_rep = back_angle
            self._max_arm_asymmetry_in_rep = arm_asymmetry

        else:
            # Real-time form coaching during movement
            if back_arch_status == "Excessive Arch":
                self.form_status = "Needs Correction"
                self.form_issue = "Excessive lower-back arch"
                self.form_correction = "You are arching your lower back during the press. This puts unnecessary stress on your lumbar spine. Brace your core tightly, keep your spine neutral, and press straight overhead without leaning back."
            elif both_arms_visible and arm_asymmetry > 25.0:
                self.form_status = "Needs Correction"
                self.form_issue = "Uneven arm movement"
                self.form_correction = "Your arms are pressing unevenly. This can create an imbalance in your shoulder movement. Push both arms upward at the same time and keep them at the same height. Keep your core stable and control the movement. Try again."
            elif self.state_machine.stage == Stage.STARTING and elbow_angle < 65.0:
                self.form_status = "Needs Correction"
                self.form_issue = "Elbows dropped too low at start"
                self.form_correction = "Your elbows are not in the correct position. Keep both elbows aligned with your shoulders, brace your core, and press the weights straight overhead. Avoid arching your lower back."
            else:
                self.form_status = "Good"
                self.form_issue = "None - Form is clean"
                self.form_correction = "Great adjustment! Your form is clean, balanced, and under control. Keep up this exact technique."

        if elbow_angle >= self.state_machine.inflection_threshold:
            extension_status = "FULL EXTENSION"
        elif elbow_angle >= 130:
            extension_status = "NEARLY EXTENDED"
        elif elbow_angle >= self.state_machine.start_threshold:
            extension_status = "PRESSING"
        else:
            extension_status = "START POSITION"

        return {
            "reps": self.valid_reps,
            "valid_reps": self.valid_reps,
            "invalid_reps": self.invalid_reps,
            "rep_incremented": rep_incremented,
            "rep_invalid": rep_invalid,
            "elbow_angle": int(elbow_angle),
            "extension_status": extension_status,
            "back_arch_status": back_arch_status,
            "form_status": self.form_status,
            "form_issue": self.form_issue,
            "form_correction": self.form_correction,
            "posture_feedback": self.form_correction if self.form_status == "Needs Correction" else "Good Form",
        }

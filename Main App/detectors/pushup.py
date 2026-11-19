import time
from core.base_exercise import BaseExercise
from core.rep_state_machine import RepetitionStateMachine, Direction, Stage
from services.config.workout_config import DETECTOR_CONFIG


class PushUpDetector(BaseExercise):
    LEFT_SHOULDER = 11
    LEFT_ELBOW = 13
    LEFT_WRIST = 15
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW = 14
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    def __init__(self):
        super().__init__()
        cfg = DETECTOR_CONFIG.get("Push-ups", {})
        self.state_machine = RepetitionStateMachine(
            start_threshold=cfg.get("start_threshold", 155.0),
            inflection_threshold=cfg.get("inflection_threshold", 90.0),
            direction=Direction.START_HIGH_INFLECT_LOW,
            hysteresis=cfg.get("hysteresis", 15.0),
            min_rep_duration=cfg.get("min_rep_duration", 0.7),
            consecutive_frames=cfg.get("consecutive_frames", 2),
        )
        self.min_visibility = cfg.get("min_visibility", 0.65)
        self.hip_sag_tolerance = cfg.get("hip_sag_tolerance", 0.08)
        self._active_side = None

        # Per-rep tracking
        self._rep_start_time = 0.0
        self._min_elbow_angle_in_rep = 180.0
        self._max_hip_sag_in_rep = 0.0
        self._min_body_angle_in_rep = 180.0

    def reset(self) -> None:
        self.state_machine.reset()
        self.reps = 0
        self.valid_reps = 0
        self.invalid_reps = 0
        self.stage = None
        self._active_side = None
        self._rep_start_time = 0.0
        self._min_elbow_angle_in_rep = 180.0
        self._max_hip_sag_in_rep = 0.0
        self._min_body_angle_in_rep = 180.0
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
            ankle_idx = self.LEFT_ANKLE
        else:
            shoulder_idx = self.RIGHT_SHOULDER
            elbow_idx = self.RIGHT_ELBOW
            wrist_idx = self.RIGHT_WRIST
            hip_idx = self.RIGHT_HIP
            ankle_idx = self.RIGHT_ANKLE

        key_landmarks_visible = self.are_visible(
            landmarks, [shoulder_idx, elbow_idx, wrist_idx], min_vis=self.min_visibility
        )

        if not key_landmarks_visible:
            self.form_status = "Unable to Assess"
            self.form_issue = "Arm landmarks occluded or outside camera view"
            self.form_correction = "Position camera to clearly capture your arms and upper body."
            return {
                "reps": self.valid_reps,
                "valid_reps": self.valid_reps,
                "invalid_reps": self.invalid_reps,
                "rep_incremented": False,
                "rep_invalid": False,
                "elbow_angle": 0,
                "body_alignment": "Straight",
                "hip_status": "LEVEL",
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

        # Body alignment and hip sag calculation
        body_angle = 180.0
        hip_deviation = 0.0
        body_visible = self.are_visible(landmarks, [shoulder_idx, hip_idx, ankle_idx], min_vis=self.min_visibility)

        if body_visible:
            body_angle = self.calculate_angle(
                self.get_point(landmarks, shoulder_idx),
                self.get_point(landmarks, hip_idx),
                self.get_point(landmarks, ankle_idx),
            )
            shoulder_y = landmarks[shoulder_idx].y
            ankle_y = landmarks[ankle_idx].y
            hip_y = landmarks[hip_idx].y
            expected_hip_y = (shoulder_y + ankle_y) / 2.0
            hip_deviation = hip_y - expected_hip_y

        if abs(hip_deviation) <= self.hip_sag_tolerance:
            hip_status = "LEVEL"
        elif hip_deviation > self.hip_sag_tolerance:
            hip_status = "SAGGING"
        else:
            hip_status = "PIKED UP"

        if body_angle > 160 and hip_status == "LEVEL":
            body_alignment = "Straight"
        elif body_angle > 140:
            body_alignment = "Slight Bend"
        else:
            body_alignment = "Poor Form"

        # Track rep progression
        if self.state_machine.stage == Stage.STARTING:
            if self._rep_start_time == 0.0:
                self._rep_start_time = current_time
                self._min_elbow_angle_in_rep = elbow_angle
                self._max_hip_sag_in_rep = hip_deviation
                self._min_body_angle_in_rep = body_angle
        else:
            self._min_elbow_angle_in_rep = min(self._min_elbow_angle_in_rep, elbow_angle)
            self._max_hip_sag_in_rep = max(self._max_hip_sag_in_rep, hip_deviation)
            self._min_body_angle_in_rep = min(self._min_body_angle_in_rep, body_angle)

        cycle_completed = self.state_machine.update(elbow_angle, current_time)
        self.stage = self.state_machine.stage_name

        rep_incremented = False
        rep_invalid = False

        if cycle_completed:
            flaws = []
            if body_visible and self._max_hip_sag_in_rep > self.hip_sag_tolerance:
                flaws.append(("Hips are sagging down", "Keep your core tight; do not let your hips sag."))
            elif body_visible and self._min_body_angle_in_rep < 150.0:
                flaws.append(("Incorrect body alignment", "Keep your body in a straight plank line."))
            if self._min_elbow_angle_in_rep > 95.0:
                flaws.append(("Incomplete elbow bend", "Lower your chest a little more."))
            if elbow_angle < 150.0:
                flaws.append(("Incomplete upward extension", "Push all the way up to full arm extension."))

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
                self.form_correction = "Good form! Rep completed."

            self._rep_start_time = current_time
            self._min_elbow_angle_in_rep = elbow_angle
            self._max_hip_sag_in_rep = hip_deviation
            self._min_body_angle_in_rep = body_angle

        else:
            # Real-time form coaching
            if body_visible and hip_status == "SAGGING":
                self.form_status = "Needs Correction"
                self.form_issue = "Hips are sagging down"
                self.form_correction = "Keep your core tight; do not let your hips sag."
            elif body_visible and hip_status == "PIKED UP":
                self.form_status = "Needs Correction"
                self.form_issue = "Hips are raised too high"
                self.form_correction = "Lower your hips into a straight plank line."
            elif elbow_angle > 95.0 and self.stage == "inflected":
                self.form_status = "Needs Correction"
                self.form_issue = "Incomplete elbow bend"
                self.form_correction = "Lower your chest a little more."
            else:
                self.form_status = "Good"
                self.form_issue = "None - Form is clean"
                self.form_correction = "Maintain this form!"

        return {
            "reps": self.valid_reps,
            "valid_reps": self.valid_reps,
            "invalid_reps": self.invalid_reps,
            "rep_incremented": rep_incremented,
            "rep_invalid": rep_invalid,
            "elbow_angle": int(elbow_angle),
            "body_alignment": body_alignment,
            "hip_status": hip_status,
            "form_status": self.form_status,
            "form_issue": self.form_issue,
            "form_correction": self.form_correction,
            "posture_feedback": self.form_correction if self.form_status == "Needs Correction" else "Good Form",
        }

import time
from core.base_exercise import BaseExercise
from core.rep_state_machine import RepetitionStateMachine, Direction, Stage
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

        # Per-rep tracking
        self._rep_start_time = 0.0
        self._min_knee_angle_in_rep = 180.0
        self._min_back_angle_in_rep = 180.0
        self._valgus_detected_in_rep = False

    def reset(self) -> None:
        self.state_machine.reset()
        self.reps = 0
        self.valid_reps = 0
        self.invalid_reps = 0
        self.stage = None
        self._active_side = None
        self._rep_start_time = 0.0
        self._min_knee_angle_in_rep = 180.0
        self._min_back_angle_in_rep = 180.0
        self._valgus_detected_in_rep = False
        self.form_status = "Good"
        self.form_issue = "None - Form is clean"
        self.form_correction = "Maintain this form!"

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

        if not key_landmarks_visible:
            self.form_status = "Unable to Assess"
            self.form_issue = "Leg landmarks occluded or outside camera view"
            self.form_correction = "Adjust camera so your hips, knees, and ankles are clearly visible."
            return {
                "reps": self.valid_reps,
                "valid_reps": self.valid_reps,
                "invalid_reps": self.invalid_reps,
                "rep_incremented": False,
                "rep_invalid": False,
                "knee_angle": 0,
                "back_angle": 0,
                "depth_status": "STANDING",
                "form_status": self.form_status,
                "form_issue": self.form_issue,
                "form_correction": self.form_correction,
                "posture_feedback": self.form_issue,
            }

        knee_angle = self.calculate_angle(
            self.get_point(landmarks, hip_idx),
            self.get_point(landmarks, knee_idx),
            self.get_point(landmarks, ankle_idx),
        )

        back_angle = 180.0
        back_visible = self.are_visible(landmarks, [shoulder_idx, hip_idx, knee_idx], min_vis=self.min_visibility)
        if back_visible:
            back_angle = self.calculate_angle(
                self.get_point(landmarks, shoulder_idx),
                self.get_point(landmarks, hip_idx),
                self.get_point(landmarks, knee_idx),
            )

        # Frontal knee valgus check if both knees and ankles visible
        both_legs_visible = self.are_visible(
            landmarks,
            [self.LEFT_KNEE, self.RIGHT_KNEE, self.LEFT_ANKLE, self.RIGHT_ANKLE],
            min_vis=0.55,
        )
        valgus_active = False
        if both_legs_visible:
            knee_dist = abs(landmarks[self.LEFT_KNEE].x - landmarks[self.RIGHT_KNEE].x)
            ankle_dist = abs(landmarks[self.LEFT_ANKLE].x - landmarks[self.RIGHT_ANKLE].x)
            if ankle_dist > 0.08 and knee_dist < (0.65 * ankle_dist):
                valgus_active = True

        # Track rep progression
        if self.state_machine.stage == Stage.STARTING:
            if self._rep_start_time == 0.0:
                self._rep_start_time = current_time
                self._min_knee_angle_in_rep = knee_angle
                self._min_back_angle_in_rep = back_angle
                self._valgus_detected_in_rep = valgus_active
        else:
            self._min_knee_angle_in_rep = min(self._min_knee_angle_in_rep, knee_angle)
            self._min_back_angle_in_rep = min(self._min_back_angle_in_rep, back_angle)
            if valgus_active:
                self._valgus_detected_in_rep = True

        cycle_completed = self.state_machine.update(knee_angle, current_time)
        self.stage = self.state_machine.stage_name

        rep_incremented = False
        rep_invalid = False

        if cycle_completed:
            flaws = []
            if back_visible and self._min_back_angle_in_rep < 120.0:
                flaws.append((
                    "Excessive forward trunk lean",
                    "Your upper body is leaning too far forward. Keep your chest lifted, engage your core, and look straight ahead throughout the squat."
                ))
            if self._min_knee_angle_in_rep > 105.0:
                flaws.append((
                    "Insufficient squat depth",
                    "Your squat depth is too shallow. Squat a little deeper for full range of motion. Slowly bend your knees and push your hips backward until you reach a comfortable depth. Keep your heels on the floor and your chest lifted."
                ))
            if self._valgus_detected_in_rep:
                flaws.append((
                    "Knees caving inward (valgus)",
                    "Your knees are caving inward during the squat. This places strain on your joints. Actively drive your knees outward in line with your toes as you descend and rise."
                ))
            if knee_angle < 155.0:
                flaws.append((
                    "Incomplete standing extension",
                    "You did not stand all the way up. Extend your hips and knees fully at the top and squeeze your glutes before your next rep."
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
            self._min_knee_angle_in_rep = knee_angle
            self._min_back_angle_in_rep = back_angle
            self._valgus_detected_in_rep = False

        else:
            # Real-time form coaching
            if back_visible and back_angle < 120.0:
                self.form_status = "Needs Correction"
                self.form_issue = "Excessive forward trunk lean"
                self.form_correction = "Your upper body is leaning too far forward. Keep your chest lifted, engage your core, and look straight ahead throughout the squat."
            elif valgus_active:
                self.form_status = "Needs Correction"
                self.form_issue = "Knees caving inward (valgus)"
                self.form_correction = "Your knees are caving inward during the squat. This places strain on your joints. Actively drive your knees outward in line with your toes as you descend and rise."
            elif knee_angle > 105.0 and self.stage == "inflected":
                self.form_status = "Needs Correction"
                self.form_issue = "Insufficient squat depth"
                self.form_correction = "Your squat depth is too shallow. Squat a little deeper for full range of motion. Slowly bend your knees and push your hips backward until you reach a comfortable depth. Keep your heels on the floor and your chest lifted."
            else:
                self.form_status = "Good"
                self.form_issue = "None - Form is clean"
                self.form_correction = "Great adjustment! Your form is clean, balanced, and under control. Keep up this exact technique."

        if knee_angle <= self.state_machine.inflection_threshold:
            depth_status = "GOOD DEPTH"
        elif knee_angle <= 135:
            depth_status = "HALF SQUAT"
        elif knee_angle >= (self.state_machine.start_threshold - 10):
            depth_status = "STANDING"
        else:
            depth_status = "DESCENDING"

        return {
            "reps": self.valid_reps,
            "valid_reps": self.valid_reps,
            "invalid_reps": self.invalid_reps,
            "rep_incremented": rep_incremented,
            "rep_invalid": rep_invalid,
            "knee_angle": int(knee_angle),
            "back_angle": int(back_angle),
            "depth_status": depth_status,
            "form_status": self.form_status,
            "form_issue": self.form_issue,
            "form_correction": self.form_correction,
            "posture_feedback": self.form_correction if self.form_status == "Needs Correction" else "Good Form",
        }

EXERCISE_OPTIONS=[
    "Squats",
    "Push-ups",
    "Biceps Curls (Dumbbell)",
    "Shoulder Press",
    "Lunges"
]


POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),       # Shoulders & Arms
    (11, 23), (12, 24), (23, 24),                           # Torso / Hips
    (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)  # Legs
]


METRICS_FIELDS = {
    "Squats": {
        "knee_angle": 0,
        "back_angle": 0,
        "depth_status": "N/A",
        "form_status": "Good",
        "form_issue": "None - Form is clean",
        "form_correction": "Maintain this form!",
        "valid_reps": 0,
        "invalid_reps": 0,
    },
    "Push-ups": {
        "elbow_angle": 0,
        "body_alignment": "N/A",
        "hip_status": "N/A",
        "form_status": "Good",
        "form_issue": "None - Form is clean",
        "form_correction": "Maintain this form!",
        "valid_reps": 0,
        "invalid_reps": 0,
    },
    "Biceps Curls (Dumbbell)": {
        "elbow_angle": 0,
        "shoulder_status": "N/A",
        "swing_status": "N/A",
        "form_status": "Good",
        "form_issue": "None - Form is clean",
        "form_correction": "Maintain this form!",
        "valid_reps": 0,
        "invalid_reps": 0,
    },
    "Shoulder Press": {
        "elbow_angle": 0,
        "extension_status": "N/A",
        "back_arch_status": "N/A",
        "form_status": "Good",
        "form_issue": "None - Form is clean",
        "form_correction": "Maintain this form!",
        "valid_reps": 0,
        "invalid_reps": 0,
    },
    "Lunges": {
        "front_knee_angle": 0,
        "torso_angle": 0,
        "balance_status": "N/A",
        "form_status": "Good",
        "form_issue": "None - Form is clean",
        "form_correction": "Maintain this form!",
        "valid_reps": 0,
        "invalid_reps": 0,
    },
}


DETECTOR_CONFIG = {
    "Squats": {
        "start_threshold": 160.0,
        "inflection_threshold": 100.0,
        "hysteresis": 15.0,
        "min_rep_duration": 0.8,
        "consecutive_frames": 2,
        "min_visibility": 0.65,
    },
    "Push-ups": {
        "start_threshold": 155.0,
        "inflection_threshold": 90.0,
        "hysteresis": 15.0,
        "min_rep_duration": 0.7,
        "consecutive_frames": 2,
        "min_visibility": 0.65,
        "hip_sag_tolerance": 0.08,
    },
    "Biceps Curls (Dumbbell)": {
        "start_threshold": 150.0,
        "inflection_threshold": 50.0,
        "hysteresis": 15.0,
        "min_rep_duration": 0.7,
        "consecutive_frames": 2,
        "min_visibility": 0.65,
        "elbow_drift_tolerance": 0.06,
        "swing_threshold": 15.0,
    },
    "Shoulder Press": {
        "start_threshold": 95.0,
        "inflection_threshold": 155.0,
        "hysteresis": 15.0,
        "min_rep_duration": 0.8,
        "consecutive_frames": 2,
        "min_visibility": 0.65,
    },
    "Lunges": {
        "start_threshold": 155.0,
        "inflection_threshold": 100.0,
        "hysteresis": 15.0,
        "min_rep_duration": 0.8,
        "consecutive_frames": 2,
        "min_visibility": 0.65,
        "balance_tolerance": 0.10,
    },
}


PROMPT = (
    "You are Apna AI Coach, a professional AI gym trainer monitoring a user's workout via live camera.\n\n"
    "### Your Role\n"
    "Provide around 10-15 words, high-energy coaching cues. You speak these aloud, so they must be natural and encouraging.\n\n"
    "### Input Format\n"
    "You receive updates in the format: 'Event: [state] Form Issue: [description]'.\n"
    "- 'Event': workout_started, set_completed, workout_completed, no_pose_detected, ongoing_form_check.\n"
    "- 'Form Issue': A technical description of a pose error (if any).\n\n"
    "### Guidelines\n"
    "1. Provide feedback in natural, short sentences. Avoid overly brief or fragmented responses.\n"
    "2. NO generic greetings or redundant questions. Focus on the workout.\n"
    "3. Use the second person (e.g., 'Straighten your back' instead of 'The user should straighten their back').\n"
    "4. Maintain a professional coaching tone and prioritize safety.\n\n"
    "### Scenario Response Styles\n"
    "- 'workout_started' -> A motivating and sharp command to begin.\n"
    "- 'workout_completed' -> A warm and encouraging closing for the session.\n"
    "- 'set_completed' -> Direct praise for finishing the set.\n"
    "- 'no_pose_detected' -> A clear instruction for the user to reposition within the camera frame.\n"
    "- 'ongoing_form_check' + Form Issue -> A precise, supportive correction for the detected error.\n"
    "- 'ongoing_form_check' (No Issue) -> Brief, energetic words of encouragement.\n"
)

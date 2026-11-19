import streamlit as st


def initial_session_defaults():
    defaults = {
        "reps": 0,
        "target_sets": 0,
        "reps_per_set": 0,
        "sets_completed": 0,
        "current_set_reps": 0,
        "workout_completed": False,
        "workout_complete": False,
        "workout_session_id": "",
        "last_notified_sets_completed": 0,
        "last_notified_workout_complete": False,
        "last_saved_sets_completed": 0,
        "set_cycle_started_at": 0.0,
        "last_exercise_type": "Squats",

        # Workout plan (set before starting)
        "workout_started": False,
        "plan_exercise": "Squats",
        "plan_sets": 3,
        "plan_reps": 10,

        # Voice & Coaching
        "voice_muted": False,
        "last_voiced_rep": 0,
        "last_voiced_set": 0,
        "last_voiced_invalid": 0,
        "audio_event_id": "",
        "last_played_audio_event_id": "",
        "audio_to_play": None,
        "coach_feedback": "",
        "voice_status": "Voice Active",
        "voice_test_audio": None,
        "voice_confirmed": None,
        "voice_test_issue": None,
        "voice_test_id": "",
        "last_played_test_id": "",

        # Audio Engine & Voice Diagnostics
        "audio_diag_event_id": "None",
        "audio_diag_event_name": "None",
        "audio_diag_issue": "None",
        "audio_diag_tts_status": "Ready",
        "audio_diag_validation_status": "Ready",
        "audio_diag_payload_bytes": 0,
        "audio_diag_queue_len": 0,
        "audio_diag_playback_status": "IDLE",
        "audio_diag_delivered_at": "N/A",
        "audio_diag_completed_at": "N/A",
        "audio_diag_mute_status": "ACTIVE",
        "audio_diag_last_error": "None",

        # Form Feedback & Quality Tracking
        "valid_reps": 0,
        "invalid_reps": 0,
        "form_status": "Good",
        "form_issue": "None - Form is clean",
        "form_correction": "Maintain this form!",

        # Common Angles
        "knee_angle": 0,
        "back_angle": 0,
        "elbow_angle": 0,
        "front_knee_angle": 0,
        "torso_angle": 0,

        # Status fields
        "depth_status": "N/A",
        "body_alignment": "N/A",
        "hip_status": "N/A",
        "shoulder_status": "N/A",
        "swing_status": "N/A",
        "extension_status": "N/A",
        "back_arch_status": "N/A",
        "balance_status": "N/A",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

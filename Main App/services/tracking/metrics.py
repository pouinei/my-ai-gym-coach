import streamlit as st
import time
import uuid
from services.config.workout_config import METRICS_FIELDS
from services.persistence.exercise_repository import add_exercise


def _apply_voice_result(result):
    if not result:
        return
    if len(result) == 3:
        audio_bytes, text, event_id = result
    elif len(result) == 2:
        audio_bytes, text = result
        event_id = uuid.uuid4().hex
    else:
        return

    st.session_state.audio_to_play = audio_bytes
    st.session_state.coach_feedback = text
    st.session_state.audio_event_id = event_id


def sync_metrics_update(context):
    if not context or not hasattr(context, "state") or not context.state.playing:
        return

    processor = getattr(context, "video_processor", None)
    if not processor:
        return

    exercise = st.session_state.get("exercise_type")
    if not exercise:
        return

    target_sets = int(st.session_state.get("target_sets", 0))
    reps_per_set = int(st.session_state.get("reps_per_set", 0))
    workout_session_id = st.session_state.get("workout_session_id")

    # Ensure processor is initialized with the current workout plan
    if hasattr(processor, "workout_manager"):
        if (
            not processor.workout_manager.is_active()
            and not processor.workout_manager.is_completed()
            and target_sets > 0
            and reps_per_set > 0
        ) or processor.workout_manager.session_id != workout_session_id:
            processor.start_workout(exercise, target_sets, reps_per_set)
            st.session_state.workout_session_id = processor.workout_manager.session_id
    else:
        processor.set_exercise(exercise)

    latest_metrics = processor.get_latest_metrics()
    if not latest_metrics:
        return

    # Extract authoritative state from processor or metrics
    raw_reps = latest_metrics.get("reps", 0) or 0
    raw_sets = latest_metrics.get("sets_completed", 0) or 0
    raw_current_reps = latest_metrics.get("current_set_reps", 0) or 0
    workout_completed = latest_metrics.get("workout_completed", False)

    # Strictly enforce boundaries and invariants
    if target_sets > 0 and reps_per_set > 0:
        sets_completed = min(raw_sets, target_sets)
        if sets_completed >= target_sets:
            workout_completed = True
            current_set_reps = reps_per_set
            total_reps = target_sets * reps_per_set
        else:
            current_set_reps = min(raw_current_reps, reps_per_set)
            total_reps = (sets_completed * reps_per_set) + current_set_reps
    else:
        sets_completed = 0
        current_set_reps = 0
        total_reps = raw_reps
        workout_completed = False

    st.session_state.reps = total_reps
    st.session_state.valid_reps = total_reps
    st.session_state.invalid_reps = latest_metrics.get("invalid_reps", 0)
    st.session_state.form_status = latest_metrics.get("form_status", "Good")
    st.session_state.form_issue = latest_metrics.get("form_issue", "None - Form is clean")
    st.session_state.form_correction = latest_metrics.get("form_correction", "Maintain this form!")
    st.session_state.sets_completed = sets_completed
    st.session_state.current_set_reps = current_set_reps
    st.session_state.workout_completed = workout_completed

    fields = METRICS_FIELDS.get(exercise)
    if fields:
        for key, default in fields.items():
            st.session_state[key] = latest_metrics.get(key, default)

    # 1. Rep completed voice notification (fires on individual repetition completion)
    last_voiced_rep = st.session_state.get("last_voiced_rep", 0)
    if total_reps > last_voiced_rep and not workout_completed:
        st.session_state.last_voiced_rep = total_reps
        # If set not completed on this rep, voice the individual rep completion
        if current_set_reps > 0 and st.session_state.get("voice_pipeline"):
            result = st.session_state.voice_pipeline.process_event(
                event="rep_completed",
                exercise=exercise,
                metrics=latest_metrics,
            )
            _apply_voice_result(result)

    # 1b. Rep invalid voice notification (fires when a repetition was rejected due to form violation)
    current_invalid = latest_metrics.get("invalid_reps", 0)
    last_voiced_invalid = st.session_state.get("last_voiced_invalid", 0)
    if current_invalid > last_voiced_invalid and not workout_completed:
        st.session_state.last_voiced_invalid = current_invalid
        if st.session_state.get("voice_pipeline"):
            result = st.session_state.voice_pipeline.process_event(
                event="rep_invalid",
                exercise=exercise,
                metrics=latest_metrics,
            )
            _apply_voice_result(result)

    # 2. Database persistence for newly completed sets (idempotent, strictly capped)
    last_saved_sets = st.session_state.get("last_saved_sets_completed", 0)
    if target_sets > 0 and reps_per_set > 0 and sets_completed > last_saved_sets:
        newly_completed = sets_completed - last_saved_sets
        now_ts = time.time()
        started_at = st.session_state.get("set_cycle_started_at", now_ts)
        time_taken = max(1, int(now_ts - started_at))
        user_id = st.session_state.get("user_id", 0)

        add_exercise(user_id, exercise, newly_completed * reps_per_set, newly_completed, time_taken)

        if not workout_completed and st.session_state.get("voice_pipeline"):
            result = st.session_state.voice_pipeline.process_event(
                event="set_completed",
                exercise=exercise,
                metrics=latest_metrics,
            )
            _apply_voice_result(result)

        st.session_state.set_cycle_started_at = now_ts
        st.session_state.last_saved_sets_completed = sets_completed

    # 3. Workout completed notification (fires exactly once upon finishing target sets)
    if workout_completed and not st.session_state.get("last_notified_workout_complete", False):
        st.session_state.last_notified_workout_complete = True

        if st.session_state.get("voice_pipeline"):
            result = st.session_state.voice_pipeline.process_event(
                event="workout_completed",
                exercise=exercise,
                metrics=latest_metrics,
            )
            _apply_voice_result(result)

    # 4. Pose detection warnings
    pose_detected = latest_metrics.get("pose_detected", True)
    if not pose_detected and st.session_state.get("voice_pipeline"):
        result = st.session_state.voice_pipeline.process_event(
            event="no_pose_detected",
            exercise=exercise,
            metrics={"issue": "No pose detected! Please step into the camera frame."},
        )
        _apply_voice_result(result)

    # 5. Form coaching and movement guidance
    if st.session_state.get("voice_pipeline"):
        result = st.session_state.voice_pipeline.process_event(
            event="ongoing_form_check",
            exercise=exercise,
            metrics=latest_metrics,
        )
        _apply_voice_result(result)

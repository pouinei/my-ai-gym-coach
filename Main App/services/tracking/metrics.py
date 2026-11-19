import streamlit as st
import time
from services.config.workout_config import METRICS_FIELDS
from services.persistence.exercise_repository import add_exercise


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
    st.session_state.sets_completed = sets_completed
    st.session_state.current_set_reps = current_set_reps
    st.session_state.workout_completed = workout_completed

    fields = METRICS_FIELDS.get(exercise)
    if fields:
        for key, default in fields.items():
            st.session_state[key] = latest_metrics.get(key, default)

    # Database persistence for newly completed sets (idempotent, strictly capped)
    last_saved_sets = st.session_state.get("last_saved_sets_completed", 0)
    if target_sets > 0 and reps_per_set > 0 and sets_completed > last_saved_sets:
        newly_completed = sets_completed - last_saved_sets
        now_ts = time.time()
        started_at = st.session_state.get("set_cycle_started_at", now_ts)
        time_taken = max(1, int(now_ts - started_at))
        user_id = st.session_state.get("user_id", 0)

        add_exercise(user_id, exercise, newly_completed * reps_per_set, newly_completed, time_taken)

        if st.session_state.get("voice_pipeline"):
            result = st.session_state.voice_pipeline.process_event(
                event="set_completed",
                exercise=exercise,
                metrics=latest_metrics,
            )
            if result:
                st.session_state.audio_to_play, st.session_state.coach_feedback = result

        st.session_state.set_cycle_started_at = now_ts
        st.session_state.last_saved_sets_completed = sets_completed

    # Workout completed notification (fires exactly once)
    if workout_completed and not st.session_state.get("last_notified_workout_complete", False):
        st.session_state.last_notified_workout_complete = True

        if st.session_state.get("voice_pipeline"):
            result = st.session_state.voice_pipeline.process_event(
                event="workout_completed",
                exercise=exercise,
                metrics=latest_metrics,
            )
            if result:
                st.session_state.audio_to_play, st.session_state.coach_feedback = result

    # Pose detection warnings
    pose_detected = latest_metrics.get("pose_detected", True)
    if not pose_detected and st.session_state.get("voice_pipeline"):
        result = st.session_state.voice_pipeline.process_event(
            event="no_pose_detected",
            exercise=exercise,
            metrics={"issue": "No pose detected! Please step into the camera frame."},
        )
        if result:
            st.session_state.audio_to_play, st.session_state.coach_feedback = result

    # Form coaching check
    if st.session_state.get("voice_pipeline"):
        result = st.session_state.voice_pipeline.process_event(
            event="ongoing_form_check",
            exercise=exercise,
            metrics=latest_metrics,
        )
        if result:
            st.session_state.audio_to_play, st.session_state.coach_feedback = result

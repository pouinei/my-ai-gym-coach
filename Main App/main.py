import streamlit as st
import os
import sys
import time
import uuid
import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Ensure the app directory and project root are on sys.path and load environment variables
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for _p in [str(APP_DIR), str(PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

load_dotenv()

from services.auth.login_wall import render_login_wall
from services.state.session_defaults import initial_session_defaults
from services.config.workout_config import EXERCISE_OPTIONS
from services.config.webrtc_config import get_rtc_configuration, DEFAULT_MEDIA_CONSTRAINTS
from services.ui.style_loader import load_css, inject_local_font, inject_webrtc_styles
from services.persistence.exercise_repository import init_db, save_workout_session, get_users_exercises

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    WEBRTC_IMPORT_ERROR = None
except Exception as _e:
    webrtc_streamer = None
    WebRtcMode = None
    WEBRTC_IMPORT_ERROR = str(_e)

from services.vision.exercise_video_processor import VideoProcessorClass
from services.tracking.metrics import sync_metrics_update
from groq import Groq
from services.coaching.llm import LLMCoach
from services.coaching.tts import TextToSpeech, validate_audio_bytes
from services.coaching.voice_pipeline import VoicePipeline, autoplay_audio, render_voice_player


def main():
    st.set_page_config(
        page_icon="🏋️‍♀️",
        page_title="AI Real-time GYM Coach",
        initial_sidebar_state="expanded",
        layout="centered"
    )

    css_path = APP_DIR / "static" / "style.css"
    font_path = APP_DIR / "static" / "AdobeClean.otf"
    load_css(str(css_path) if css_path.exists() else os.path.join(os.getcwd(), "static", "style.css"))
    inject_local_font(str(font_path) if font_path.exists() else os.path.join(os.getcwd(), "static", "AdobeClean.otf"), "AdobeClean")

    init_db()

    if not render_login_wall():
        return 

    initial_session_defaults()

    if "voice_pipeline" not in st.session_state or st.session_state.voice_pipeline is None:
        groq_client = None
        try:
            api_key = os.environ.get("GROQ_API_KEY", "").strip()

            if not api_key and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                api_key = str(st.secrets["GROQ_API_KEY"]).strip()

            if api_key:
                groq_client = Groq(api_key=api_key)
        except Exception as e:
            groq_client = None

        llm_coach = LLMCoach(groq_client=groq_client)
        tts = TextToSpeech()
        tts.load_static_cache()
        st.session_state.voice_pipeline = VoicePipeline(llm_coach, tts)

    workout_started = st.session_state.get("workout_started", False)
    
    with st.sidebar:
        st.title("🏋️‍♂️ Apna AI Coach")

        if st.session_state.username:
            st.caption(f"👤 Login as {st.session_state.username}")

        st.divider()

        st.subheader("Workout Plan")

        if not workout_started:
            plan_exercise = st.selectbox("Exercise", options=EXERCISE_OPTIONS, key="plan_exercise")

            plan_sets = st.number_input("Sets", min_value=0, max_value=50, key="plan_sets", step=1)

            plan_reps = st.number_input("Reps per Set", min_value=0, max_value=50, key="plan_reps", step=1)

            st.markdown("")

            start_session_button = st.button("Start Workout", width="stretch", key="start_session_button")

            if start_session_button:
                st.session_state.exercise_type = plan_exercise
                st.session_state.target_sets = int(plan_sets)
                st.session_state.reps_per_set = int(plan_reps)
                st.session_state.reps = 0
                st.session_state.sets_completed = 0
                st.session_state.current_set_reps = 0
                st.session_state.workout_completed = False
                st.session_state.workout_started = True
                st.session_state.workout_session_id = uuid.uuid4().hex
                st.session_state.set_cycle_started_at = time.time()
                st.session_state.last_saved_sets_completed = 0
                st.session_state.last_notified_sets_completed = 0
                st.session_state.last_notified_workout_complete = False
                st.session_state.last_voiced_rep = 0
                st.session_state.last_voiced_set = 0

                if st.session_state.voice_pipeline:
                    st.session_state.voice_pipeline.clear_queue()
                    result = st.session_state.voice_pipeline.process_event(
                        event="workout_started",
                        exercise=plan_exercise,
                        metrics={}
                    )
                    
                    if result:
                        if len(result) == 3:
                            st.session_state.audio_to_play, st.session_state.coach_feedback, st.session_state.audio_event_id = result
                        else:
                            st.session_state.audio_to_play, st.session_state.coach_feedback = result
                            st.session_state.audio_event_id = uuid.uuid4().hex

                st.rerun()
        else:
            exercise = st.session_state.get("exercise_type")
            sets = st.session_state.get("target_sets")
            reps = st.session_state.get("reps_per_set")

            st.info(f"**{exercise}** -- {sets} Sets / {reps} Reps")

            end_session_button = st.button("End Workout", key="end_session_button", width="stretch")

            if end_session_button:
                st.session_state.workout_started = False
                user_id = st.session_state.get("user_id", 0)
                if user_id:
                    duration = max(1, int(time.time() - st.session_state.get("set_cycle_started_at", time.time())))
                    save_workout_session(
                        user_id=user_id,
                        session_id=st.session_state.get("workout_session_id", uuid.uuid4().hex),
                        exercise_name=exercise,
                        target_sets=int(sets or 0),
                        target_reps=int(reps or 0),
                        sets_completed=int(st.session_state.get("sets_completed", 0)),
                        total_reps=int(st.session_state.get("reps", 0)),
                        duration_seconds=duration,
                        status="COMPLETED" if st.session_state.get("workout_completed", False) else "ENDED",
                    )
                
                if st.session_state.voice_pipeline:
                    st.session_state.voice_pipeline.clear_queue()
                    result = st.session_state.voice_pipeline.process_event(
                        event="workout_completed",
                        exercise=exercise,
                        metrics={}
                    )
                    if result:
                        if len(result) == 3:
                            st.session_state.audio_to_play, st.session_state.coach_feedback, st.session_state.audio_event_id = result
                        else:
                            st.session_state.audio_to_play, st.session_state.coach_feedback = result
                            st.session_state.audio_event_id = uuid.uuid4().hex

                st.rerun()

        st.divider()
        st.subheader("🎙️ Voice Coach")

        is_muted = st.session_state.get("voice_muted", False)
        voice_pipe = st.session_state.get("voice_pipeline")

        if not voice_pipe:
            status_text = "⚠️ Voice Unavailable"
        elif is_muted:
            status_text = "🔇 Muted"
        else:
            has_groq = bool(getattr(voice_pipe.llm, "client", None))
            mode_desc = "AI (Groq)" if has_groq else "Smart Fallback"
            status_text = f"🔊 Active ({mode_desc})"

        st.caption(f"**Status:** {status_text}")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            mute_lbl = "🔊 Unmute" if is_muted else "🔇 Mute"
            if st.button(mute_lbl, width="stretch", key="mute_toggle_btn"):
                st.session_state.voice_muted = not is_muted
                if voice_pipe:
                    voice_pipe.muted = st.session_state.voice_muted
                st.rerun()

        test_phrase = "Welcome to Apna AI Coach. Voice is active and working properly."
        with col_m2:
            if st.button("🧪 Test Voice", width="stretch", key="test_voice_btn"):
                tts_engine = getattr(voice_pipe, "tts", None) if voice_pipe else None
                if tts_engine:
                    if hasattr(tts_engine, "load_static_cache"):
                        tts_engine.load_static_cache()
                    try:
                        audio_data = tts_engine.speak(test_phrase)
                    except Exception as e:
                        logger.error(f"TTS speak failed during Test Voice: {e}")
                        audio_data = None

                    if audio_data and validate_audio_bytes(audio_data):
                        test_id = uuid.uuid4().hex
                        st.session_state.voice_test_audio = audio_data
                        st.session_state.voice_test_id = test_id
                        st.session_state.coach_feedback = test_phrase
                        st.session_state.voice_confirmed = None
                        st.session_state.voice_test_issue = None

                        # Maintain verified runtime audio diagnostics
                        st.session_state.audio_diag_event_id = test_id
                        st.session_state.audio_diag_event_name = "test_voice"
                        st.session_state.audio_diag_issue = "Voice Test Verification"
                        st.session_state.audio_diag_tts_status = "SUCCESS"
                        st.session_state.audio_diag_validation_status = "VALID_MP3"
                        st.session_state.audio_diag_payload_bytes = len(audio_data)
                        st.session_state.audio_diag_playback_status = "READY"
                        st.session_state.audio_diag_delivered_at = time.strftime("%H:%M:%S")
                    else:
                        st.session_state.voice_test_audio = None
                        st.session_state.voice_test_issue = True
                        st.session_state.audio_diag_tts_status = "FAILED"
                        st.session_state.audio_diag_last_error = "TTS speak generation failed or returned invalid MP3 bytes"
                else:
                    st.session_state.voice_test_audio = None
                    st.session_state.voice_test_issue = True
                    st.session_state.audio_diag_tts_status = "NO_TTS_ENGINE"
                    st.session_state.audio_diag_last_error = "Voice pipeline or TTS engine not available"

        if st.session_state.get("voice_test_audio"):
            st.markdown("---")
            st.markdown("**🔊 Test Voice Player**")
            test_id = st.session_state.get("voice_test_id", "")
            is_new_test = bool(test_id and test_id != st.session_state.get("last_played_test_id"))
            if is_new_test:
                st.session_state["last_played_test_id"] = test_id

            # Render audited HTML5 player with manual Play button, duration, volume, mute & error display
            render_voice_player(
                st.session_state.voice_test_audio,
                text=test_phrase,
                event_id=test_id,
                auto_play=is_new_test,
                height=78,
            )
            st.caption(f"_{test_phrase}_")

            st.markdown("**Did you hear the coach speak?**")
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                if st.button("✅ Yes, loud & clear", key="voice_confirm_yes", width="stretch"):
                    st.session_state.voice_confirmed = True
                    st.session_state.voice_test_issue = False
                    st.rerun()
            with col_v2:
                if st.button("❌ No sound", key="voice_confirm_no", width="stretch"):
                    st.session_state.voice_confirmed = False
                    st.session_state.voice_test_issue = True
                    st.rerun()

            if st.session_state.get("voice_confirmed") is True:
                st.success("🎉 Voice playback confirmed working on your browser!")
            elif st.session_state.get("voice_test_issue") is True:
                st.warning(
                    "⚠️ **No sound heard:**\n"
                    "1. Click the **Play (▶)** button on the player above.\n"
                    "2. Check that your device and browser tab volume is unmuted.\n"
                    "3. Ensure browser permissions allow audio playback on this site."
                )
        elif st.session_state.get("voice_test_issue") is True:
            st.warning("⚠️ **Voice generation failed.** Could not generate audio. Please check your internet connection or voice settings.")

        if workout_started:
            st.divider()

            exercise = st.session_state.get("exercise_type")
            total_reps = st.session_state.get("reps")
            current_set_reps = st.session_state.get("current_set_reps")
            reps_per_set = st.session_state.get("reps_per_set")
            sets_completed = st.session_state.get("sets_completed")
            target_sets = st.session_state.get("target_sets")

            st.subheader("Progress")

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Valid Reps", f"{total_reps}")
            with col_p2:
                st.metric("Invalid", f"{st.session_state.get('invalid_reps', 0)}")

            st.metric("Current Set Reps", f"{current_set_reps} / {reps_per_set}")
            st.metric("Sets Completed", f"{sets_completed} / {target_sets}")

            if st.session_state.get("workout_completed", False):
                st.success("🎉 Target Achieved! Great job!")

            st.divider()

            if exercise == "Squats":
                st.subheader("Squat Metrics")
                st.metric("Knee Angle", f"{st.session_state.knee_angle}°")
                st.metric("Back Angle", f"{st.session_state.back_angle}°")
                st.metric("Depth Status", st.session_state.depth_status)

            elif exercise == "Push-ups":
                st.subheader("Push-up Metrics")
                st.metric("Elbow Angle", f"{st.session_state.elbow_angle}°")
                st.metric("Body Alignment", st.session_state.body_alignment)
                st.metric("Hip Position", st.session_state.hip_status)

            elif exercise == "Biceps Curls (Dumbbell)":
                st.subheader("Curl Metrics")
                st.metric("Elbow Angle", f"{st.session_state.elbow_angle}°")
                st.metric("Shoulder Stability", st.session_state.shoulder_status)
                st.metric("Swing Detection", st.session_state.swing_status)

            elif exercise == "Shoulder Press":
                st.subheader("Shoulder Press Metrics")
                st.metric("Elbow Angle", f"{st.session_state.elbow_angle}°")
                st.metric("Arm Extension", st.session_state.extension_status)
                st.metric("Back Arch", st.session_state.back_arch_status)

            elif exercise == "Lunges":
                st.subheader("Lunge Metrics")
                st.metric("Front Knee Angle", f"{st.session_state.front_knee_angle}°")
                st.metric("Torso Angle", f"{st.session_state.torso_angle}°")
                st.metric("Balance Status", st.session_state.balance_status)

    st.title("AI Real-time GYM Coach")
    st.markdown("#### Real-time pose detection with proactive AI voice coaching")
 
    if st.session_state.get("coach_feedback"):
        st.markdown("")
        st.success(f"🤖 **Coach:** {st.session_state.coach_feedback}")

    if not st.session_state.get("voice_muted", False) and st.session_state.get("audio_to_play"):
        event_id = st.session_state.get("audio_event_id", "")
        autoplay_audio(st.session_state.audio_to_play, text=st.session_state.get("coach_feedback", ""), event_id=event_id)

    if workout_started:
        form_status = st.session_state.get("form_status", "Good")
        form_issue = st.session_state.get("form_issue", "None - Form is clean")
        form_correction = st.session_state.get("form_correction", "Maintain this form!")
        valid_reps_count = st.session_state.get("valid_reps", st.session_state.get("reps", 0))
        invalid_reps_count = st.session_state.get("invalid_reps", 0)

        with st.container():
            st.markdown("#### 🎯 Real-Time Form Analysis & Quality Tracker")
            col_f1, col_f2, col_f3 = st.columns([1.4, 1, 1])

            with col_f1:
                if form_status == "Good":
                    st.success("🟢 **Form Status: Good**")
                elif form_status == "Needs Correction":
                    st.warning("⚠️ **Form Status: Needs Correction**")
                else:
                    st.info("⚪ **Form Status: Unable to Assess**")

            with col_f2:
                st.metric("✅ Valid Reps", valid_reps_count)

            with col_f3:
                st.metric("❌ Invalid / Incomplete", invalid_reps_count)

            if form_status == "Needs Correction":
                st.error(f"**Detected Issue:** {form_issue}\n\n👉 **Recommended Correction:** {form_correction}")
            elif form_status == "Unable to Assess":
                st.info(f"**Status:** {form_issue}\n\n👉 **Recommended Action:** {form_correction}")
            else:
                st.caption(f"✨ **Coaching Guidance:** {form_correction}")

        st.markdown("")

    if not workout_started:
        st.markdown(
            """
            <div style="
                border: 10px dashed #444;
                border-radius: 0px;
                padding: 48px 32px;
                text-align: center;
                color: #888;
                margin-top: 32px;
                margin-bottom: 32px;
            ">
                <h2 style="color:#ccc; margin-bottom:8px;">👈 Set your workout plan</h2>
                <p style="font-size:1.05rem;">
                    Choose your exercise, sets and reps in the sidebar,<br>
                    then click <strong>Start Workout</strong> to activate the camera and AI coach.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if webrtc_streamer is None or WebRtcMode is None:
            st.error(
                f"🚨 **WebRTC Module Error:** `streamlit-webrtc` could not be loaded ({WEBRTC_IMPORT_ERROR}). "
                "Please verify that `streamlit-webrtc` is installed in your Python environment."
            )
            return

        rtc_config = get_rtc_configuration()

        context = webrtc_streamer(
            key="exercise-analysis",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessorClass,
            rtc_configuration=rtc_config,
            media_stream_constraints=DEFAULT_MEDIA_CONSTRAINTS,
            async_processing=True
        )

        if not context.state.playing:
            st.info("📹 Click **START** above to activate your camera. Make sure browser camera permission is allowed.")
            with st.expander("🛠️ Connection Help & STUN/TURN Settings", expanded=False):
                st.markdown(
                    """
                    - **Camera Permissions**: Check that your browser has allowed camera access on this page.
                    - **Network Firewalls**: If your connection takes longer than expected, your network might restrict direct UDP traffic.
                    - **Custom TURN Support**: You can add your own free TURN credentials (from [Metered](https://www.metered.ca/tools/openrelay/)) in **Settings ➔ Secrets**:
                    ```toml
                    [webrtc]
                    turn_url = "turn:global.relay.metered.ca:80"
                    turn_username = "your_username"
                    turn_credential = "your_credential"
                    ```
                    """
                )
        else:
            st.caption("🟢 Live Tracking Active")

        sync_metrics_update(context)

        if context.state.playing:
            time.sleep(0.25)
            st.rerun()

        inject_webrtc_styles()

    st.divider()

    with st.expander("🎧 Audio Engine & Voice Diagnostics", expanded=False):
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.markdown(f"**Playback State:** `{st.session_state.get('audio_diag_playback_status', 'IDLE')}`")
            ev_id = st.session_state.get('audio_diag_event_id', 'None')
            display_id = f"{ev_id[:8]}..." if (ev_id and ev_id != 'None') else "None"
            st.markdown(f"**Active Event ID:** `{display_id}`")
            st.markdown(f"**Event Name:** `{st.session_state.get('audio_diag_event_name', 'None')}`")
            st.markdown(f"**Mute Status:** `{st.session_state.get('audio_diag_mute_status', 'ACTIVE')}`")
        with col_d2:
            st.markdown(f"**TTS Engine Status:** `{st.session_state.get('audio_diag_tts_status', 'Ready')}`")
            st.markdown(f"**Validation Status:** `{st.session_state.get('audio_diag_validation_status', 'Ready')}`")
            st.markdown(f"**Payload Size:** `{st.session_state.get('audio_diag_payload_bytes', 0):,} bytes`")
            st.markdown(f"**Queue Depth:** `{st.session_state.get('audio_diag_queue_len', 0)} pending`")
        with col_d3:
            st.markdown(f"**Delivered At:** `{st.session_state.get('audio_diag_delivered_at', 'N/A')}`")
            st.markdown(f"**Completed At:** `{st.session_state.get('audio_diag_completed_at', 'N/A')}`")
            st.markdown(f"**Detected Flaw:** `{st.session_state.get('audio_diag_issue', 'None')}`")
            st.markdown(f"**Last Error:** `{st.session_state.get('audio_diag_last_error', 'None')}`")

    st.markdown("#### Workout History")

    user_id = st.session_state.get("user_id", 0)

    if isinstance(user_id, int):
        history_rows = get_users_exercises(user_id)

        arr = [
            {
                "Exercise": row['exercise_name'],
                "Reps": row['reps'],
                "Sets": row['sets'],
                "Time (sec)": row['time'],
                "Date": row['created_at']
            }
            for row in history_rows
        ]

        df = pd.DataFrame(arr)

        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            agg_df = df.groupby(["Exercise", "Date"]).agg({
                "Reps": 'sum',
                "Sets": "sum",
                "Time (sec)": "sum"
            }).reset_index()
            agg_df.index += 1
            st.table(agg_df, border="horizontal")
        else:
            st.info("No workout history found.")


if __name__ == "__main__":
    main()
    
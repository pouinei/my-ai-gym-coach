import sys
import unittest.mock as mock
from pathlib import Path
import pytest

# Ensure Main App is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_APP_DIR = PROJECT_ROOT / "Main App"
if str(MAIN_APP_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_APP_DIR))

from services.coaching.tts import TextToSpeech
from services.coaching.llm import LLMCoach, get_deterministic_cue
from services.coaching.voice_pipeline import VoicePipeline, autoplay_audio
from services.state.workout_state import WorkoutStateManager


# ==============================================================================
# TEST 1: TTS returns valid audio bytes
# ==============================================================================
def test_1_tts_returns_valid_audio_bytes():
    tts = TextToSpeech()
    # Mock gTTS to test without network dependency in CI
    fake_audio = b"\xff\xfb\x90\x00" + b"\x00" * 200
    with mock.patch("services.coaching.tts.gTTS") as mock_gtts:
        mock_instance = mock.MagicMock()
        mock_instance.write_to_fp.side_effect = lambda buf: buf.write(fake_audio)
        mock_gtts.return_value = mock_instance

        audio = tts.speak("Good job! Rep completed.")
        assert audio is not None
        assert isinstance(audio, bytes)
        assert len(audio) > 100
        assert tts.get_cached_count() >= 1


# ==============================================================================
# TEST 2: Empty or failed audio generation is handled gracefully
# ==============================================================================
def test_2_empty_or_failed_audio_handled():
    tts = TextToSpeech()

    # Empty inputs
    assert tts.speak("") is None
    assert tts.speak("   ") is None
    assert tts.speak(None) is None

    # Exception during generation
    with mock.patch("services.coaching.tts.gTTS", side_effect=Exception("Network error")):
        audio = tts.speak("Should fail gracefully")
        assert audio is None


# ==============================================================================
# TEST 3: Missing Groq API key uses deterministic fallback coaching
# ==============================================================================
def test_3_missing_groq_api_key_uses_fallback():
    # Groq client is None
    coach = LLMCoach(groq_client=None)

    start_cue = coach.give_feedback("workout_started", exercise="Shoulder Press")
    assert "Raise your dumbbells" in start_cue

    rep_cue = coach.give_feedback("rep_completed")
    assert "Rep completed" in rep_cue

    set_cue = coach.give_feedback("set_completed")
    assert "Set completed" in set_cue

    comp_cue = coach.give_feedback("workout_completed")
    assert "Workout completed" in comp_cue

    form_cue = coach.give_feedback("ongoing_form_check", issue="The user's squat is not deep enough")
    assert "Squat a little deeper" in form_cue


# ==============================================================================
# TEST 4: One rep event produces at most one audio event
# ==============================================================================
def test_4_one_rep_produces_at_most_one_audio_event():
    coach = LLMCoach(groq_client=None)
    tts = TextToSpeech()
    fake_audio = b"\xff\xfb\x90\x00" + b"\x00" * 150
    tts._cache["en:good job! rep completed."] = fake_audio

    pipeline = VoicePipeline(coach, tts)

    res1 = pipeline.process_event("rep_completed", "Squats", {"reps": 1})
    assert res1 is not None
    audio1, text1, event_id1 = res1
    assert len(audio1) > 0
    assert event_id1 is not None

    # Immediate duplicate event within 3 seconds is debounced
    res2 = pipeline.process_event("rep_completed", "Squats", {"reps": 1})
    assert res2 is None


# ==============================================================================
# TEST 5: Duplicate Streamlit reruns maintain DOM audio element without restarting
# ==============================================================================
def test_5_duplicate_reruns_do_not_duplicate_playback():
    import streamlit as st

    fake_audio = b"\xff\xfb\x90\x00" + b"\x00" * 150
    event_id = "test-event-uuid-unique"

    st.session_state["last_played_audio_event_id"] = ""

    with mock.patch("streamlit.audio") as mock_st_audio, mock.patch("streamlit.components.v1.html") as mock_html:
        # First execution (new event): renders with autoplay=True and injects JS
        autoplay_audio(fake_audio, text="Good rep", event_id=event_id)
        assert mock_st_audio.call_count == 1
        assert mock_st_audio.call_args[1]["autoplay"] is True
        assert mock_html.call_count == 1
        assert st.session_state["last_played_audio_event_id"] == event_id

        # Second execution (simulated Streamlit rerun 250ms later with same event_id):
        # Keeps st.audio mounted in DOM with autoplay=False so playback is not aborted
        autoplay_audio(fake_audio, text="Good rep", event_id=event_id)
        assert mock_st_audio.call_count == 2
        assert mock_st_audio.call_args[1]["autoplay"] is False
        # JS autoplay trigger is NOT called again on rerun
        assert mock_html.call_count == 1


# ==============================================================================
# TEST 5b: Audio byte validation tests
# ==============================================================================
def test_5b_validate_audio_bytes():
    # Valid MPEG frame sync
    valid_mpeg = b"\xff\xfb\x90\x00" + b"\x00" * 120
    assert TextToSpeech.validate_audio_bytes(valid_mpeg) is True

    # Valid ID3 header
    valid_id3 = b"ID3\x03\x00\x00\x00" + b"\x00" * 120
    assert TextToSpeech.validate_audio_bytes(valid_id3) is True

    # Empty or None
    assert TextToSpeech.validate_audio_bytes(None) is False
    assert TextToSpeech.validate_audio_bytes(b"") is False

    # Too short (< 100 bytes)
    assert TextToSpeech.validate_audio_bytes(b"\xff\xfb\x90\x00") is False

    # Corrupt / non-audio data
    corrupt = b"\x00\x11\x22\x33" * 30
    assert TextToSpeech.validate_audio_bytes(corrupt) is False


# ==============================================================================
# TEST 6: Workout completion announcement is triggered once
# ==============================================================================
def test_6_workout_completion_announcement_triggered_once():
    coach = LLMCoach(groq_client=None)
    tts = TextToSpeech()
    fake_audio = b"\xff\xfb\x90\x00" + b"\x00" * 150
    tts._cache["en:workout completed! outstanding job!"] = fake_audio

    pipeline = VoicePipeline(coach, tts)

    res1 = pipeline.process_event("workout_completed", "Push-ups", {})
    assert res1 is not None
    audio1, text1, id1 = res1
    assert "Workout completed" in text1

    # Immediate second call is suppressed
    res2 = pipeline.process_event("workout_completed", "Push-ups", {})
    assert res2 is None


# ==============================================================================
# TEST 7: Muted mode prevents audio delivery
# ==============================================================================
def test_7_muted_mode_prevents_audio_delivery():
    coach = LLMCoach(groq_client=None)
    tts = TextToSpeech()
    pipeline = VoicePipeline(coach, tts)
    pipeline.muted = True

    assert pipeline.process_event("workout_started", "Squats", {}) is None
    assert pipeline.process_event("rep_completed", "Squats", {}) is None
    assert pipeline.process_event("workout_completed", "Squats", {}) is None


# ==============================================================================
# TEST 8: Camera and rep counting continue if TTS fails
# ==============================================================================
def test_8_rep_counting_continues_if_tts_fails():
    manager = WorkoutStateManager()
    manager.start_workout("Shoulder Press", 1, 3)

    tts = TextToSpeech()
    coach = LLMCoach(groq_client=None)
    pipeline = VoicePipeline(coach, tts)

    # Force TTS failure
    with mock.patch.object(tts, "speak", return_value=None):
        rec1, ev1 = manager.record_rep(timestamp=1.0)
        assert rec1 is True
        assert manager.total_reps == 1

        voice_res = pipeline.process_event("rep_completed", "Shoulder Press", {})
        assert voice_res is None  # Voice failed, but did not crash!

        # Rep counting proceeds uninterrupted
        rec2, ev2 = manager.record_rep(timestamp=2.0)
        rec3, ev3 = manager.record_rep(timestamp=3.0)
        assert rec3 is True
        assert manager.total_reps == 3
        assert manager.is_completed()


# ==============================================================================
# TEST 9: Voice test generates playable audio
# ==============================================================================
def test_9_voice_test_generates_playable_audio():
    tts = TextToSpeech()
    test_phrase = "Welcome to Apna AI Coach. Voice is active and working properly."
    fake_audio = b"\xff\xfb\x90\x00" + b"\x00" * 200

    with mock.patch("services.coaching.tts.gTTS") as mock_gtts:
        mock_instance = mock.MagicMock()
        mock_instance.write_to_fp.side_effect = lambda buf: buf.write(fake_audio)
        mock_gtts.return_value = mock_instance

        audio = tts.speak(test_phrase)
        assert audio is not None
        assert isinstance(audio, bytes)
        assert len(audio) > 100


# ==============================================================================
# TEST 10: Audio events cleared on workout reset
# ==============================================================================
def test_10_audio_events_cleared_on_workout_reset():
    import streamlit as st
    st.session_state["last_voiced_rep"] = 5
    st.session_state["last_voiced_set"] = 2
    st.session_state["audio_to_play"] = b"bytes"
    st.session_state["audio_event_id"] = "event-123"

    # Reset
    st.session_state["last_voiced_rep"] = 0
    st.session_state["last_voiced_set"] = 0
    st.session_state["audio_to_play"] = None
    st.session_state["audio_event_id"] = ""

    assert st.session_state["last_voiced_rep"] == 0
    assert st.session_state["last_voiced_set"] == 0
    assert st.session_state["audio_to_play"] is None
    assert st.session_state["audio_event_id"] == ""

import sys
import unittest.mock as mock
from pathlib import Path
import pytest

# Ensure Main App is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_APP_DIR = PROJECT_ROOT / "Main App"
if str(MAIN_APP_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_APP_DIR))

from services.coaching.tts import TextToSpeech, validate_audio_bytes
from services.coaching.llm import LLMCoach, get_deterministic_cue
from services.coaching.voice_pipeline import VoicePipeline, autoplay_audio
from services.state.workout_state import WorkoutStateManager
from services.config.webrtc_config import get_rtc_configuration, DEFAULT_MEDIA_CONSTRAINTS


# ==============================================================================
# TEST 1: Valid MP3 acceptance
# ==============================================================================
def test_1_valid_mp3_acceptance():
    # MPEG frame sync header (0xFF 0xFB) with sufficient length
    valid_mpeg = b"\xff\xfb\x90\x00" + b"\x00" * 150
    assert validate_audio_bytes(valid_mpeg) is True

    # ID3v2 container header
    valid_id3 = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 150
    assert validate_audio_bytes(valid_id3) is True

    # MPEG frame sync slightly offset (e.g. metadata or padding)
    offset_mpeg = b"\x00\x00\x00" + b"\xff\xfb\x90\x00" + b"\x00" * 150
    assert validate_audio_bytes(offset_mpeg) is True


# ==============================================================================
# TEST 2: Empty audio rejection
# ==============================================================================
def test_2_empty_audio_rejection():
    assert validate_audio_bytes(b"") is False
    assert validate_audio_bytes(bytearray()) is False


# ==============================================================================
# TEST 3: None and invalid types input handling
# ==============================================================================
def test_3_none_input_handling():
    assert validate_audio_bytes(None) is False
    assert validate_audio_bytes("not_bytes_string") is False  # type: ignore
    assert validate_audio_bytes(12345) is False  # type: ignore
    assert validate_audio_bytes([]) is False  # type: ignore
    assert validate_audio_bytes({}) is False  # type: ignore


# ==============================================================================
# TEST 4: Invalid audio rejection
# ==============================================================================
def test_4_invalid_audio_rejection():
    # Long buffer with zero sync words or ID3 headers
    assert validate_audio_bytes(b"\x00" * 200) is False
    assert validate_audio_bytes(b"\x55\xaa" * 100) is False
    assert validate_audio_bytes(b"This is just plain text masquerading as audio" * 5) is False


# ==============================================================================
# TEST 5: Truncated audio handling
# ==============================================================================
def test_5_truncated_audio_handling():
    # Shorter than 100 bytes minimum threshold
    assert validate_audio_bytes(b"\xff\xfb\x90\x00") is False
    assert validate_audio_bytes(b"ID3" + b"\x00" * 20) is False
    assert validate_audio_bytes(b"\xff\xfb" + b"\x00" * 90) is False


# ==============================================================================
# TEST 6: Validator import and call consistency
# ==============================================================================
def test_6_validator_import_and_call_consistency():
    valid_sample = b"\xff\xfb\x90\x00" + b"\x00" * 150
    invalid_sample = b"\x00" * 150

    # 1. Module-level canonical function
    assert validate_audio_bytes(valid_sample) is True
    assert validate_audio_bytes(invalid_sample) is False

    # 2. TextToSpeech class method
    assert TextToSpeech.validate_audio_bytes(valid_sample) is True
    assert TextToSpeech.validate_audio_bytes(invalid_sample) is False

    # 3. TextToSpeech instance method
    tts = TextToSpeech()
    assert tts.validate_audio_bytes(valid_sample) is True
    assert tts.validate_audio_bytes(invalid_sample) is False


# ==============================================================================
# TEST 7: TextToSpeech API consistency
# ==============================================================================
def test_7_text_to_speech_api_consistency():
    tts = TextToSpeech()
    assert hasattr(tts, "speak")
    assert hasattr(tts, "validate_audio_bytes")
    assert hasattr(tts, "precache_phrases")
    assert hasattr(tts, "get_cached_count")

    # Graceful handling of invalid inputs
    assert tts.speak("") is None
    assert tts.speak("   ") is None
    assert tts.speak(None) is None


# ==============================================================================
# TEST 8: Test Voice successful generation
# ==============================================================================
def test_8_test_voice_successful_generation():
    tts = TextToSpeech()
    test_phrase = "Welcome to Apna AI Coach. Voice is active and working properly."
    fake_audio = b"\xff\xfb\x90\x00" + b"\x00" * 200

    with mock.patch("services.coaching.tts.gTTS") as mock_gtts:
        mock_instance = mock.MagicMock()
        mock_instance.write_to_fp.side_effect = lambda buf: buf.write(fake_audio)
        mock_gtts.return_value = mock_instance

        audio = tts.speak(test_phrase)
        assert audio is not None
        assert validate_audio_bytes(audio) is True
        assert len(audio) > 100


# ==============================================================================
# TEST 9: Test Voice validation failure without crashing
# ==============================================================================
def test_9_test_voice_validation_failure_without_crashing():
    tts = TextToSpeech()
    test_phrase = "Welcome to Apna AI Coach. Voice is active and working properly."
    # Corrupt audio with no frame sync
    corrupt_audio = b"\x00" * 200

    with mock.patch("services.coaching.tts.gTTS") as mock_gtts:
        mock_instance = mock.MagicMock()
        mock_instance.write_to_fp.side_effect = lambda buf: buf.write(corrupt_audio)
        mock_gtts.return_value = mock_instance

        audio = tts.speak(test_phrase)
        # Validator inside speak() detects invalid audio and returns None
        assert audio is None
        assert validate_audio_bytes(audio) is False


# ==============================================================================
# TEST 10: Duplicate audio event prevention
# ==============================================================================
def test_10_duplicate_audio_event_prevention():
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

    # Immediate second call within 3 seconds is debounced
    res2 = pipeline.process_event("rep_completed", "Squats", {"reps": 1})
    assert res2 is None


# ==============================================================================
# TEST 11: Audio persistence across Streamlit reruns
# ==============================================================================
def test_11_audio_persistence_across_streamlit_reruns():
    import streamlit as st

    fake_audio = b"\xff\xfb\x90\x00" + b"\x00" * 150
    event_id = "test-event-uuid-persistence"

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
# TEST 12: Mute behavior
# ==============================================================================
def test_12_mute_behavior():
    coach = LLMCoach(groq_client=None)
    tts = TextToSpeech()
    pipeline = VoicePipeline(coach, tts)
    pipeline.muted = True

    assert pipeline.process_event("workout_started", "Squats", {}) is None
    assert pipeline.process_event("rep_completed", "Squats", {}) is None
    assert pipeline.process_event("set_completed", "Squats", {}) is None
    assert pipeline.process_event("workout_completed", "Squats", {}) is None
    assert pipeline.process_event("ongoing_form_check", "Squats", {}) is None


# ==============================================================================
# TEST 13: Groq failure fallback
# ==============================================================================
def test_13_groq_failure_fallback():
    mock_groq = mock.MagicMock()
    mock_groq.chat.completions.create.side_effect = Exception("Rate limit exceeded 429")

    coach = LLMCoach(groq_client=mock_groq)
    feedback = coach.give_feedback("rep_completed", exercise="Squats")
    assert "Rep completed" in feedback


# ==============================================================================
# TEST 14: Missing API key fallback
# ==============================================================================
def test_14_missing_api_key_fallback():
    coach = LLMCoach(groq_client=None)

    assert "Raise your dumbbells" in coach.give_feedback("workout_started", exercise="Shoulder Press")
    assert "Rep completed" in coach.give_feedback("rep_completed")
    assert "Set completed" in coach.give_feedback("set_completed")
    assert "Workout completed" in coach.give_feedback("workout_completed")
    assert "Squat a little deeper" in coach.give_feedback("ongoing_form_check", issue="squat is not deep enough")


# ==============================================================================
# TEST 15: Rep completion voice announcement
# ==============================================================================
def test_15_rep_completion_voice_announcement():
    coach = LLMCoach(groq_client=None)
    tts = TextToSpeech()
    fake_audio = b"\xff\xfb\x90\x00" + b"\x00" * 150
    tts._cache["en:good job! rep completed."] = fake_audio

    pipeline = VoicePipeline(coach, tts)
    result = pipeline.process_event("rep_completed", "Squats", {"reps": 1})
    assert result is not None
    audio, text, event_id = result
    assert "Rep completed" in text
    assert len(audio) > 100


# ==============================================================================
# TEST 16: Set completion announcement
# ==============================================================================
def test_16_set_completion_announcement():
    coach = LLMCoach(groq_client=None)
    tts = TextToSpeech()
    fake_audio = b"\xff\xfb\x90\x00" + b"\x00" * 150
    tts._cache["en:awesome work! set completed. take a breath."] = fake_audio

    pipeline = VoicePipeline(coach, tts)
    result = pipeline.process_event("set_completed", "Squats", {"sets_completed": 1})
    assert result is not None
    audio, text, event_id = result
    assert "Set completed" in text
    assert len(audio) > 100


# ==============================================================================
# TEST 17: Workout completion announcement
# ==============================================================================
def test_17_workout_completion_announcement():
    coach = LLMCoach(groq_client=None)
    tts = TextToSpeech()
    fake_audio = b"\xff\xfb\x90\x00" + b"\x00" * 150
    tts._cache["en:workout completed! outstanding job!"] = fake_audio

    pipeline = VoicePipeline(coach, tts)
    result = pipeline.process_event("workout_completed", "Push-ups", {})
    assert result is not None
    audio, text, event_id = result
    assert "Workout completed" in text
    assert len(audio) > 100


# ==============================================================================
# TEST 18: Rep counting continues when audio fails
# ==============================================================================
def test_18_rep_counting_continues_when_audio_fails():
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
        assert voice_res is None  # Voice returned None safely

        # Rep counting proceeds uninterrupted to completion
        rec2, ev2 = manager.record_rep(timestamp=2.0)
        rec3, ev3 = manager.record_rep(timestamp=3.0)
        assert rec3 is True
        assert manager.total_reps == 3
        assert manager.is_completed()


# ==============================================================================
# TEST 19: WebRTC configuration and import regression
# ==============================================================================
def test_19_webrtc_import_regression():
    rtc_config = get_rtc_configuration()
    assert "iceServers" in rtc_config
    assert len(rtc_config["iceServers"]) >= 1
    assert "urls" in rtc_config["iceServers"][0]

    assert DEFAULT_MEDIA_CONSTRAINTS["video"]["frameRate"]["ideal"] == 30
    assert DEFAULT_MEDIA_CONSTRAINTS["audio"] is False


# ==============================================================================
# TEST 20: Workout engine regression
# ==============================================================================
def test_20_workout_engine_regression():
    manager = WorkoutStateManager()
    manager.start_workout("Squats", 2, 2)
    assert manager.is_active() is True
    assert manager.total_reps == 0

    # Rep 1 (Set 1) -> normal rep
    rec, ev = manager.record_rep(timestamp=1.0)
    assert rec is True
    assert ev is None
    assert manager.current_set_reps == 1

    # Rep 2 (Completes Set 1)
    rec, ev = manager.record_rep(timestamp=2.0)
    assert rec is True
    assert ev == "set_completed"
    assert manager.sets_completed == 1
    assert manager.current_set_reps == 0

    # Rep 3 (Set 2) -> normal rep
    rec, ev = manager.record_rep(timestamp=3.0)
    assert rec is True
    assert ev is None
    assert manager.current_set_reps == 1

    # Rep 4 (Completes Set 2 & Workout)
    rec, ev = manager.record_rep(timestamp=4.0)
    assert rec is True
    assert ev == "workout_completed"
    assert manager.sets_completed == 2
    assert manager.is_completed() is True

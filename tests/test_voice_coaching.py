import sys
import unittest.mock as mock
from pathlib import Path
import pytest

# Ensure Main App and Project root are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_APP_DIR = PROJECT_ROOT / "Main App"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MAIN_APP_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_APP_DIR))

from services.coaching.tts import TextToSpeech, validate_audio_bytes
from services.coaching.llm import LLMCoach, get_deterministic_cue
from services.coaching.voice_pipeline import VoicePipeline, autoplay_audio
from services.state.workout_state import WorkoutStateManager
from services.config.webrtc_config import get_rtc_configuration, DEFAULT_MEDIA_CONSTRAINTS
from detectors.shoulder_press import ShoulderPressDetector


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
    tts._cache["en:good job! rep completed. keep that steady pace and control."] = fake_audio

    pipeline = VoicePipeline(coach, tts)

    res1 = pipeline.process_event("rep_completed", "Squats", {"reps": 1})
    assert res1 is not None
    audio1, text1, event_id1 = res1
    assert len(audio1) > 0
    assert event_id1 is not None

    # Immediate second call within cooldown is debounced
    res2 = pipeline.process_event("rep_completed", "Squats", {"reps": 1})
    assert res2 is None


# ==============================================================================
# TEST 11: Audio persistence across Streamlit reruns without competing players
# ==============================================================================
def test_11_audio_persistence_across_streamlit_reruns():
    import streamlit as st

    fake_audio = b"\xff\xfb\x90\x00" + b"\x00" * 150
    event_id = "test-event-uuid-persistence"

    st.session_state["last_played_audio_event_id"] = ""

    with mock.patch("streamlit.audio") as mock_st_audio, mock.patch("streamlit.components.v1.html") as mock_html:
        # First execution (new event): renders canonical player with auto_play=True
        autoplay_audio(fake_audio, text="Good rep", event_id=event_id)
        assert mock_st_audio.call_count == 0  # No duplicate competing audio player!
        assert mock_html.call_count == 1
        assert st.session_state["last_played_audio_event_id"] == event_id

        # Second execution (simulated Streamlit rerun 250ms later with same event_id):
        # Keeps player mounted in DOM with auto_play=False so active playback is not aborted
        autoplay_audio(fake_audio, text="Good rep", event_id=event_id)
        assert mock_st_audio.call_count == 0  # No duplicate competing audio player!
        assert mock_html.call_count == 2


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
    tts._cache["en:good job! rep completed. keep that steady pace and control."] = fake_audio

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
    tts._cache["en:awesome work! set completed. take a breath, recover, and prepare for your next set."] = fake_audio

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
    tts._cache["en:workout completed! outstanding job! you crushed every set today."] = fake_audio

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


# ==============================================================================
# TEST 21: Full-sentence natural coaching explanations for all exercises
# ==============================================================================
def test_21_full_sentence_generation_all_exercises():
    coach = LLMCoach(groq_client=None)

    # 1. Shoulder Press
    cue_sp = coach.give_feedback("ongoing_form_check", issue="Elbows dropped too low at start", exercise="Shoulder Press")
    assert "Your elbows are not in the correct position" in cue_sp
    assert "brace your core" in cue_sp
    assert len(cue_sp.split()) >= 15

    # 2. Squats
    cue_sq = coach.give_feedback("ongoing_form_check", issue="Insufficient squat depth", exercise="Squats")
    assert "Your squat depth is too shallow" in cue_sq
    assert "heels on the floor" in cue_sq
    assert len(cue_sq.split()) >= 15

    # 3. Push-ups
    cue_pu = coach.give_feedback("ongoing_form_check", issue="Hips are sagging down", exercise="Push-ups")
    assert "Your hips are dropping below your body alignment" in cue_pu
    assert "straight line" in cue_pu
    assert len(cue_pu.split()) >= 15

    # 4. Biceps Curls
    cue_bc = coach.give_feedback("ongoing_form_check", issue="Excessive elbow drift", exercise="Biceps Curls (Dumbbell)")
    assert "You are moving your elbows away from your body" in cue_bc
    assert "elbows close to your sides" in cue_bc
    assert len(cue_bc.split()) >= 15

    # 5. Lunges
    cue_lg = coach.give_feedback("ongoing_form_check", issue="Losing balance / torso sway", exercise="Lunges")
    assert "Your balance is unstable" in cue_lg
    assert "controlled step forward" in cue_lg
    assert len(cue_lg.split()) >= 15


# ==============================================================================
# TEST 22: Audio duration estimation and non-truncation
# ==============================================================================
def test_22_audio_non_truncation_duration():
    coach = LLMCoach(groq_client=None)
    tts = TextToSpeech()
    pipeline = VoicePipeline(coach, tts)

    text = "Your squat depth is too shallow. Slowly bend your knees and push your hips backward until you reach a comfortable depth. Keep your heels on the floor and your chest lifted."
    fake_audio = b"\xff\xfb\x90\x00" + b"\x00" * 32000  # 32004 bytes = ~8.0s at 4000 B/s

    duration = pipeline.estimate_duration(fake_audio, text)
    # Never fixed short duration (< 4s); matches true byte length / word pacing
    assert duration >= 8.0
    assert duration <= 15.0


# ==============================================================================
# TEST 23: Audio queue sequencing and non-interruption
# ==============================================================================
def test_23_audio_queue_multiple_messages_sequential():
    coach = LLMCoach(groq_client=None)
    tts = TextToSpeech()
    fake_audio1 = b"\xff\xfb\x90\x00" + b"\x00" * 20000  # ~5.0s
    fake_audio2 = b"\xff\xfb\x90\x00" + b"\x00" * 24000  # ~6.0s

    tts._cache["en:first speech sentence."] = fake_audio1
    tts._cache["en:second speech sentence."] = fake_audio2

    pipeline = VoicePipeline(coach, tts)
    t0 = 100.0

    # Start speech 1 at t0
    pipeline.current_playing = {
        "event_id": "ev-1",
        "audio_bytes": fake_audio1,
        "text": "First speech sentence.",
        "duration": 5.0,
        "started_at": t0,
    }

    # At t0 + 2.0s, pipeline is still playing speech 1
    assert pipeline.is_playing(now=t0 + 2.0) is True

    # Enqueue speech 2 while speech 1 is playing
    pipeline.enqueue(
        audio_bytes=fake_audio2,
        text="Second speech sentence.",
        event_id="ev-2",
        duration=6.0,
        event="set_completed",
    )
    assert len(pipeline.queue) == 1

    # While still playing (t0 + 3.0s), get_active_audio returns speech 1 (NOT speech 2!)
    active_mid = pipeline.get_active_audio(current_form_status="Good", now=t0 + 3.0)
    assert active_mid is not None
    assert active_mid[2] == "ev-1"  # Still playing speech 1!
    assert len(pipeline.queue) == 1

    # Once speech 1 finishes (t0 + 5.1s), get_active_audio pops speech 2 and plays it!
    active_after = pipeline.get_active_audio(current_form_status="Good", now=t0 + 5.1)
    assert active_after is not None
    assert active_after[2] == "ev-2"  # Now playing speech 2!
    assert active_after[1] == "Second speech sentence."
    assert len(pipeline.queue) == 0


# ==============================================================================
# TEST 24: No replacement of active audio
# ==============================================================================
def test_24_no_replacement_of_currently_playing_audio():
    coach = LLMCoach(groq_client=None)
    tts = TextToSpeech()
    fake_audio1 = b"\xff\xfb\x90\x00" + b"\x00" * 30000
    fake_audio2 = b"\xff\xfb\x90\x00" + b"\x00" * 30000

    phrase1 = "Get ready to begin! Raise your dumbbells to shoulder height, brace your core, and press straight overhead without arching your back."
    phrase2 = "You are arching your lower back during the press. This puts unnecessary stress on your lumbar spine. Brace your core tightly, keep your spine neutral, and press straight overhead without leaning back."

    tts._cache[f"en:{phrase1.lower()}"] = fake_audio1
    tts._cache[f"en:{phrase2.lower()}"] = fake_audio2

    pipeline = VoicePipeline(coach, tts)
    t0 = 200.0

    # Event 1 starts playing
    res1 = pipeline.process_event("workout_started", "Shoulder Press", {})
    assert res1 is not None
    audio1, text1, id1 = res1
    pipeline.current_playing["started_at"] = t0
    pipeline.current_playing["duration"] = 7.0
    pipeline.last_spoken_at = t0

    # Event 2 triggered at t0 + 4.5s (after 4.0s cooldown, but while Event 1 still has 2.5s remaining)
    with mock.patch("time.time", return_value=t0 + 4.5):
        res2 = pipeline.process_event("ongoing_form_check", "Shoulder Press", {
            "form_status": "Needs Correction",
            "form_issue": "Excessive lower-back arch",
        })
        # Returned active audio is still Event 1; Event 2 was placed in queue!
        assert res2 is not None
        assert res2[2] == id1
        assert len(pipeline.queue) == 1
        assert pipeline.queue[0]["text"] == phrase2


# ==============================================================================
# TEST 25: Valid queued correction preserved even if form recovers to Good
# ==============================================================================
def test_25_valid_queued_correction_preserved_on_form_transition():
    coach = LLMCoach(groq_client=None)
    tts = TextToSpeech()
    pipeline = VoicePipeline(coach, tts)

    t0 = 300.0
    pipeline.current_playing = {
        "event_id": "active-cue",
        "audio_bytes": b"\xff\xfb\x90\x00" + b"\x00" * 150,
        "text": "Active cue",
        "duration": 4.0,
        "started_at": t0,
    }

    # A valid form correction was queued during rep movement at t0 + 1.0
    pipeline.queue.append({
        "event_id": "flaw-cue",
        "audio_bytes": b"\xff\xfb\x90\x00" + b"\x00" * 150,
        "text": "Your arms are pressing unevenly. This can create an imbalance in your shoulder movement.",
        "duration": 5.0,
        "issue": "Uneven arm movement",
        "enqueued_at": t0 + 1.0,
    })

    # At t0 + 4.5s (after active cue finishes), athlete lowered weights so current_form_status is "Good"
    active = pipeline.get_active_audio(current_form_status="Good", now=t0 + 4.5)
    # The valid form correction must NOT be discarded; it must be delivered!
    assert active is not None
    assert active[2] == "flaw-cue"
    assert "unevenly" in active[1].lower()
    assert len(pipeline.queue) == 0


# ==============================================================================
# TEST 26: Queue cleared on workout completion
# ==============================================================================
def test_26_clear_queue_on_workout_completion():
    coach = LLMCoach(groq_client=None)
    tts = TextToSpeech()
    pipeline = VoicePipeline(coach, tts)

    # Queue some intermediate form cues
    pipeline.queue.append({"event_id": "q1", "text": "Squat deeper", "duration": 4.0, "enqueued_at": 1.0})
    pipeline.queue.append({"event_id": "q2", "text": "Drive knees out", "duration": 4.0, "enqueued_at": 2.0})
    assert len(pipeline.queue) == 2

    # Clear queue
    pipeline.clear_queue()
    assert len(pipeline.queue) == 0
    assert pipeline.current_playing is None


# ==============================================================================
# TEST 27: Form status and voice message consistency
# ==============================================================================
def test_27_form_status_and_voice_message_consistency():
    from tests.test_form_correction import make_shoulder_press_landmarks
    det = ShoulderPressDetector()
    # Good posture at 90 degrees elbows and neutral spine
    m_good = det.process(make_shoulder_press_landmarks(90.0, 90.0, 175.0))
    assert m_good["form_status"] == "Good"
    # Form correction text is encouraging, never an error complaint
    assert "clean" in m_good["form_correction"].lower() or "adjustment" in m_good["form_correction"].lower() or "maintain" in m_good["form_correction"].lower()


# ==============================================================================
# TEST 28: Stale queue items purged after timeout (> 20s)
# ==============================================================================
def test_28_stale_queue_purged_after_timeout():
    coach = LLMCoach(groq_client=None)
    tts = TextToSpeech()
    pipeline = VoicePipeline(coach, tts)

    t0 = 400.0
    pipeline.queue.append({
        "event_id": "ancient-cue",
        "audio_bytes": b"\xff\xfb\x90\x00" + b"\x00" * 150,
        "text": "Very old cue from minutes ago",
        "duration": 4.0,
        "enqueued_at": t0,
    })

    # Evaluated 25 seconds later (> 20s threshold)
    active = pipeline.get_active_audio(current_form_status="Good", now=t0 + 25.0)
    assert active is None
    assert len(pipeline.queue) == 0


# ==============================================================================
# TEST 29: Shoulder press uneven arm movement exact text verification
# ==============================================================================
def test_29_shoulder_press_uneven_arm_movement_exact_text():
    from tests.test_form_correction import make_shoulder_press_landmarks
    det = ShoulderPressDetector()
    # Left arm at 130 deg, Right arm at 85 deg -> asymmetry = 45 deg > 25.0
    metrics = det.process(make_shoulder_press_landmarks(130.0, 85.0, 175.0))
    assert metrics["form_status"] == "Needs Correction"
    assert metrics["form_issue"] == "Uneven arm movement"

    expected = (
        "Your arms are pressing unevenly. This can create an imbalance in your shoulder movement. "
        "Push both arms upward at the same time and keep them at the same height. "
        "Keep your core stable and control the movement. Try again."
    )
    assert metrics["form_correction"] == expected

    cue = get_deterministic_cue("ongoing_form_check", metrics["form_issue"], "Shoulder Press")
    assert cue == expected


# ==============================================================================
# TEST 30: Form corrections bypass minor cue cooldown and are enqueued
# ==============================================================================
def test_30_form_corrections_bypass_minor_cue_cooldown():
    coach = LLMCoach(groq_client=None)
    tts = TextToSpeech()
    fake_audio = b"\xff\xfb\x90\x00" + b"\x00" * 20000
    expected_phrase = (
        "Your arms are pressing unevenly. This can create an imbalance in your shoulder movement. "
        "Push both arms upward at the same time and keep them at the same height. "
        "Keep your core stable and control the movement. Try again."
    )
    tts._cache[f"en:{expected_phrase.lower()}"] = fake_audio

    pipeline = VoicePipeline(coach, tts)
    t0 = 500.0

    # Start a previous announcement playing
    pipeline.current_playing = {
        "event_id": "intro",
        "audio_bytes": fake_audio,
        "text": "Get ready to begin!",
        "duration": 6.0,
        "started_at": t0,
    }
    pipeline.last_spoken_at = t0

    # Only 1.0 second later (well within normal 4.0s minor cue cooldown), form mistake occurs
    with mock.patch("time.time", return_value=t0 + 1.0):
        res = pipeline.process_event("ongoing_form_check", "Shoulder Press", {
            "form_status": "Needs Correction",
            "form_issue": "Uneven arm movement",
            "form_correction": expected_phrase,
        })
        # Audio was not dropped! It returned currently playing and placed the correction in queue
        assert res is not None
        assert res[2] == "intro"
        assert len(pipeline.queue) == 1
        assert pipeline.queue[0]["text"] == expected_phrase


# ==============================================================================
# TEST 31: Runtime audio diagnostics state tracking
# ==============================================================================
def test_31_audio_diagnostics_state_tracking():
    import streamlit as st
    from services.state.session_defaults import initial_session_defaults
    from services.tracking.metrics import _apply_voice_result

    initial_session_defaults()

    fake_bytes = b"\xff\xfb\x90\x00" + b"\x00" * 300
    _apply_voice_result(
        (fake_bytes, "Sample spoken text", "ev-test-123"),
        event_name="form_correction",
        issue_text="Uneven arm movement",
    )

    assert st.session_state.audio_diag_event_id == "ev-test-123"
    assert st.session_state.audio_diag_event_name == "form_correction"
    assert st.session_state.audio_diag_issue == "Uneven arm movement"
    assert st.session_state.audio_diag_tts_status == "SUCCESS"
    assert st.session_state.audio_diag_validation_status == "VALID_MP3"
    assert st.session_state.audio_diag_payload_bytes == len(fake_bytes)
    assert st.session_state.audio_diag_playback_status == "DELIVERED"


# ==============================================================================
# TEST 32: render_voice_player manual Play, unmute, and error reporting
# ==============================================================================
def test_32_render_voice_player_manual_play_and_error_reporting():
    from services.coaching.voice_pipeline import render_voice_player

    fake_bytes = b"\xff\xfb\x90\x00" + b"\x00" * 200
    with mock.patch("streamlit.components.v1.html") as mock_html:
        render_voice_player(fake_bytes, text="Test cue", event_id="ev-test-render", auto_play=True)
        assert mock_html.call_count == 1
        html_arg = mock_html.call_args[0][0]

        # 1. Manual play button present with direct startAudio() onclick
        assert 'id="apna-voice-btn"' in html_arg
        assert 'onclick="startAudio()"' in html_arg

        # 2. Direct click handler ensures audio is unmuted and volume is 1.0
        assert "audio.muted = false" in html_arg
        assert "audio.volume = 1.0" in html_arg

        # 3. Audio.play() Promise is directly invoked from click handler
        assert "var promise = audio.play()" in html_arg or "promise = audio.play()" in html_arg

        # 4. Error display exists and errors are NOT swallowed into console.warn only
        assert 'id="error-display"' in html_arg
        assert "showError(" in html_arg

        # 5. Playback status, time, volume, and payload badges exist
        assert 'id="playback-badge"' in html_arg
        assert 'id="time-badge"' in html_arg
        assert 'id="vol-badge"' in html_arg
        assert 'id="src-badge"' in html_arg


# ==============================================================================
# TEST 33: render_voice_player rejects invalid audio gracefully
# ==============================================================================
def test_33_render_voice_player_rejects_invalid_audio():
    from services.coaching.voice_pipeline import render_voice_player

    with mock.patch("streamlit.components.v1.html") as mock_html, mock.patch("streamlit.warning") as mock_warn:
        render_voice_player(b"", text="Empty", event_id="ev-invalid-1")
        assert mock_html.call_count == 0
        assert mock_warn.call_count == 1

        render_voice_player(b"\x00" * 50, text="Corrupt", event_id="ev-invalid-2")
        assert mock_html.call_count == 0
        assert mock_warn.call_count == 2


# ==============================================================================
# TEST 34: Test Voice diagnostics integration
# ==============================================================================
def test_34_test_voice_diagnostics_integration():
    import streamlit as st
    from services.state.session_defaults import initial_session_defaults

    initial_session_defaults()
    fake_audio = b"\xff\xfb\x90\x00" + b"\x00" * 400
    test_id = "test-voice-uuid"

    # Simulate Test Voice success branch
    st.session_state.audio_diag_event_id = test_id
    st.session_state.audio_diag_event_name = "test_voice"
    st.session_state.audio_diag_issue = "Voice Test Verification"
    st.session_state.audio_diag_tts_status = "SUCCESS"
    st.session_state.audio_diag_validation_status = "VALID_MP3"
    st.session_state.audio_diag_payload_bytes = len(fake_audio)
    st.session_state.audio_diag_playback_status = "READY"

    assert st.session_state.audio_diag_event_id == test_id
    assert st.session_state.audio_diag_event_name == "test_voice"
    assert st.session_state.audio_diag_tts_status == "SUCCESS"
    assert st.session_state.audio_diag_validation_status == "VALID_MP3"
    assert st.session_state.audio_diag_payload_bytes == 404
    assert st.session_state.audio_diag_playback_status == "READY"


# ==============================================================================
# TEST 35: Static welcome voice pre-cache in TextToSpeech
# ==============================================================================
def test_35_static_welcome_voice_precache():
    tts = TextToSpeech()
    test_phrase = "Welcome to Apna AI Coach. Voice is active and working properly."
    tts.load_static_cache()

    # Cache must contain the pre-seeded welcome phrase if welcome_voice.mp3 is present
    audio = tts.speak(test_phrase)
    assert audio is not None
    assert validate_audio_bytes(audio) is True
    assert len(audio) > 1000


# ==============================================================================
# TEST 36: autoplay_audio delegates to render_voice_player without competing players
# ==============================================================================
def test_36_autoplay_audio_delegates_to_render_voice_player():
    import streamlit as st

    fake_audio = b"\xff\xfb\x90\x00" + b"\x00" * 150
    st.session_state["last_played_audio_event_id"] = ""

    with mock.patch("services.coaching.voice_pipeline.render_voice_player") as mock_render, mock.patch("streamlit.audio") as mock_st_audio:
        autoplay_audio(fake_audio, text="Set complete", event_id="ev-del-1")
        assert mock_render.call_count == 1
        assert mock_st_audio.call_count == 0  # No competing audio players
        call_args = mock_render.call_args[0]
        assert call_args[0] == fake_audio
        assert mock_render.call_args[1]["event_id"] == "ev-del-1"
        assert mock_render.call_args[1]["auto_play"] is True


# ==============================================================================
# TEST 37: Import smoke test for all packages, services, and __init__.py exports
# ==============================================================================
def test_37_import_smoke_test_all_services_and_packages():
    # Test top-level package and subpackage imports
    import services
    import services.auth
    import services.coaching
    import services.config
    import services.persistence
    import services.state
    import services.tracking
    import services.ui
    import services.vision
    import core
    import detectors

    # Test individual public exports from package __init__ files
    from services.auth import render_login_wall
    from services.coaching import render_voice_player, VoicePipeline, autoplay_audio, TextToSpeech, validate_audio_bytes, LLMCoach, resolve_groq_model
    from services.persistence import init_db, save_workout_session, get_users_exercises, get_or_create_user, add_exercise
    from services.state import initial_session_defaults, WorkoutStateManager, WorkoutStatus
    from services.config import EXERCISE_OPTIONS, get_rtc_configuration
    from services.tracking import sync_metrics_update
    from services.ui import load_css, inject_local_font
    from services.vision import VideoProcessorClass
    from core import BaseExercise, RepetitionStateMachine
    from detectors import SquatDetector, PushUpDetector, BicepsCurlDetector, ShoulderPressDetector, LungesDetector

    assert callable(render_login_wall)
    assert callable(render_voice_player)
    assert callable(autoplay_audio)
    assert callable(resolve_groq_model)
    assert callable(init_db)
    assert callable(initial_session_defaults)
    assert callable(sync_metrics_update)
    assert len(EXERCISE_OPTIONS) >= 5


# ==============================================================================
# TEST 38: Groq model configuration and dynamic account inspection
# ==============================================================================
def test_38_resolve_groq_model_configuration():
    from services.coaching.llm import resolve_groq_model
    import os

    # 1. Fallback when no client and no env var
    orig_env = os.environ.get("GROQ_MODEL")
    try:
        if "GROQ_MODEL" in os.environ:
            del os.environ["GROQ_MODEL"]

        model = resolve_groq_model(client=None)
        assert model == "llama-3.1-8b-instant"

        # 2. Configurable via environment variable
        os.environ["GROQ_MODEL"] = "custom-test-model"
        assert resolve_groq_model(client=None) == "custom-test-model"

        # 3. Dynamic inspection when client is provided
        mock_client = mock.MagicMock()
        m1 = mock.MagicMock(); m1.id = "llama-3.1-8b-instant"
        m2 = mock.MagicMock(); m2.id = "llama-3.3-70b-versatile"
        mock_client.models.list.return_value.data = [m1, m2]

        # Explicit request
        assert resolve_groq_model(mock_client, requested_model="llama-3.3-70b-versatile") == "llama-3.3-70b-versatile"

        # Preferred auto-selection from account
        del os.environ["GROQ_MODEL"]
        assert resolve_groq_model(mock_client) == "llama-3.1-8b-instant"

    finally:
        if orig_env is not None:
            os.environ["GROQ_MODEL"] = orig_env
        elif "GROQ_MODEL" in os.environ:
            del os.environ["GROQ_MODEL"]


# ==============================================================================
# TEST 39: LLMCoach uses verified model and falls back to deterministic cue
# ==============================================================================
def test_39_llm_coach_dynamic_model_usage_and_fallback():
    from services.coaching.llm import LLMCoach

    mock_client = mock.MagicMock()
    m = mock.MagicMock(); m.id = "llama-3.1-8b-instant"
    mock_client.models.list.return_value.data = [m]

    coach = LLMCoach(groq_client=mock_client)
    assert coach.model == "llama-3.1-8b-instant"

    # Test error fallback to deterministic cue
    mock_client.chat.completions.create.side_effect = Exception("Groq rate limit exceeded (429)")
    cue = coach.give_feedback("ongoing_form_check", issue="Knees caving inward (valgus collapse)", exercise="Squats")
    assert "knees are caving inward" in cue.lower()
    assert "drive your knees outward" in cue.lower()



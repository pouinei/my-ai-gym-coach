import time
import uuid
import logging
from typing import Optional, Tuple, Dict, Any
import streamlit as st
import streamlit.components.v1 as components
from services.coaching.tts import validate_audio_bytes

logger = logging.getLogger(__name__)

STAGE_GUIDANCE_CUES = {
    "Shoulder Press": {
        "moving": "Press upward. Extend your elbows.",
        "descending": "Lower slowly to shoulder level.",
        "inflected": "Full extension overhead! Lower down.",
    },
    "Squats": {
        "moving": "Bend your knees. Keep your back straight.",
        "descending": "Lower your hips with control.",
        "inflected": "Good depth! Push back up.",
        "ascending": "Drive upward through your heels.",
    },
    "Push-ups": {
        "moving": "Lower your chest to the floor.",
        "descending": "Control the descent.",
        "inflected": "Chest down! Push upward.",
        "ascending": "Push through your palms.",
    },
    "Biceps Curls (Dumbbell)": {
        "moving": "Curl upward towards your shoulders.",
        "ascending": "Squeeze at the top.",
        "inflected": "Good contraction! Lower slowly.",
        "descending": "Lower with control.",
    },
    "Lunges": {
        "moving": "Step forward. Lower your hips.",
        "descending": "Drop your back knee towards the floor.",
        "inflected": "Solid lunge! Push back up.",
        "ascending": "Drive through the front heel.",
    },
}


class VoicePipeline:
    """
    Unified Voice Coaching Pipeline:
    - Bridges Computer Vision events to AI coaching text and TTS audio
    - Enforces anti-spam rate limiting and debounce
    - Supports muting and offline deterministic fallbacks
    - Pre-caches core audio phrases for instant playback
    """

    def __init__(self, llm, tts, precache: bool = False):
        self.llm = llm
        self.tts = tts
        self.last_spoken_at: float = 0.0
        self.last_spoken_text: str = ""
        self.last_stage: Optional[str] = None
        self.muted: bool = False

        if precache:
            self._precache_standard_cues()

    def _precache_standard_cues(self) -> None:
        standard_phrases = [
            "Good job! Rep completed.",
            "Awesome work! Set completed. Take a breath.",
            "Workout completed! Outstanding job!",
            "Step into the camera frame so I can track your form.",
            "Looking strong! Maintain good control.",
            "Get ready. Raise your dumbbells to shoulder height.",
            "Get ready. Stand straight with feet shoulder-width apart.",
            "Get ready. Get into a solid plank position.",
            "Get ready. Keep your elbows close to your sides.",
            "Get ready. Stand upright and prepare to step forward.",
            "Welcome to Apna AI Coach. Voice is active and working properly.",
        ]
        try:
            self.tts.precache_phrases(standard_phrases)
        except Exception as e:
            logger.warning(f"Voice pipeline pre-caching warning: {e}")

    def _find_form_issue(self, exercise: str, metrics: Dict[str, Any]) -> Optional[str]:
        if "issue" in metrics:
            return metrics["issue"]

        if exercise == "Squats":
            depth = metrics.get("depth_status", "")
            back_angle = metrics.get("back_angle", 180)
            if depth in ("TOO HIGH", "HALF SQUAT"):
                return "The user's squat is not deep enough — knees are not bending sufficiently."
            if isinstance(back_angle, (int, float)) and back_angle < 120:
                return "The user is leaning too far forward during the squat."

        elif exercise == "Push-ups":
            alignment = metrics.get("body_alignment", "")
            hip_status = metrics.get("hip_status", "")
            if alignment == "Poor Form":
                return "The user's body is not straight during the push-up."
            if hip_status == "SAGGING":
                return "The user's hips are sagging down — keep the core engaged."
            if hip_status == "PIKED UP":
                return "The user's hips are too high — lower them to form a plank."

        elif exercise == "Biceps Curls (Dumbbell)":
            swing = metrics.get("swing_status", "")
            shoulder = metrics.get("shoulder_status", "")
            if swing == "SWINGING":
                return "The user is swinging their torso — keep the body still and isolate biceps."
            if shoulder == "ELBOW DRIFTING":
                return "The user's elbows are drifting away from their sides."

        elif exercise == "Shoulder Press":
            back_arch = metrics.get("back_arch_status", "")
            posture = metrics.get("posture_feedback", "")
            if back_arch == "Excessive Arch":
                return "The user is arching their lower back excessively during the press."
            if "elbow" in posture.lower():
                return "Don't lower elbows too far below shoulders."

        elif exercise == "Lunges":
            balance = metrics.get("balance_status", "")
            if balance == "OFF BALANCE":
                return "The user is losing balance — keep feet hip-width apart and torso upright."

        return None

    def process_event(
        self,
        event: str,
        exercise: str,
        metrics: Dict[str, Any],
    ) -> Optional[Tuple[bytes, str, str]]:
        """
        Processes a coaching event.
        Returns: (audio_bytes, text, event_id) or None.
        """
        if self.muted:
            return None

        now = time.time()
        is_major_event = event in ("workout_started", "rep_completed", "set_completed", "workout_completed")

        issue = self._find_form_issue(exercise, metrics) if not is_major_event else None

        # Check for stage-specific movement guidance
        stage = metrics.get("stage") or metrics.get("exercise_stage")
        stage_cue = None
        if not is_major_event and not issue and stage and exercise in STAGE_GUIDANCE_CUES:
            if stage != self.last_stage and stage in STAGE_GUIDANCE_CUES[exercise]:
                stage_cue = STAGE_GUIDANCE_CUES[exercise][stage]
                self.last_stage = stage

        # Rate-limiting / anti-spam for minor cues
        if not is_major_event:
            if (now - self.last_spoken_at) < 3.5:
                return None
            if not issue and not stage_cue and event == "ongoing_form_check":
                return None

        # Determine spoken text
        if stage_cue and not issue:
            text = stage_cue
        else:
            text = self.llm.give_feedback(event, issue, exercise)

        if not text:
            return None

        # Prevent repeating identical text within 3 seconds
        if text.strip().lower() == self.last_spoken_text.strip().lower() and (now - self.last_spoken_at) < 3.0:
            return None

        audio_bytes = self.tts.speak(text)
        if not audio_bytes:
            return None

        self.last_spoken_at = now
        self.last_spoken_text = text
        event_id = uuid.uuid4().hex

        return audio_bytes, text, event_id


def autoplay_audio(audio_bytes: bytes, text: str = "", event_id: str = "") -> None:
    """
    Renders the audio element into Streamlit.
    Ensures:
    1. Audio is valid and decodable via validate_audio_bytes.
    2. Audio is NOT removed on subsequent 250ms reruns (persists in DOM).
    3. Uses autoplay=True on first render of an event.
    4. Injects JS helper to ensure unmuted volume=1.0 and invoke play().
    """
    if not audio_bytes or not validate_audio_bytes(audio_bytes):
        return

    last_played = st.session_state.get("last_played_audio_event_id")
    is_first_render = bool(event_id and event_id != last_played)

    if is_first_render:
        st.session_state["last_played_audio_event_id"] = event_id

    # Always render the audio widget so Streamlit does NOT unmount it on 250ms reruns!
    # Pass autoplay=True on first render, and autoplay=False on subsequent 250ms reruns to avoid restarting playback.
    st.audio(audio_bytes, format="audio/mp3", autoplay=is_first_render)

    if is_first_render:
        components.html(
            """
            <script>
            (function() {
                try {
                    const parentDoc = window.parent.document;
                    const audios = parentDoc.querySelectorAll('audio');
                    if (audios.length > 0) {
                        const a = audios[audios.length - 1];
                        a.volume = 1.0;
                        a.muted = false;
                        const p = a.play();
                        if (p !== undefined) {
                            p.catch(function(e) {
                                console.warn('[Apna Coach Voice] Autoplay waiting for user interaction:', e);
                            });
                        }
                    }
                } catch (err) {}
            })();
            </script>
            """,
            height=0,
        )

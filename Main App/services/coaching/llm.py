import logging
from typing import Optional
from services.config.workout_config import PROMPT

logger = logging.getLogger(__name__)

# Standard deterministic coaching cues for offline/fallback mode
DEFAULT_COACHING_CUES = {
    "workout_started": "Get ready to begin! Let's crush this workout.",
    "rep_completed": "Good job! Rep completed.",
    "set_completed": "Awesome work! Set completed. Take a breath.",
    "workout_completed": "Workout completed! Outstanding job!",
    "no_pose_detected": "Step into the camera frame so I can track your form.",
    "ongoing_form_check": "Looking strong! Maintain good control.",
}

EXERCISE_START_CUES = {
    "Shoulder Press": "Get ready. Raise your dumbbells to shoulder height.",
    "Squats": "Get ready. Stand straight with feet shoulder-width apart.",
    "Push-ups": "Get ready. Get into a solid plank position.",
    "Biceps Curls (Dumbbell)": "Get ready. Keep your elbows close to your sides.",
    "Lunges": "Get ready. Stand upright and prepare to step forward.",
}


def get_deterministic_cue(event: str, issue: Optional[str] = None, exercise: Optional[str] = None) -> str:
    """Provides high-energy, concise coaching cues without external API dependency."""
    if event == "workout_started" and exercise in EXERCISE_START_CUES:
        return EXERCISE_START_CUES[exercise]

    if issue:
        cleaned_issue = issue.lower()
        if "deep" in cleaned_issue:
            return "Squat a little deeper for full range of motion."
        elif "arch" in cleaned_issue or "leaning" in cleaned_issue:
            return "Brace your core and keep your back straight."
        elif "sag" in cleaned_issue:
            return "Keep your core tight; do not let your hips sag."
        elif "pike" in cleaned_issue or "high" in cleaned_issue:
            return "Lower your hips into a straight plank line."
        elif "swing" in cleaned_issue:
            return "Keep your torso still; avoid using momentum."
        elif "drift" in cleaned_issue or "elbow" in cleaned_issue:
            return "Keep your elbows pinned to your sides."
        elif "balance" in cleaned_issue:
            return "Maintain balance and keep your torso upright."
        elif "frame" in cleaned_issue or "camera" in cleaned_issue:
            return "Please step into the camera frame."
        return issue

    return DEFAULT_COACHING_CUES.get(event, "Keep going! Good form.")


class LLMCoach:
    """
    Hybrid AI Coach:
    - Uses Groq LLaMA-3.3-70B for dynamic, high-energy coaching when available
    - Automatically falls back to deterministic local cues on rate limits, errors, or offline mode
    - Never throws an uncaught exception
    """

    def __init__(self, groq_client=None):
        self.client = groq_client
        self.history = []
        self.system_prompt = PROMPT

    def give_feedback(self, event: str, issue: Optional[str] = None, exercise: Optional[str] = None) -> str:
        fallback = get_deterministic_cue(event, issue, exercise)

        if not self.client:
            return fallback

        prompt = f"Event: {event}"
        if exercise:
            prompt += f" Exercise: {exercise}"
        if issue:
            prompt += f" Form Issue: {issue}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.history[-6:],
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.4,
                timeout=3.0,
            )
            text = response.choices[0].message.content.strip()
            # Clean up quotation marks or markdown
            text = text.replace('"', '').replace('*', '').strip()
            self.history.append({"role": "assistant", "content": text})
            return text if text else fallback
        except Exception as e:
            logger.warning(f"Groq API call failed ({e}); using local deterministic coaching cue.")
            return fallback

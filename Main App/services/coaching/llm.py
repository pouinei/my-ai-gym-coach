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

    if event == "rep_invalid":
        if issue:
            return f"Rep not counted: {issue}"
        return "Rep not counted. Please check your form."

    if issue:
        cleaned_issue = issue.lower()
        if "arch" in cleaned_issue:
            return "Don't arch your lower back. Keep your core tight."
        elif "uneven" in cleaned_issue or "asymmetry" in cleaned_issue:
            return "Move both arms evenly."
        elif "fast" in cleaned_issue or "speed" in cleaned_issue:
            return "Slow down your movement."
        elif "valgus" in cleaned_issue or "align" in cleaned_issue or ("knee" in cleaned_issue and "toe" in cleaned_issue):
            return "Keep your knees aligned with your toes."
        elif "extend" in cleaned_issue or "extension" in cleaned_issue:
            if "elbow" in cleaned_issue or "arm" in cleaned_issue:
                return "Extend your arms fully overhead."
            elif "stand" in cleaned_issue or "standing" in cleaned_issue:
                return "Stand all the way up."
            return "Push all the way up to full extension."
        elif "deep" in cleaned_issue or "depth" in cleaned_issue:
            return "Squat a little deeper for full range of motion."
        elif "sag" in cleaned_issue:
            return "Keep your core tight; do not let your hips sag."
        elif "pike" in cleaned_issue or "high" in cleaned_issue:
            return "Lower your hips into a straight plank line."
        elif "swing" in cleaned_issue or "momentum" in cleaned_issue:
            return "Avoid swinging your body; isolate your muscles."
        elif "drift" in cleaned_issue or "elbow" in cleaned_issue:
            return "Keep your elbows pinned to your sides."
        elif "lower" in cleaned_issue or "chest" in cleaned_issue:
            return "Lower yourself a little more."
        elif "curl" in cleaned_issue:
            return "Curl the weight all the way to the top."
        elif "balance" in cleaned_issue or "sway" in cleaned_issue:
            return "Maintain balance and keep your torso upright."
        elif "frame" in cleaned_issue or "camera" in cleaned_issue:
            return "Please step into the camera frame."
        return issue

    return DEFAULT_COACHING_CUES.get(event, "Good form! Keep going.")


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

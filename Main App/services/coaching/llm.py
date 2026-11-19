import logging
from typing import Optional
from services.config.workout_config import PROMPT

logger = logging.getLogger(__name__)

# Standard natural, full-sentence coaching cues for offline/fallback mode
DEFAULT_COACHING_CUES = {
    "workout_started": "Get ready to begin! Let's crush this workout with full focus and great form.",
    "rep_completed": "Good job! Rep completed. Keep that steady pace and control.",
    "set_completed": "Awesome work! Set completed. Take a breath, recover, and prepare for your next set.",
    "workout_completed": "Workout completed! Outstanding job! You crushed every set today.",
    "no_pose_detected": "Key body landmarks are outside the camera frame. Step back into the camera frame so I can track your full body.",
    "ongoing_form_check": "Looking strong! Maintain good control and keep your breathing steady.",
    "form_recovered": "Great adjustment! Your form is clean, balanced, and under control. Keep up this exact technique.",
}

EXERCISE_START_CUES = {
    "Shoulder Press": "Get ready to begin! Raise your dumbbells to shoulder height, brace your core, and press straight overhead without arching your back.",
    "Squats": "Get ready to begin! Stand straight with feet shoulder-width apart, keep your chest lifted, and prepare to squat with control.",
    "Push-ups": "Get ready to begin! Get into a solid plank position with your hands shoulder-width apart and your core tightly engaged.",
    "Biceps Curls (Dumbbell)": "Get ready to begin! Keep your elbows close to your sides, stand upright, and prepare to curl the weights with strict form.",
    "Lunges": "Get ready to begin! Stand upright and prepare to step forward, keeping your balance centered and your torso tall.",
}


def get_deterministic_cue(event: str, issue: Optional[str] = None, exercise: Optional[str] = None) -> str:
    """Provides complete, natural, trainer-style coaching explanations without external API dependency."""
    if event == "workout_started" and exercise in EXERCISE_START_CUES:
        return EXERCISE_START_CUES[exercise]

    if event == "rep_invalid":
        if issue:
            cleaned = issue.strip()
            # If issue is already an explanation, wrap into full rep invalid context
            if len(cleaned.split()) > 6:
                return f"That repetition did not count. {cleaned}"
            return f"That repetition did not count because of {cleaned.lower()}. Reset your posture and perform the next rep with clean form."
        return "That repetition did not count due to incomplete range of motion. Reset your posture and perform the next rep with clean form."

    if issue:
        cleaned_issue = issue.lower()

        # If issue already contains a complete explanation, return it directly
        if len(issue.split()) >= 10 and (issue.endswith(".") or issue.endswith("!")):
            return issue

        # Exercise Form Corrections (Full Trainer Explanations)
        if "arch" in cleaned_issue:
            return "You are arching your lower back during the press. This puts unnecessary stress on your lumbar spine. Brace your core tightly, keep your spine neutral, and press straight overhead without leaning back."
        elif "uneven" in cleaned_issue or "asymmetry" in cleaned_issue:
            return "Your arms are pressing unevenly. Balance your strength by pushing both dumbbells upward at the exact same time and height."
        elif "fast" in cleaned_issue or "speed" in cleaned_issue:
            return "You are pressing too quickly. Control the movement with a steady two-second press up and a controlled lowering down."
        elif "valgus" in cleaned_issue or "caved" in cleaned_issue or ("knee" in cleaned_issue and "inward" in cleaned_issue):
            return "Your knees are caving inward during the squat. This places strain on your joints. Actively drive your knees outward in line with your toes as you descend and rise."
        elif "lean" in cleaned_issue:
            return "Your upper body is leaning too far forward. Keep your chest lifted, engage your core, and look straight ahead throughout the squat."
        elif "deep" in cleaned_issue or "depth" in cleaned_issue or "shallow" in cleaned_issue:
            return "Your squat depth is too shallow. Squat a little deeper for full range of motion. Slowly bend your knees and push your hips backward until you reach a comfortable depth. Keep your heels on the floor and your chest lifted."
        elif "sag" in cleaned_issue:
            return "Your hips are dropping below your body alignment. Engage your core and maintain a straight line from your shoulders to your heels. Lower your chest slowly and push back up."
        elif "pike" in cleaned_issue or ("hip" in cleaned_issue and "high" in cleaned_issue):
            return "Your hips are lifted too high in the air. Lower your hips so your body forms a straight, solid plank line from your head to your heels."
        elif "swing" in cleaned_issue or "momentum" in cleaned_issue:
            return "You are swinging your torso to lift the weights. Stand tall, brace your core, and isolate your biceps by moving only your forearms."
        elif "drift" in cleaned_issue:
            return "You are moving your elbows away from your body. Keep your elbows close to your sides, avoid swinging your torso, and curl the weight slowly."
        elif "too low" in cleaned_issue or ("elbow" in cleaned_issue and ("start" in cleaned_issue or "drop" in cleaned_issue)):
            return "Your elbows are not in the correct position. Keep both elbows aligned with your shoulders, brace your core, and press the weights straight overhead. Avoid arching your lower back."
        elif "balance" in cleaned_issue or "sway" in cleaned_issue or "unstable" in cleaned_issue:
            return "Your balance is unstable. Take a controlled step forward, keep your upper body upright, and lower yourself slowly while maintaining comfortable knee alignment."
        elif "past toes" in cleaned_issue or ("knee" in cleaned_issue and "toe" in cleaned_issue):
            return "Your front knee is tracking too far forward over your toes. Take a slightly larger stride forward so your front knee stays directly above your ankle."
        elif "extend" in cleaned_issue or "extension" in cleaned_issue or "lockout" in cleaned_issue:
            if "elbow" in cleaned_issue or "arm" in cleaned_issue or "overhead" in cleaned_issue:
                return "Your arms are not extending fully overhead. To complete the full movement, press the weights all the way up until your elbows are fully locked out overhead."
            elif "stand" in cleaned_issue or "standing" in cleaned_issue or "hip" in cleaned_issue:
                return "You did not stand all the way up. Extend your hips and knees fully at the top and squeeze your glutes before your next rep."
            return "You didn't push all the way to the top. Fully extend your arms and lock out your elbows at the top of the plank."
        elif "lower" in cleaned_issue or "chest" in cleaned_issue or "bend" in cleaned_issue:
            return "You are not lowering your chest enough. Bend your elbows until your chest is a few inches from the floor for full chest and triceps activation."
        elif "curl" in cleaned_issue or "contraction" in cleaned_issue:
            return "You are not curling the weight high enough. Lift the weight all the way up toward your shoulder to achieve a full bicep contraction."
        elif "frame" in cleaned_issue or "camera" in cleaned_issue or "occlu" in cleaned_issue:
            return "Key body landmarks are partially hidden or outside the camera view. Step back into camera frame so your entire body from head to feet is clearly visible."
        elif "clean" in cleaned_issue or "strong" in cleaned_issue or "good" in cleaned_issue:
            return "Great adjustment! Your form is clean, balanced, and under control. Keep up this exact technique."

        return issue

    return DEFAULT_COACHING_CUES.get(event, "Great job! Keep up this exact technique and stay focused.")


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

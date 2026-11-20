import time
import uuid
import base64
import logging
from typing import Optional, Tuple, Dict, Any, List
import streamlit as st
import streamlit.components.v1 as components
from services.coaching.tts import validate_audio_bytes
from services.coaching.llm import get_deterministic_cue

logger = logging.getLogger(__name__)

STAGE_GUIDANCE_CUES = {
    "Shoulder Press": {
        "moving": "Press upward with control and extend your elbows toward full lockout.",
        "descending": "Lower the weights slowly back to shoulder height.",
        "inflected": "Full extension overhead! Now lower down with control.",
    },
    "Squats": {
        "moving": "Bend your knees and keep your chest elevated.",
        "descending": "Lower your hips smoothly until your thighs reach parallel.",
        "inflected": "Great depth! Push upward through your heels.",
        "ascending": "Drive upward through your heels to a full stand.",
    },
    "Push-ups": {
        "moving": "Lower your chest smoothly toward the floor.",
        "descending": "Control the descent while keeping your core engaged.",
        "inflected": "Chest down! Press firmly upward to full plank.",
        "ascending": "Push through your palms and maintain a straight plank.",
    },
    "Biceps Curls (Dumbbell)": {
        "moving": "Curl upward toward your shoulders without moving your elbows.",
        "ascending": "Squeeze your biceps at the top contraction.",
        "inflected": "Great contraction! Lower the dumbbells slowly.",
        "descending": "Lower with steady control for a full stretch.",
    },
    "Lunges": {
        "moving": "Step forward with control and lower your hips.",
        "descending": "Drop your back knee toward the floor while keeping your torso upright.",
        "inflected": "Solid lunge depth! Drive back up through your front heel.",
        "ascending": "Drive through the front heel to return to a full stand.",
    },
}


class VoicePipeline:
    """
    Unified Voice Coaching Pipeline with Natural Delivery & Audio Queue:
    - Bridges Computer Vision events to complete, natural AI coaching text and TTS audio
    - Enforces non-interruption audio queue: never truncates or replaces active speech
    - Provides smart coaching: detailed explanation on first detection, focused cues on repetition
    - Automatically purges stale error messages when form recovers to Good
    - Supports muting, offline deterministic fallbacks, and Streamlit rerun persistence
    """

    def __init__(self, llm, tts, precache: bool = False):
        self.llm = llm
        self.tts = tts
        self.last_spoken_at: float = 0.0
        self.last_spoken_text: str = ""
        self.last_stage: Optional[str] = None
        self.muted: bool = False
        self.had_form_issue: bool = False

        # Queue Management
        self.queue: List[Dict[str, Any]] = []
        self.current_playing: Optional[Dict[str, Any]] = None

        # Form Correction Tracking & Anti-Spam
        self.last_form_issue: Optional[str] = None
        self.last_form_issue_time: float = 0.0

        if precache:
            self._precache_standard_cues()

    def estimate_duration(self, audio_bytes: Optional[bytes], text: str = "") -> float:
        """
        Estimates the speaking duration in seconds.
        gTTS produces CBR 32 kbps (4,000 bytes/sec) MP3.
        Also checks word count (approx 2.3 words/sec).
        """
        byte_dur = (len(audio_bytes) / 4000.0) if audio_bytes else 0.0
        word_dur = (len(text.split()) / 2.3 + 0.6) if text else 0.0
        return max(2.5, byte_dur, word_dur)

    def is_playing(self, now: Optional[float] = None) -> bool:
        """Returns True if the current audio event is actively playing in the browser."""
        if not self.current_playing:
            return False
        if now is None:
            now = time.time()
        started_at = self.current_playing.get("started_at", 0.0)
        duration = self.current_playing.get("duration", 0.0)
        return now < (started_at + duration)

    def enqueue(
        self,
        audio_bytes: bytes,
        text: str,
        event_id: str,
        duration: float,
        event: str,
        issue: Optional[str] = None,
        exercise: Optional[str] = None,
    ) -> None:
        """Queues an audio announcement for playback after current audio finishes."""
        cleaned_text = text.strip().lower()
        # Prevent identical messages from piling up in the queue
        if any(item["text"].strip().lower() == cleaned_text for item in self.queue):
            return
        # Bound maximum queue size to avoid stale backlogs
        if len(self.queue) >= 4:
            self.queue.pop(0)

        self.queue.append({
            "event_id": event_id,
            "audio_bytes": audio_bytes,
            "text": text,
            "duration": duration,
            "event": event,
            "issue": issue,
            "exercise": exercise,
            "enqueued_at": time.time(),
        })

    def clear_queue(self) -> None:
        """Clears all pending announcements and resets playback state on workout completion/reset."""
        self.queue.clear()
        self.current_playing = None

    def get_active_audio(self, current_form_status: str = "Good", now: Optional[float] = None) -> Optional[Tuple[bytes, str, str]]:
        """
        Servicing loop for the audio queue:
        - If current audio is still playing, keeps it mounted in DOM.
        - When current audio finishes, dequeues the next announcement in FIFO order.
        - Preserves valid corrections from current workout movements. Only drops items older than 20s.
        """
        if self.muted:
            return None
        if now is None:
            now = time.time()

        # If current audio is still playing, keep it mounted
        if self.is_playing(now):
            return (
                self.current_playing["audio_bytes"],
                self.current_playing["text"],
                self.current_playing["event_id"],
            )

        # Current audio has finished; check the queue
        while self.queue:
            item = self.queue.pop(0)
            enqueued_at = item.get("enqueued_at", now)

            # If item is older than 20 seconds, discard as stale
            if (now - enqueued_at) > 20.0:
                continue

            # Valid item ready for playback
            item["started_at"] = now
            self.current_playing = item
            self.last_spoken_at = now
            self.last_spoken_text = item["text"]
            return (item["audio_bytes"], item["text"], item["event_id"])

        self.current_playing = None
        return None

    def _precache_standard_cues(self) -> None:
        standard_phrases = [
            "Good job! Rep completed. Keep that steady pace and control.",
            "Awesome work! Set completed. Take a breath, recover, and prepare for your next set.",
            "Workout completed! Outstanding job! You crushed every set today.",
            "Key body landmarks are outside the camera frame. Step back into the camera frame so I can track your full body.",
            "Looking strong! Maintain good control and keep your breathing steady.",
            "Great adjustment! Your form is clean, balanced, and under control. Keep up this exact technique.",
            "Welcome to Apna AI Coach. Voice is active and working properly.",
        ]
        try:
            self.tts.precache_phrases(standard_phrases)
        except Exception as e:
            logger.warning(f"Voice pipeline pre-caching warning: {e}")

    def _find_form_issue(self, exercise: str, metrics: Dict[str, Any]) -> Optional[str]:
        # Direct check from detector's form analysis
        if metrics.get("form_status") == "Needs Correction":
            return metrics.get("form_correction") or metrics.get("form_issue")

        if "issue" in metrics:
            return metrics["issue"]

        if exercise == "Squats":
            depth = metrics.get("depth_status", "")
            back_angle = metrics.get("back_angle", 180)
            if depth in ("TOO HIGH", "HALF SQUAT"):
                return "Your squat depth is too shallow. Squat a little deeper for full range of motion. Bend your knees and push your hips backward until thighs are parallel to the floor."
            if isinstance(back_angle, (int, float)) and back_angle < 120:
                return "Your upper body is leaning too far forward. Keep your chest lifted, engage your core, and look straight ahead throughout the squat."

        elif exercise == "Push-ups":
            alignment = metrics.get("body_alignment", "")
            hip_status = metrics.get("hip_status", "")
            if alignment == "Poor Form":
                return "Your body alignment is bending out of line. Engage your core tightly and maintain a solid straight line from head to heels."
            if hip_status == "SAGGING":
                return "Your hips are dropping below your body alignment. Engage your core and maintain a straight line from your shoulders to your heels."
            if hip_status == "PIKED UP":
                return "Your hips are lifted too high in the air. Lower your hips so your body forms a straight, solid plank line from your head to your heels."

        elif exercise == "Biceps Curls (Dumbbell)":
            swing = metrics.get("swing_status", "")
            shoulder = metrics.get("shoulder_status", "")
            if swing == "SWINGING":
                return "You are swinging your torso to lift the weights. Stand tall, brace your core, and isolate your biceps by moving only your forearms."
            if shoulder == "ELBOW DRIFTING":
                return "You are moving your elbows away from your body. Keep your elbows close to your sides, avoid swinging your torso, and curl the weight slowly."

        elif exercise == "Shoulder Press":
            arm_asymmetry = metrics.get("arm_asymmetry", 0.0)
            if isinstance(arm_asymmetry, (int, float)) and arm_asymmetry > 25.0:
                return "Your arms are pressing unevenly. This can create an imbalance in your shoulder movement. Push both arms upward at the same time and keep them at the same height. Keep your core stable and control the movement. Try again."
            back_arch = metrics.get("back_arch_status", "")
            if back_arch == "Excessive Arch":
                return "You are arching your lower back during the press. This puts unnecessary stress on your lumbar spine. Brace your core tightly, keep your spine neutral, and press straight overhead without leaning back."

        elif exercise == "Lunges":
            balance = metrics.get("balance_status", "")
            if balance == "OFF BALANCE":
                return "Your balance is unstable. Take a controlled step forward, keep your upper body upright, and lower yourself slowly while maintaining comfortable knee alignment."

        return None

    def process_event(
        self,
        event: str,
        exercise: str,
        metrics: Dict[str, Any],
    ) -> Optional[Tuple[bytes, str, str]]:
        """
        Processes a coaching event with complete sentence formulation and audio queue management.
        Returns: (audio_bytes, text, event_id) or None.
        """
        if self.muted:
            return None

        now = time.time()
        is_major_event = event in ("workout_started", "rep_completed", "set_completed", "workout_completed", "rep_invalid")

        issue = self._find_form_issue(exercise, metrics) if not is_major_event else None

        if issue:
            self.had_form_issue = True

        # Check for stage-specific movement guidance
        stage = metrics.get("stage") or metrics.get("exercise_stage")
        stage_cue = None
        if not is_major_event and not issue and stage and exercise in STAGE_GUIDANCE_CUES:
            if stage != self.last_stage and stage in STAGE_GUIDANCE_CUES[exercise]:
                stage_cue = STAGE_GUIDANCE_CUES[exercise][stage]
                self.last_stage = stage

        # Rate-limiting / anti-spam / deduplication
        if is_major_event:
            pass
        elif issue:
            # Form corrections: high priority!
            # Never throttle form corrections using generic stage-cue cooldown (last_spoken_at).
            # Deduplicate only if identical issue was processed within 6.0 seconds.
            if self.last_form_issue == issue and (now - self.last_form_issue_time) < 6.0:
                return None
            self.last_form_issue = issue
            self.last_form_issue_time = now
        else:
            if (now - self.last_spoken_at) < 4.0:
                return None

            # Resume positive feedback when form was corrected
            if not stage_cue and self.had_form_issue and metrics.get("form_status") == "Good":
                self.had_form_issue = False
                issue = "Great adjustment! Your form is clean, balanced, and under control. Keep up this exact technique."

            if not issue and not stage_cue and event == "ongoing_form_check":
                return None

        # Determine spoken text
        if event == "rep_invalid":
            reason = metrics.get("form_correction") or metrics.get("form_issue", "check your form")
            text = get_deterministic_cue("rep_invalid", reason, exercise)
        elif stage_cue and not issue:
            text = stage_cue
        else:
            text = self.llm.give_feedback(event, issue, exercise)

        if not text:
            return None

        # Prevent repeating identical text within 6.0 seconds
        if text.strip().lower() == self.last_spoken_text.strip().lower() and (now - self.last_spoken_at) < 6.0:
            return None

        audio_bytes = self.tts.speak(text)
        if not audio_bytes:
            return None

        duration = self.estimate_duration(audio_bytes, text)
        event_id = uuid.uuid4().hex

        # Non-interruption queue: if audio is currently playing in browser, enqueue!
        if self.is_playing(now):
            self.enqueue(audio_bytes, text, event_id, duration, event, issue, exercise)
            # Return current playing audio so player remains mounted
            return (
                self.current_playing["audio_bytes"],
                self.current_playing["text"],
                self.current_playing["event_id"],
            )

        # Idle: start playing immediately
        self.current_playing = {
            "event_id": event_id,
            "audio_bytes": audio_bytes,
            "text": text,
            "duration": duration,
            "event": event,
            "issue": issue,
            "exercise": exercise,
            "started_at": now,
        }
        self.last_spoken_at = now
        self.last_spoken_text = text

        return audio_bytes, text, event_id


def render_voice_player(audio_bytes: bytes, text: str = "", event_id: str = "", auto_play: bool = True, height: int = 78) -> None:
    """
    Renders an interactive, audited HTML5 audio player directly inside Streamlit:
    - Triggers audio.play() directly from the manual Play button click handler with volume 1.0 and unmuted assurance.
    - Resumes suspended Web Audio API AudioContext if needed.
    - Displays actual playback errors visibly in the UI instead of silently swallowing exceptions.
    - Shows real-time source payload, duration, volume, mute, and playback state badges.
    - Automatically attempts autoplay upon initial render with graceful fallback to manual play button.
    """
    if not audio_bytes or not validate_audio_bytes(audio_bytes):
        st.warning("⚠️ Invalid or undecodable audio format.")
        return

    b64 = base64.b64encode(audio_bytes).decode("ascii")
    payload_size = len(audio_bytes)
    safe_id = "".join(c for c in (event_id or "default") if c.isalnum() or c in "-_")
    autoplay_attr = "autoplay" if auto_play else ""

    html_code = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0;
    padding: 2px 4px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: transparent;
    color: #e0e0e0;
  }}
  .player-box {{
    background: #141923;
    border: 1px solid #2d3748;
    border-radius: 6px;
    padding: 6px 10px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}
  .controls-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }}
  .play-btn {{
    background: #ff4b4b;
    color: #ffffff;
    border: none;
    padding: 5px 12px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    transition: background 0.15s ease;
  }}
  .play-btn:hover {{
    background: #ff3333;
  }}
  .badge {{
    font-size: 11px;
    padding: 2px 5px;
    border-radius: 3px;
    background: #232b3b;
    color: #cbd5e0;
  }}
  .status-badge {{
    font-size: 11px;
    padding: 2px 5px;
    border-radius: 3px;
    font-weight: bold;
  }}
  .status-idle {{ background: #4a5568; color: #fff; }}
  .status-playing {{ background: #22543d; color: #9ae6b4; }}
  .status-error {{ background: #742a2a; color: #feb2b2; }}
  .status-blocked {{ background: #7b341e; color: #fbd38d; }}
  .error-box {{
    color: #feb2b2;
    font-size: 11px;
    background: #441a1a;
    padding: 3px 6px;
    border-radius: 3px;
    border-left: 3px solid #f56565;
    display: none;
    margin-top: 2px;
  }}
  .progress-bar-container {{
    width: 100%;
    height: 3px;
    background: #232b3b;
    border-radius: 2px;
    overflow: hidden;
  }}
  .progress-bar {{
    height: 100%;
    width: 0%;
    background: #ff4b4b;
    transition: width 0.1s linear;
  }}
</style>
</head>
<body>
<div class="player-box" id="apna-player-{safe_id}">
  <audio id="apna-voice-el" preload="auto" src="data:audio/mp3;base64,{b64}" {autoplay_attr}></audio>
  
  <div class="controls-row">
    <button id="apna-voice-btn" class="play-btn" onclick="startAudio()">
      <span id="btn-icon">▶</span> <span id="btn-text">Play Coach Voice</span>
    </button>
    <span id="playback-badge" class="status-badge status-idle">IDLE</span>
    <span class="badge" id="time-badge">0.0s / 0.0s</span>
    <span class="badge" id="vol-badge">Vol: 100%</span>
    <span class="badge" id="src-badge">{payload_size:,} B</span>
  </div>

  <div class="progress-bar-container">
    <div id="playback-progress" class="progress-bar"></div>
  </div>

  <div id="error-display" class="error-box"></div>
</div>

<script>
var audio = document.getElementById('apna-voice-el');
var btn = document.getElementById('apna-voice-btn');
var btnIcon = document.getElementById('btn-icon');
var btnText = document.getElementById('btn-text');
var badge = document.getElementById('playback-badge');
var timeBadge = document.getElementById('time-badge');
var volBadge = document.getElementById('vol-badge');
var errBox = document.getElementById('error-display');
var progress = document.getElementById('playback-progress');

function setStatus(st, text) {{
  if (!badge) return;
  badge.className = 'status-badge status-' + st;
  badge.textContent = text;
}}

function showError(msg) {{
  if (!errBox) return;
  errBox.style.display = 'block';
  errBox.textContent = '⚠️ ' + msg;
  setStatus('error', 'ERROR');
}}

function clearError() {{
  if (!errBox) return;
  errBox.style.display = 'none';
  errBox.textContent = '';
}}

function updateTime() {{
  if (!audio) return;
  var cur = audio.currentTime || 0;
  var dur = audio.duration || 0;
  if (timeBadge) timeBadge.textContent = cur.toFixed(1) + 's / ' + (dur ? dur.toFixed(1) : '?') + 's';
  if (volBadge) volBadge.textContent = 'Vol: ' + Math.round((audio.volume || 1) * 100) + '%' + (audio.muted ? ' (M)' : '');
  if (dur > 0 && progress) {{
    progress.style.width = Math.min(100, (cur / dur) * 100) + '%';
  }}
}}

function startAudio() {{
  clearError();
  if (!audio) {{
    showError('Audio element missing from DOM.');
    return;
  }}

  // 1. Ensure volume and unmute directly in click handler
  audio.muted = false;
  audio.volume = 1.0;

  // 2. Resume Web Audio AudioContext if suspended
  try {{
    var AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (AudioCtx) {{
      var ctx = new AudioCtx();
      if (ctx.state === 'suspended') {{
        ctx.resume();
      }}
    }}
  }} catch (e) {{
    console.warn('[Apna Audio] AudioContext init warning:', e);
  }}

  // 3. Also unmute and trigger top-level audio elements if reachable
  try {{
    if (window.parent && window.parent.document) {{
      var pAudios = window.parent.document.querySelectorAll('audio');
      for (var i = 0; i < pAudios.length; i++) {{
        pAudios[i].muted = false;
        pAudios[i].volume = 1.0;
        var pp = pAudios[i].play();
        if (pp !== undefined) pp.catch(function() {{}});
      }}
    }}
  }} catch (e) {{}}

  // 4. Trigger audio.play() directly from the click handler
  var promise = audio.play();
  if (promise !== undefined) {{
    promise.then(function() {{
      setStatus('playing', 'PLAYING');
      if (btnIcon) btnIcon.textContent = '⏸';
      if (btnText) btnText.textContent = 'Pause';
      btn.onclick = handlePause;
      updateTime();
    }}).catch(function(err) {{
      var errMsg = 'Playback failed: ' + (err.name || 'Error') + ' - ' + (err.message || 'Unknown browser restriction');
      if (err.name === 'NotAllowedError') {{
        errMsg = 'Browser blocked playback. Click Play again or allow audio permission.';
      }}
      showError(errMsg);
    }});
  }}
}}

function handlePause() {{
  if (audio) {{
    audio.pause();
    setStatus('idle', 'PAUSED');
    if (btnIcon) btnIcon.textContent = '▶';
    if (btnText) btnText.textContent = 'Resume';
    btn.onclick = startAudio;
  }}
}}

if (audio) {{
  audio.addEventListener('play', function() {{
    setStatus('playing', 'PLAYING');
    if (btnIcon) btnIcon.textContent = '⏸';
    if (btnText) btnText.textContent = 'Pause';
    btn.onclick = handlePause;
  }});
  audio.addEventListener('pause', function() {{
    if (!audio.ended) {{
      setStatus('idle', 'PAUSED');
      if (btnIcon) btnIcon.textContent = '▶';
      if (btnText) btnText.textContent = 'Resume';
      btn.onclick = startAudio;
    }}
  }});
  audio.addEventListener('ended', function() {{
    setStatus('idle', 'COMPLETED');
    if (btnIcon) btnIcon.textContent = '🔄';
    if (btnText) btnText.textContent = 'Replay Voice';
    btn.onclick = startAudio;
    if (progress) progress.style.width = '100%';
  }});
  audio.addEventListener('timeupdate', updateTime);
  audio.addEventListener('loadedmetadata', updateTime);
  audio.addEventListener('error', function(e) {{
    var code = audio.error ? audio.error.code : 'unknown';
    var msg = audio.error ? audio.error.message : '';
    showError('Media decode error (Code ' + code + '): ' + (msg || 'Corrupted or unplayable format'));
  }});

  // Attempt initial autoplay
  audio.muted = false;
  audio.volume = 1.0;
  var autoPromise = audio.play();
  if (autoPromise !== undefined) {{
    autoPromise.then(function() {{
      setStatus('playing', 'PLAYING');
    }}).catch(function(err) {{
      setStatus('blocked', 'CLICK PLAY');
      console.warn('[Apna Audio] Autoplay blocked by browser policy:', err);
    }});
  }}
}}
</script>
</body>
</html>"""

    components.html(html_code, height=height)


def autoplay_audio(audio_bytes: bytes, text: str = "", event_id: str = "") -> None:
    """
    Renders the audio element into Streamlit with browser-level non-interruption protection.
    Ensures:
    1. Audio is valid and decodable via validate_audio_bytes.
    2. Keeps audio mounted in DOM during Streamlit 250ms reruns without restarting.
    3. Triggers autoplay=True on first render of a new event_id.
    4. Preserves native HTML5 Play/Pause/Volume controls if browser blocks autoplay.
    5. Injects an audited HTML5 player with direct manual Play button, volume/mute assurance, and visible errors.
    """
    if not audio_bytes or not validate_audio_bytes(audio_bytes):
        return

    last_played = st.session_state.get("last_played_audio_event_id")
    is_first_render = bool(event_id and event_id != last_played)

    if is_first_render:
        st.session_state["last_played_audio_event_id"] = event_id

    # Always render the audio widget so Streamlit does NOT unmount it on 250ms reruns!
    st.audio(audio_bytes, format="audio/mp3", autoplay=is_first_render)

    if is_first_render:
        render_voice_player(audio_bytes, text=text, event_id=event_id, auto_play=True, height=78)

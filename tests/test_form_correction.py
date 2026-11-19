import math
import sys
import unittest.mock as mock
from pathlib import Path
import pytest

# Ensure Main App is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_APP_DIR = PROJECT_ROOT / "Main App"
if str(MAIN_APP_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_APP_DIR))

from detectors.shoulder_press import ShoulderPressDetector
from detectors.squat import SquatDetector
from detectors.pushup import PushUpDetector
from detectors.biceps_curl import BicepsCurlDetector
from detectors.lunges import LungesDetector
from services.state.workout_state import WorkoutStateManager
from services.coaching.tts import TextToSpeech
from services.coaching.llm import LLMCoach
from services.coaching.voice_pipeline import VoicePipeline


class MockLandmark:
    def __init__(self, x=0.5, y=0.5, z=0.0, visibility=0.95):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def create_landmarks(num_landmarks=33, visibility=0.95):
    return [MockLandmark(0.5, 0.5, 0.0, visibility) for _ in range(num_landmarks)]


# ==============================================================================
# 1. SHOULDER PRESS FORM CHECKS & REP VALIDATION
# ==============================================================================
def make_shoulder_press_landmarks(l_angle=90.0, r_angle=90.0, back_angle=175.0, start_y=0.3):
    lm = create_landmarks()
    hip_y = start_y + 0.35
    knee_y = hip_y + 0.35
    lm[23] = MockLandmark(0.4, hip_y)
    lm[24] = MockLandmark(0.6, hip_y)
    lm[25] = MockLandmark(0.4, knee_y)
    lm[26] = MockLandmark(0.6, knee_y)

    # Shoulders (adjust for back arch)
    if back_angle < 170.0:
        rad_b = math.radians(180 - back_angle)
        l_sh_x = 0.4 + 0.35 * math.sin(rad_b)
        l_sh_y = hip_y - 0.35 * math.cos(rad_b)
        r_sh_x = 0.6 + 0.35 * math.sin(rad_b)
        r_sh_y = hip_y - 0.35 * math.cos(rad_b)
    else:
        l_sh_x, l_sh_y = 0.4, start_y
        r_sh_x, r_sh_y = 0.6, start_y

    lm[11] = MockLandmark(l_sh_x, l_sh_y)
    lm[12] = MockLandmark(r_sh_x, r_sh_y)

    # Left Arm: Shoulder (11), Elbow (13), Wrist (15)
    lm[13] = MockLandmark(l_sh_x, l_sh_y + 0.2)
    rad_l = math.radians(180 - l_angle)
    lm[15] = MockLandmark(l_sh_x + 0.2 * math.sin(rad_l), (l_sh_y + 0.2) + 0.2 * math.cos(rad_l))

    # Right Arm: Shoulder (12), Elbow (14), Wrist (16)
    lm[14] = MockLandmark(r_sh_x, r_sh_y + 0.2)
    rad_r = math.radians(180 - r_angle)
    lm[16] = MockLandmark(r_sh_x - 0.2 * math.sin(rad_r), (r_sh_y + 0.2) + 0.2 * math.cos(rad_r))

    return lm


def test_shoulder_press_form_checks():
    det = ShoulderPressDetector()

    # Good form: elbow at 90°, back straight (175°)
    lm = make_shoulder_press_landmarks(l_angle=90.0, r_angle=90.0, back_angle=175.0)
    m = det.process(lm, current_time=1.0)
    assert m["form_status"] == "Good"
    assert m["back_arch_status"] == "Neutral"

    # Excessive lower-back arch (back angle < 145°)
    det.reset()
    lm_arch = make_shoulder_press_landmarks(l_angle=90.0, r_angle=90.0, back_angle=135.0)
    m_arch = det.process(lm_arch, current_time=1.0)
    assert m_arch["form_status"] == "Needs Correction"
    assert "arch" in m_arch["form_issue"].lower()
    assert "back" in m_arch["form_correction"].lower()

    # Uneven arms asymmetry (> 25° difference)
    det.reset()
    lm_uneven = make_shoulder_press_landmarks(l_angle=140.0, r_angle=95.0, back_angle=175.0)
    m_uneven = det.process(lm_uneven, current_time=1.0)
    assert m_uneven["form_status"] == "Needs Correction"
    assert "uneven" in m_uneven["form_issue"].lower()

    # Elbows dropped too low at start (< 65°)
    det.reset()
    lm_low = make_shoulder_press_landmarks(l_angle=55.0, r_angle=55.0, back_angle=175.0)
    m_low = det.process(lm_low, current_time=1.0)
    assert m_low["form_status"] == "Needs Correction"
    assert "too low" in m_low["form_issue"].lower()


def test_shoulder_press_rep_validation():
    det = ShoulderPressDetector()

    # Valid rep: start at 90°, press to 160°, return to 90° with neutral back
    det.process(make_shoulder_press_landmarks(90.0, 90.0, 175.0), current_time=1.0)
    det.process(make_shoulder_press_landmarks(125.0, 125.0, 175.0), current_time=1.3)
    det.process(make_shoulder_press_landmarks(160.0, 160.0, 175.0), current_time=1.6)
    det.process(make_shoulder_press_landmarks(160.0, 160.0, 175.0), current_time=1.7)
    det.process(make_shoulder_press_landmarks(125.0, 125.0, 175.0), current_time=2.1)
    det.process(make_shoulder_press_landmarks(90.0, 90.0, 175.0), current_time=2.5)
    m = det.process(make_shoulder_press_landmarks(90.0, 90.0, 175.0), current_time=2.6)

    assert m["rep_incremented"] is True
    assert m["rep_invalid"] is False
    assert m["valid_reps"] == 1
    assert m["invalid_reps"] == 0

    # Invalid rep: presses with excessive lower-back arch (back angle 130°)
    det.process(make_shoulder_press_landmarks(90.0, 90.0, 130.0), current_time=3.0)
    det.process(make_shoulder_press_landmarks(125.0, 125.0, 130.0), current_time=3.3)
    det.process(make_shoulder_press_landmarks(160.0, 160.0, 130.0), current_time=3.6)
    det.process(make_shoulder_press_landmarks(160.0, 160.0, 130.0), current_time=3.7)
    det.process(make_shoulder_press_landmarks(125.0, 125.0, 130.0), current_time=4.1)
    det.process(make_shoulder_press_landmarks(90.0, 90.0, 130.0), current_time=4.5)
    m_bad = det.process(make_shoulder_press_landmarks(90.0, 90.0, 130.0), current_time=4.6)

    assert m_bad["rep_incremented"] is False
    assert m_bad["rep_invalid"] is True
    assert m_bad["valid_reps"] == 1  # Did not increment!
    assert m_bad["invalid_reps"] == 1  # Increment invalid!
    assert "arch" in m_bad["form_issue"].lower()


# ==============================================================================
# 2. SQUATS FORM CHECKS & REP VALIDATION
# ==============================================================================
def make_squat_landmarks(knee_angle=165.0, back_angle=175.0, valgus=False):
    lm = create_landmarks()
    hip_x, hip_y = 0.4, 0.4
    knee_x, knee_y = 0.4, 0.65

    lm[23] = MockLandmark(hip_x, hip_y)
    lm[24] = MockLandmark(0.6, hip_y)
    lm[25] = MockLandmark(knee_x, knee_y)
    lm[26] = MockLandmark(0.6, knee_y)

    # Ankle position for knee angle
    rad_k = math.radians(180 - knee_angle)
    lm[27] = MockLandmark(knee_x + 0.25 * math.sin(rad_k), knee_y + 0.25 * math.cos(rad_k))
    lm[28] = MockLandmark(0.6 - 0.25 * math.sin(rad_k), knee_y + 0.25 * math.cos(rad_k))

    # Shoulder position for back angle
    rad_b = math.radians(180 - back_angle)
    lm[11] = MockLandmark(hip_x + 0.25 * math.sin(rad_b), hip_y - 0.25 * math.cos(rad_b))
    lm[12] = MockLandmark(0.6 - 0.25 * math.sin(rad_b), hip_y - 0.25 * math.cos(rad_b))

    if valgus:
        # Knees caved in: knees close, ankles apart
        lm[25].x = 0.48
        lm[26].x = 0.52
        lm[27].x = 0.35
        lm[28].x = 0.65

    return lm


def test_squats_form_checks():
    det = SquatDetector()

    # Good form: standing upright (knee 165°, back 175°)
    lm_good = make_squat_landmarks(knee_angle=165.0, back_angle=175.0)
    m = det.process(lm_good, current_time=1.0)
    assert m["form_status"] == "Good"

    # Excessive forward trunk lean (back angle < 120°)
    det.reset()
    lm_lean = make_squat_landmarks(knee_angle=130.0, back_angle=110.0)
    m_lean = det.process(lm_lean, current_time=1.0)
    assert m_lean["form_status"] == "Needs Correction"
    assert "lean" in m_lean["form_issue"].lower()

    # Knee valgus (knees caving inward)
    det.reset()
    lm_valgus = make_squat_landmarks(knee_angle=130.0, back_angle=160.0, valgus=True)
    m_valgus = det.process(lm_valgus, current_time=1.0)
    assert m_valgus["form_status"] == "Needs Correction"
    assert "valgus" in m_valgus["form_issue"].lower()


def test_squats_rep_validation():
    det = SquatDetector()

    # Valid rep: drops to 95° depth, returns to 165°, back upright
    det.process(make_squat_landmarks(165.0, 175.0), current_time=1.0)
    det.process(make_squat_landmarks(130.0, 175.0), current_time=1.3)
    det.process(make_squat_landmarks(95.0, 175.0), current_time=1.6)
    det.process(make_squat_landmarks(95.0, 175.0), current_time=1.7)
    det.process(make_squat_landmarks(130.0, 175.0), current_time=2.1)
    det.process(make_squat_landmarks(165.0, 175.0), current_time=2.5)
    m = det.process(make_squat_landmarks(165.0, 175.0), current_time=2.6)

    assert m["rep_incremented"] is True
    assert m["rep_invalid"] is False
    assert m["valid_reps"] == 1
    assert m["invalid_reps"] == 0

    # Invalid rep: drops with excessive forward lean (back 105°)
    det.process(make_squat_landmarks(165.0, 105.0), current_time=3.0)
    det.process(make_squat_landmarks(130.0, 105.0), current_time=3.3)
    det.process(make_squat_landmarks(95.0, 105.0), current_time=3.6)
    det.process(make_squat_landmarks(95.0, 105.0), current_time=3.7)
    det.process(make_squat_landmarks(130.0, 105.0), current_time=4.1)
    det.process(make_squat_landmarks(165.0, 105.0), current_time=4.5)
    m_bad = det.process(make_squat_landmarks(165.0, 105.0), current_time=4.6)

    assert m_bad["rep_incremented"] is False
    assert m_bad["rep_invalid"] is True
    assert m_bad["valid_reps"] == 1  # Not incremented!
    assert m_bad["invalid_reps"] == 1
    assert "lean" in m_bad["form_issue"].lower()


# ==============================================================================
# 3. PUSH-UPS FORM CHECKS & REP VALIDATION
# ==============================================================================
def make_pushup_landmarks(elbow_angle=160.0, sag=0.0):
    lm = create_landmarks()
    lm[11] = MockLandmark(0.2, 0.5)  # Shoulder
    lm[13] = MockLandmark(0.2, 0.7)  # Elbow
    rad_e = math.radians(180 - elbow_angle)
    lm[15] = MockLandmark(0.2 + 0.2 * math.sin(rad_e), 0.7 + 0.2 * math.cos(rad_e))

    # Body line: Shoulder at (0.2, 0.5), Ankle at (0.8, 0.5)
    # Expected hip y is 0.5
    lm[23] = MockLandmark(0.5, 0.5 + sag)
    lm[27] = MockLandmark(0.8, 0.5)
    return lm


def test_pushups_form_checks():
    det = PushUpDetector()

    # Good form: elbow 160°, level hips
    lm_good = make_pushup_landmarks(160.0, sag=0.0)
    m = det.process(lm_good, current_time=1.0)
    assert m["form_status"] == "Good"
    assert m["hip_status"] == "LEVEL"

    # Sagging hips
    det.reset()
    lm_sag = make_pushup_landmarks(160.0, sag=0.12)
    m_sag = det.process(lm_sag, current_time=1.0)
    assert m_sag["form_status"] == "Needs Correction"
    assert m_sag["hip_status"] == "SAGGING"
    assert "sag" in m_sag["form_issue"].lower()

    # Piked hips
    det.reset()
    lm_pike = make_pushup_landmarks(160.0, sag=-0.12)
    m_pike = det.process(lm_pike, current_time=1.0)
    assert m_pike["form_status"] == "Needs Correction"
    assert m_pike["hip_status"] == "PIKED UP"
    assert "high" in m_pike["form_issue"].lower()


def test_pushups_rep_validation():
    det = PushUpDetector()

    # Valid rep: drops to 85° depth with level hips, pushes back up
    det.process(make_pushup_landmarks(160.0, sag=0.0), current_time=1.0)
    det.process(make_pushup_landmarks(125.0, sag=0.0), current_time=1.3)
    det.process(make_pushup_landmarks(85.0, sag=0.0), current_time=1.6)
    det.process(make_pushup_landmarks(85.0, sag=0.0), current_time=1.7)
    det.process(make_pushup_landmarks(125.0, sag=0.0), current_time=2.1)
    det.process(make_pushup_landmarks(160.0, sag=0.0), current_time=2.5)
    m = det.process(make_pushup_landmarks(160.0, sag=0.0), current_time=2.6)

    assert m["rep_incremented"] is True
    assert m["rep_invalid"] is False
    assert m["valid_reps"] == 1
    assert m["invalid_reps"] == 0

    # Invalid rep: drops to 85° but hips sag throughout rep
    det.process(make_pushup_landmarks(160.0, sag=0.12), current_time=3.0)
    det.process(make_pushup_landmarks(125.0, sag=0.12), current_time=3.3)
    det.process(make_pushup_landmarks(85.0, sag=0.12), current_time=3.6)
    det.process(make_pushup_landmarks(85.0, sag=0.12), current_time=3.7)
    det.process(make_pushup_landmarks(125.0, sag=0.12), current_time=4.1)
    det.process(make_pushup_landmarks(160.0, sag=0.12), current_time=4.5)
    m_bad = det.process(make_pushup_landmarks(160.0, sag=0.12), current_time=4.6)

    assert m_bad["rep_incremented"] is False
    assert m_bad["rep_invalid"] is True
    assert m_bad["valid_reps"] == 1
    assert m_bad["invalid_reps"] == 1
    assert "sag" in m_bad["form_issue"].lower()


# ==============================================================================
# 4. BICEPS CURLS FORM CHECKS & REP VALIDATION
# ==============================================================================
def make_curl_landmarks(elbow_angle=160.0, drift=0.01, torso_shift=0.0):
    lm = create_landmarks()
    lm[11] = MockLandmark(0.4 + torso_shift, 0.3)
    lm[12] = MockLandmark(0.6 + torso_shift, 0.3)
    lm[23] = MockLandmark(0.4, 0.6)
    lm[24] = MockLandmark(0.6, 0.6)

    # Arm tracking on left side
    sh_x = 0.4 + torso_shift
    lm[13] = MockLandmark(sh_x + drift, 0.5)
    rad_e = math.radians(180 - elbow_angle)
    lm[15] = MockLandmark((sh_x + drift) + 0.2 * math.sin(rad_e), 0.5 + 0.2 * math.cos(rad_e))
    return lm


def test_biceps_curls_form_checks():
    det = BicepsCurlDetector()

    # Good form: elbow extended 160°, stable elbow, no swing
    lm_good = make_curl_landmarks(160.0, drift=0.01, torso_shift=0.0)
    m = det.process(lm_good, current_time=1.0)
    assert m["form_status"] == "Good"
    assert m["shoulder_status"] == "STABLE"
    assert m["swing_status"] == "NO SWING"

    # Elbow drifting (> 0.07)
    det.reset()
    lm_drift = make_curl_landmarks(160.0, drift=0.12, torso_shift=0.0)
    m_drift = det.process(lm_drift, current_time=1.0)
    assert m_drift["form_status"] == "Needs Correction"
    assert m_drift["shoulder_status"] == "ELBOW DRIFTING"
    assert "drift" in m_drift["form_issue"].lower()

    # Torso swinging
    det.reset()
    lm_swing = make_curl_landmarks(160.0, drift=0.01, torso_shift=0.15)
    m_swing = det.process(lm_swing, current_time=1.0)
    assert m_swing["form_status"] == "Needs Correction"
    assert m_swing["swing_status"] == "SWINGING"
    assert "momentum" in m_swing["form_issue"].lower() or "swing" in m_swing["form_issue"].lower()


def test_biceps_curls_rep_validation():
    det = BicepsCurlDetector()

    # Valid rep: curls to 45°, lowers to 160°, stable posture
    det.process(make_curl_landmarks(160.0), current_time=1.0)
    det.process(make_curl_landmarks(110.0), current_time=1.3)
    det.process(make_curl_landmarks(45.0), current_time=1.6)
    det.process(make_curl_landmarks(45.0), current_time=1.7)
    det.process(make_curl_landmarks(110.0), current_time=2.1)
    det.process(make_curl_landmarks(160.0), current_time=2.5)
    m = det.process(make_curl_landmarks(160.0), current_time=2.6)

    assert m["rep_incremented"] is True
    assert m["rep_invalid"] is False
    assert m["valid_reps"] == 1
    assert m["invalid_reps"] == 0

    # Invalid rep: curls with swinging torso
    det.process(make_curl_landmarks(160.0, torso_shift=0.15), current_time=3.0)
    det.process(make_curl_landmarks(110.0, torso_shift=0.15), current_time=3.3)
    det.process(make_curl_landmarks(45.0, torso_shift=0.15), current_time=3.6)
    det.process(make_curl_landmarks(45.0, torso_shift=0.15), current_time=3.7)
    det.process(make_curl_landmarks(110.0, torso_shift=0.15), current_time=4.1)
    det.process(make_curl_landmarks(160.0, torso_shift=0.15), current_time=4.5)
    m_bad = det.process(make_curl_landmarks(160.0, torso_shift=0.15), current_time=4.6)

    assert m_bad["rep_incremented"] is False
    assert m_bad["rep_invalid"] is True
    assert m_bad["valid_reps"] == 1
    assert m_bad["invalid_reps"] == 1
    assert "swing" in m_bad["form_issue"].lower() or "momentum" in m_bad["form_issue"].lower()


# ==============================================================================
# 5. LUNGES FORM CHECKS & REP VALIDATION
# ==============================================================================
def make_lunge_landmarks(front_knee_angle=160.0, torso_shift=0.0):
    lm = create_landmarks()
    # Torso
    lm[11] = MockLandmark(0.4 + torso_shift, 0.3)
    lm[12] = MockLandmark(0.6 + torso_shift, 0.3)
    lm[23] = MockLandmark(0.4, 0.6)
    lm[24] = MockLandmark(0.6, 0.6)

    # Left leg (front active leg)
    lm[25] = MockLandmark(0.4, 0.75)
    rad_f = math.radians(180 - front_knee_angle)
    lm[27] = MockLandmark(0.4 + 0.2 * math.sin(rad_f), 0.75 + 0.2 * math.cos(rad_f))

    # Right leg (back leg extended 175°)
    lm[26] = MockLandmark(0.6, 0.75)
    rad_b = math.radians(180 - 175.0)
    lm[28] = MockLandmark(0.6 + 0.2 * math.sin(rad_b), 0.75 + 0.2 * math.cos(rad_b))
    return lm


def test_lunges_form_checks():
    det = LungesDetector()

    # Good form: front knee 160°, upright posture, balanced
    lm_good = make_lunge_landmarks(160.0, torso_shift=0.0)
    m = det.process(lm_good, current_time=1.0)
    assert m["form_status"] == "Good"
    assert m["balance_status"] == "BALANCED"

    # Balance loss
    det.reset()
    lm_balance = make_lunge_landmarks(160.0, torso_shift=0.20)
    m_balance = det.process(lm_balance, current_time=1.0)
    assert m_balance["form_status"] == "Needs Correction"
    assert m_balance["balance_status"] == "OFF BALANCE"
    assert "balance" in m_balance["form_issue"].lower()

    # Knee pushed past toes (angle < 75°)
    det.reset()
    lm_toes = make_lunge_landmarks(68.0, torso_shift=0.0)
    m_toes = det.process(lm_toes, current_time=1.0)
    assert m_toes["form_status"] == "Needs Correction"
    assert "past toes" in m_toes["form_issue"].lower()


def test_lunges_rep_validation():
    det = LungesDetector()

    # Valid rep: drops to 95° depth, returns to 160° standing
    det.process(make_lunge_landmarks(160.0), current_time=1.0)
    det.process(make_lunge_landmarks(125.0), current_time=1.3)
    det.process(make_lunge_landmarks(95.0), current_time=1.6)
    det.process(make_lunge_landmarks(95.0), current_time=1.7)
    det.process(make_lunge_landmarks(125.0), current_time=2.1)
    det.process(make_lunge_landmarks(160.0), current_time=2.5)
    m = det.process(make_lunge_landmarks(160.0), current_time=2.6)

    assert m["rep_incremented"] is True
    assert m["rep_invalid"] is False
    assert m["valid_reps"] == 1
    assert m["invalid_reps"] == 0

    # Invalid rep: drops with balance loss
    det.process(make_lunge_landmarks(160.0, torso_shift=0.20), current_time=3.0)
    det.process(make_lunge_landmarks(125.0, torso_shift=0.20), current_time=3.3)
    det.process(make_lunge_landmarks(95.0, torso_shift=0.20), current_time=3.6)
    det.process(make_lunge_landmarks(95.0, torso_shift=0.20), current_time=3.7)
    det.process(make_lunge_landmarks(125.0, torso_shift=0.20), current_time=4.1)
    det.process(make_lunge_landmarks(160.0, torso_shift=0.20), current_time=4.5)
    m_bad = det.process(make_lunge_landmarks(160.0, torso_shift=0.20), current_time=4.6)

    assert m_bad["rep_incremented"] is False
    assert m_bad["rep_invalid"] is True
    assert m_bad["valid_reps"] == 1
    assert m_bad["invalid_reps"] == 1
    assert "balance" in m_bad["form_issue"].lower()


# ==============================================================================
# 6. OCCLUSION AND NOISY LANDMARKS (Unable to Assess)
# ==============================================================================
def test_occluded_landmarks_safety():
    for det_class in [ShoulderPressDetector, SquatDetector, PushUpDetector, BicepsCurlDetector, LungesDetector]:
        det = det_class()
        occluded_lm = create_landmarks(visibility=0.10)
        m = det.process(occluded_lm, current_time=1.0)
        assert m["form_status"] == "Unable to Assess"
        assert m["rep_incremented"] is False


# ==============================================================================
# 7. WORKOUT STATE MANAGER: VALID VS INVALID REPS
# ==============================================================================
def test_workout_state_manager_valid_vs_invalid():
    manager = WorkoutStateManager()
    manager.start_workout("Squats", 1, 3)

    # Valid rep 1
    rec, ev = manager.record_rep(timestamp=1.0)
    assert rec is True
    assert manager.total_reps == 1
    assert manager.invalid_reps == 0

    # Rejected movement attempt (invalid rep)
    manager.record_invalid_rep(reason="Insufficient squat depth", correction="Squat a little deeper.")
    assert manager.total_reps == 1  # Valid reps DID NOT increment!
    assert manager.invalid_reps == 1
    assert manager.last_form_status == "Needs Correction"
    assert "depth" in manager.last_form_issue.lower()

    # Valid rep 2
    rec, ev = manager.record_rep(timestamp=2.0)
    assert rec is True
    assert manager.total_reps == 2
    assert manager.invalid_reps == 1

    # Snapshot check
    snap = manager.get_snapshot()
    assert snap["valid_reps"] == 2
    assert snap["total_reps"] == 2
    assert snap["invalid_reps"] == 1
    assert snap["is_completed"] is False


# ==============================================================================
# 8. VOICE CORRECTION EVENT AND DEBOUNCE
# ==============================================================================
def test_voice_correction_generation_and_cooldown():
    coach = LLMCoach(groq_client=None)
    tts = TextToSpeech()
    fake_audio = b"\xff\xfb\x90\x00" + b"\x00" * 150
    tts._cache["en:don't arch your lower back. keep your core tight."] = fake_audio
    tts._cache["en:good form! rep completed."] = fake_audio

    pipeline = VoicePipeline(coach, tts)

    # First form violation generates audio correction
    metrics_flaw = {
        "form_status": "Needs Correction",
        "form_issue": "Excessive lower-back arch",
        "form_correction": "Don't arch your lower back. Brace your core.",
    }
    res1 = pipeline.process_event("ongoing_form_check", "Shoulder Press", metrics_flaw)
    assert res1 is not None
    audio1, text1, id1 = res1
    assert "arch" in text1.lower()

    # Immediate second call within 3.5s cooldown is suppressed
    res2 = pipeline.process_event("ongoing_form_check", "Shoulder Press", metrics_flaw)
    assert res2 is None

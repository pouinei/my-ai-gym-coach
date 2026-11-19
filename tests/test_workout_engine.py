import os
import sys
import tempfile
import sqlite3
from pathlib import Path
import pytest

# Ensure Main App is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_APP_DIR = PROJECT_ROOT / "Main App"
if str(MAIN_APP_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_APP_DIR))

from core.rep_state_machine import RepetitionStateMachine, Direction, Stage
from services.state.workout_state import WorkoutStateManager, WorkoutStatus
from detectors.shoulder_press import ShoulderPressDetector
from detectors.squat import SquatDetector
from detectors.pushup import PushUpDetector
from detectors.biceps_curl import BicepsCurlDetector
from detectors.lunges import LungesDetector
from services.persistence.exercise_repository import init_db, save_workout_session, add_exercise, get_users_sessions, get_users_exercises


class MockLandmark:
    def __init__(self, x=0.5, y=0.5, z=0.0, visibility=0.9):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def create_landmarks(num_landmarks=33, visibility=0.95):
    return [MockLandmark(0.5, 0.5, 0.0, visibility) for _ in range(num_landmarks)]


def set_arm_landmarks(landmarks, side="left", elbow_angle_deg=90.0, shoulder_y=0.3, visibility=0.95):
    """
    Sets landmarks for shoulder, elbow, wrist to achieve a specific elbow angle.
    """
    import math
    if side == "left":
        s_idx, e_idx, w_idx = 11, 13, 15
    else:
        s_idx, e_idx, w_idx = 12, 14, 16

    landmarks[s_idx] = MockLandmark(0.5, shoulder_y, 0.0, visibility)
    landmarks[e_idx] = MockLandmark(0.5, shoulder_y + 0.2, 0.0, visibility)

    # Calculate wrist position for desired angle at elbow
    rad = math.radians(elbow_angle_deg)
    # Vector from elbow to shoulder is (0, -0.2)
    # Rotate vector by (180 - elbow_angle)
    dx = 0.2 * math.sin(math.pi - rad)
    dy = -0.2 * math.cos(math.pi - rad)
    landmarks[w_idx] = MockLandmark(0.5 + dx, (shoulder_y + 0.2) + dy, 0.0, visibility)


# ==============================================================================
# TEST A: Basic 1x3 workout
# 1 set x 3 reps -> terminates at exactly 3 reps and 1 set
# ==============================================================================
def test_a_basic_1x3_workout():
    manager = WorkoutStateManager()
    manager.start_workout(exercise_name="Shoulder Press", target_sets=1, reps_per_set=3)

    assert manager.is_active()
    assert manager.total_reps == 0
    assert manager.sets_completed == 0

    # Rep 1
    recorded, event = manager.record_rep(timestamp=1.0)
    assert recorded is True
    assert event is None
    assert manager.total_reps == 1
    assert manager.current_set_reps == 1
    assert manager.sets_completed == 0

    # Rep 2
    recorded, event = manager.record_rep(timestamp=2.0)
    assert recorded is True
    assert event is None
    assert manager.total_reps == 2
    assert manager.current_set_reps == 2
    assert manager.sets_completed == 0

    # Rep 3 -> Workout completed!
    recorded, event = manager.record_rep(timestamp=3.0)
    assert recorded is True
    assert event == "workout_completed"
    assert manager.total_reps == 3
    assert manager.current_set_reps == 3
    assert manager.sets_completed == 1
    assert manager.is_completed()
    assert manager.status == WorkoutStatus.COMPLETED


# ==============================================================================
# TEST B: Duplicate frame rejection
# Consecutive identical frames do not trigger multiple reps
# ==============================================================================
def test_b_duplicate_frame_rejection():
    sm = RepetitionStateMachine(
        start_threshold=160.0,
        inflection_threshold=100.0,
        direction=Direction.START_HIGH_INFLECT_LOW,
        min_rep_duration=0.8,
        consecutive_frames=2,
    )

    t = 0.0
    # User standing still at 170° for 20 identical frames
    for _ in range(20):
        t += 0.033
        assert sm.update(170.0, current_time=t) is False

    assert sm.reps == 0

    # User descends to 90° and stays there for 20 frames
    for _ in range(20):
        t += 0.033
        assert sm.update(90.0, current_time=t) is False

    assert sm.reps == 0
    assert sm.stage == Stage.INFLECTED


# ==============================================================================
# TEST C: Incomplete movement rejection (Half-rep)
# Half-rep does not increment counter
# ==============================================================================
def test_c_incomplete_movement_rejection():
    sm = RepetitionStateMachine(
        start_threshold=160.0,
        inflection_threshold=100.0,
        direction=Direction.START_HIGH_INFLECT_LOW,
        min_rep_duration=0.8,
        consecutive_frames=2,
    )

    t = 1.0
    sm.update(165.0, current_time=t)

    # Half squat: drops to 125° (does not reach <= 100°)
    t += 0.3
    sm.update(135.0, current_time=t)
    t += 0.3
    sm.update(125.0, current_time=t)
    t += 0.3
    sm.update(125.0, current_time=t)

    # Returns to standing at 165°
    t += 0.3
    sm.update(150.0, current_time=t)
    t += 0.3
    completed = sm.update(165.0, current_time=t)
    t += 0.3
    completed2 = sm.update(165.0, current_time=t)

    assert completed is False
    assert completed2 is False
    assert sm.reps == 0  # Half rep rejected!


# ==============================================================================
# TEST D: Noisy pose estimation
# Rapid frame-to-frame angle jitter around threshold is filtered
# ==============================================================================
def test_d_noisy_pose_estimation_filtered():
    sm = RepetitionStateMachine(
        start_threshold=95.0,
        inflection_threshold=155.0,
        direction=Direction.START_LOW_INFLECT_HIGH,
        min_rep_duration=0.8,
        consecutive_frames=2,
    )

    t = 1.0
    # Jittering around 95°: 94, 96, 95, 96, 94
    for val in [94.0, 96.0, 95.0, 96.0, 94.0, 95.0]:
        t += 0.033
        completed = sm.update(val, current_time=t)
        assert completed is False

    assert sm.reps == 0


# ==============================================================================
# TEST E: Tracking loss resilience
# Temporary landmark drop / low confidence does not trigger false rep
# ==============================================================================
def test_e_tracking_loss_resilience():
    detector = ShoulderPressDetector()
    landmarks = create_landmarks()

    # Normal starting position with high visibility
    set_arm_landmarks(landmarks, side="left", elbow_angle_deg=90.0, visibility=0.9)
    res = detector.process(landmarks, current_time=1.0)
    assert res["reps"] == 0

    # Sudden tracking loss: landmarks visibility drops to 0.1
    for idx in [11, 13, 15, 12, 14, 16]:
        landmarks[idx].visibility = 0.1

    # In tracking loss, angle math or noise should not increment
    res = detector.process(landmarks, current_time=1.1)
    assert res["reps"] == 0

    # Tracking restores
    for idx in [11, 13, 15, 12, 14, 16]:
        landmarks[idx].visibility = 0.9
    set_arm_landmarks(landmarks, side="left", elbow_angle_deg=90.0, visibility=0.9)
    res = detector.process(landmarks, current_time=1.2)
    assert res["reps"] == 0


# ==============================================================================
# TEST F: Extra movement after workout completion
# Reps after 1x3 are strictly ignored, total remains 3
# ==============================================================================
def test_f_extra_movement_after_completion():
    manager = WorkoutStateManager()
    manager.start_workout(exercise_name="Squats", target_sets=1, reps_per_set=3)

    # Complete 3 reps
    manager.record_rep(timestamp=1.0)
    manager.record_rep(timestamp=2.0)
    recorded, event = manager.record_rep(timestamp=3.0)
    assert manager.is_completed()
    assert manager.total_reps == 3

    # Additional movements after completion (e.g. stepping away or putting down weights)
    rec1, ev1 = manager.record_rep(timestamp=4.0)
    rec2, ev2 = manager.record_rep(timestamp=5.0)
    rec3, ev3 = manager.record_rep(timestamp=6.0)

    assert rec1 is False
    assert rec2 is False
    assert rec3 is False
    assert ev1 is None
    assert manager.total_reps == 3
    assert manager.sets_completed == 1
    assert manager.current_set_reps == 3


# ==============================================================================
# TEST G: Workout reset
# Reset restores 0, start new workout counts from 0 cleanly
# ==============================================================================
def test_g_workout_reset():
    detector = SquatDetector()
    manager = WorkoutStateManager()

    manager.start_workout(exercise_name="Squats", target_sets=1, reps_per_set=3)
    manager.record_rep(timestamp=1.0)
    assert manager.total_reps == 1

    # Reset
    detector.reset()
    manager.reset()

    assert detector.reps == 0
    assert manager.total_reps == 0
    assert manager.sets_completed == 0
    assert manager.current_set_reps == 0
    assert manager.status == WorkoutStatus.IDLE

    # Start fresh workout
    manager.start_workout(exercise_name="Squats", target_sets=1, reps_per_set=3)
    assert manager.total_reps == 0
    assert manager.is_active()


# ==============================================================================
# TEST H: Duplicate end workout calls
# Idempotent database saving; only 1 session saved
# ==============================================================================
def test_h_duplicate_end_workout_calls():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        init_db(db_path=":memory:")
        # Execute table setup directly on this memory connection
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS exercises (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                exercise_name TEXT NOT NULL,
                reps INTEGER NOT NULL DEFAULT 0,
                sets INTEGER NOT NULL DEFAULT 0,
                time INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workout_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL REFERENCES users(id),
                exercise_name TEXT NOT NULL,
                target_sets INTEGER NOT NULL DEFAULT 0,
                target_reps INTEGER NOT NULL DEFAULT 0,
                sets_completed INTEGER NOT NULL DEFAULT 0,
                total_reps INTEGER NOT NULL DEFAULT 0,
                duration_seconds INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'COMPLETED',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create user
        conn.execute("INSERT INTO users (id, username) VALUES (1, 'testuser')")
        conn.commit()

        session_id = "test-session-uuid-12345"

        # End workout call 1
        save_workout_session(
            user_id=1,
            session_id=session_id,
            exercise_name="Squats",
            target_sets=1,
            target_reps=3,
            sets_completed=1,
            total_reps=3,
            duration_seconds=45,
            status="COMPLETED",
            conn=conn,
        )

        # End workout call 2 (duplicate click)
        save_workout_session(
            user_id=1,
            session_id=session_id,
            exercise_name="Squats",
            target_sets=1,
            target_reps=3,
            sets_completed=1,
            total_reps=3,
            duration_seconds=45,
            status="COMPLETED",
            conn=conn,
        )

        sessions = get_users_sessions(user_id=1, conn=conn)
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == session_id
        assert sessions[0]["total_reps"] == 3
        assert sessions[0]["sets_completed"] == 1
    finally:
        conn.close()


# ==============================================================================
# TEST I: Multi-set workout (2 sets x 3 reps)
# Correctly transitions set 1 -> set 2 -> complete
# ==============================================================================
def test_i_multi_set_workout():
    manager = WorkoutStateManager()
    manager.start_workout(exercise_name="Push-ups", target_sets=2, reps_per_set=3)

    # Set 1
    manager.record_rep(timestamp=1.0)
    manager.record_rep(timestamp=2.0)
    rec, ev = manager.record_rep(timestamp=3.0)

    assert rec is True
    assert ev == "set_completed"
    assert manager.sets_completed == 1
    assert manager.current_set_reps == 0
    assert manager.total_reps == 3
    assert manager.is_active()

    # Set 2
    manager.record_rep(timestamp=4.0)
    manager.record_rep(timestamp=5.0)
    rec, ev = manager.record_rep(timestamp=6.0)

    assert rec is True
    assert ev == "workout_completed"
    assert manager.sets_completed == 2
    assert manager.current_set_reps == 3
    assert manager.total_reps == 6
    assert manager.is_completed()


# ==============================================================================
# TEST J: Set overflow prevention
# Impossible to display 2/1 sets or 8/3 reps
# ==============================================================================
def test_j_set_overflow_prevention():
    manager = WorkoutStateManager()
    manager.start_workout(exercise_name="Shoulder Press", target_sets=1, reps_per_set=3)

    # Simulate 8 incoming rep events
    for i in range(1, 9):
        manager.record_rep(timestamp=float(i))

    snapshot = manager.get_snapshot()

    # Enforce strictly:
    assert snapshot["sets_completed"] == 1  # NEVER 2
    assert snapshot["current_set_reps"] == 3  # NEVER 2/3
    assert snapshot["total_reps"] == 3       # NEVER 8
    assert snapshot["target_sets"] == 1
    assert snapshot["reps_per_set"] == 3
    assert snapshot["is_completed"] is True


# ==============================================================================
# TEST K: Full Repetition Cycle on All 5 Detectors
# ==============================================================================
def test_k_all_detectors_full_rep():
    # 1. Squats
    sq = SquatDetector()
    sm = sq.state_machine
    assert sm.update(165.0, current_time=1.0) is False
    assert sm.update(140.0, current_time=1.2) is False
    assert sm.update(95.0, current_time=1.4) is False
    assert sm.update(95.0, current_time=1.5) is False
    assert sm.stage == Stage.INFLECTED
    assert sm.update(120.0, current_time=1.8) is False
    assert sm.update(162.0, current_time=2.4) is False
    assert sm.update(162.0, current_time=2.5) is True
    assert sm.reps == 1

    # 2. Shoulder Press
    sp = ShoulderPressDetector()
    sm = sp.state_machine
    assert sm.update(90.0, current_time=1.0) is False
    assert sm.update(115.0, current_time=1.2) is False
    assert sm.update(160.0, current_time=1.4) is False
    assert sm.update(160.0, current_time=1.5) is False
    assert sm.stage == Stage.INFLECTED
    assert sm.update(135.0, current_time=1.8) is False
    assert sm.update(90.0, current_time=2.4) is False
    assert sm.update(90.0, current_time=2.5) is True
    assert sm.reps == 1

    # 3. Push-ups
    pu = PushUpDetector()
    sm = pu.state_machine
    assert sm.update(160.0, current_time=1.0) is False
    assert sm.update(85.0, current_time=1.4) is False
    assert sm.update(85.0, current_time=1.5) is False
    assert sm.update(160.0, current_time=2.4) is False
    assert sm.update(160.0, current_time=2.5) is True
    assert sm.reps == 1

    # 4. Biceps Curl
    bc = BicepsCurlDetector()
    sm = bc.state_machine
    assert sm.update(155.0, current_time=1.0) is False
    assert sm.update(45.0, current_time=1.4) is False
    assert sm.update(45.0, current_time=1.5) is False
    assert sm.update(155.0, current_time=2.4) is False
    assert sm.update(155.0, current_time=2.5) is True
    assert sm.reps == 1

    # 5. Lunges
    lg = LungesDetector()
    sm = lg.state_machine
    assert sm.update(160.0, current_time=1.0) is False
    assert sm.update(95.0, current_time=1.4) is False
    assert sm.update(95.0, current_time=1.5) is False
    assert sm.update(160.0, current_time=2.4) is False
    assert sm.update(160.0, current_time=2.5) is True
    assert sm.reps == 1


# ==============================================================================
# TEST L: Metrics Sync Strict Capping and Invariant Protection
# ==============================================================================
def test_l_sync_metrics_safety():
    import streamlit as st
    from services.persistence.exercise_repository import init_db
    from services.tracking.metrics import sync_metrics_update

    init_db()

    class MockContextState:
        playing = True

    class MockProcessor:
        def __init__(self):
            self.workout_manager = WorkoutStateManager()
            self.workout_manager.start_workout("Shoulder Press", 1, 3)
            # Simulate processor holding over-reported raw metrics
            self._metrics = {
                "reps": 8,
                "sets_completed": 2,
                "current_set_reps": 2,
                "workout_completed": True,
            }

        def set_exercise(self, ex):
            pass

        def start_workout(self, ex, sets, reps):
            self.workout_manager.start_workout(ex, sets, reps)

        def get_latest_metrics(self):
            return self._metrics.copy()

    class MockContext:
        state = MockContextState()
        video_processor = MockProcessor()

    st.session_state["exercise_type"] = "Shoulder Press"
    st.session_state["target_sets"] = 1
    st.session_state["reps_per_set"] = 3
    st.session_state["reps"] = 0
    st.session_state["sets_completed"] = 0
    st.session_state["current_set_reps"] = 0
    st.session_state["workout_completed"] = False
    st.session_state["workout_session_id"] = "fixed-session-id"

    ctx = MockContext()
    sync_metrics_update(ctx)

    # Invariants MUST hold:
    assert st.session_state["sets_completed"] == 1
    assert st.session_state["current_set_reps"] == 3
    assert st.session_state["reps"] == 3
    assert st.session_state["workout_completed"] is True


# ==============================================================================
# TEST M: WebRTC Symbols and Configuration Availability
# ==============================================================================
def test_m_webrtc_symbols_and_config():
    import main
    assert hasattr(main, "webrtc_streamer")
    assert main.webrtc_streamer is not None
    assert hasattr(main, "WebRtcMode")
    assert main.WebRtcMode is not None
    assert hasattr(main, "VideoProcessorClass")
    assert hasattr(main, "sync_metrics_update")

    from services.config.webrtc_config import get_rtc_configuration, DEFAULT_MEDIA_CONSTRAINTS
    cfg = get_rtc_configuration()
    assert "iceServers" in cfg
    assert len(cfg["iceServers"]) > 0
    assert "video" in DEFAULT_MEDIA_CONSTRAINTS



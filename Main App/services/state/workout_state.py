import time
import uuid
import threading
from enum import Enum
from typing import Optional, Tuple, Dict, Any


class WorkoutStatus(Enum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    ENDED = "ENDED"


class WorkoutStateManager:
    """
    Authoritative, thread-safe Workout State Manager.
    Enforces strict invariants:
    - 0 <= current_set_reps <= reps_per_set
    - 0 <= sets_completed <= target_sets
    - total_reps <= target_sets * reps_per_set
    - Once COMPLETED or ENDED, further rep counting is strictly rejected.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.session_id: str = uuid.uuid4().hex
        self.exercise_name: str = "Squats"
        self.target_sets: int = 0
        self.reps_per_set: int = 0
        self.sets_completed: int = 0
        self.current_set_reps: int = 0
        self.total_reps: int = 0
        self.status: WorkoutStatus = WorkoutStatus.IDLE
        self.started_at: float = 0.0
        self.completed_at: float = 0.0
        self.set_started_at: float = 0.0
        self.last_rep_timestamp: float = 0.0

    def start_workout(self, exercise_name: str, target_sets: int, reps_per_set: int) -> None:
        with self._lock:
            self.session_id = uuid.uuid4().hex
            self.exercise_name = exercise_name
            self.target_sets = max(0, int(target_sets))
            self.reps_per_set = max(0, int(reps_per_set))
            self.sets_completed = 0
            self.current_set_reps = 0
            self.total_reps = 0
            now = time.time()
            self.started_at = now
            self.set_started_at = now
            self.completed_at = 0.0
            self.last_rep_timestamp = 0.0

            if self.target_sets > 0 and self.reps_per_set > 0:
                self.status = WorkoutStatus.ACTIVE
            else:
                self.status = WorkoutStatus.IDLE

    def record_rep(self, timestamp: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        """
        Record a single repetition.
        Returns (recorded: bool, event_name: Optional[str]).
        Events: 'set_completed', 'workout_completed', or None.
        """
        with self._lock:
            if self.status != WorkoutStatus.ACTIVE:
                return False, None

            if timestamp is None:
                timestamp = time.time()

            self.last_rep_timestamp = timestamp

            # Increment within set
            self.current_set_reps += 1
            self.total_reps += 1
            event = None

            # Check if current set has reached target reps
            if self.current_set_reps >= self.reps_per_set:
                self.sets_completed += 1

                # Check if entire workout is complete
                if self.sets_completed >= self.target_sets:
                    self.sets_completed = self.target_sets
                    self.current_set_reps = self.reps_per_set
                    self.total_reps = self.target_sets * self.reps_per_set
                    self.status = WorkoutStatus.COMPLETED
                    self.completed_at = timestamp
                    event = "workout_completed"
                else:
                    # Set completed, prepare for next set
                    event = "set_completed"
                    self.current_set_reps = 0
                    self.set_started_at = timestamp

            # Invariant assertions
            self._validate_invariants()
            return True, event

    def end_workout(self) -> Dict[str, Any]:
        with self._lock:
            if self.status not in (WorkoutStatus.COMPLETED, WorkoutStatus.ENDED):
                self.status = WorkoutStatus.ENDED
                self.completed_at = time.time()

            return self.get_snapshot()

    def reset(self) -> None:
        with self._lock:
            self.session_id = uuid.uuid4().hex
            self.target_sets = 0
            self.reps_per_set = 0
            self.sets_completed = 0
            self.current_set_reps = 0
            self.total_reps = 0
            self.status = WorkoutStatus.IDLE
            self.started_at = 0.0
            self.completed_at = 0.0
            self.set_started_at = 0.0
            self.last_rep_timestamp = 0.0

    def is_active(self) -> bool:
        with self._lock:
            return self.status == WorkoutStatus.ACTIVE

    def is_completed(self) -> bool:
        with self._lock:
            return self.status == WorkoutStatus.COMPLETED

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            elapsed = 0.0
            if self.started_at > 0:
                end_time = self.completed_at if self.completed_at > 0 else time.time()
                elapsed = max(0.0, end_time - self.started_at)

            return {
                "session_id": self.session_id,
                "exercise_name": self.exercise_name,
                "target_sets": self.target_sets,
                "reps_per_set": self.reps_per_set,
                "sets_completed": self.sets_completed,
                "current_set_reps": self.current_set_reps,
                "total_reps": self.total_reps,
                "status": self.status.value,
                "is_active": self.status == WorkoutStatus.ACTIVE,
                "is_completed": self.status == WorkoutStatus.COMPLETED,
                "elapsed_seconds": int(elapsed),
                "started_at": self.started_at,
                "completed_at": self.completed_at,
            }

    def _validate_invariants(self) -> None:
        if self.target_sets > 0:
            assert 0 <= self.sets_completed <= self.target_sets, (
                f"Invariant violation: sets_completed={self.sets_completed} exceeds target_sets={self.target_sets}"
            )
        if self.reps_per_set > 0:
            assert 0 <= self.current_set_reps <= self.reps_per_set, (
                f"Invariant violation: current_set_reps={self.current_set_reps} exceeds reps_per_set={self.reps_per_set}"
            )
        if self.status == WorkoutStatus.COMPLETED:
            assert self.sets_completed == self.target_sets, (
                f"Invariant violation: completed workout sets={self.sets_completed} != target={self.target_sets}"
            )

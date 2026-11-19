import time
from enum import Enum
from typing import Optional


class Direction(Enum):
    START_HIGH_INFLECT_LOW = 1  # Squat, Push-up, Biceps Curl, Lunges
    START_LOW_INFLECT_HIGH = 2  # Shoulder Press


class Stage(Enum):
    STARTING = "starting"
    MOVING_TO_INFLECTION = "moving"
    INFLECTED = "inflected"
    RETURNING = "returning"


class RepetitionStateMachine:
    """
    Deterministic repetition counter using a multi-state machine with:
    - Directional awareness (high->low->high or low->high->low)
    - Hysteresis bands to prevent oscillation at thresholds
    - Consecutive frame confirmation to eliminate single-frame glitches
    - Minimum duration (debounce) between completed repetitions
    - Half-rep and noisy jump rejection
    """

    def __init__(
        self,
        start_threshold: float,
        inflection_threshold: float,
        direction: Direction = Direction.START_HIGH_INFLECT_LOW,
        hysteresis: float = 15.0,
        min_rep_duration: float = 0.8,
        consecutive_frames: int = 2,
    ):
        self.start_threshold = float(start_threshold)
        self.inflection_threshold = float(inflection_threshold)
        self.direction = direction
        self.hysteresis = float(hysteresis)
        self.min_rep_duration = float(min_rep_duration)
        self.required_consecutive_frames = int(consecutive_frames)

        self.stage = Stage.STARTING
        self.reps = 0
        self.last_rep_time = 0.0
        self._consecutive_count = 0
        self._last_angle = None

    def reset(self) -> None:
        self.stage = Stage.STARTING
        self.reps = 0
        self.last_rep_time = 0.0
        self._consecutive_count = 0
        self._last_angle = None

    @property
    def stage_name(self) -> str:
        if self.stage == Stage.STARTING:
            return "up" if self.direction == Direction.START_HIGH_INFLECT_LOW else "down"
        elif self.stage == Stage.INFLECTED:
            return "down" if self.direction == Direction.START_HIGH_INFLECT_LOW else "up"
        elif self.stage == Stage.MOVING_TO_INFLECTION:
            return "descending" if self.direction == Direction.START_HIGH_INFLECT_LOW else "ascending"
        elif self.stage == Stage.RETURNING:
            return "ascending" if self.direction == Direction.START_HIGH_INFLECT_LOW else "descending"
        return "idle"

    def update(self, angle: float, current_time: Optional[float] = None) -> bool:
        """
        Process the current joint angle.
        Returns True if a repetition was completed on this update, False otherwise.
        """
        if current_time is None:
            current_time = time.time()

        if angle is None or angle < 0 or angle > 180:
            return False

        self._last_angle = angle
        rep_completed = False

        if self.direction == Direction.START_HIGH_INFLECT_LOW:
            rep_completed = self._update_high_low_high(angle, current_time)
        else:
            rep_completed = self._update_low_high_low(angle, current_time)

        return rep_completed

    def _update_high_low_high(self, angle: float, current_time: float) -> bool:
        """
        Movement starts at high angle (standing/extended),
        drops to low angle (inflection/deep), and returns to high angle.
        Example: Squats, Pushups, Biceps Curls, Lunges.
        """
        rep_completed = False

        if self.stage == Stage.STARTING:
            # Moving towards inflection: must drop below start_threshold - hysteresis
            if angle <= (self.start_threshold - self.hysteresis):
                self.stage = Stage.MOVING_TO_INFLECTION
                self._consecutive_count = 0

        if self.stage == Stage.MOVING_TO_INFLECTION:
            # Check if reaching inflection point
            if angle <= self.inflection_threshold:
                self._consecutive_count += 1
                if self._consecutive_count >= self.required_consecutive_frames:
                    self.stage = Stage.INFLECTED
                    self._consecutive_count = 0
            else:
                self._consecutive_count = 0
                # If aborted and returned to start without reaching inflection (half-rep)
                if angle >= (self.start_threshold - 5.0):
                    self.stage = Stage.STARTING

        elif self.stage == Stage.INFLECTED:
            # Must rise above inflection + hysteresis to begin return
            if angle >= (self.inflection_threshold + self.hysteresis):
                self.stage = Stage.RETURNING
                self._consecutive_count = 0

        if self.stage == Stage.RETURNING:
            # Check if reached start position
            if angle >= (self.start_threshold - 5.0):
                self._consecutive_count += 1
                if self._consecutive_count >= self.required_consecutive_frames:
                    time_since_last = current_time - self.last_rep_time
                    if time_since_last >= self.min_rep_duration:
                        self.reps += 1
                        self.last_rep_time = current_time
                        rep_completed = True
                    self.stage = Stage.STARTING
                    self._consecutive_count = 0
            else:
                self._consecutive_count = 0
                # If reversed back down before completing return
                if angle <= self.inflection_threshold:
                    self.stage = Stage.INFLECTED

        return rep_completed

    def _update_low_high_low(self, angle: float, current_time: float) -> bool:
        """
        Movement starts at low angle (hands at shoulder height),
        presses up to high angle (overhead extension), and returns down to low angle.
        Example: Shoulder Press.
        """
        rep_completed = False

        if self.stage == Stage.STARTING:
            # Moving towards overhead: must rise above start_threshold + hysteresis
            if angle >= (self.start_threshold + self.hysteresis):
                self.stage = Stage.MOVING_TO_INFLECTION
                self._consecutive_count = 0

        if self.stage == Stage.MOVING_TO_INFLECTION:
            # Check if reaching overhead extension
            if angle >= self.inflection_threshold:
                self._consecutive_count += 1
                if self._consecutive_count >= self.required_consecutive_frames:
                    self.stage = Stage.INFLECTED
                    self._consecutive_count = 0
            else:
                self._consecutive_count = 0
                # If aborted and returned to start without full extension
                if angle <= (self.start_threshold + 5.0):
                    self.stage = Stage.STARTING

        elif self.stage == Stage.INFLECTED:
            # Must lower below inflection - hysteresis to begin return
            if angle <= (self.inflection_threshold - self.hysteresis):
                self.stage = Stage.RETURNING
                self._consecutive_count = 0

        if self.stage == Stage.RETURNING:
            # Check if returned to shoulder level
            if angle <= (self.start_threshold + 5.0):
                self._consecutive_count += 1
                if self._consecutive_count >= self.required_consecutive_frames:
                    time_since_last = current_time - self.last_rep_time
                    if time_since_last >= self.min_rep_duration:
                        self.reps += 1
                        self.last_rep_time = current_time
                        rep_completed = True
                    self.stage = Stage.STARTING
                    self._consecutive_count = 0
            else:
                self._consecutive_count = 0
                # If pressed back up before completing lowering
                if angle >= self.inflection_threshold:
                    self.stage = Stage.INFLECTED

        return rep_completed

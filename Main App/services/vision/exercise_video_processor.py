import os
import cv2
import av
import numpy as np
import mediapipe as mp
import threading
from pathlib import Path
from streamlit_webrtc import VideoProcessorBase
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from detectors.squat import SquatDetector
from detectors.pushup import PushUpDetector
from detectors.biceps_curl import BicepsCurlDetector
from detectors.shoulder_press import ShoulderPressDetector
from detectors.lunges import LungesDetector
from services.config.workout_config import POSE_CONNECTIONS
from services.state.workout_state import WorkoutStateManager


class VideoProcessorClass(VideoProcessorBase):
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_metrics = None
        self._exercise_type = "Squats"
        self.workout_manager = WorkoutStateManager()

        base_dir = Path(__file__).resolve().parent.parent.parent
        resolved_model_path = base_dir / "ml_models" / "pose_landmarker_full.task"
        model_path = str(resolved_model_path) if resolved_model_path.exists() else os.path.join(os.getcwd(), "ml_models", "pose_landmarker_full.task")
        base_option = python.BaseOptions(model_asset_path=model_path)

        options = vision.PoseLandmarkerOptions(
            base_options=base_option,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            output_segmentation_masks=False,
        )

        self._landmarker = vision.PoseLandmarker.create_from_options(options)

        self._detectors = {
            "Squats": SquatDetector(),
            "Push-ups": PushUpDetector(),
            "Biceps Curls (Dumbbell)": BicepsCurlDetector(),
            "Shoulder Press": ShoulderPressDetector(),
            "Lunges": LungesDetector(),
        }

        self._frame_timestamps_ms = 0

    def start_workout(self, exercise_name: str, target_sets: int, reps_per_set: int) -> None:
        with self._lock:
            self._exercise_type = exercise_name
            self.workout_manager.start_workout(exercise_name, target_sets, reps_per_set)
            if exercise_name in self._detectors:
                self._detectors[exercise_name].reset()

    def stop_workout(self) -> None:
        with self._lock:
            self.workout_manager.end_workout()

    def reset_detectors(self) -> None:
        with self._lock:
            for det in self._detectors.values():
                det.reset()
            self.workout_manager.reset()

    def set_latest_metrics(self, metrics):
        with self._lock:
            self._latest_metrics = metrics.copy()

    def get_latest_metrics(self):
        with self._lock:
            return None if self._latest_metrics is None else self._latest_metrics.copy()

    def set_exercise(self, exercise_type):
        with self._lock:
            if self._exercise_type != exercise_type:
                self._exercise_type = exercise_type
                if exercise_type in self._detectors:
                    self._detectors[exercise_type].reset()

    def get_exercise(self):
        with self._lock:
            return self._exercise_type

    def _draw_skeleton(self, img, landmarks):
        h, w = img.shape[:2]

        for start_idx, end_idx in POSE_CONNECTIONS:
            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]

            if p1.visibility > 0.7 and p2.visibility > 0.7:
                cv2.line(
                    img,
                    (int(p1.x * w), int(p1.y * h)),
                    (int(p2.x * w), int(p2.y * h)),
                    (0, 255, 0),
                    8,
                )

        for lm in landmarks:
            if lm.visibility > 0.7:
                cv2.circle(
                    img,
                    (int(lm.x * w), int(lm.y * h)),
                    8,
                    (255, 0, 0),
                    -1,
                )

    def _draw_no_pose_warnings(self, img):
        cv2.putText(
            img,
            "NO POSE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            "PLEASE FACE THE CAMERA",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def _draw_completion_banner(self, img):
        h, w = img.shape[:2]
        center_y = h // 2
        overlay = img.copy()
        cv2.rectangle(overlay, (20, center_y - 45), (w - 20, center_y + 45), (0, 140, 0), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
        cv2.putText(
            img,
            "WORKOUT COMPLETED!",
            (35, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "GREAT JOB - TARGET ACHIEVED",
            (35, center_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_overlays(self, img, metrics, ex_type):
        if metrics.get("workout_completed", False):
            self._draw_completion_banner(img)
            return

        if ex_type == "Squats":
            self._draw_squats_overlays(img, metrics)
        elif ex_type == "Push-ups":
            self._draw_pushup_overlays(img, metrics)
        elif ex_type == "Biceps Curls (Dumbbell)":
            self._draw_curl_overlays(img, metrics)
        elif ex_type == "Shoulder Press":
            self._draw_press_overlays(img, metrics)
        elif ex_type == "Lunges":
            self._draw_lunge_overlays(img, metrics)

    def _draw_squats_overlays(self, img, metrics):
        h, _ = img.shape[:2]
        depth = metrics.get("depth_status", "N/A")
        cv2.putText(
            img,
            f"DEPTH: {depth}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    def _draw_pushup_overlays(self, img, metrics):
        h, _ = img.shape[:2]
        body = metrics.get("body_alignment", "N/A")
        hip = metrics.get("hip_status", "N/A")
        cv2.putText(
            img,
            f"BODY: {body} | HIP: {hip}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    def _draw_curl_overlays(self, img, metrics):
        h, _ = img.shape[:2]
        swing = metrics.get("swing_status", "N/A")
        cv2.putText(
            img,
            f"SWING: {swing}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    def _draw_press_overlays(self, img, metrics):
        h, _ = img.shape[:2]
        ext = metrics.get("extension_status", "N/A")
        back = metrics.get("back_arch_status", "N/A")
        cv2.putText(
            img,
            f"EXT: {ext} | BACK: {back}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    def _draw_lunge_overlays(self, img, metrics):
        h, _ = img.shape[:2]
        bal = metrics.get("balance_status", "N/A")
        cv2.putText(
            img,
            f"BALANCE: {bal}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    def recv(self, frame):
        image = np.asarray(
            cv2.flip(frame.to_ndarray(format="bgr24"), 1),
            dtype=np.uint8,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )

        self._frame_timestamps_ms += 30
        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamps_ms)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            self._draw_skeleton(image, landmarks)
            ex_type = self.get_exercise()
            detector = self._detectors.get(ex_type)

            if detector:
                metrics = detector.process(landmarks)

                # Process repetition through workout manager if active
                if metrics.get("rep_incremented", False):
                    self.workout_manager.record_rep()

                # Authoritative state synchronization from WorkoutStateManager
                snapshot = self.workout_manager.get_snapshot()
                metrics["reps"] = snapshot["total_reps"]
                metrics["sets_completed"] = snapshot["sets_completed"]
                metrics["current_set_reps"] = snapshot["current_set_reps"]
                metrics["workout_completed"] = snapshot["is_completed"]
                metrics["workout_status"] = snapshot["status"]
                metrics["session_id"] = snapshot["session_id"]
                metrics["pose_detected"] = True

                self._draw_overlays(image, metrics, ex_type)
                self.set_latest_metrics(metrics)
        else:
            self._draw_no_pose_warnings(image)
            with self._lock:
                if self._latest_metrics is not None:
                    self._latest_metrics["pose_detected"] = False
                else:
                    self._latest_metrics = {"pose_detected": False}

        return av.VideoFrame.from_ndarray(image, format="bgr24")

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class HandOutput:
    cursor_xy: Optional[Tuple[int, int]]
    gesture: str


class HandDetector:
    def __init__(self):
        base = python.BaseOptions(model_asset_path=self._model_path())
        options = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def _model_path(self) -> str:
        return "assets/models/hand_landmarker.task"

    def process(self, frame_bgr) -> HandOutput:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        res = self.landmarker.detect(mp_image)
        if not res.hand_landmarks:
            return HandOutput(cursor_xy=None, gesture="NONE")

        lm = res.hand_landmarks[0]

        def pt(i):
            return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

        idx_tip = pt(8)
        idx_pip = pt(6)
        mid_tip = pt(12)
        mid_pip = pt(10)
        ring_tip = pt(16)
        ring_pip = pt(14)
        pinky_tip = pt(20)
        pinky_pip = pt(18)

        index_up = idx_tip[1] < idx_pip[1] - 6
        middle_up = mid_tip[1] < mid_pip[1] - 6
        ring_up = ring_tip[1] < ring_pip[1] - 6
        pinky_up = pinky_tip[1] < pinky_pip[1] - 6

        thumb_tip = pt(4)

        pinch_dist = np.linalg.norm(idx_tip - thumb_tip)
        pinch = pinch_dist < 35

        up_count = sum([index_up, middle_up, ring_up, pinky_up])

        if pinch:
            gesture = "PINCH"
        elif up_count == 0:
            gesture = "FIST"
        elif index_up and up_count == 1:
            gesture = "INDEX_UP"
        elif up_count >= 3:
            gesture = "OPEN_PALM"
        else:
            gesture = "OTHER"
        # =========================
        # Gesture classification
        # =========================
        # Here we map hand landmarks to high-level gestures.
        # You can extend this logic to add new gestures such as:
        # - TWO_FINGERS_UP (index + middle)
        # - THUMB_UP / THUMB_DOWN
        # - CUSTOM_SHAPE (e.g. circle gesture)
        #
        # IMPORTANT:
        # This detector should ONLY decide "what gesture is visible".
        # Do NOT bind actions here. Actions are handled later via bindings.

        cursor_xy = (int(idx_tip[0]), int(idx_tip[1]))
        return HandOutput(cursor_xy=cursor_xy, gesture=gesture)


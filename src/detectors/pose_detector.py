from dataclasses import dataclass

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class PoseOutput:
    pose_event: str


class PoseDetector:
    def __init__(self):
        base = python.BaseOptions(model_asset_path=self._model_path())
        options = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._hands_up_hold = 0

    def _model_path(self) -> str:
        return "assets/models/pose_landmarker_full.task"

    def process(self, frame_bgr) -> PoseOutput:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        res = self.landmarker.detect(mp_image)
        if not res.pose_landmarks:
            self._hands_up_hold = 0
            return PoseOutput(pose_event="NONE")

        lm = res.pose_landmarks[0]

        L_WRIST = 15
        R_WRIST = 16
        L_SHOULDER = 11
        R_SHOULDER = 12

        lw_y = lm[L_WRIST].y
        rw_y = lm[R_WRIST].y
        ls_y = lm[L_SHOULDER].y
        rs_y = lm[R_SHOULDER].y

        hands_up = (lw_y < ls_y) and (rw_y < rs_y)

        if hands_up:
            self._hands_up_hold += 1
        else:
            self._hands_up_hold = 0

        if self._hands_up_hold >= 5:
            self._hands_up_hold = 0
            return PoseOutput(pose_event="HANDS_UP")

        return PoseOutput(pose_event="NONE")

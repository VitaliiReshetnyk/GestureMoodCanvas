from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.core.paths import MODELS
from src.core.config import CFG


@dataclass
class FaceOutput:
    expression_state: str
    face_bbox: Optional[Tuple[int, int, int, int]]
    angry_score: float
    happy_score: float
    surprise_score: float


class FaceDetector:
    def __init__(self):
        base = python.BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task"))
        options = vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        self._angry_ema = 0.0
        self._happy_ema = 0.0
        self._surprise_ema = 0.0

        self._hold = 0
        self._held_state = "NEUTRAL"

    def process(self, frame_bgr) -> FaceOutput:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        res = self.landmarker.detect(mp_image)
        if not res.face_landmarks:
            self._angry_ema = 0.0
            self._happy_ema = 0.0
            self._surprise_ema = 0.0
            self._hold = 0
            self._held_state = "NEUTRAL"
            return FaceOutput("NONE", None, 0.0, 0.0, 0.0)

        # ---- bbox from landmarks ----
        lm = res.face_landmarks[0]
        xs = [int(p.x * w) for p in lm]
        ys = [int(p.y * h) for p in lm]
        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2, y2 = min(w - 1, max(xs)), min(h - 1, max(ys))
        bbox = (x1, y1, x2, y2)

        # ---- blendshapes -> dict(name -> score) ----
        bs: Dict[str, float] = {}
        if res.face_blendshapes and len(res.face_blendshapes) > 0:
            cats = res.face_blendshapes[0]
            for c in cats:
                # c.category_name, c.score
                bs[c.category_name] = float(c.score)

        def g(name: str) -> float:
            return bs.get(name, 0.0)

        # ======================================================
        # Facial expression scores (blendshapes-based)
        # ======================================================
        # Each expression is represented as a normalized score [0..1].
        # Scores are computed from MediaPipe Face Blendshapes.
        #
        # To add a new expression:
        # 1) Identify relevant blendshapes (see MediaPipe documentation)
        # 2) Combine them into a new *_raw score
        # 3) Add EMA smoothing
        # 4) Include it in the state selection logic below

        # ---- robust scores (0..1) ----
        # HAPPY: mouth corners up + smile
        happy_raw = 0.50 * (g("mouthSmileLeft") + g("mouthSmileRight")) \
                  + 0.35 * (g("mouthCornerPullLeft") + g("mouthCornerPullRight")) \
                  + 0.15 * (g("cheekSquintLeft") + g("cheekSquintRight"))
        happy_raw = max(0.0, min(1.0, happy_raw))

        # ANGRY: brow down + brow inner up/down patterns + nose sneer
        angry_raw = 0.55 * (g("browDownLeft") + g("browDownRight")) \
                  + 0.25 * (g("browInnerUp")) \
                  + 0.20 * (g("noseSneerLeft") + g("noseSneerRight"))
        angry_raw = max(0.0, min(1.0, angry_raw))

        # SURPRISED: jaw open + mouth open + brows up
        surprise_raw = 0.60 * g("jawOpen") \
                     + 0.25 * (g("mouthFunnel") + g("mouthPucker")) \
                     + 0.15 * (g("browOuterUpLeft") + g("browOuterUpRight"))
        surprise_raw = max(0.0, min(1.0, surprise_raw))

        # ---- EMA smoothing ----
        self._happy_ema = 0.30 * happy_raw + 0.70 * self._happy_ema
        self._angry_ema = 0.30 * angry_raw + 0.70 * self._angry_ema
        self._surprise_ema = 0.30 * surprise_raw + 0.70 * self._surprise_ema

        # ======================================================
        # Expression state selection
        # ======================================================
        # Only ONE expression state is active at a time.
        # Priority order matters and can be adjusted.
        #
        # Current priority:
        # SURPRISED > ANGRY > HAPPY > NEUTRAL
        #
        # This prevents rapid flickering between states.

        # ---- state selection + hold ----
        if self._hold > 0:
            state = self._held_state
            self._hold -= 1
        else:
            state = "NEUTRAL"
            if self._surprise_ema > CFG.surprise_on:
                state = "SURPRISED"
            elif self._angry_ema > CFG.angry_on:
                state = "ANGRY"
            elif self._happy_ema > CFG.happy_on:
                state = "HAPPY"

            if state != "NEUTRAL":
                self._held_state = state
                self._hold = CFG.expr_hold_frames

        return FaceOutput(
            expression_state=state,
            face_bbox=bbox,
            angry_score=float(self._angry_ema),
            happy_score=float(self._happy_ema),
            surprise_score=float(self._surprise_ema),
        )

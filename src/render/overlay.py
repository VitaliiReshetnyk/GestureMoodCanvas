import os
from typing import Tuple, Optional

import cv2
import numpy as np


class OverlayRenderer:
    def __init__(self, icon_dir: str):
        self.icon_dir = icon_dir
        self._cache = {}

    def _load_icon(self, name: str) -> Optional[np.ndarray]:
        if name in self._cache:
            return self._cache[name]

        path = os.path.join(self.icon_dir, name)
        if not os.path.exists(path):
            self._cache[name] = None
            return None

        icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if icon is None:
            self._cache[name] = None
            return None

        # Ensure RGBA
        if len(icon.shape) == 2:
            # grayscale -> BGR
            icon = cv2.cvtColor(icon, cv2.COLOR_GRAY2BGR)

        if icon.shape[2] == 3:
            # BGR -> BGRA with full opacity
            b, g, r = cv2.split(icon)
            a = np.full((icon.shape[0], icon.shape[1]), 255, dtype=np.uint8)
            icon = cv2.merge((b, g, r, a))

        self._cache[name] = icon
        return icon

    # ======================================================
    # Visual overlays
    # ======================================================
    # Icons are rendered above the detected face bounding box.
    # This method can be extended to support:
    # - Multiple icons at once
    # - Animated icons
    # - Icons following head movement in 3D

    def draw_icon_above_face(self, frame_bgr, face_bbox: Tuple[int, int, int, int], icon_name: str, scale: float = 0.18):
        icon = self._load_icon(icon_name)
        if icon is None:
            return frame_bgr

        x1, y1, x2, y2 = face_bbox
        fw = max(1, x2 - x1)
        fh = max(1, y2 - y1)

        target_w = int(fw * scale)
        target_w = max(26, min(180, target_w))

        ih, iw = icon.shape[:2]
        aspect = ih / max(1, iw)
        target_h = int(target_w * aspect)

        icon_rs = cv2.resize(icon, (target_w, target_h), interpolation=cv2.INTER_AREA)

        cx = x1 + fw // 2
        px = int(cx - target_w // 2)
        py = int(y1 - target_h - int(0.08 * fh))

        # Clamp to frame so it is always visible
        px = max(0, min(frame_bgr.shape[1] - target_w, px))
        py = max(0, min(frame_bgr.shape[0] - target_h, py))

        return self._alpha_blend(frame_bgr, icon_rs, px, py)

    # Alpha blending utility.
    # Supports both RGBA icons and RGB images (auto-converted).
    def _alpha_blend(self, frame_bgr, icon_bgra, x: int, y: int):
        if icon_bgra is None or icon_bgra.shape[2] != 4:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        ih, iw = icon_bgra.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + iw)
        y2 = min(h, y + ih)

        if x1 >= x2 or y1 >= y2:
            return frame_bgr

        roi = frame_bgr[y1:y2, x1:x2]
        icon_crop = icon_bgra[(y1 - y):(y2 - y), (x1 - x):(x2 - x)]

        bgr = icon_crop[..., :3].astype(np.float32)
        a = (icon_crop[..., 3:4].astype(np.float32) / 255.0)

        out = (1.0 - a) * roi.astype(np.float32) + a * bgr
        frame_bgr[y1:y2, x1:x2] = out.astype(np.uint8)
        return frame_bgr

    def draw_hud(self, frame_bgr, state, expr_state: str, hand_gesture: str, pose_event: str,
                 angry_score: float, happy_score: float, surprise_score: float):
        box_w, box_h = 560, 95
        x0, y0 = 8, 8
        x1, y1 = x0 + box_w, y0 + box_h

        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        frame_bgr = cv2.addWeighted(overlay, 0.35, frame_bgr, 0.65, 0)

        lines = [
            f"mode={'ERASE' if state.erase_mode else 'DRAW'}   draw={'ON' if state.draw_enabled else 'OFF'}   hand={hand_gesture}   pose={pose_event}",
            f"expr={expr_state}   angry={angry_score:.2f}   happy={happy_score:.2f}   surprise={surprise_score:.2f}",
            "ESC: quit   c: clear"
        ]

        y = 32
        for line in lines:
            cv2.putText(frame_bgr, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            y += 26

        return frame_bgr

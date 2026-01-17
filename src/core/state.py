import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional

from .actions import Action
from src.core.config import CFG

@dataclass
class AppState:
    width: int
    height: int

    draw_enabled: bool = False
    erase_mode: bool = False
    request_clear: bool = False

    color_idx: int = 0
    thickness: int = 6

    _cursor: Optional[Tuple[int, int]] = None
    _prev_cursor: Optional[Tuple[int, int]] = None

    _ema_xy: Optional[np.ndarray] = None
    ema_alpha: float = CFG.ema_alpha_cursor

    palette: Tuple[Tuple[int, int, int], ...] = field(default_factory=lambda: (
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
    ))

    def resize(self, w: int, h: int):
        self.width = w
        self.height = h
        self._cursor = None
        self._prev_cursor = None
        self._ema_xy = None

    def update_from_actions(self, actions):
        for a in actions:
            if a == Action.DRAW_ON:
                self.draw_enabled = True
            elif a == Action.DRAW_OFF:
                self.draw_enabled = False
                self._prev_cursor = None
            elif a == Action.CLEAR_CANVAS:
                self.request_clear = True
                self._prev_cursor = None
            elif a == Action.TOGGLE_ERASE:
                self.erase_mode = not self.erase_mode
                self._prev_cursor = None
            elif a == Action.NEXT_COLOR:
                self.color_idx = (self.color_idx + 1) % len(self.palette)
            elif a == Action.THICKER:
                self.thickness = min(40, self.thickness + 2)
            elif a == Action.THINNER:
                self.thickness = max(1, self.thickness - 2)

    def update_cursor(self, xy: Tuple[int, int]):
        x, y = xy
        x = int(np.clip(x, 0, self.width - 1))
        y = int(np.clip(y, 0, self.height - 1))

        v = np.array([x, y], dtype=np.float32)
        if self._ema_xy is None:
            self._ema_xy = v
        else:
            self._ema_xy = self.ema_alpha * v + (1.0 - self.ema_alpha) * self._ema_xy

        sx, sy = int(self._ema_xy[0]), int(self._ema_xy[1])
        self._cursor = (sx, sy)

    def paint(self, canvas):
        if self._cursor is None:
            return

        if self._prev_cursor is None:
            self._prev_cursor = self._cursor
            return

        x1, y1 = self._prev_cursor
        x2, y2 = self._cursor

        if self.erase_mode:
            color = (0, 0, 0)
            t = max(12, self.thickness + 8)
        else:
            color = self.palette[self.color_idx]
            t = self.thickness

        import cv2
        cv2.line(canvas, (x1, y1), (x2, y2), color, t, cv2.LINE_AA)
        self._prev_cursor = self._cursor

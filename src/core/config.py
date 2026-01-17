from dataclasses import dataclass

@dataclass
class Config:
    ema_alpha_cursor: float = 0.35

    angry_on: float = 0.55
    happy_on: float = 0.55
    surprise_on: float = 0.55

    expr_hold_frames: int = 12

    gesture_stable_frames: int = 4

CFG = Config()

from enum import Enum, auto


class Action(Enum):
    DRAW_ON = auto()
    DRAW_OFF = auto()
    CLEAR_CANVAS = auto()
    TOGGLE_ERASE = auto()
    NEXT_COLOR = auto()
    THICKER = auto()
    THINNER = auto()

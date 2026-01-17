# ======================================================
# Gesture / Pose / Expression bindings
# ======================================================
# This file defines how detected gestures, poses and
# facial expressions are mapped to high-level actions
# or visual effects.
#
# To add a new interaction:
# 1) Detect a new gesture / pose / expression
# 2) Add a new Action (if needed) in actions.py
# 3) Bind it here without touching the detectors
#
# This separation keeps the system modular and scalable.


from dataclasses import dataclass
from typing import Dict

from .actions import Action


@dataclass
class Bindings:
    hand_gesture_to_action: Dict[str, Action]
    pose_event_to_action: Dict[str, Action]
    expression_to_icon: Dict[str, str]


def default_bindings() -> Bindings:
    return Bindings(
        # Hand gesture -> Action mapping
        # Example extensions:
        # "TWO_FINGERS_UP": Action.UNDO
        # "THUMB_UP": Action.THICKER
        # "THUMB_DOWN": Action.THINNER
        hand_gesture_to_action={
            "INDEX_UP": Action.DRAW_ON,
            "FIST": Action.DRAW_OFF,
            "PINCH": Action.TOGGLE_ERASE,
            "OPEN_PALM": Action.NEXT_COLOR,
        },
        pose_event_to_action={
            "HANDS_UP": Action.CLEAR_CANVAS,
        },
        # Facial expression -> Icon mapping
        # You can add more expressions such as:
        # "SAD": "rain.png"
        # "CONFUSED": "question.png"
        # "EXCITED": "star.png"
        #
        # Icons are rendered as overlays above the head.

        expression_to_icon={
            "ANGRY": "thermo.png",
            "HAPPY": "sparkle.png",
            "SURPRISED": "wow.png",
        },
    )

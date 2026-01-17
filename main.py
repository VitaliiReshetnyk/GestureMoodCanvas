import cv2
import numpy as np

from src.detectors.hand_detector import HandDetector
from src.detectors.face_detector import FaceDetector
from src.detectors.pose_detector import PoseDetector

from src.core.state import AppState
from src.core.bindings import default_bindings
from src.core.paths import ICONS
from src.render.overlay import OverlayRenderer


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Cannot read from webcam")

    h, w = frame.shape[:2]
    state = AppState(width=w, height=h)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    hand = HandDetector()
    face = FaceDetector()
    pose = PoseDetector()

    bindings = default_bindings()
    renderer = OverlayRenderer(icon_dir=str(ICONS))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if canvas.shape[0] != h or canvas.shape[1] != w:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            state.resize(w, h)

        hand_out = hand.process(frame)
        face_out = face.process(frame)
        pose_out = pose.process(frame)

        # ======================================================
        # High-level interaction logic
        # ======================================================
        # Detectors produce signals (gestures, poses, expressions).
        # Here we:
        # 1) Map them to actions
        # 2) Update application state
        # 3) Render the result
        #
        # This is the main control loop of the application.

        actions = []
        if hand_out.gesture in bindings.hand_gesture_to_action:
            actions.append(bindings.hand_gesture_to_action[hand_out.gesture])

        if pose_out.pose_event in bindings.pose_event_to_action:
            actions.append(bindings.pose_event_to_action[pose_out.pose_event])

        expr_state = face_out.expression_state
        icon_name = bindings.expression_to_icon.get(expr_state, None)

        # Apply all actions for this frame.
        # Multiple actions can be triggered simultaneously
        # (e.g. drawing + color change).

        state.update_from_actions(actions)

        if state.draw_enabled and hand_out.cursor_xy is not None:
            state.update_cursor(hand_out.cursor_xy)
            state.paint(canvas)

        if state.request_clear:
            canvas[:] = 0
            state.request_clear = False

        composed = cv2.addWeighted(frame, 0.75, canvas, 0.85, 0.0)

        # draw icon for non-neutral states (and when mapping exists)
        if face_out.face_bbox is not None and icon_name is not None and expr_state not in ("NONE", "NEUTRAL"):
            composed = renderer.draw_icon_above_face(
                composed,
                face_bbox=face_out.face_bbox,
                icon_name=icon_name,
                scale=0.18
            )

        composed = renderer.draw_hud(
            composed,
            state,
            expr_state,
            hand_out.gesture,
            pose_out.pose_event,
            face_out.angry_score,
            face_out.happy_score,
            face_out.surprise_score
        )

        cv2.imshow("GestureMoodCanvas", composed)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        if key == ord('c'):
            canvas[:] = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

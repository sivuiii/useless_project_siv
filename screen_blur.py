# eye_blur_dynamic.py
import sys
import time
import math
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import mss
from PyQt5 import QtCore, QtGui, QtWidgets

# ---------------- CONFIG ----------------
FPS = 18                      # UI update rate
BLUR_DOWNSCALE = 0.25         # blur on a small image, then upscale (speed)
GAUSSIAN_KERNEL = (21, 21)    # blur kernel (kept moderate because downscaling amplifies blur)
SMOOTHING_ALPHA = 0.6         # smoothing for blend factor (0..1) higher = smoother
EYE_FRAMES_SMOOTH = 3         # average over last N frames for EAR
WEBCAM_RES = (640, 480)       # webcam capture size for face detection
CLICK_THROUGH = True          # True => window won't capture mouse clicks (pass-through)
# EAR calibration: typical EAR open ~0.26-0.32, closed ~0.12-0.18 — tune per camera/person
EAR_MIN = 0.12   # EAR at fully closed (maps to blur_factor = 0.0)
EAR_MAX = 0.30   # EAR at wide open (maps to blur_factor = 1.0)
# ----------------------------------------

mp_face_mesh = mp.solutions.face_mesh
FACE_MESH = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# landmark groups for drawing
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [300, 293, 334, 296, 336]
LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
FACE_OUT = [10, 109, 67, 103, 54, 21, 162, 127, 234, 93]

# Helper: convert OpenCV BGR to QImage
def cv_to_qimage(cv_img):
    h, w = cv_img.shape[:2]
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)

def eye_aspect_ratio(landmarks, indices):
    # expects landmarks list of (x,y)
    a = np.linalg.norm(np.array(landmarks[indices[1]]) - np.array(landmarks[indices[5]]))
    b = np.linalg.norm(np.array(landmarks[indices[2]]) - np.array(landmarks[indices[4]]))
    c = np.linalg.norm(np.array(landmarks[indices[0]]) - np.array(landmarks[indices[3]])) + 1e-8
    ear = (a + b) / (2.0 * c)
    return ear

class FullOverlay(QtWidgets.QWidget):
    def __init__(self, width, height):
        super().__init__()
        flags = (QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint)
        if CLICK_THROUGH:
            flags |= QtCore.Qt.WindowTransparentForInput
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # show fullscreen on primary screen
        self.showFullScreen()
        self.setFixedSize(width, height)
        self._pix = None
        self._lock = QtCore.QMutex()

    def set_frame(self, frame_bgr):
        """Thread-safe update of the current frame (BGR cv2 image)"""
        with QtCore.QMutexLocker(self._lock):
            self._pix = QtGui.QPixmap.fromImage(cv_to_qimage(frame_bgr))
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        with QtCore.QMutexLocker(self._lock):
            pix = self._pix
        if pix is None:
            painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 0))
        else:
            painter.drawPixmap(0, 0, pix)
        painter.end()

class EyeBlurController(QtCore.QObject):
    def __init__(self, overlay):
        super().__init__()
        self.overlay = overlay

        # desktop capture
        self.sct = mss.mss()
        mon = self.sct.monitors[1]  # primary
        self.mon = {'left': mon['left'], 'top': mon['top'], 'width': mon['width'], 'height': mon['height']}
        self.screen_w = self.mon['width']
        self.screen_h = self.mon['height']

        # webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_RES[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_RES[1])

        # ear smoothing queue
        self.ear_queue = deque(maxlen=EYE_FRAMES_SMOOTH)
        self.smoothed_blend = 1.0  # initial assume eyes open -> blur on startup

        # pre-allocated blurred cache
        self.last_desktop = None
        self.last_blurred = None
        self.last_desktop_ts = 0.0
        self.blur_down_w = max(2, int(self.screen_w * BLUR_DOWNSCALE))
        self.blur_down_h = max(2, int(self.screen_h * BLUR_DOWNSCALE))

    def start(self, fps=FPS):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(int(1000 / fps))

    def update(self):
        # 1) capture desktop (always raw)
        s = self.sct.grab(self.mon)
        desktop_bgra = np.array(s)  # BGRA
        desktop_bgr = cv2.cvtColor(desktop_bgra, cv2.COLOR_BGRA2BGR)

        # if desktop changed resolution or timestamp, recompute blurred (but we always recompute each frame here)
        # 2) compute single blurred layer (on downscaled image for perf)
        small = cv2.resize(desktop_bgr, (self.blur_down_w, self.blur_down_h), interpolation=cv2.INTER_AREA)
        small_blur = cv2.GaussianBlur(small, GAUSSIAN_KERNEL, 0)
        # upscale back to full
        blurred_full = cv2.resize(small_blur, (self.screen_w, self.screen_h), interpolation=cv2.INTER_LINEAR)

        # 3) read webcam frame and compute EAR
        ret, cam = self.cap.read()
        if not ret:
            return
        cam_h, cam_w = cam.shape[:2]
        rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
        results = FACE_MESH.process(rgb)

        avg_ear = None
        face_landmarks_px = None

        if results.multi_face_landmarks:
            face_lm = results.multi_face_landmarks[0]
            landmarks = [(int(p.x * cam_w), int(p.y * cam_h)) for p in face_lm.landmark]
            face_landmarks_px = landmarks
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)
            avg_ear = (left_ear + right_ear) / 2.0
            self.ear_queue.append(avg_ear)
        else:
            # no face -> assume eyes open so maximum blur (put a high EAR)
            self.ear_queue.append(EAR_MAX)

        # 4) compute blend factor from EAR
        # map EAR in [EAR_MIN, EAR_MAX] -> openness in [0,1]
        recent = list(self.ear_queue)
        if len(recent) > 0:
            ear_mean = sum(recent) / len(recent)
        else:
            ear_mean = EAR_MAX

        # clamp and normalize
        norm = (ear_mean - EAR_MIN) / (EAR_MAX - EAR_MIN + 1e-8)
        norm = max(0.0, min(1.0, norm))  # 0 = closed, 1 = wide open

        # blur strength = norm (open->more blur). We want value in [0..1]
        target_blend = norm

        # smooth the blend to avoid flicker
        self.smoothed_blend = SMOOTHING_ALPHA * self.smoothed_blend + (1 - SMOOTHING_ALPHA) * target_blend
        blend = float(self.smoothed_blend)

        # 5) single-layer blend: blended = blurred * blend + original * (1-blend)
        blended = cv2.addWeighted(blurred_full, blend, desktop_bgr, 1.0 - blend, 0)

        # 6) draw face mesh and HUD on blended (map webcam landmarks to screen)
        draw_img = blended

        if face_landmarks_px:
            # project landmarks roughly from webcam to screen
            for (lx, ly) in face_landmarks_px:
                sx = int((lx / cam_w) * self.screen_w)
                sy = int((ly / cam_h) * self.screen_h)
                cv2.circle(draw_img, (sx, sy), 1, (0, 255, 255), -1)

            # draw outline, eyes, eyebrows, lips (stylized)
            def poly_draw(idxs, color=(0, 255, 0), thick=1):
                pts = [face_landmarks_px[i] for i in idxs]
                mapped = [(int((x / cam_w) * self.screen_w), int((y / cam_h) * self.screen_h)) for (x, y) in pts]
                for i in range(len(mapped) - 1):
                    cv2.line(draw_img, mapped[i], mapped[i + 1], color, thick, cv2.LINE_AA)

            poly_draw(FACE_OUT, (200, 200, 200), 1)
            poly_draw(LEFT_EYE_IDX, (0, 255, 0), 2)
            poly_draw(RIGHT_EYE_IDX, (0, 255, 0), 2)
            poly_draw(LEFT_EYEBROW, (255, 200, 0), 2)
            poly_draw(RIGHT_EYEBROW, (255, 200, 0), 2)
            poly_draw(LIPS, (0, 120, 255), 2)

        # 7) small HUD showing blend percentage
        perc = int(blend * 100)
        cv2.putText(draw_img, f"Blur: {perc}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # 8) send final frame to overlay window
        self.overlay.set_frame(draw_img)

    def stop(self):
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self.sct.close()
        except Exception:
            pass
        try:
            FACE_MESH.close()
        except Exception:
            pass

def run_app():
    app = QtWidgets.QApplication(sys.argv)
    # get screen geometry
    screen = app.primaryScreen()
    geom = screen.geometry()
    overlay = FullOverlay(geom.width(), geom.height())
    controller = EyeBlurController(overlay)
    controller.start(FPS)

    # Quit on ESC key (global)
    def check_escape():
        # uses OpenCV waitKey for simplicity
        if cv2.waitKey(1) & 0xFF == 27:
            controller.stop()
            QtWidgets.QApplication.quit()

    esc_timer = QtCore.QTimer()
    esc_timer.timeout.connect(check_escape)
    esc_timer.start(40)

    sys.exit(app.exec_())

if __name__ == "__main__":
    run_app()

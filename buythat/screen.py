"""Find a TV/monitor screen in a phone photo and warp it to a flat, front-on frame."""
from dataclasses import dataclass

import cv2
import numpy as np

OUT_W, OUT_H = 1280, 720


@dataclass
class ScreenResult:
    frame: np.ndarray  # rectified BGR image (or the original if no screen found)
    quad: list | None  # 4 corner points in the source image, TL, TR, BR, BL
    found: bool


def _order(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype=np.float32)


def _candidates(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 90)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        for eps in (0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                yield approx.reshape(4, 2).astype(np.float32)
                break


def _score(quad: np.ndarray, img_area: float) -> float:
    area = cv2.contourArea(quad)
    frac = area / img_area
    if frac < 0.12 or frac > 0.97:
        return -1
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if h == 0:
        return -1
    aspect = w / h
    aspect_penalty = min(abs(aspect - 16 / 9), abs(aspect - 4 / 3)) / (16 / 9)
    if aspect_penalty > 0.45:
        return -1
    return frac * (1 - aspect_penalty)


def find_screen(img: np.ndarray) -> ScreenResult:
    h, w = img.shape[:2]
    scale = 900 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img
    s = scale if scale < 1 else 1.0
    area = small.shape[0] * small.shape[1]
    best, best_score = None, 0.0
    for q in _candidates(small):
        q = _order(q)
        sc = _score(q, area)
        if sc > best_score:
            best, best_score = q, sc
    if best is None:
        return ScreenResult(frame=img, quad=None, found=False)
    quad = best / s
    dst = np.array([[0, 0], [OUT_W - 1, 0], [OUT_W - 1, OUT_H - 1], [0, OUT_H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img, M, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC)
    return ScreenResult(frame=suppress_moire(warped), quad=quad.round(1).tolist(), found=True)


def suppress_moire(img: np.ndarray) -> np.ndarray:
    """Camera-of-a-screen photos carry pixel-grid moire; a light bilateral pass removes most of it."""
    return cv2.bilateralFilter(img, 7, 40, 7)

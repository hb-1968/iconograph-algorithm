"""Align dominant edges with the image axes.

Score Hough-line orientations over [-45, 45) degrees in 0.5-degree steps, then
rotate and reframe. Input and output are square RGB uint8 images on white.
"""

import numpy as np
import cv2
from skimage.transform import hough_line, hough_line_peaks

from iconograph.application.common import frame_with_margin


def align(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    hspace, theta_axis, rho_axis = hough_line(edges)
    accum, peak_angles, _ = hough_line_peaks(
        hspace, theta_axis, rho_axis,
        threshold=0.25 * hspace.max(),
        num_peaks=200,
    )

    alphas = np.deg2rad(np.arange(-45.0, 45.0, 0.5))
    rotated = peak_angles[None, :] + alphas[:, None]
    residual = ((rotated + np.pi / 4) % (np.pi / 2)) - np.pi / 4
    weights = np.cos(2*residual) ** 2
    scores = (weights * accum[None, :]).sum(axis=1)
    alpha_best = alphas[int(np.argmax(scores))]

    if scores.max() < 0.05 * float(accum.sum()) or abs(np.rad2deg(alpha_best)) < 0.25:
        return img.copy()
    
    alpha_deg = float(np.rad2deg(alpha_best))
    S = img.shape[0]
    M = cv2.getRotationMatrix2D(
        center=(S / 2.0, S / 2.0),
        angle=alpha_deg,
        scale=1.0,
    )

    rotated = cv2.warpAffine(
        img, M, (S, S),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return frame_with_margin(rotated)
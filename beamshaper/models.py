"""Physikalisches Modell: rotierter 2D-Supergauss.

    I(x, y) = A * exp(-(r^2)^m) + B,
    r^2 = (x'/wa)^2 + (y'/wb)^2,   (x', y') = um theta gedrehte Koordinaten

m = 1 ist ein normaler Gauss (wa, wb sind dann die 1/e-Radien der Intensitaet,
nicht die 1/e^2-Radien!). Groessere m -> flacheres Plateau, steilere Kanten.
Konvention nach dem Fit: wa >= wb, theta = Winkel der langen Achse gegen die
x-Achse des Bildes (Bild-y zeigt nach unten), theta in (-90 deg, 90 deg].
"""
from __future__ import annotations

import numpy as np


def supergaussian_2d(coords, A, x0, y0, wa, wb, theta, m, B):
    x, y = coords
    ct, st = np.cos(theta), np.sin(theta)
    dx, dy = x - x0, y - y0
    xp = dx * ct + dy * st
    yp = -dx * st + dy * ct
    r2 = (xp / wa) ** 2 + (yp / wb) ** 2
    return A * np.exp(-(r2 ** m)) + B


def radius_at_level(w: float, m: float, level: float) -> float:
    """Halbachse, bei der die Intensitaet auf `level` (0..1) des Maximums gefallen ist."""
    if not 0.0 < level < 1.0:
        raise ValueError("level muss zwischen 0 und 1 liegen")
    return w * (-np.log(level)) ** (1.0 / (2.0 * m))


def ellipse_radius_along(a: float, b: float, theta: float, phi: float) -> float:
    """Abstand Zentrum -> Ellipsenrand in Richtung phi (Bildkoordinaten).

    Das ist die Stelle, an der ein Schnitt unter dem Winkel phi den Level
    kreuzt. (Nicht zu verwechseln mit der Bounding-Box-Halbbreite
    sqrt((a cos)^2 + (b sin)^2), die in WaistAnalysis.py verwendet wurde.)
    """
    d = phi - theta
    return a * b / np.sqrt((b * np.cos(d)) ** 2 + (a * np.sin(d)) ** 2)


def normalize_axes(wa, wb, theta, swa=0.0, swb=0.0):
    """Erzwingt wa >= wb und theta in (-pi/2, pi/2]."""
    if wb > wa:
        wa, wb = wb, wa
        swa, swb = swb, swa
        theta = theta + np.pi / 2
    theta = (theta + np.pi / 2) % np.pi - np.pi / 2
    if theta <= -np.pi / 2:
        theta += np.pi
    return wa, wb, theta, swa, swb

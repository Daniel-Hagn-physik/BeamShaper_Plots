"""Laden von Kamerabildern als float-Array (Graustufen).

Neben Kamerabildern lassen sich simulierte 2D-Profile laden (.npz aus
Multitone_Simulation, Weighted_Multitone_Lens_GUI.py, Knopf "Export 2D profile
for BeamShaperPlots"). Sie bringen ihre eigene Pixelgroesse mit (um pro Pixel in
der Objektebene) und sind auf das Maximum 1 normiert.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

SIM_SUFFIXES = (".npz",)
IMAGE_SUFFIXES = (".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg") + SIM_SUFFIXES


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def is_background_file(path: str | Path) -> bool:
    """Dateien mit 'background'/'hintergrund' im Namen werden im Batch uebersprungen."""
    name = Path(path).stem.lower()
    return "background" in name or "hintergrund" in name


def is_simulation_file(path: str | Path) -> bool:
    """Simuliertes 2D-Profil (.npz aus Multitone_Simulation)?"""
    return Path(path).suffix.lower() in SIM_SUFFIXES


def load_simulation(path: str | Path) -> dict:
    """Laedt ein simuliertes 2D-Profil.

    Rueckgabe: {"image": 2D float64 (Zeile 0 = oben, wie Kamerabild),
                "um_per_px": float, "params": dict, "saturation": 1.0}
    Das Bild ist auf sein Maximum normiert; die Skala reicht daher bis 1.
    """
    import json
    with np.load(path, allow_pickle=False) as d:
        if "image" not in d or "um_per_px" not in d:
            raise ValueError(f"{Path(path).name}: kein Simulationsprofil "
                             "(erwartet 'image' und 'um_per_px')")
        img = np.asarray(d["image"], dtype=np.float64)
        um = float(d["um_per_px"])
        params = {}
        if "params" in d:
            try:
                params = json.loads(str(d["params"]))
            except ValueError:
                params = {}
    if img.ndim != 2 or um <= 0:
        raise ValueError(f"{Path(path).name}: ungueltiges Simulationsprofil")
    peak = float(np.nanmax(img))
    if peak > 0:
        img = img / peak
    return {"image": img, "um_per_px": um, "params": params, "saturation": 1.0}


def load_image(path: str | Path) -> tuple[np.ndarray, float]:
    """Laedt ein Bild als 2D-float-Array.

    Returns
    -------
    img : np.ndarray (ny, nx), float64
    saturation_value : float
        Groesster darstellbarer Wert (255 bei 8 bit, 65535 bei 16 bit ...).
        Wird fuer die Uebersteuerungs-Warnung gebraucht.
    """
    if is_simulation_file(path):
        sim = load_simulation(path)
        return sim["image"], sim["saturation"]
    with Image.open(path) as im:
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.asarray(im, dtype=np.float64)
            # 12-bit-Kameras speichern oft in 16-bit-Containern
            sat = 4095.0 if arr.max() <= 4095 else 65535.0
        elif im.mode == "F":
            arr = np.asarray(im, dtype=np.float64)
            sat = float(np.nanmax(arr))
        else:
            arr = np.asarray(im.convert("L"), dtype=np.float64)
            sat = 255.0
    return arr, sat


def list_images(folder: str | Path, skip_background: bool = True) -> list[Path]:
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.is_file() and is_image_file(p))
    if skip_background:
        files = [p for p in files if not is_background_file(p)]
    return files

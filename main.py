"""Startet die BeamShaperPlots-GUI.

    python main.py                 # fragt beim Start nach der Datei (Ordner Messdaten)
    python main.py <bild|ordner>   # direkt laden

Stuerzt das Programm ab, steht die Ursache in crash.log (neben dieser Datei)
und in der Konsole.
"""
import datetime
import faulthandler
import sys
from pathlib import Path


LOG_PATH = Path(__file__).resolve().with_name("crash.log")
_log = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
_log.write(f"\n=== Start {datetime.datetime.now():%Y-%m-%d %H:%M:%S}, "
           f"Python {sys.version.split()[0]} ===\n")
# schreibt bei harten Abstuerzen (auch in C-Code/Qt) den Python-Stack ins Log
faulthandler.enable(file=_log, all_threads=True)

from beamshaper import gui  # noqa: E402  (erst nach faulthandler importieren)

gui.LOG_FILE = _log

if __name__ == "__main__":
    sys.exit(gui.main())
"""Locating the files the application reads and writes.

Part of the flashlight simulator core; see core/__init__.py for the
public surface.
"""

import json
import os
import sys

# The core package sits one level below the application root, and the
# settings, the catalogue and the .ui file all live at that root rather
# than inside the package. Anchoring here keeps them out of it.
APPLICATION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource_path(relative_path: str) -> str:
    """Returns the absolute path of a read-only asset shipped with the app.

    Args:
        relative_path: Path of the asset relative to the application root.

    Returns:
        The path inside the PyInstaller bundle when frozen, otherwise the
        path next to flashlight-sim.py.
    """
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(APPLICATION_ROOT, relative_path)


def user_data_path(relative_path: str) -> str:
    """Returns the absolute path of a writable file that lives beside the app.

    Anything the operator edits (settings, the hardware library, exported plots)
    must not be written into the PyInstaller bundle: under --onefile the bundle
    is a temporary directory wiped on exit, and under --onedir it may sit on
    read-only media.

    Args:
        relative_path: Path of the file relative to the application folder.

    Returns:
        The path next to the executable when frozen, otherwise the path
        next to flashlight-sim.py.
    """
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = APPLICATION_ROOT
    return os.path.join(base, relative_path)


def _read_json(path: str) -> dict:
    """Reads and parses a UTF-8 JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, data: dict) -> None:
    """Writes a dict to disk as indented UTF-8 JSON."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)

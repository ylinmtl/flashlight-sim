"""Exporting the beam as an IES photometric file.

Part of the flashlight simulator core; see core/__init__.py for the
public surface.
"""

import math
import os
import time

import numpy as np

from .config import SimulationConfig


# ==============================================================================
# SECTION 8: IES PHOTOMETRIC EXPORT
# ==============================================================================


def _bilinear_sample(grid: np.ndarray, row: np.ndarray, col: np.ndarray):
    """Reads a grid at fractional indices, blending the four nearest cells.

    Args:
        grid: The 2D array to read.
        row: Fractional row indices, any shape.
        col: Fractional column indices, matching row.

    Returns:
        An array of samples shaped like row.
    """
    rows, columns = grid.shape
    row_low = np.clip(np.floor(row), 0, rows - 1).astype(np.int64)
    col_low = np.clip(np.floor(col), 0, columns - 1).astype(np.int64)
    row_high = np.clip(row_low + 1, 0, rows - 1)
    col_high = np.clip(col_low + 1, 0, columns - 1)

    row_fraction = np.clip(row - row_low, 0.0, 1.0)
    col_fraction = np.clip(col - col_low, 0.0, 1.0)

    lower = (grid[row_low, col_low] * (1.0 - col_fraction)
             + grid[row_low, col_high] * col_fraction)
    upper = (grid[row_high, col_low] * (1.0 - col_fraction)
             + grid[row_high, col_high] * col_fraction)
    return lower * (1.0 - row_fraction) + upper * row_fraction


def beam_candela_grid(wall_lux: np.ndarray, config: SimulationConfig):
    """Turns the simulated wall illuminance into a candela distribution.

    The tracer lands light on a flat wall, so a point away from the axis is
    both further from the head and struck at a slant. Undoing the two gives the
    luminous intensity leaving the head in that direction:

        I = E * r^3 / z

    where r is the distance from the head to the wall point and z the distance
    to the wall along the axis. One factor of r^2 is the inverse square law and
    the remaining r / z is 1 / cos of the angle the ray meets the wall at.

    Sampling on a full turn of horizontal angles is what preserves asymmetry: a
    square die, a chamfered one or an emitter pushed off the axis all give a
    beam that differs from one side to the other, and a single plane would
    average that away.

    Args:
        wall_lux: Illuminance grid from simulate_wall_illuminance.
        config: Active configuration.

    Returns:
        (vertical_deg, horizontal_deg, candela), with candela indexed
        [horizontal, vertical] as the IES file orders it.
    """
    grid_res = wall_lux.shape[0]
    wall_radius = config.wall_radius_m
    distance = config.target_distance_m

    # Only the inscribed circle of the wall is covered at every azimuth, so the
    # sweep stops at its half angle. The corners reach further but only in four
    # directions, which would read as spurious asymmetry.
    covered = math.degrees(math.atan(wall_radius / distance))
    limit = min(float(config.ies_max_vertical_angle_deg), covered)

    vertical = np.arange(0.0, limit + 1e-9, float(config.ies_vertical_step_deg))
    horizontal = np.arange(0.0, 360.0 + 1e-9, float(config.ies_horizontal_step_deg))

    polar = np.radians(vertical)[None, :]
    azimuth = np.radians(horizontal)[:, None]
    offset = distance * np.tan(polar)
    wall_x = offset * np.cos(azimuth)
    wall_y = offset * np.sin(azimuth)

    # Grid cells are counted from the low corner, so a cell centre sits half a
    # cell in; the same half cell comes back off to get a fractional index.
    cell = (2.0 * wall_radius) / grid_res
    col = (wall_x + wall_radius) / cell - 0.5
    row = (wall_y + wall_radius) / cell - 0.5

    ray_length = np.sqrt(offset ** 2 + distance ** 2)
    candela = (_bilinear_sample(wall_lux, row, col)
               * ray_length ** 3 / distance)

    inside = (np.abs(wall_x) <= wall_radius) & (np.abs(wall_y) <= wall_radius)
    return vertical, horizontal, np.where(inside, candela, 0.0)


def _wrap_numbers(values, per_line: int = 12) -> str:
    """Formats numbers into wrapped lines, keeping them well under 132 columns.

    Args:
        values: Numbers to write.
        per_line: How many to put on each line.

    Returns:
        The formatted block, newline terminated.
    """
    text = ""
    for start in range(0, len(values), per_line):
        chunk = values[start:start + per_line]
        text += " ".join(f"{value:.6g}" for value in chunk) + "\n"
    return text


def write_ies_file(path: str, vertical_deg, horizontal_deg, candela,
                   header: dict) -> None:
    """Writes an IESNA LM-63-2002 photometric file.

    Type C photometry with a full turn of horizontal angles, which is the form
    Unreal Engine and Blender both read and the only one that can carry an
    asymmetric beam.

    Args:
        path: File to write.
        vertical_deg: Vertical angles from the beam axis outwards.
        horizontal_deg: Horizontal angles, 0 to 360.
        candela: Intensities indexed [horizontal, vertical].
        header: Keys "luminaire", "lamp", "catalogue", "lumens", "watts" and
            "notes", a list of extra comment lines.
    """
    lines = ["IESNA:LM-63-2002",
             "[TEST] Simulated, not measured",
             "[TESTLAB] flashlight-sim ray tracer",
             f"[ISSUEDATE] {time.strftime('%Y-%m-%d')}",
             "[MANUFAC] flashlight-sim",
             f"[LUMCAT] {header.get('catalogue', '')}",
             f"[LUMINAIRE] {header.get('luminaire', '')}",
             f"[LAMP] {header.get('lamp', '')}"]
    lines += [f"[MORE] {note}" for note in header.get("notes", [])]
    lines.append("TILT=NONE")

    # 1 lamp, absolute photometry, type C, metres, and a point sized luminaire:
    # the profile is a far field distribution, so no physical size is claimed.
    lines.append(f"1 {header.get('lumens', 0.0):.1f} 1.0 {len(vertical_deg)} "
                 f"{len(horizontal_deg)} 1 2 0 0 0")
    lines.append(f"1.0 1.0 {header.get('watts', 0.0):.2f}")

    body = _wrap_numbers(list(vertical_deg)) + _wrap_numbers(list(horizontal_deg))
    # One block per horizontal plane, each running out along the vertical angles.
    for plane in candela:
        body += _wrap_numbers(list(plane))

    with open(path, "w", encoding="ascii", errors="replace") as handle:
        handle.write("\n".join(lines) + "\n" + body)


def export_beam_ies(path: str, wall_lux: np.ndarray, config: SimulationConfig,
                    header: dict) -> str:
    """Builds the candela distribution and writes it out as an IES file.

    Args:
        path: File to write.
        wall_lux: Illuminance grid from simulate_wall_illuminance.
        config: Active configuration.
        header: Passed through to write_ies_file.

    Returns:
        A one line summary of what was written, for the log.
    """
    vertical, horizontal, candela = beam_candela_grid(wall_lux, config)
    header = dict(header)
    header.setdefault("notes", []).append(
        f"Sampled from a simulated wall at {config.target_distance_m} m; "
        f"vertical angles run to {vertical[-1]:.1f} deg, the limit of the "
        f"simulated field of view")
    write_ies_file(path, vertical, horizontal, candela, header)
    return (f"IES written: {os.path.basename(path)} "
            f"({len(vertical)} x {len(horizontal)} angles, "
            f"peak {candela.max():,.0f} cd)")

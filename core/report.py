"""Rendering the beam and writing the result files.

Part of the flashlight simulator core; see core/__init__.py for the
public surface.
"""

import csv
import math
import os
from typing import NamedTuple, Optional

import numpy as np

# Agg keeps Matplotlib off the GUI thread and stops it opening
# detached windows. It must be chosen before pyplot is imported.
import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from .config import (CancelCallback, LogCallback, ProgressCallback,
                     SimulationConfig)
from .hardware import HardwareLibrary, spec_or_default
from .optics import (NO_EMITTER_OFFSET, EmitterOffset, calculate_lumens,
                     forward_voltage, get_sim_geometry)
from .photometry import export_beam_ies
from .simulation import (BeamMetrics, angular_sampling_warnings,
                         apply_camera_exposure_and_tonemap,
                         get_beam_metrics, simulate_wall_illuminance)


# Characters Windows reserves for paths and wildcards. Hardware names are free
# text and end up inside export filenames, so they have to be replaced first.
_RESERVED_FILENAME_CHARS = '<>:"/\\|?*'


def draw_human_silhouette(ax, person_x: float, person_y_bottom: float,
                          person_height_m: float) -> None:
    """Draws a dashed stick figure on the wall plot for scale.

    Args:
        ax: Axes to draw on.
        person_x: Horizontal centre of the figure, in metres.
        person_y_bottom: Height of the figure's feet, in metres.
        person_height_m: Overall height, in metres.
    """
    # Proportions of the head, torso, legs and arms relative to full height.
    head_r, torso_w, torso_h, leg_w, leg_h, arm_w, arm_h = (
        person_height_m * ratio for ratio in (0.08, 0.25, 0.35, 0.08, 0.45, 0.06, 0.40))
    style = dict(ec="#FFFF00", fc="none", alpha=0.4, lw=1.0, ls="--")

    shoulder_y = person_y_bottom + leg_h + torso_h
    ax.add_patch(patches.Circle((person_x, shoulder_y + head_r), head_r, **style))
    ax.add_patch(patches.Rectangle(
        (person_x - torso_w / 2, person_y_bottom + leg_h), torso_w, torso_h, **style))
    ax.add_patch(patches.Rectangle(
        (person_x - torso_w / 2, person_y_bottom), leg_w, leg_h, **style))
    ax.add_patch(patches.Rectangle(
        (person_x + torso_w / 2 - leg_w, person_y_bottom), leg_w, leg_h, **style))
    ax.add_patch(patches.Rectangle(
        (person_x - torso_w / 2 - arm_w, shoulder_y - arm_h), arm_w, arm_h, **style))
    ax.add_patch(patches.Rectangle(
        (person_x + torso_w / 2, shoulder_y - arm_h), arm_w, arm_h, **style))


def _style_dark_axes(ax) -> None:
    """Applies the shared dark plot styling to one Axes."""
    ax.set_facecolor("black")
    ax.tick_params(colors="#CCCCCC", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#555555")


def render_intensity_profile(slice_lux: np.ndarray, dist_array: np.ndarray,
                             suffix_name: str, title_str: str, save_path: Optional[str],
                             config: SimulationConfig) -> None:
    """Renders and optionally saves one intensity profile through the beam.

    Args:
        slice_lux: Illuminance along the slice, in lux.
        dist_array: Distance from the beam centre for each sample, in metres.
        suffix_name: Label for the slice, e.g. "X-Axis"; also the filename suffix.
        title_str: Title block shared with the wall shot.
        save_path: Wall shot path the profile filename is derived from, or None.
        config: Active configuration.
    """
    slice_cd = slice_lux * (config.target_distance_m ** 2)
    angles = np.degrees(np.arctan(dist_array / config.target_distance_m))

    figure, ax = plt.subplots(figsize=(10, 5), facecolor="black")
    _style_dark_axes(ax)

    ax.plot(angles, slice_cd, color="#FFFF00", linewidth=1.5)
    ax.fill_between(angles, slice_cd, color="#FFFF00", alpha=0.1)

    ax.set_xlim(-config.plot_fov_deg / 2.0, config.plot_fov_deg / 2.0)
    ax.set_ylim(0, max(np.max(slice_cd) * 1.05, 1))
    ax.set_xlabel("Angle (Degrees)", color="#CCCCCC", fontsize=11, labelpad=10)
    ax.set_ylabel("Intensity (Candela)", color="#CCCCCC", fontsize=11, labelpad=10)
    ax.grid(True, color="#333333", linestyle="--", alpha=0.5)

    plt.title(f"{title_str}\n[Intensity Profile: {suffix_name}]", color="#CCCCCC", pad=15)
    plt.tight_layout()

    if save_path and config.export_plots:
        base, extension = os.path.splitext(save_path)
        plt.savefig(f"{base}_{suffix_name}{extension}", facecolor="black",
                    edgecolor="none", dpi=150, bbox_inches="tight")

    # This figure is never handed back to the GUI, so release it or the Agg
    # backend accumulates one per profile for the life of the process.
    plt.close(figure)


def _format_exposure_caption(config: SimulationConfig) -> str:
    """Describes the active camera exposure in one line."""
    if config.use_auto_exposure:
        return f"Exposure: Auto (EV {config.auto_exposure_compensation_ev:+.1f})"

    if config.cam_shutter_speed_s < 1.0:
        shutter = "1/" + str(int(1.0 / config.cam_shutter_speed_s))
    else:
        shutter = config.cam_shutter_speed_s
    return f"Exposure: ISO {config.cam_iso} | f/{config.cam_f_stop} | {shutter}s"


def _format_beam_geometry(metrics: BeamMetrics, config: SimulationConfig) -> str:
    """Formats the beam measurements shown in the corner of the wall shot."""
    distance = config.target_distance_m
    return (f"Spill Angle: {metrics.spill_angle_deg:.1f}°\n"
            f"Spill Ø @ {distance}m: {metrics.spill_diameter_m:.2f}m\n"
            f"Corona Angle: {metrics.corona_angle_deg:.1f}°\n"
            f"Corona Ø @ {distance}m: {metrics.corona_diameter_m:.2f}m\n"
            f"Hotspot Angle: {metrics.hotspot_angle_deg:.1f}°\n"
            f"Hotspot Ø @ {distance}m: {metrics.hotspot_diameter_m:.2f}m\n"
            f"Cd/Lm Ratio: {metrics.candela_per_lumen:.1f} cd/lm\n")


def _format_output_modes(emitter: dict, max_amps: float, max_cd: float,
                         total_flux: float, delivered_flux: float,
                         config: SimulationConfig) -> str:
    """Builds the output table for the usual 1/10/35/100 percent drive levels.

    Lumens displayed are the "delivered" lumens that survive the reflector, lens,
    and gasket to land inside the canvas. Output scales with drive current, and
    intensity is scaled from the simulated maximum the same way, since the beam
    shape does not change.

    Args:
        emitter: Emitter specs.
        max_amps: Maximum drive current.
        max_cd: Peak intensity at that current.
        total_flux: Output leaving the die at that current.
        delivered_flux: Output reaching the wall at that current.
        config: Active configuration.

    Returns:
        The formatted table.
    """
    survival = delivered_flux / total_flux if total_flux else 0.0
    table = (" Mode | Amps |  Lumens |  Candela | Throw \n"
             + "-" * 45 + "\n")
    for fraction in (0.01, 0.10, 0.35, 1.0):
        amps = max_amps * fraction
        lumens = calculate_lumens(emitter, amps, config)
        candela = max_cd * (lumens / total_flux) if total_flux else 0.0
        table += (f"{int(fraction * 100):>4}% | {amps:>4.1f} | "
                  f"{int(lumens * survival):>7,} | "
                  f"{int(candela):>8,} | {int(np.sqrt(candela * 4)):>4,}m\n")
    return table


class WallShot(NamedTuple):
    """Everything needed to draw a wall shot, kept apart from the trace.

    The camera turns illuminance into a picture, and nothing about that feeds
    back into the ray tracing. Holding these four values means the exposure can
    be changed and the shot redrawn in a moment, instead of tracing the whole
    ray budget again to look at the same beam differently.

    Attributes:
        wall_lux: Illuminance on the wall, in lux.
        title: The header block for the figure.
        geometry_text: The beam geometry overlay.
        modes_text: The output table overlay.
        filename: What to call the PNG if it is exported again.
    """

    wall_lux: np.ndarray
    title: str
    geometry_text: str
    modes_text: str
    filename: str


def render_wall_shot(shot: WallShot, config: SimulationConfig,
                     save_path: Optional[str] = None,
                     always_save: bool = False):
    """Draws a wall shot at the current camera settings.

    Args:
        shot: The stored render inputs from a completed simulation.
        config: Active configuration, for the camera settings.
        save_path: PNG to write, or None to render without saving.
        always_save: Writes the file even when export_plots is off. That
            setting governs whether a run leaves files behind on its own;
            asking for the shot to be exported is a separate instruction
            and should not be silently ignored because of it.

    Returns:
        The Matplotlib figure.
    """
    return _render_wall_shot(
        apply_camera_exposure_and_tonemap(shot.wall_lux, config),
        shot.title, shot.geometry_text, shot.modes_text, config, save_path,
        always_save)


def _render_wall_shot(render_data: np.ndarray, title_str: str, geometry_text: str,
                      modes_text: str, config: SimulationConfig,
                      save_path: Optional[str], always_save: bool = False):
    """Renders the simulated photograph of the beam on the wall.

    Args:
        render_data: Tone mapped image in the range 0-1.
        title_str: Two-line hardware and results header.
        geometry_text: Beam measurements for the bottom left overlay.
        modes_text: Output table for the bottom right overlay.
        config: Active configuration.
        save_path: Where to write the PNG, or None to only return the figure.

    Returns:
        The Matplotlib figure, which the GUI embeds in its canvas.
    """
    figure, ax = plt.subplots(figsize=(10, 10), facecolor="black")
    _style_dark_axes(ax)

    ax.imshow(render_data,
              extent=[-config.wall_radius_m, config.wall_radius_m,
                      -config.wall_radius_m, config.wall_radius_m],
              cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.set(xlim=(-config.plot_radius_m, config.plot_radius_m),
           ylim=(-config.plot_radius_m, config.plot_radius_m))
    ax.set_xlabel("Horizontal Distance (m)", color="#CCCCCC", fontsize=11, labelpad=10)
    ax.set_ylabel("Vertical Distance (m)", color="#CCCCCC", fontsize=11, labelpad=10)

    if config.show_human_silhouette:
        # Feet at 65% of the figure's height below the beam axis.
        draw_human_silhouette(ax, 0.0, -1.75 * 0.65, 1.75)

    overlay = dict(facecolor="black", alpha=0.7, edgecolor="none", pad=6)
    ax.text(0.02, 0.02, geometry_text.strip(), transform=ax.transAxes,
            color="#CCCCCC", fontsize=9, va="bottom", bbox=overlay)
    ax.text(0.98, 0.02, modes_text.strip(), transform=ax.transAxes,
            color="#CCCCCC", fontsize=9, family="monospace",
            ha="right", va="bottom", bbox=overlay)

    mm_per_pixel = ((2.0 * config.wall_radius_m) / config.sim_grid_res) * 1000.0
    plt.figtext(0.5, 0.015,
                f"Canvas FOV: {config.canvas_fov_deg}° | Plot FOV: {config.plot_fov_deg}° | "
                f"Grid Res: {mm_per_pixel:.1f} mm/px | [{_format_exposure_caption(config)}]",
                color="#CCCCCC", fontsize=9, ha="center", va="bottom",
                bbox=dict(facecolor="black", alpha=0.7, edgecolor="none", pad=4))
    # The header runs to two lines and the hardware names can be long,
    # so the axes give up a strip at the top rather than let it clip.
    plt.title(title_str, color="#CCCCCC", fontsize=10, pad=12)
    plt.tight_layout(rect=[0.01, 0.05, 0.99, 0.95])

    if save_path and (always_save or config.export_plots):
        plt.savefig(save_path, facecolor="black", edgecolor="none",
                    dpi=150, bbox_inches="tight")

    return figure


def generate_flashlight_plot(emitter_name: str, reflector_name: str, gasket_name: str,
                             finish_type: str, config: SimulationConfig,
                             library: HardwareLibrary,
                             log_callback: LogCallback = None,
                             progress_callback: ProgressCallback = None,
                             is_cancelled_callback: CancelCallback = None,
                             save_path: str = None,
                             emitter_offset: EmitterOffset = NO_EMITTER_OFFSET):
    """Simulates one hardware combination and renders the requested plots.

    Args:
        emitter_name: Emitter to simulate.
        reflector_name: Reflector to simulate.
        gasket_name: Gasket to simulate.
        finish_type: "smooth" or "orange_peel".
        config: Active configuration.
        library: Hardware catalogue the three names are looked up in.
        log_callback: Receives progress text.
        progress_callback: Receives completion percentage.
        is_cancelled_callback: Polled during tracing; returns True to stop.
        save_path: PNG path for the wall shot. Intensity profiles derive their
            filenames from it. None renders without saving.
        emitter_offset: Centring error of the emitter for this run. It belongs
            to the build rather than to any catalogue part, so it is passed in
            here instead of being read from the reflector specs.

    Returns:
        (figure, results, shot): the wall shot figure (None when plot_wall_shot is
        off) and a dict of headline results, or (None, None) if cancelled.
    """
    reflector = library.get("reflector", reflector_name)
    emitter = library.get("emitter", emitter_name)
    gasket = library.get("gasket", gasket_name)
    max_amps = emitter["max_current_amps"]

    geom = get_sim_geometry(reflector, emitter, gasket, finish_type, config,
                            emitter_offset)
    illumination = simulate_wall_illuminance(
        geom, emitter, max_amps, finish_type, config,
        log_callback, progress_callback, is_cancelled_callback)

    if illumination is None:
        return None, None, None  # Cancelled.

    # Peak intensity, and the ANSI throw distance where it falls to 0.25 lux.
    max_cd = np.max(illumination.total_lux) * (config.target_distance_m ** 2)
    throw_m = int(np.sqrt(max_cd / 0.25))

    metrics = get_beam_metrics(illumination.total_lux, illumination.hotspot_lux,
                               illumination.spill_lux, max_cd,
                               illumination.delivered_lumens, config)

    # The hardware line carries the three names and nothing else: which part is
    # which is obvious from the names themselves. The finish stays with the
    # reflector it belongs to, because it changes the result.
    title_str = (
        f"{reflector_name} ({finish_type.replace('_', ' ').title()}) | "
        f"{emitter_name} | {gasket_name}\n"
        f"Distance: {config.target_distance_m}m | "
        f"Bore: {geom['effective_d_hole']:.1f}mm | "
        f"Focus: {geom['focus_delta']:+.2f}mm | "
        f"Max Intensity: {int(max_cd):,} cd | Throw: {throw_m:,}m")

    # A centred emitter is the normal case, so it is left out of the title.
    offset_text = emitter_offset.describe()
    if offset_text:
        title_str += f" | {offset_text}"

    shot = WallShot(illumination.total_lux, title_str,
                    _format_beam_geometry(metrics, config),
                    _format_output_modes(emitter, max_amps, max_cd,
                                         illumination.total_lumens,
                                         illumination.delivered_lumens,
                                         config),
                    _plot_filename(reflector_name, emitter_name,
                                   gasket_name, finish_type))

    figure = None
    if config.plot_wall_shot:
        if log_callback:
            log_callback("Rendering final camera visualization...")
        figure = render_wall_shot(shot, config, save_path)

    if config.export_ies and save_path:
        watts = max_amps * forward_voltage(emitter, max_amps, config)
        summary = export_beam_ies(
            os.path.splitext(save_path)[0] + ".ies", illumination.total_lux,
            config, {
                "catalogue": reflector_name,
                "luminaire": (f"{reflector_name} | {emitter_name} | "
                              f"{gasket_name}"),
                "lamp": emitter_name,
                "lumens": illumination.total_lumens,
                "watts": watts,
                "notes": [
                    f"Finish: {finish_type.replace('_', ' ').title()}",
                    f"Peak intensity: {int(max_cd):,} cd",
                ] + ([emitter_offset.describe()]
                     if emitter_offset.describe() else []),
            })
        if log_callback:
            log_callback(summary)

    # Slices through the centre of the wall, in metres from the beam axis.
    axis_distance = np.linspace(-config.wall_radius_m, config.wall_radius_m,
                                config.sim_grid_res)
    centre = int((config.sim_grid_res - 1) / 2.0)

    if config.plot_intensity_x:
        render_intensity_profile(illumination.total_lux[centre, :], axis_distance,
                                 "X-Axis", title_str, save_path, config)
    if config.plot_intensity_y:
        render_intensity_profile(illumination.total_lux[:, centre], axis_distance,
                                 "Y-Axis", title_str, save_path, config)
    if config.plot_intensity_45:
        diagonal_distance = np.linspace(-config.wall_radius_m * math.sqrt(2),
                                        config.wall_radius_m * math.sqrt(2),
                                        config.sim_grid_res)
        render_intensity_profile(np.diagonal(illumination.total_lux), diagonal_distance,
                                 "45-Deg", title_str, save_path, config)

    return figure, {
        "Reflector": reflector_name,
        "Emitter": emitter_name,
        "Gasket": gasket_name,
        "Finish": finish_type.upper(),
        "Max Candela (cd)": int(max_cd),
        "Throw (m)": int(throw_m),
        "Total Lumens": int(illumination.total_lumens),
        "Wall Lumens": int(illumination.delivered_lumens),
        "Spill Angle (deg)": round(metrics.spill_angle_deg, 1),
        "Corona Angle (deg)": round(metrics.corona_angle_deg, 1),
        "Hotspot Angle (deg)": round(metrics.hotspot_angle_deg, 1),
        "Cd/Lm Ratio": round(metrics.candela_per_lumen, 1),
    }, shot
# ==============================================================================
# 6. API EXECUTION ENTRY POINT
# ==============================================================================

CSV_HEADERS = [
    "Reflector", "Emitter", "Gasket", "Finish", "Max Candela (cd)", "Throw (m)",
    "Total Lumens", "Wall Lumens", "Spill Angle (deg)", "Corona Angle (deg)",
    "Hotspot Angle (deg)", "Cd/Lm Ratio",
]


def _results_key(row: dict) -> tuple:
    """Identity of one result row: the hardware combination it describes."""
    return (row["Reflector"], row["Emitter"], row.get("Gasket", "None"), row["Finish"])


def _read_existing_results(csv_filepath: str) -> dict:
    """Loads previously exported results so a new run tops them up.

    Args:
        csv_filepath: CSV written by an earlier run, which need not exist.

    Returns:
        Result rows keyed by hardware combination; empty if there is no file.
    """
    if not os.path.exists(csv_filepath):
        return {}
    with open(csv_filepath, mode="r", newline="") as handle:
        return {_results_key(row): row for row in csv.DictReader(handle)}


def _list_valid_combinations(library: HardwareLibrary) -> list:
    """Enumerates every hardware combination worth simulating in batch mode.

    Combinations where the emitter is too large for the reflector are skipped:
    a footprint wider than a third of the reflector diameter cannot produce a
    meaningful beam.

    Args:
        library: Hardware catalogue to enumerate.

    Returns:
        (reflector, emitter, gasket, finish) tuples.
    """
    combinations = []
    for reflector_name in library.names("reflector"):
        reflector = library.get("reflector", reflector_name)
        for emitter_name in library.names("emitter"):
            emitter = library.get("emitter", emitter_name)
            footprint_diagonal = np.sqrt(emitter["footprint_x_mm"] ** 2
                                         + emitter["footprint_y_mm"] ** 2)
            if footprint_diagonal > (reflector["diameter_mm"] / 3.0):
                continue
            for gasket_name in library.names("gasket"):
                for finish in ("smooth", "orange_peel"):
                    combinations.append(
                        (reflector_name, emitter_name, gasket_name, finish))
    return combinations


def _sanitise_filename_part(name: str) -> str:
    """Makes one hardware name safe to embed in a filename.

    Catalogue names are free text and routinely contain characters Windows
    reserves for paths, for example "Convoy M1/M21B 7mm". Left alone, the slash
    is read as a directory separator and the save fails on a folder that does
    not exist.

    Args:
        name: Hardware name as it appears in the catalogue.

    Returns:
        The name with reserved and control characters replaced by hyphens, and
        with the trailing dots and spaces Windows refuses to store stripped off.
    """
    cleaned = "".join("-" if character in _RESERVED_FILENAME_CHARS or ord(character) < 32
                      else character
                      for character in name)
    return cleaned.strip(" .") or "unnamed"


def _plot_filename(reflector: str, emitter: str, gasket: str, finish: str) -> str:
    """Builds the PNG filename for one hardware combination."""
    finish_tag = "OP" if finish == "orange_peel" else "SMO"
    return "_".join((_sanitise_filename_part(reflector),
                     _sanitise_filename_part(emitter),
                     _sanitise_filename_part(gasket),
                     finish_tag)) + ".png"


def run_simulation_job(config: SimulationConfig, library: HardwareLibrary,
                       active_reflector: str, active_emitter: str, active_gasket: str,
                       finish: str,
                       log_callback: LogCallback = None,
                       progress_callback: ProgressCallback = None,
                       is_cancelled_callback: CancelCallback = None,
                       emitter_offset: EmitterOffset = NO_EMITTER_OFFSET):
    """Runs the simulator: one hardware combination, or the whole batch.

    config.generate_all_plots selects between rendering the given combination
    and sweeping every valid combination in the library. Results are merged into
    the CSV for the current distance and exposure, so repeated runs accumulate.

    Args:
        config: Active configuration.
        library: Hardware catalogue.
        active_reflector: Reflector for a single render.
        active_emitter: Emitter for a single render.
        active_gasket: Gasket for a single render.
        finish: "smooth" or "orange_peel" for a single render.
        log_callback: Receives progress text.
        progress_callback: Receives completion percentage.
        is_cancelled_callback: Polled during tracing; returns True to stop.
        emitter_offset: Centring error of the emitter. It describes the build
            rather than any one part, so in batch mode it applies to every
            combination swept.

    Returns:
        (figure, results, shot): the wall shot figure for a single render
        (None in batch mode), every result row keyed by hardware
        combination, and the render inputs so the shot can be redrawn at a
        different exposure without tracing again. (None, None, None) if the
        run was cancelled.
    """
    output_dir = config.resolved_output_directory
    os.makedirs(output_dir, exist_ok=True)

    # Sampling problems are silent otherwise: the run finishes and the
    # numbers look plausible, they are just not symmetric.
    for warning in angular_sampling_warnings(config):
        if log_callback:
            log_callback(f"Warning: {warning}")

    if config.use_auto_exposure:
        exposure_id = f"Auto_EV_{config.auto_exposure_compensation_ev:+.1f}"
    else:
        shutter = ("1_" + str(int(1.0 / config.cam_shutter_speed_s))
                   if config.cam_shutter_speed_s < 1.0 else config.cam_shutter_speed_s)
        exposure_id = f"ISO{config.cam_iso}_f{config.cam_f_stop}_{shutter}s"

    csv_filepath = os.path.join(
        output_dir, f"sim_results_{config.target_distance_m}m_{exposure_id}.csv")
    results = _read_existing_results(csv_filepath) if config.export_csv else {}

    if config.generate_all_plots:
        if log_callback:
            log_callback(f"Batch generation enabled. Outputting to: {output_dir}")

        combinations = _list_valid_combinations(library)
        for position, (reflector, emitter, gasket, combo_finish) in enumerate(combinations, 1):
            if is_cancelled_callback and is_cancelled_callback():
                return None, None, None
            if log_callback:
                log_callback(f"[{position}/{len(combinations)}] Rendering {reflector} + "
                             f"{emitter} + {gasket} ({combo_finish.upper()})...")

            _, metrics, _ = generate_flashlight_plot(
                emitter, reflector, gasket, combo_finish, config, library,
                log_callback, progress_callback, is_cancelled_callback,
                os.path.join(output_dir,
                             _plot_filename(reflector, emitter, gasket, combo_finish)),
                emitter_offset)
            if metrics:
                results[_results_key(metrics)] = metrics

            # Batch figures are never displayed, so drop them instead of
            # climbing through memory one 10x10 inch canvas at a time.
            plt.close("all")

        if log_callback:
            log_callback("Batch generation complete!")
        figure, shot = None, None

    else:
        if log_callback:
            log_callback(f"Starting specific render: {active_reflector} + "
                         f"{active_emitter} + {active_gasket}")

        figure, metrics, shot = generate_flashlight_plot(
            active_emitter, active_reflector, active_gasket, finish, config, library,
            log_callback, progress_callback, is_cancelled_callback,
            os.path.join(output_dir, _plot_filename(
                active_reflector, active_emitter, active_gasket, finish)),
            emitter_offset)
        if metrics:
            results[_results_key(metrics)] = metrics
            if log_callback:
                log_callback("Simulation complete.")

    if config.export_csv:
        with open(csv_filepath, mode="w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
            writer.writeheader()
            writer.writerows(results.values())

    return figure, results, shot
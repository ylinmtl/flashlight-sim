"""The settings file and the values read from it.

Part of the flashlight simulator core; see core/__init__.py for the
public surface.
"""

import math
import os
from typing import Callable, Dict, List, Optional

from .paths import _read_json, _write_json, resource_path, user_data_path


# Callback aliases used throughout the public API.
LogCallback = Optional[Callable[[str], None]]
ProgressCallback = Optional[Callable[[float], None]]
CancelCallback = Optional[Callable[[], bool]]
# Settings that record which hardware the operator last picked. They live in the
# settings file for convenience but are never restored from the template.
_SELECTION_KEYS = frozenset({
    "active_emitter_name",
    "active_reflector_name",
    "active_gasket_name",
    "reflector_finish",
})
# ==============================================================================
# 1. CONFIGURATION & DATA MANAGEMENT
# ==============================================================================


class SimulationConfig:
    """Simulation constraints, thresholds and camera settings, backed by JSON.

    Settings are grouped into categories on disk so the file stays readable, but
    are exposed as flat attributes (``config.sim_grid_res``) because the math
    engine reads them inside tight loops. _CATEGORIES maps between the two
    layouts; any setting missing from it is kept at the root of the file.

    Attributes:
        filepath: Writable JSON file holding the live settings.
        default_filepath: Read-only template shipped with the application.
        restored_settings: Names of the settings the last load had to take from
            the template because the user's file predates them.
    """

    _CATEGORIES = {
        "Output & Rendering": [
            "generate_all_plots", "show_human_silhouette", "plot_wall_shot",
            "plot_intensity_x", "plot_intensity_y", "plot_intensity_45",
            "batch_output_directory", "export_csv", "export_plots",
        ],
        "Simulation Space & Constraints": [
            "use_gpu", "max_multiple_reflections", "use_reflector_opening",
            "target_distance_m", "canvas_fov_deg", "plot_fov_deg",
        ],
        "Camera Settings": [
            "use_auto_exposure", "auto_exposure_compensation_ev",
            "cam_iso", "cam_f_stop", "cam_shutter_speed_s",
        ],
        "Resolution & Angular Density": [
            "sim_grid_res", "sim_emitter_elements", "sim_theta_step_deg",
            "sim_phi_step_deg", "sim_theta_min_deg", "sim_theta_max_deg",
            "sim_phi_min_deg", "sim_phi_max_deg",
        ],
        "IES Export": [
            "export_ies", "ies_vertical_step_deg",
            "ies_horizontal_step_deg", "ies_max_vertical_angle_deg",
        ],
        "Material Defaults & Thresholds": [
            "default_reflectivity_smooth", "default_reflectivity_op",
            "default_reflectivity_cylinder", "default_gasket_reflectivity",
            "default_op_blur_strength", "default_op_factor",
            "default_transmissivity_lens", "default_surface_finish",
            "spill_visible_threshold_lux", "corona_visible_threshold",
            "hotspot_fwhm_threshold", "default_gasket_thickness_mm",
            "default_gasket_total_height_mm",
            "default_gasket_inner_diameter_mm",
            "default_gasket_wall_shape",
            "default_reflector_wall_thickness_mm",
            "default_reflector_base_thickness_mm", "default_focus_offset_mm",
            "default_opening_diameter_mm", "default_dome_size_mm",
            "default_refractive_index", "default_emitter_shape",
            "default_emitter_output_mode", "default_max_lumens",
            "default_forward_voltage_v", "default_vf_turn_on_v",
            "default_vf_scale", "default_base_efficacy_lm_w",
            "default_droop_factor",
        ],
    }

    def __init__(self, filepath: str = None, default_filepath: str = None):
        """Loads settings, topping them up from the template where needed.

        The two files deliberately live in different places:

            simulation_settings.json  the operator's own copy, always beside
                flashlight-sim.py when run from source and beside the .exe when
                frozen, because it is written to.
            default_settings.json     the read-only template, next to the
                source when run from source and inside the PyInstaller bundle
                when frozen, because it ships with the application.

        Args:
            filepath: Override for the writable settings file.
            default_filepath: Override for the read-only template.
        """
        self.filepath = filepath or user_data_path("simulation_settings.json")
        self.default_filepath = default_filepath or resource_path("default_settings.json")
        self.load_settings()

    @staticmethod
    def _flatten(data: dict):
        """Yields (key, value) pairs from a category-nested settings dict.

        Args:
            data: Settings as stored on disk.

        Yields:
            One (key, value) pair per setting. Scalars stored at the root of
            the file, such as the active hardware selection, are passed through
            unchanged.
        """
        for key, value in data.items():
            if isinstance(value, dict):
                yield from value.items()
            else:
                yield key, value

    def load_settings(self) -> None:
        """Reads the settings file, topping it up from the shipped template.

        A settings file written by an earlier version of the application has no
        entry for any setting added since. Rather than reject it, or leave the
        engine reading an attribute that does not exist, the template is loaded
        first and the user's file layered over it. Old files therefore keep
        every value the operator chose and silently gain the defaults for
        whatever is new. Which settings had to be restored is recorded in
        restored_settings so the caller can report the upgrade.

        Raises:
            FileNotFoundError: If neither the settings file nor the template
                can be found.
        """
        has_settings = os.path.exists(self.filepath)
        has_template = os.path.exists(self.default_filepath)
        if not has_settings and not has_template:
            raise FileNotFoundError(
                f"CRITICAL: Missing both '{self.filepath}' and '{self.default_filepath}'."
            )

        defaults = (dict(self._flatten(_read_json(self.default_filepath)))
                    if has_template else {})
        stored = (dict(self._flatten(_read_json(self.filepath)))
                  if has_settings else {})

        # Underscored so save_settings leaves it out of the file it writes. An
        # absent settings file is a first run rather than an upgrade, so there
        # is nothing to report in that case even though everything is new.
        self._restored_settings = (sorted(set(defaults) - set(stored))
                                   if has_settings else [])

        # The user's own values win; the template only fills the gaps. Settings
        # the template no longer lists are kept, so nothing is lost either way.
        for key, value in {**defaults, **stored}.items():
            setattr(self, key, value)

        # Write straight back so a first run materialises the user's own copy
        # and an upgraded one keeps the settings it has just gained.
        self.save_settings()

    def save_settings(self) -> None:
        """Writes the flat attributes back to disk in the category layout."""
        pending = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith("_") and key not in ("filepath", "default_filepath")
        }

        nested = {}
        for category, keys in self._CATEGORIES.items():
            nested[category] = {key: pending.pop(key) for key in keys if key in pending}
        nested.update(pending)  # Anything unmapped stays at the root of the file.

        _write_json(self.filepath, nested)

    def reset_to_defaults(self) -> None:
        """Restores every tunable from the shipped template, in memory only.

        The active hardware selection is left untouched, and nothing is written
        to disk until save_settings is called.

        Raises:
            FileNotFoundError: If the template is missing.
        """
        if not os.path.exists(self.default_filepath):
            raise FileNotFoundError(f"Could not find '{self.default_filepath}'.")

        for key, value in self._flatten(_read_json(self.default_filepath)):
            if key not in _SELECTION_KEYS:
                setattr(self, key, value)

    @property
    def restored_settings(self) -> List[str]:
        """Settings the last load had to take from the shipped template.

        Empty unless the settings file on disk was written by an older version
        of the application, so a first run, which has no settings file at all,
        reports nothing. A property rather than a plain attribute so that
        save_settings, which serialises everything in __dict__, never sees it.
        """
        return list(self._restored_settings)

    @property
    def wall_radius_m(self) -> float:
        """Half-width of the simulated wall, from the canvas field of view."""
        return self.target_distance_m * math.tan(math.radians(self.canvas_fov_deg / 2.0))

    @property
    def plot_radius_m(self) -> float:
        """Half-width of the visible plot area, from the plot field of view."""
        return self.target_distance_m * math.tan(math.radians(self.plot_fov_deg / 2.0))

    @property
    def resolved_output_directory(self) -> str:
        """batch_output_directory anchored to the app folder, never the CWD.

        A frozen app launched from a shortcut inherits an arbitrary working
        directory, so a relative path would otherwise scatter exports.
        """
        out_dir = self.batch_output_directory
        return out_dir if os.path.isabs(out_dir) else user_data_path(out_dir)

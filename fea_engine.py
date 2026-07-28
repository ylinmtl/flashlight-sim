"""Finite element ray-tracing engine for flashlight beam simulation.

The engine fires rays from a subdivided LED die, refracts them through the
silicone dome, bounces them off the parabolic reflector, the reflector's centre
bore and the gasket, then accumulates the surviving flux on a virtual wall to
produce an illuminance map and the usual beam figures (candela, throw, hotspot
and spill angles).

Rays are traced by a Numba kernel that runs on CUDA when a usable GPU toolchain
is present and on the CPU otherwise. Importing this module also configures the
CUDA toolkit bundled by PyInstaller, so it must be imported before Numba.

Typical usage:

    config = SimulationConfig()
    library = HardwareLibrary()
    figure, results = run_simulation_job(
        config, library, "Convoy M3", "Luminus SFT40 6500K", "9mm 5050", "smooth")
"""

import copy
import csv
import json
import math
import os
import shutil
import sys
import time
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

# Callback aliases used throughout the public API.
LogCallback = Optional[Callable[[str], None]]
ProgressCallback = Optional[Callable[[float], None]]
CancelCallback = Optional[Callable[[], bool]]

# The three interchangeable hardware categories, and the JSON section each one
# is stored under in hardware_library.json.
HARDWARE_KINDS = ("emitter", "reflector", "gasket")
_JSON_SECTION_BY_KIND = {
    "emitter": "emitters",
    "reflector": "reflectors",
    "gasket": "gaskets",
}

# Settings that record which hardware the operator last picked. They live in the
# settings file for convenience but are never restored from the template.
_SELECTION_KEYS = frozenset({
    "active_emitter_name",
    "active_reflector_name",
    "active_gasket_name",
    "reflector_finish",
})

# Surfaces a ray can land on. Numba freezes module level integers as compile
# time constants, which is why these are plain ints rather than an Enum.
_HIT_NONE = 0
_HIT_PARABOLA = 1
_HIT_CYLINDER = 2
_HIT_ABSORBED = 3
_HIT_GASKET = 4

# Returned by solve_quadratic when no usable root exists. Large enough that the
# nearest-hit comparisons always reject it.
_NO_ROOT = 1e9

# Characters Windows reserves for paths and wildcards. Hardware names are free
# text and end up inside export filenames, so they have to be replaced first.
_RESERVED_FILENAME_CHARS = '<>:"/\\|?*'

# CUDA launch geometry. 256 threads per block suits every architecture the
# simulator targets.
_THREADS_PER_BLOCK = 256


def resource_path(relative_path: str) -> str:
    """Returns the absolute path of a read-only asset shipped with the app.

    Args:
        relative_path: Path of the asset relative to the application root.

    Returns:
        The path inside the PyInstaller bundle when frozen, otherwise the path
        next to this source file.
    """
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


def user_data_path(relative_path: str) -> str:
    """Returns the absolute path of a writable file that lives beside the app.

    Anything the operator edits (settings, the hardware library, exported plots)
    must not be written into the PyInstaller bundle: under --onefile the bundle
    is a temporary directory wiped on exit, and under --onedir it may sit on
    read-only media.

    Args:
        relative_path: Path of the file relative to the application folder.

    Returns:
        The path next to the executable when frozen, otherwise the path next to
        this source file.
    """
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)


def _bootstrap_bundled_cuda() -> list:
    """Points Numba at the CUDA toolkit bundled inside a frozen build.

    Numba locates the toolkit through CUDA_HOME (falling back to CUDA_PATH) and
    expects a genuine toolkit layout underneath it:

        <CUDA_HOME>/nvvm/bin/nvvm64_*.dll
        <CUDA_HOME>/nvvm/libdevice/libdevice.*.bc
        <CUDA_HOME>/bin/cudart64_*.dll  (plus ptxas.exe and zlibwapi.dll)

    When that lookup fails Numba falls back to the bare name "nvvm.dll", which a
    frozen build cannot resolve, and the first kernel launch dies with
    NvvmSupportError. Python 3.8 and later ignore PATH when loading DLLs, so the
    directories must also be registered with os.add_dll_directory.

    Returns:
        The directory handles returned by os.add_dll_directory. They are kept
        alive for the lifetime of the process; dropping them un-registers the
        directories.
    """
    handles = []
    if not (getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")):
        return handles

    bundle_dir = sys._MEIPASS
    cuda_bin = os.path.join(bundle_dir, "bin")
    nvvm_bin = os.path.join(bundle_dir, "nvvm", "bin")

    if os.path.isdir(nvvm_bin):
        os.environ["CUDA_HOME"] = bundle_dir
        os.environ["CUDA_PATH"] = bundle_dir

    for directory in (cuda_bin, nvvm_bin):
        if not os.path.isdir(directory):
            continue
        # PATH still matters for child processes such as ptxas.exe.
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            try:
                handles.append(os.add_dll_directory(directory))
            except OSError:
                pass
    return handles


# Must run before Numba is imported: Numba caches its toolkit path lookup
# the first time it is asked for it. The handles are kept in a module
# level name on purpose. Nothing reads them, but dropping the reference
# would let the loaded DLLs be released again.
_CUDA_DLL_DIRECTORIES = _bootstrap_bundled_cuda()

# Agg keeps Matplotlib off the GUI thread and stops it opening detached windows.
import matplotlib  # noqa: E402  (import order is deliberate, see above)

matplotlib.use("Agg")

import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from numba import cuda, njit  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402


def _read_json(path: str) -> dict:
    """Reads and parses a UTF-8 JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, data: dict) -> None:
    """Writes a dict to disk as indented UTF-8 JSON."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)


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
            "sim_phi_min_deg", "sim_phi_max_deg", "lumen_calc_step_deg",
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


# ==============================================================================
# 2. HARDWARE LIBRARIES
# ==============================================================================


# Every hardware spec that has a default in the settings file, as
# kind -> {spec: setting}. A catalogue entry written before a spec existed has
# no entry for it at all, so HardwareLibrary.restore_missing_specs copies the
# value in from the settings when the application starts. This is also the
# table get_sim_geometry reads its own fall backs from, so the default for a
# spec is defined in exactly one place.
#
# A spec absent from this table is mandatory: an entry without it describes no
# real part, so the engine raises KeyError rather than inventing a number.
# Specs the engine ignores entirely, such as a gasket's outer diameter, are
# left alone and simply carried along.
# Die outlines the tracer knows how to sample. "square" fills a plain
# rectangle, "round" masks a circle out of it, and "polygon" masks an arbitrary
# outline given by the die_outline spec, which is how anything else is modelled:
# the chamfered corners of an SFT60, the corner notches of an SFT40, and so on.
DIE_SHAPES = ("square", "round", "polygon")

# Reflector surface treatments. The finish selects which of the
# reflector's two reflectivity figures applies to the parabola.
SURFACE_FINISHES = ("smooth", "orange_peel")

# Shape of the gasket wall that surrounds the emitter. A round wall
# leaves a circular aperture; a square one follows the die package.
GASKET_WALL_SHAPES = ("round", "square")

# How an emitter's light output is described. Advanced works it out
# from the electrical specs; simple takes a rated figure at maximum
# current and scales it, which needs one number instead of four.
OUTPUT_MODES = ("simple", "advanced")

# Specs a later version renamed, as kind -> {old name: (new name, factor)}.
# thickness_diameter_mm was the total reduction in diameter, both walls at
# once. wall_thickness_mm is a single wall, so a catalogue written before the
# change holds a number that has to be halved as it is carried across.
# Without this the same figure would quietly model a wall twice as thick.
RENAMED_SPECS = {
    "reflector": {"thickness_diameter_mm": ("wall_thickness_mm", 0.5)},
    "gasket": {
        "gasket_thickness_mm": ("thickness_mm", 1.0),
        "gasket_total_height_mm": ("total_height_mm", 1.0),
        "gasket_opening_mm": ("inner_diameter_mm", 1.0),
    },
}

# Probes per axis used to work out how much of a boundary cell lies on the die.
# Eight gives sixty four probes per cell, which places the emitting area of a
# circle or a chamfered die within about 0.2% of its true value. The cost is a
# few milliseconds once per simulation, so accuracy is worth more than speed
# here. Raising it further converges roughly in proportion to 1 / probes.
DIE_SUBSAMPLES = 8

SPEC_DEFAULT_SETTINGS = {
    "reflector": {
        "opening_diameter_mm": "default_opening_diameter_mm",
        "focus_offset_mm": "default_focus_offset_mm",
        "wall_thickness_mm": "default_reflector_wall_thickness_mm",
        "thickness_height_mm": "default_reflector_base_thickness_mm",
        "reflectivity_smooth": "default_reflectivity_smooth",
        "reflectivity_op": "default_reflectivity_op",
        "reflectivity_cylinder": "default_reflectivity_cylinder",
        "gasket_reflectivity": "default_gasket_reflectivity",
        "OP_Factor": "default_op_factor",
        "transmissivity_lens": "default_transmissivity_lens",
        "surface_finish": "default_surface_finish",
    },
    "emitter": {
        "output_mode": "default_emitter_output_mode",
        "max_lumens": "default_max_lumens",
        "forward_voltage_v": "default_forward_voltage_v",
        "vf_turn_on_v": "default_vf_turn_on_v",
        "vf_scale": "default_vf_scale",
        "base_efficacy_lm_w": "default_base_efficacy_lm_w",
        "droop_factor": "default_droop_factor",
        "dome_size_mm": "default_dome_size_mm",
        "refractive_index": "default_refractive_index",
        "shape": "default_emitter_shape",
    },
    "gasket": {
        "inner_diameter_mm": "default_gasket_inner_diameter_mm",
        "wall_shape": "default_gasket_wall_shape",
        "thickness_mm": "default_gasket_thickness_mm",
        "total_height_mm": "default_gasket_total_height_mm",
    },
}


def emitter_footprint_diagonal(emitter: dict) -> float:
    """Returns the diagonal across an emitter's package, in millimetres.

    This is the narrowest hole the package can pass through, so it is the
    smallest bore a reflector can have and still clear the emitter.

    Args:
        emitter: Emitter specs.

    Returns:
        The diagonal in millimetres.
    """
    return math.sqrt(float(emitter["footprint_x_mm"]) ** 2
                     + float(emitter["footprint_y_mm"]) ** 2)


def effective_bore_diameter(reflector: dict, emitter: dict,
                            config: "SimulationConfig") -> float:
    """Returns the reflector bore the tracer will actually use.

    Three cases, and the GUI and the tracer both come here so they cannot
    disagree about which one applies:

    * An opening of zero has never been measured, so the emitter's footprint
      diagonal stands in for it. Forcing the catalogue value cannot override
      this, because there is no measured value to force.
    * With use_reflector_opening set, a measured opening is used as it stands,
      even where the emitter will not fit through it.
    * Otherwise the bore is opened out to clear the emitter if it has to be.

    Args:
        reflector: Reflector specs.
        emitter: Emitter specs.
        config: Active configuration.

    Returns:
        The bore diameter in millimetres.
    """
    bore = float(spec_or_default(reflector, "reflector", "opening_diameter_mm",
                                 config))
    diagonal = emitter_footprint_diagonal(emitter)
    if bore <= 0.0:
        return diagonal
    return bore if config.use_reflector_opening else max(bore, diagonal)


def spec_or_default(specs: dict, kind: str, name: str, config: "SimulationConfig"):
    """Reads one hardware spec, falling back to its default in the settings.

    Args:
        specs: Specs of one catalogue entry.
        kind: One of HARDWARE_KINDS.
        name: Spec to read.
        config: Active configuration, holding the defaults.

    Returns:
        The stored value, or the settings default when the entry predates the
        spec.

    Raises:
        KeyError: If the spec is mandatory, meaning it has no default in
            SPEC_DEFAULT_SETTINGS, and the entry does not carry it.
    """
    if name in specs:
        return specs[name]
    return getattr(config, SPEC_DEFAULT_SETTINGS[kind][name])


class HardwareLibrary:
    """The catalogue of emitters, reflectors and gaskets, backed by JSON.

    Every method takes a ``kind`` from HARDWARE_KINDS, since the three
    catalogues behave identically and differ only in which specs they hold.

    Attributes:
        filepath: Writable JSON file holding the catalogue, or None on a
            detached copy from copy_for_run(), which is never written.
        default_filepath: Read-only catalogue shipped with the application.
    """

    def __init__(self, filepath: str = None):
        """Loads the catalogue, seeding it from the shipped copy if needed.

        Args:
            filepath: Override for the writable catalogue file.
        """
        self.filepath = filepath or user_data_path("hardware_library.json")
        self.default_filepath = resource_path("hardware_library.json")
        self._catalogues: Dict[str, dict] = {kind: {} for kind in HARDWARE_KINDS}
        self.load_database()

    def _catalogue(self, kind: str) -> dict:
        """Returns the live dict for one hardware kind.

        Args:
            kind: One of HARDWARE_KINDS.

        Returns:
            The name -> specs mapping for that kind.

        Raises:
            ValueError: If kind is not a known hardware kind.
        """
        try:
            return self._catalogues[kind]
        except KeyError:
            raise ValueError(
                f"Unknown hardware kind '{kind}'; expected one of {HARDWARE_KINDS}."
            ) from None

    def load_database(self) -> None:
        """Reads the catalogue from disk.

        On the first run of a frozen build the read-only copy inside the bundle
        is used to seed the writable copy beside the executable.

        Raises:
            FileNotFoundError: If no catalogue can be found.
        """
        if (not os.path.exists(self.filepath)
                and os.path.exists(self.default_filepath)
                and os.path.abspath(self.filepath) != os.path.abspath(self.default_filepath)):
            shutil.copy2(self.default_filepath, self.filepath)

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Could not find {self.filepath}.")

        data = _read_json(self.filepath)
        for kind in HARDWARE_KINDS:
            self._catalogues[kind] = data.get(_JSON_SECTION_BY_KIND[kind], {})

    def save_database(self) -> None:
        """Writes all three catalogues back to disk.

        Raises:
            RuntimeError: If this is a detached copy from copy_for_run(),
                which exists only to be read by one simulation.
        """
        if self.filepath is None:
            raise RuntimeError(
                "This catalogue is a detached copy made for a single simulation "
                "run and has no file to be written to.")

        _write_json(self.filepath, {
            _JSON_SECTION_BY_KIND[kind]: self._catalogues[kind] for kind in HARDWARE_KINDS
        })

    def names(self, kind: str) -> List[str]:
        """Returns the names of every entry of one kind, case-insensitively sorted."""
        return sorted(self._catalogue(kind).keys(), key=str.casefold)

    def get(self, kind: str, name: str) -> dict:
        """Returns the live specs dict for one entry.

        The dict is not copied, so callers that mutate it change the in-memory
        catalogue. Use save() to persist, or apply_overrides() to change the
        specs for a single run.

        Args:
            kind: One of HARDWARE_KINDS.
            name: Name of the entry.

        Returns:
            The specs mapping for that entry.

        Raises:
            KeyError: If no entry of that kind has that name.
        """
        return self._catalogue(kind)[name]

    def save(self, kind: str, name: str, specs: dict) -> None:
        """Adds or updates one entry and writes the catalogue to disk.

        The given specs are merged over whatever is stored, rather than
        replacing it. The editing form only shows the specs it knows about, so
        a straight replacement would silently drop anything else the entry
        carries, including specs restored by restore_missing_specs and specs
        the engine does not read at all.

        Args:
            kind: One of HARDWARE_KINDS.
            name: Name of the entry to add or update.
            specs: Specs to store.
        """
        self._catalogue(kind).setdefault(name, {}).update(specs)
        self.save_database()

    def delete(self, kind: str, name: str) -> None:
        """Removes one entry, if present, and writes the catalogue to disk."""
        catalogue = self._catalogue(kind)
        if name in catalogue:
            del catalogue[name]
            self.save_database()

    def import_new_entries(self) -> Dict[str, List[str]]:
        """Adds catalogue entries the shipped library has gained, by name.

        restore_missing_specs only tops up variables inside entries that
        already exist under a given name; it has no way to notice a whole new
        reflector, emitter or gasket that a later release adds to
        hardware_library.json, because the user's catalogue has never heard of
        that name. This is the counterpart that closes that gap: every name
        present in the shipped copy but absent from the live one is copied
        across whole. An entry the operator already has, whether shipped or
        their own, is never touched, so nothing they have customised is
        overwritten.

        Run once at start up, immediately before restore_missing_specs so a
        newly imported entry is caught by that pass too, in case the shipped
        copy is ever a version or two behind the running code.

        Returns:
            Names added, keyed by kind. Empty when the catalogue already has
            every name the shipped library does, or when there is no separate
            shipped copy to compare against.
        """
        if not os.path.exists(self.default_filepath):
            return {}
        if os.path.abspath(self.filepath) == os.path.abspath(self.default_filepath):
            return {}  # Running from source: both paths are the same file.

        shipped = _read_json(self.default_filepath)
        added = {}
        for kind in HARDWARE_KINDS:
            shipped_entries = shipped.get(_JSON_SECTION_BY_KIND[kind], {})
            live = self._catalogue(kind)
            new_names = sorted(name for name in shipped_entries if name not in live)
            for name in new_names:
                live[name] = copy.deepcopy(shipped_entries[name])
            if new_names:
                added[kind] = new_names

        if added:
            self.save_database()
        return added

    def rename_legacy_specs(self) -> Dict[str, List[str]]:
        """Carries specs a later version renamed across to their new names.

        Run once at start up, before restore_missing_specs, because a spec that
        has been carried across must not then be treated as missing and filled
        with a default. Where the meaning changed as well as the name the
        stored value is converted, which is the whole point: a wall thickness
        that used to mean both walls at once would otherwise be read as one
        wall and quietly double the thickness of every reflector.

        The old key is removed, so the rename happens once and a second start
        up finds nothing to do.

        Returns:
            The renames applied, as "kind/name" -> ["old -> new", ...]. Empty
            when the catalogue is already current.
        """
        renamed = {}
        for kind, replacements in RENAMED_SPECS.items():
            for name, specs in self._catalogue(kind).items():
                applied = []
                for old_name, (new_name, factor) in replacements.items():
                    if old_name not in specs:
                        continue
                    value = specs.pop(old_name)
                    # An explicit value already under the new name wins; the
                    # old one is just dropped.
                    if new_name not in specs:
                        specs[new_name] = round(value * factor, 6)
                    applied.append(f"{old_name} -> {new_name}")
                if applied:
                    renamed[f"{kind}/{name}"] = applied

        if renamed:
            self.save_database()
        return renamed

    def restore_missing_specs(self, config: "SimulationConfig") -> Dict[str, List[str]]:
        """Fills in every spec any catalogue entry is missing, from the settings.

        Run once at start up, this compares every entry against
        SPEC_DEFAULT_SETTINGS rather than only checking whether the catalogue
        file exists, so an entry written by an older version quietly gains the
        specs added since instead of falling back to a default on every read.
        The catalogue is written back only when something actually changed.

        Args:
            config: Active configuration, holding the defaults.

        Returns:
            The specs restored, keyed by "kind/name". Empty when the catalogue
            already carries every spec.
        """
        restored = {}
        for kind, spec_settings in SPEC_DEFAULT_SETTINGS.items():
            for name, specs in self._catalogue(kind).items():
                added = [spec for spec in spec_settings if spec not in specs]
                for spec in added:
                    specs[spec] = getattr(config, spec_settings[spec])
                if added:
                    restored[f"{kind}/{name}"] = added

        if restored:
            self.save_database()
        return restored

    def copy_for_run(self) -> "HardwareLibrary":
        """Returns a detached copy of the catalogue for one simulation run.

        Unsaved edits typed into the form apply to the run that is starting and
        to nothing else. Patching the live catalogue would arm the next
        save_database() to write them out, so a run works on its own deep copy
        instead. The copy carries no file path, which makes it an error for
        anything holding it to try to write it.

        Returns:
            A HardwareLibrary holding a deep copy of every entry, suitable for
            apply_overrides and for reading, but not for saving.
        """
        detached = HardwareLibrary.__new__(HardwareLibrary)
        detached.filepath = None
        detached.default_filepath = None
        detached._catalogues = copy.deepcopy(self._catalogues)
        return detached

    def apply_overrides(self, kind: str, name: str, specs: dict) -> None:
        """Merges edited specs into one entry of a detached run copy.

        This is how the GUI lets an operator tweak a value for a single run
        without editing the catalogue. It refuses to touch the live catalogue,
        because an override left there is indistinguishable from a saved value
        and would be written out by the next save of any entry, of any kind.

        Args:
            kind: One of HARDWARE_KINDS.
            name: Name of the entry to patch.
            specs: Specs to merge over the stored ones.

        Raises:
            RuntimeError: If called on the live catalogue rather than on a
                detached copy from copy_for_run().
        """
        if self.filepath is not None:
            raise RuntimeError(
                "apply_overrides() works only on a detached copy from "
                "copy_for_run(); overriding the live catalogue would leak "
                "unsaved edits into the next save.")

        self._catalogue(kind)[name].update(specs)


# ==============================================================================
# 3. HELPERS & HARDWARE INTERPOLATION
# ==============================================================================


def lambertian_intensity(theta_rad: np.ndarray) -> np.ndarray:
    """Relative intensity of a Lambertian emitter at the given polar angles.

    Args:
        theta_rad: Polar angles measured from the optical axis, in radians.

    Returns:
        cos(theta) for angles within the forward hemisphere, 0 outside it.
    """
    intensity = np.cos(theta_rad)
    intensity[np.abs(np.degrees(theta_rad)) > 90.0] = 0.0
    return intensity


def forward_voltage(emitter: dict, current_amps: float,
                    config: "SimulationConfig") -> float:
    """Returns the emitter's forward voltage at a drive current.

    Args:
        emitter: Emitter specs.
        current_amps: Drive current in amps.
        config: Active configuration, for the mode and its defaults.

    Returns:
        Forward voltage in volts.
    """
    if spec_or_default(emitter, "emitter", "output_mode", config) == "simple":
        return float(spec_or_default(emitter, "emitter", "forward_voltage_v", config))

    return float(spec_or_default(emitter, "emitter", "vf_turn_on_v", config)
                 + spec_or_default(emitter, "emitter", "vf_scale", config)
                 * math.log(current_amps + 1.0))


def calculate_lumens(emitter: dict, current_amps: float,
                     config: "SimulationConfig") -> float:
    """Estimates total luminous flux for an emitter at a drive current.

    Which of the two models applies is the emitter's own choice:

    * "advanced" works from the electrical specs. Forward voltage is modelled
      logarithmically and efficacy decays exponentially with current, so the
      efficiency lost at high drive is accounted for.
    * "simple" takes the rated output at maximum current and scales it in
      proportion to current. It has no way to know about droop, so it reads
      high in the middle of the range, but it only needs the one figure that
      every datasheet prints.

    Args:
        emitter: Emitter specs.
        current_amps: Drive current in amps.
        config: Active configuration, for the mode and its defaults.

    Returns:
        Total output in lumens.
    """
    if spec_or_default(emitter, "emitter", "output_mode", config) == "simple":
        max_amps = float(emitter["max_current_amps"])
        rated = float(spec_or_default(emitter, "emitter", "max_lumens", config))
        return rated * (current_amps / max_amps) if max_amps > 0.0 else 0.0

    power_watts = current_amps * forward_voltage(emitter, current_amps, config)
    efficacy = (spec_or_default(emitter, "emitter", "base_efficacy_lm_w", config)
                * np.exp(-spec_or_default(emitter, "emitter", "droop_factor", config)
                         * current_amps))
    return power_watts * efficacy


class EmitterOffset(NamedTuple):
    """How far the emitter sits off the reflector's optical axis, in polar form.

    A centring error is read off a beam shot as a direction and a distance, so
    that is how it is described here. The angle is measured in the plane of the
    wall image: 0 degrees points straight up and the angle increases clockwise,
    which puts 90 degrees to the right.

    This is a property of the build being simulated rather than of any catalogue
    part, so it is passed into a run and never stored with the reflector.

    Attributes:
        distance_mm: Distance from the axis in millimetres; 0 is centred.
        angle_deg: Direction of the offset in degrees clockwise from straight up.
    """

    distance_mm: float = 0.0
    angle_deg: float = 0.0

    @property
    def x_mm(self) -> float:
        """Horizontal component of the offset, positive towards +x."""
        return self.distance_mm * math.sin(math.radians(self.angle_deg))

    @property
    def y_mm(self) -> float:
        """Vertical component of the offset, positive towards +y."""
        return self.distance_mm * math.cos(math.radians(self.angle_deg))

    def describe(self) -> str:
        """Summarises the offset for a plot title, or "" when it is centred."""
        if self.distance_mm == 0.0:
            return ""
        return f"Emitter Offset: {self.distance_mm:.2f}mm @ {self.angle_deg:.1f}°"


# A perfectly centred emitter: the default wherever an offset is not supplied.
NO_EMITTER_OFFSET = EmitterOffset()


def get_sim_geometry(reflector: dict, emitter: dict, gasket: dict, finish: str,
                     config: SimulationConfig,
                     emitter_offset: EmitterOffset = NO_EMITTER_OFFSET) -> dict:
    """Derives the ray-tracing geometry from a hardware combination.

    Everything is expressed in millimetres in a coordinate system whose origin
    sits at the focus of the parabola, with +z pointing out of the flashlight.

    Args:
        reflector: Reflector specs.
        emitter: Emitter specs.
        gasket: Gasket specs.
        finish: Either "smooth" or "orange_peel"; selects which reflectivity of
            the reflector applies.
        config: Active configuration, used for any spec the hardware omits.
        emitter_offset: Centring error of the emitter. Only the emitter package
            moves: the reflector and the gasket stay on the axis, because both
            are located by the head rather than by the emitter.

    Returns:
        A dict of scalars consumed by the tracing kernels, plus three values
        used for reporting: effective_d_hole, focus_delta and op_multiplier.
    """
    # Inner diameter of the reflective surface. The spec is one wall, so it
    # comes off the diameter twice, once on each side.
    inner_diameter = reflector["diameter_mm"] - 2.0 * spec_or_default(
        reflector, "reflector", "wall_thickness_mm", config)
    radius_max = inner_diameter / 2.0
    total_height = reflector["height_mm"]

    bore_diameter = spec_or_default(
        reflector, "reflector", "opening_diameter_mm", config)
    focus_offset_mm = spec_or_default(
        reflector, "reflector", "focus_offset_mm", config)
    base_thickness_mm = spec_or_default(
        reflector, "reflector", "thickness_height_mm", config)

    gasket_thickness_mm = spec_or_default(gasket, "gasket", "thickness_mm", config)
    gasket_total_height_mm = spec_or_default(
        gasket, "gasket", "total_height_mm", config)
    gasket_opening_mm = spec_or_default(
        gasket, "gasket", "inner_diameter_mm", config)

    effective_d_hole = effective_bore_diameter(reflector, emitter, config)
    r_hole = effective_d_hole / 2.0

    # Focal length of the parabola that is `total_height` deep and `radius_max`
    # wide, shifted by the operator's focus offset.
    effective_height = total_height - focus_offset_mm
    focal_length = (-effective_height
                    + math.sqrt(effective_height ** 2 + radius_max ** 2)) / 2.0

    # Vertical cuts: the reflector floor, the top of the bore and the mouth.
    z_bottom = focal_length - focus_offset_mm
    z_min_cut = z_bottom + base_thickness_mm
    z_max_cut = z_bottom + total_height
    z_intersect = (r_hole ** 2) / (4.0 * focal_length)
    z_hole_top = float(max(z_intersect, z_min_cut))

    # The gasket only protrudes above the floor by the part that is not
    # compressed under the emitter board.
    z_gasket_top = z_bottom + max(0.0, gasket_total_height_mm - gasket_thickness_mm)

    if gasket_opening_mm > 0.0:
        # Round aperture: a single radius describes it.
        r_gasket = gasket_opening_mm / 2.0
        gasket_x_half, gasket_y_half = 0.0, 0.0
        is_cylindrical_gasket = 1
    else:
        # Rectangular aperture: it hugs the emitter footprint.
        r_gasket = 0.0
        gasket_x_half = emitter["footprint_x_mm"] / 2.0
        gasket_y_half = emitter["footprint_y_mm"] / 2.0
        is_cylindrical_gasket = 0

    # Height of the light emitting surface once the gasket is compressed.
    ez_base = z_bottom + (emitter["height_mm"] - gasket_thickness_mm)

    # dome_size_mm of -1 means "as wide as the narrowest footprint edge".
    dome_input = spec_or_default(emitter, "emitter", "dome_size_mm", config)
    dome_diameter = (min(emitter["footprint_x_mm"], emitter["footprint_y_mm"])
                     if dome_input == -1 else max(0.0, dome_input))

    reflectivity_parabola = spec_or_default(
        reflector, "reflector",
        "reflectivity_op" if finish == "orange_peel" else "reflectivity_smooth",
        config)

    return {
        "focal_length": focal_length,
        "z_bottom": z_bottom,
        "z_min_cut": z_min_cut,
        "z_hole_top": z_hole_top,
        "z_max_cut": z_max_cut,
        "radius_max": radius_max,
        "r_hole": r_hole,
        "z_gasket_top": z_gasket_top,
        "r_gasket": r_gasket,
        "gasket_x_half": gasket_x_half,
        "gasket_y_half": gasket_y_half,
        "is_cylindrical_gasket": is_cylindrical_gasket,
        "ez_base": ez_base,
        "dome_radius": dome_diameter / 2.0,
        "refractive_index": spec_or_default(
            emitter, "emitter", "refractive_index", config),
        "refl_para": reflectivity_parabola,
        "refl_cyl": spec_or_default(
            reflector, "reflector", "reflectivity_cylinder", config),
        "refl_gask": spec_or_default(
            reflector, "reflector", "gasket_reflectivity", config),
        # Fraction of flux surviving the lens.
        "transmissivity_lens": spec_or_default(
            reflector, "reflector", "transmissivity_lens", config),
        "emitter_offset_x": emitter_offset.x_mm,
        "emitter_offset_y": emitter_offset.y_mm,
        # Reported, not traced:
        "effective_d_hole": effective_d_hole,
        "focus_delta": ez_base - focal_length,
        "op_multiplier": spec_or_default(reflector, "reflector", "OP_Factor", config),
    }


# ==============================================================================
# 4. MATH & FINITE ELEMENT ANALYSIS (FEA) ENGINE
# ==============================================================================


@njit
def solve_quadratic(a, b, c):
    """Solves a*t^2 + b*t + c = 0 for ray/surface intersections.

    Args:
        a, b, c: Quadratic coefficients.

    Returns:
        The two roots, smallest first, or (_NO_ROOT, _NO_ROOT) when the ray runs
        parallel to the surface or misses it entirely.
    """
    if a < 1e-8:
        return _NO_ROOT, _NO_ROOT

    discriminant = b ** 2 - 4.0 * a * c
    if discriminant < 0.0:
        return _NO_ROOT, _NO_ROOT

    root = math.sqrt(discriminant)
    return (-b - root) / (2.0 * a), (-b + root) / (2.0 * a)


@njit
def apply_dome_refraction(ex, ey, ez, vx, vy, vz, dome_radius, refractive_index):
    """Refracts a ray as it leaves the emitter's silicone dome.

    The dome is a hemisphere of radius dome_radius resting on the emitter
    surface and centred on the middle of the die, so the plane the die elements
    sit on is the hemisphere's flat base and ez is the height of its centre. A
    ray leaves a die element inside the silicone, meets the curved surface from
    within, and is bent in three dimensions by the vector form of Snell's law
    about the surface normal at the exit point.

    Args:
        ex, ey, ez: Ray origin in millimetres. ez is the emitter surface, which
            is also the height of the hemisphere's centre.
        vx, vy, vz: Direction of the ray; expected to be a unit vector.
        dome_radius: Hemisphere radius in millimetres.
        refractive_index: Dome index divided by the index of air.

    Returns:
        (blocked, x, y, z, vx, vy, vz): the exit point on the dome and the
        refracted direction. A ray that never meets the silicone is returned
        unchanged and unblocked; a ray trapped by total internal reflection is
        returned with blocked=True, and the caller drops it.
    """
    # Ray origin relative to the centre of the dome, which sits at (0, 0, ez).
    # The die plane passes through that centre, so the local height is zero.
    origin_x, origin_y, origin_z = ex, ey, 0.0

    # Intersect the ray with the sphere the hemisphere is part of.
    _, t_exit = solve_quadratic(
        vx ** 2 + vy ** 2 + vz ** 2,
        2.0 * (origin_x * vx + origin_y * vy + origin_z * vz),
        origin_x ** 2 + origin_y ** 2 + origin_z ** 2 - dome_radius ** 2)

    # A ray starting inside the silicone always leaves through the far root. No
    # root, or only roots behind the origin, means the element sits outside the
    # dome footprint and the ray never enters the silicone at all.
    if t_exit >= _NO_ROOT or t_exit <= 0.0:
        return False, ex, ey, ez, vx, vy, vz

    hit_x = origin_x + t_exit * vx
    hit_y = origin_y + t_exit * vy
    hit_z = origin_z + t_exit * vz

    # Below the flat base the ray is leaving through the die rather than the
    # curved surface, so nothing is refracted; the reflector floor absorbs it.
    if hit_z < 0.0:
        return False, ex, ey, ez, vx, vy, vz

    # Outward normal of a sphere is the radius through the hit point.
    nx = hit_x / dome_radius
    ny = hit_y / dome_radius
    nz = hit_z / dome_radius

    cos_in = vx * nx + vy * ny + vz * nz
    ratio = refractive_index
    cos_out_sq = 1.0 - ratio ** 2 * (1.0 - cos_in ** 2)

    if cos_out_sq < 0.0:
        # Past the critical angle: the ray is totally internally reflected and
        # is treated as lost inside the package.
        return True, ex, ey, ez, vx, vy, vz

    # Snell's law in vector form: the refracted ray keeps the tangential part of
    # the incoming direction, scaled by the index ratio, and is re-tilted along
    # the normal by the difference in the two cosines.
    cos_out = math.sqrt(cos_out_sq)
    bend = ratio * cos_in - cos_out
    vx = ratio * vx - bend * nx
    vy = ratio * vy - bend * ny
    vz = ratio * vz - bend * nz

    magnitude = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    return (False, hit_x, hit_y, ez + hit_z,
            vx / magnitude, vy / magnitude, vz / magnitude)


@njit
def process_single_ray(ex, ey, ez_base, vx, vy, vz, flux,
                       focal_length, z_bottom, z_min_cut, z_hole_top, z_max_cut,
                       radius_max, r_hole, target_z_mm, grid_res, wall_radius_m,
                       reflectivity_parabola, reflectivity_cylinder, reflectivity_gasket,
                       dome_radius, refractive_index, max_multiple_reflections,
                       z_gasket_top, r_gasket, gasket_x_half, gasket_y_half,
                       is_cylindrical_gasket,
                       emitter_offset_x, emitter_offset_y, transmissivity_lens):
    """Traces one ray from the die until it lands on the wall or is absorbed.

    Each pass through the loop finds the nearest surface along the ray, then
    either reflects off it (losing flux to its reflectivity), stops there, or,
    when nothing is hit, lets the ray escape towards the wall.

    Args:
        ex, ey, ez_base: Ray origin in millimetres.
        vx, vy, vz: Unit direction of the ray.
        flux: Luminous flux carried by the ray, in lumens.
        focal_length: Focal length of the parabola.
        z_bottom, z_min_cut, z_hole_top, z_max_cut: Vertical cuts of the
            reflector: floor, top of the base, top of the bore, and mouth.
        radius_max: Radius of the reflector mouth.
        r_hole: Radius of the centre bore.
        target_z_mm: Distance to the wall in millimetres.
        grid_res: Width of the square accumulation grid in pixels.
        wall_radius_m: Half-width of the simulated wall in metres.
        reflectivity_parabola, reflectivity_cylinder, reflectivity_gasket:
            Surviving fraction of flux per bounce off each surface.
        dome_radius: Dome radius, or 0 for a dedomed emitter.
        refractive_index: Dome refractive index.
        max_multiple_reflections: Extra bounces allowed beyond the first.
        z_gasket_top: Height of the exposed part of the gasket.
        r_gasket: Aperture radius for a round gasket.
        gasket_x_half, gasket_y_half: Half extents for a rectangular gasket.
        is_cylindrical_gasket: 1 for a round aperture, 0 for a rectangular one.
        emitter_offset_x, emitter_offset_y: Centring error of the emitter
            package in millimetres, as Cartesian components.
        transmissivity_lens: Fraction of flux surviving the lens.

    Returns:
        (flux, row, col, bounces) for a ray that reaches the wall, otherwise
        (0.0, -1, -1, -1).
    """
    blocked = False
    if dome_radius > 0.0:
        # A domed emitter fires into silicone, so the ray is refracted at the
        # dome surface before it sees any of the reflector.
        blocked, ex, ey, ez_base, vx, vy, vz = apply_dome_refraction(
            ex, ey, ez_base, vx, vy, vz, dome_radius, refractive_index)

    # Running ray state: position, direction and remaining flux. The centring
    # error moves the whole emitter package, dome included, so the shift is
    # applied after refraction, which is solved in emitter-local coordinates.
    # Translating a ray moves where it starts and leaves its direction alone.
    px, py, pz = ex + emitter_offset_x, ey + emitter_offset_y, ez_base
    dx, dy, dz = vx, vy, vz
    remaining_flux = flux

    bounces = 0
    bin_size = (2.0 * wall_radius_m) / grid_res

    while not blocked:
        hit_type, t_hit = _HIT_NONE, _NO_ROOT

        # --- Centre bore wall (a cylinder of radius r_hole) ---
        t_first, t_second = solve_quadratic(
            dx ** 2 + dy ** 2,
            2.0 * (px * dx + py * dy),
            px ** 2 + py ** 2 - r_hole ** 2)
        for t_bore in (t_first, t_second):
            if 1e-4 < t_bore < t_hit and z_bottom <= (pz + t_bore * dz) <= z_hole_top:
                t_hit, hit_type = t_bore, _HIT_CYLINDER

        # --- Downward facing planes ---
        if dz < 0.0:
            # Flat annulus around the bore, only present when the bore does not
            # reach the parabola before the base thickness does.
            if z_hole_top == z_min_cut and pz > z_min_cut:
                t_plane = (z_min_cut - pz) / dz
                if (1e-4 < t_plane < t_hit
                        and (px + t_plane * dx) ** 2 + (py + t_plane * dy) ** 2 > r_hole ** 2):
                    t_hit, hit_type = t_plane, _HIT_ABSORBED

            # Top face of the gasket, i.e. everything outside its aperture.
            if z_gasket_top > z_bottom and pz > z_gasket_top:
                t_plane = (z_gasket_top - pz) / dz
                if 1e-4 < t_plane < t_hit:
                    hit_x, hit_y = px + t_plane * dx, py + t_plane * dy
                    if ((is_cylindrical_gasket == 1
                         and hit_x ** 2 + hit_y ** 2 >= r_gasket ** 2)
                            or (is_cylindrical_gasket == 0
                                and (abs(hit_x) >= gasket_x_half
                                     or abs(hit_y) >= gasket_y_half))):
                        t_hit, hit_type = t_plane, _HIT_GASKET

            # The reflector floor absorbs anything that gets this far.
            if pz > z_bottom:
                t_plane = (z_bottom - pz) / dz
                if 1e-4 < t_plane < t_hit:
                    t_hit, hit_type = t_plane, _HIT_ABSORBED

        # --- Upward through the floor from inside the bore ---
        elif dz > 0.0 and pz < z_bottom:
            t_plane = (z_bottom - pz) / dz
            if (1e-4 < t_plane < t_hit
                    and (px + t_plane * dx) ** 2 + (py + t_plane * dy) ** 2 > r_hole ** 2):
                t_hit, hit_type = t_plane, _HIT_ABSORBED

        # --- Parabolic reflector (x^2 + y^2 = 4*f*z) ---
        t_first, t_second = solve_quadratic(
            dx ** 2 + dy ** 2,
            2.0 * (px * dx + py * dy) - 4.0 * focal_length * dz,
            px ** 2 + py ** 2 - 4.0 * focal_length * pz)
        for t_para in (t_first, t_second):
            if 1e-4 < t_para < t_hit and z_hole_top <= (pz + t_para * dz) <= z_max_cut:
                t_hit, hit_type = t_para, _HIT_PARABOLA

        # --- Inner wall of the gasket aperture ---
        if z_gasket_top > z_bottom:
            if is_cylindrical_gasket == 1:
                _, t_far = solve_quadratic(
                    dx ** 2 + dy ** 2,
                    2.0 * (px * dx + py * dy),
                    px ** 2 + py ** 2 - r_gasket ** 2)
                if 1e-4 < t_far < t_hit and (pz + t_far * dz) <= z_gasket_top:
                    t_hit, hit_type = t_far, _HIT_GASKET
            else:
                # Rectangular aperture: test the two facing side walls.
                if abs(dx) > 1e-8:
                    t_x = (gasket_x_half * (1.0 if dx > 0.0 else -1.0) - px) / dx
                    if (1e-4 < t_x < t_hit and (pz + t_x * dz) <= z_gasket_top
                            and -gasket_y_half <= (py + t_x * dy) <= gasket_y_half):
                        t_hit, hit_type = t_x, _HIT_GASKET
                if abs(dy) > 1e-8:
                    t_y = (gasket_y_half * (1.0 if dy > 0.0 else -1.0) - py) / dy
                    if (1e-4 < t_y < t_hit and (pz + t_y * dz) <= z_gasket_top
                            and -gasket_x_half <= (px + t_y * dx) <= gasket_x_half):
                        t_hit, hit_type = t_y, _HIT_GASKET

        # --- Resolve the nearest hit ---
        if hit_type == _HIT_PARABOLA or hit_type == _HIT_CYLINDER or hit_type == _HIT_GASKET:
            if bounces >= max_multiple_reflections + 1:
                return 0.0, -1, -1, -1
            bounces += 1

            hit_x, hit_y, hit_z = px + t_hit * dx, py + t_hit * dy, pz + t_hit * dz

            if hit_type == _HIT_PARABOLA:
                # Surface normal of x^2 + y^2 - 4*f*z = 0, pointing inwards.
                nx, ny, nz = -hit_x, -hit_y, 2.0 * focal_length
                reflectivity = reflectivity_parabola
            elif hit_type == _HIT_CYLINDER:
                nx, ny, nz = -hit_x, -hit_y, 0.0
                reflectivity = reflectivity_cylinder
            else:
                if abs(hit_z - z_gasket_top) < 1e-4 and dz < 0.0:
                    nx, ny, nz = 0.0, 0.0, 1.0  # Landed on the gasket's top face.
                elif is_cylindrical_gasket == 1:
                    nx, ny, nz = -hit_x, -hit_y, 0.0
                elif abs(abs(hit_x) - gasket_x_half) < 1e-4:
                    nx, ny, nz = (-1.0 if hit_x > 0.0 else 1.0), 0.0, 0.0
                else:
                    nx, ny, nz = 0.0, (-1.0 if hit_y > 0.0 else 1.0), 0.0
                reflectivity = reflectivity_gasket

            magnitude = math.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
            nx, ny, nz = nx / magnitude, ny / magnitude, nz / magnitude

            # Mirror the direction about the surface normal.
            projection = dx * nx + dy * ny + dz * nz
            dx = dx - 2.0 * projection * nx
            dy = dy - 2.0 * projection * ny
            dz = dz - 2.0 * projection * nz
            px, py, pz = hit_x, hit_y, hit_z
            remaining_flux *= reflectivity

        elif hit_type == _HIT_ABSORBED:
            return 0.0, -1, -1, -1

        else:
            # Nothing left to hit: the ray either clears the mouth or is
            # clipped by the reflector wall on its way out.
            if dz > 0.0:
                t_mouth = (z_max_cut - pz) / dz
                exit_x = px + t_mouth * dx
                exit_y = py + t_mouth * dy

                if math.sqrt(exit_x ** 2 + exit_y ** 2) <= radius_max + 1e-4:
                    # Every ray leaving the head crosses the lens, whether it
                    # bounced off the reflector or is spill straight from the
                    # die, so both are attenuated at this single point. The
                    # loss is a plain scalar for now; making it depend on the
                    # path length through the glass means using the direction
                    # (dx, dy, dz), which is in hand right here.
                    remaining_flux *= transmissivity_lens

                    t_wall = (target_z_mm - pz) / dz
                    col = int((((px + t_wall * dx) / 1000.0) + wall_radius_m) / bin_size)
                    row = int((((py + t_wall * dy) / 1000.0) + wall_radius_m) / bin_size)

                    if 0 <= col < grid_res and 0 <= row < grid_res:
                        return remaining_flux, row, col, bounces
            return 0.0, -1, -1, -1

    return 0.0, -1, -1, -1


@cuda.jit
def ray_trace_kernel_gpu(start_idx, end_idx, element_x, element_y,
                         element_weight, ray_vx, ray_vy, ray_vz, ray_flux,
                         focal_length, ez_base, z_bottom, z_min_cut, z_hole_top,
                         z_max_cut, radius_max, r_hole, target_z_mm, grid_res,
                         wall_radius_m, reflectivity_parabola, reflectivity_cylinder,
                         reflectivity_gasket, dome_radius, refractive_index,
                         max_multiple_reflections, z_gasket_top, r_gasket,
                         gasket_x_half, gasket_y_half, is_cylindrical_gasket,
                         emitter_offset_x, emitter_offset_y, transmissivity_lens,
                         hotspot_grid, spill_grid):
    """Traces one (die element, ray direction) pair per CUDA thread.

    The work is a flat range of indices so it can be dispatched in chunks; see
    _build_kernel_args for the argument order and execute_tracers for the
    chunking. Reflected and direct light are accumulated separately so the two
    can be blurred independently later.

    Args:
        start_idx: First flat work index this launch is responsible for.
        end_idx: One past the last work index.
        element_x, element_y: Die element coordinates in millimetres.
        element_weight: Share of the emitter's flux carried by each
            element, proportional to the die area it stands for. Sums to 1.
        ray_vx, ray_vy, ray_vz: Unit ray directions.
        ray_flux: Flux carried by each direction, in lumens.
        hotspot_grid: Accumulator for rays that bounced at least once.
        spill_grid: Accumulator for rays that reached the wall directly.
        Remaining args: see process_single_ray.
    """
    index = cuda.grid(1) + start_idx
    if index >= end_idx:
        return

    rays_per_element = ray_vx.shape[0]
    element_idx = index // rays_per_element
    ray_idx = index % rays_per_element

    final_flux, row, col, bounces = process_single_ray(
        element_x[element_idx], element_y[element_idx], ez_base,
        ray_vx[ray_idx], ray_vy[ray_idx], ray_vz[ray_idx],
        ray_flux[ray_idx] * element_weight[element_idx],
        focal_length, z_bottom, z_min_cut, z_hole_top, z_max_cut,
        radius_max, r_hole, target_z_mm, grid_res, wall_radius_m,
        reflectivity_parabola, reflectivity_cylinder, reflectivity_gasket,
        dome_radius, refractive_index, max_multiple_reflections,
        z_gasket_top, r_gasket, gasket_x_half, gasket_y_half, is_cylindrical_gasket,
        emitter_offset_x, emitter_offset_y, transmissivity_lens)

    if row != -1 and col != -1:
        cuda.atomic.add(hotspot_grid if bounces > 0 else spill_grid, (row, col), final_flux)


@njit
def ray_trace_kernel_cpu(start_idx, end_idx, element_x, element_y,
                         element_weight, ray_vx, ray_vy, ray_vz, ray_flux,
                         focal_length, ez_base, z_bottom, z_min_cut, z_hole_top,
                         z_max_cut, radius_max, r_hole, target_z_mm, grid_res,
                         wall_radius_m, reflectivity_parabola, reflectivity_cylinder,
                         reflectivity_gasket, dome_radius, refractive_index,
                         max_multiple_reflections, z_gasket_top, r_gasket,
                         gasket_x_half, gasket_y_half, is_cylindrical_gasket,
                         emitter_offset_x, emitter_offset_y, transmissivity_lens,
                         hotspot_grid, spill_grid):
    """CPU twin of ray_trace_kernel_gpu; walks the index range in a loop.

    Args:
        Identical to ray_trace_kernel_gpu.
    """
    rays_per_element = ray_vx.shape[0]
    for index in range(start_idx, end_idx):
        element_idx = index // rays_per_element
        ray_idx = index % rays_per_element

        final_flux, row, col, bounces = process_single_ray(
            element_x[element_idx], element_y[element_idx], ez_base,
            ray_vx[ray_idx], ray_vy[ray_idx], ray_vz[ray_idx],
        ray_flux[ray_idx] * element_weight[element_idx],
            focal_length, z_bottom, z_min_cut, z_hole_top, z_max_cut,
            radius_max, r_hole, target_z_mm, grid_res, wall_radius_m,
            reflectivity_parabola, reflectivity_cylinder, reflectivity_gasket,
            dome_radius, refractive_index, max_multiple_reflections,
            z_gasket_top, r_gasket, gasket_x_half, gasket_y_half, is_cylindrical_gasket,
            emitter_offset_x, emitter_offset_y, transmissivity_lens)

        if row != -1 and col != -1:
            if bounces > 0:
                hotspot_grid[row, col] += final_flux
            else:
                spill_grid[row, col] += final_flux


def _build_kernel_args(element_x, element_y, element_weight,
                      ray_vx, ray_vy, ray_vz, ray_flux,
                       geom: dict, config: SimulationConfig, target_z_mm: float,
                       hotspot_grid, spill_grid) -> tuple:
    """Packs every kernel argument into one tuple, in the kernels' parameter order.

    Both kernels are launched as ``kernel(start, end, *args)``, so this is the
    single place where the order is defined. The explicit float()/int() casts
    keep Numba from recompiling the kernel when a JSON setting happens to load
    as an int rather than a float.

    Args:
        element_x, element_y: Die element coordinates, C-contiguous float64.
        element_weight: Flux share per element, C-contiguous float64.
        ray_vx, ray_vy, ray_vz: Unit ray directions, C-contiguous float64.
        ray_flux: Flux per ray direction, C-contiguous float64.
        geom: Output of get_sim_geometry.
        config: Active configuration.
        target_z_mm: Distance to the wall in millimetres.
        hotspot_grid, spill_grid: Accumulators, on the host or the device.

    Returns:
        The argument tuple to splat into a kernel launch.
    """
    return (
        element_x, element_y, element_weight, ray_vx, ray_vy, ray_vz, ray_flux,
        float(geom["focal_length"]), float(geom["ez_base"]), float(geom["z_bottom"]),
        float(geom["z_min_cut"]), float(geom["z_hole_top"]), float(geom["z_max_cut"]),
        float(geom["radius_max"]), float(geom["r_hole"]), float(target_z_mm),
        int(config.sim_grid_res), float(config.wall_radius_m),
        float(geom["refl_para"]), float(geom["refl_cyl"]), float(geom["refl_gask"]),
        float(geom["dome_radius"]), float(geom["refractive_index"]),
        int(config.max_multiple_reflections),
        float(geom["z_gasket_top"]), float(geom["r_gasket"]),
        float(geom["gasket_x_half"]), float(geom["gasket_y_half"]),
        int(geom["is_cylindrical_gasket"]),
        float(geom["emitter_offset_x"]), float(geom["emitter_offset_y"]),
        float(geom["transmissivity_lens"]),
        hotspot_grid, spill_grid,
    )


def execute_tracers(is_gpu: bool, kernel, total_threads: int, args: tuple,
                    log_callback: LogCallback = None,
                    progress_callback: ProgressCallback = None,
                    is_cancelled_callback: CancelCallback = None) -> None:
    """Runs a tracing kernel over the whole workload in cancellable chunks.

    A small calibration slice is traced first. It absorbs the JIT compile and
    measures the throughput, which is then used to size the remaining chunks so
    each one takes roughly half a second, keeping progress reporting smooth and
    cancellation responsive.

    Args:
        is_gpu: True to launch as a CUDA kernel, False to call it directly.
        kernel: ray_trace_kernel_gpu or ray_trace_kernel_cpu.
        total_threads: Total number of (element, ray) pairs to trace.
        args: Argument tuple from _build_kernel_args.
        log_callback: Receives progress text.
        progress_callback: Receives completion percentage.
        is_cancelled_callback: Polled between chunks; returns True to stop.
    """
    def launch(start_idx, end_idx):
        """Runs one slice of the workload."""
        if is_gpu:
            blocks = ((end_idx - start_idx) + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
            kernel[blocks, _THREADS_PER_BLOCK](start_idx, end_idx, *args)
            cuda.synchronize()
        else:
            kernel(start_idx, end_idx, *args)

    calibration_size = min(max(int(total_threads * 0.02), 250_000), total_threads - 1)

    if log_callback:
        log_callback(f"[{'CUDA' if is_gpu else 'CPU'} FEA Engine] Compiling & Calibrating...")

    started_at = time.time()
    launch(0, 1)                                  # One ray on its own: JIT compile.
    launch(1, 1 + calibration_size)               # Timed sample.
    elapsed = time.time() - started_at

    rays_per_sec = calibration_size / elapsed if elapsed > 0 else 1
    remaining = total_threads - (1 + calibration_size)
    predicted_time = remaining / rays_per_sec if rays_per_sec > 0 else 0

    if log_callback:
        log_callback(f"Done. ({rays_per_sec:,.0f} rays/sec) | "
                     f"Predicted completion: ~{predicted_time:.1f} s")

    if remaining <= 0:
        return

    chunk_size = max(int(rays_per_sec * 0.5), 100_000)
    for start_idx in range(1 + calibration_size, total_threads, chunk_size):
        if is_cancelled_callback and is_cancelled_callback():
            return

        end_idx = min(start_idx + chunk_size, total_threads)
        launch(start_idx, end_idx)

        if progress_callback:
            progress_callback((end_idx / total_threads) * 100.0)


_CUDA_STATUS: Optional[Tuple[bool, str]] = None


def probe_cuda_toolchain() -> Tuple[bool, str]:
    """Checks that the whole GPU toolchain works, not just that a device exists.

    cuda.get_current_device() only touches the driver, which ships with the
    display driver and is always present, so it says nothing about whether
    libNVVM and libdevice can be loaded. Compiling and running a throwaway
    kernel is the only check that exercises every piece, which matters most in a
    frozen build where the CUDA toolkit is bundled by hand.

    Returns:
        (True, "") when the GPU is usable, otherwise (False, reason). The result
        is cached for the lifetime of the process.
    """
    global _CUDA_STATUS
    if _CUDA_STATUS is not None:
        return _CUDA_STATUS

    try:
        if not cuda.is_available():
            raise RuntimeError("No CUDA-capable device is visible to the driver.")

        @cuda.jit
        def _probe_kernel(values):
            index = cuda.grid(1)
            if index < values.size:
                values[index] += 1.0

        device_values = cuda.to_device(np.zeros(1, dtype=np.float64))
        _probe_kernel[1, 1](device_values)
        cuda.synchronize()

        if device_values.copy_to_host()[0] != 1.0:
            raise RuntimeError("Probe kernel returned an incorrect result.")

        _CUDA_STATUS = (True, "")
    except Exception as error:  # Any failure here simply means "use the CPU".
        _CUDA_STATUS = (False, f"{type(error).__name__}: {error}")

    return _CUDA_STATUS


class WallIllumination(NamedTuple):
    """Illuminance maps produced by a single trace, in lux."""

    total_lux: np.ndarray
    hotspot_lux: np.ndarray
    spill_lux: np.ndarray
    total_lumens: float


def _points_in_polygon(points_x: np.ndarray, points_y: np.ndarray,
                       vertices: np.ndarray, tolerance: float) -> np.ndarray:
    """Tests which points fall inside a polygon, counting the boundary as in.

    An even-odd ray cast decides the interior, which handles concave outlines
    such as a notched die correctly. A ray cast is undefined for a point lying
    exactly on an edge, and the sampling grid puts points exactly on the edges
    of an axis-aligned die, so those are detected separately and added back.

    Args:
        points_x: Flat array of x coordinates.
        points_y: Flat array of y coordinates, same length as points_x.
        vertices: (N, 2) array of polygon corners in order. The outline is
            closed implicitly, so the first corner is not repeated.
        tolerance: Distance within which a point counts as on the boundary.

    Returns:
        A boolean array, one entry per point.
    """
    px = points_x.reshape(-1, 1)
    py = points_y.reshape(-1, 1)
    x0, y0 = vertices[:, 0], vertices[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)

    # Even-odd ray cast along +x. Only edges straddling the ray can cross it,
    # so horizontal edges, where the division would be degenerate, never count.
    straddles = (y0 > py) != (y1 > py)
    safe_dy = np.where(y1 != y0, y1 - y0, 1.0)
    crossing_x = x0 + (py - y0) * (x1 - x0) / safe_dy
    interior = np.sum(straddles & (px < crossing_x), axis=1) % 2 == 1

    # Shortest distance to each edge, for the points sitting on the outline.
    edge_dx, edge_dy = x1 - x0, y1 - y0
    length_sq = edge_dx ** 2 + edge_dy ** 2
    safe_length_sq = np.where(length_sq > 0.0, length_sq, 1.0)
    along = np.clip(((px - x0) * edge_dx + (py - y0) * edge_dy) / safe_length_sq,
                    0.0, 1.0)
    on_edge = np.min(np.hypot(px - (x0 + along * edge_dx),
                              py - (y0 + along * edge_dy)), axis=1) <= tolerance

    return interior | on_edge


def emitter_die_outline(emitter: dict, shape: str) -> Optional[np.ndarray]:
    """Returns the validated die outline for a polygon emitter.

    Args:
        emitter: Emitter specs.
        shape: Die shape, already resolved against the settings default.

    Returns:
        An (N, 2) array of vertices in millimetres relative to the die centre,
        or None for a shape that does not need one.

    Raises:
        ValueError: If the shape is not one of DIE_SHAPES, or if a polygon die
            has an outline that is missing, malformed or too small to enclose
            an area.
    """
    if shape not in DIE_SHAPES:
        raise ValueError(
            f"Unknown emitter die shape {shape!r}. Use one of "
            f"{', '.join(DIE_SHAPES)}. Anything that is not a plain rectangle "
            f"or circle is modelled as 'polygon' with a die_outline.")

    if shape != "polygon":
        return None

    outline = emitter.get("die_outline")
    if isinstance(outline, str):
        # Hand-edited catalogues, and the GUI's input box, hold pasted JSON.
        try:
            outline = json.loads(outline)
        except json.JSONDecodeError as error:
            raise ValueError(f"die_outline is not valid JSON: {error}") from error

    vertices = np.asarray(outline if outline else [], dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
        raise ValueError(
            "A polygon die needs die_outline set to at least three [x, y] "
            "vertex pairs, in millimetres relative to the die centre.")
    return vertices


def _cell_bounds(samples: np.ndarray, low: float, high: float):
    """Returns the span of the grid cell each sample point stands for.

    Cell edges sit halfway between neighbouring samples, so the cells tile the
    die's bounding box exactly. A sample on the perimeter therefore owns only
    the half cell that lies inside the die, which is what stops the edge of a
    die from being over weighted.

    Args:
        samples: Sorted sample coordinates along one axis.
        low: Lower edge of the bounding box on this axis.
        high: Upper edge of the bounding box on this axis.

    Returns:
        (lower, upper) arrays, one entry per sample.
    """
    if samples.size == 1:
        return np.array([low]), np.array([high])

    midpoints = (samples[:-1] + samples[1:]) / 2.0
    return (np.concatenate(([low], midpoints)),
            np.concatenate((midpoints, [high])))


def _cell_coverage(x_lo, x_hi, y_lo, y_hi, inside_test, subsamples: int):
    """Measures what fraction of each grid cell lies on the emitting surface.

    Every cell is probed on a regular sub grid and the hits are counted. Doing
    it by sampling rather than by clipping the outline analytically means one
    piece of code covers circles, convex outlines and concave ones alike, and
    the answer converges predictably as subsamples rises.

    Args:
        x_lo, x_hi: Cell edges along x, one entry per column.
        y_lo, y_hi: Cell edges along y, one entry per row.
        inside_test: Callable taking flat x and y arrays and returning a
            boolean array.
        subsamples: Probes per axis within each cell.

    Returns:
        A (rows, columns) array of fractions between 0 and 1.
    """
    steps = (np.arange(subsamples) + 0.5) / subsamples
    probe_x = (x_lo[:, None] + steps[None, :] * (x_hi - x_lo)[:, None]).ravel()
    probe_y = (y_lo[:, None] + steps[None, :] * (y_hi - y_lo)[:, None]).ravel()

    mesh_x, mesh_y = np.meshgrid(probe_x, probe_y)
    inside = inside_test(mesh_x.ravel(), mesh_y.ravel())
    return inside.reshape(len(y_lo), subsamples,
                          len(x_lo), subsamples).mean(axis=(1, 3))


def _build_emitter_elements(emitter: dict, elements_per_side: int, shape: str,
                            outline: Optional[np.ndarray] = None,
                            subsamples: int = DIE_SUBSAMPLES):
    """Subdivides the light emitting surface into area weighted point sources.

    The die's bounding box is covered with a grid of sample points, the
    outermost of which sit exactly on the perimeter. Each point stands for the
    cell around it and carries the share of the emitter's flux that the area of
    its cell deserves. An edge point owns half a cell and a corner point a
    quarter, so the perimeter is sampled without being over weighted, and where
    a curved or angled edge cuts through a cell only the part inside the die
    counts. This is the same rule for every shape: a rectangle, a circle and an
    arbitrary outline differ only in how much of each cell is covered.

    Args:
        emitter: Emitter specs.
        elements_per_side: Grid resolution across the die.
        shape: Die shape, already resolved against the settings default.
        outline: Vertices from emitter_die_outline, required for "polygon".
            The outline is authoritative for a polygon die, so the grid spans
            its bounding box and die_length_mm and die_width_mm are ignored.
        subsamples: Probes per axis used to measure a partly covered cell.

    Returns:
        (x, y, weight) arrays. The weights are areas in square millimetres and
        sum to the area of the emitting surface. Cells lying entirely off the
        die are dropped, so the arrays are shorter than elements_per_side^2 for
        any shape that does not fill its bounding box.

    Raises:
        ValueError: If the outline covers none of the sample cells, which means
            it is too thin to resolve at this grid resolution.
    """
    if shape == "polygon":
        min_x, min_y = outline.min(axis=0)
        max_x, max_y = outline.max(axis=0)
        tolerance = 1e-9 * max(max_x - min_x, max_y - min_y, 1.0)

        def inside_test(probe_x, probe_y):
            """True where a probe lies on the polygon die."""
            return _points_in_polygon(probe_x, probe_y, outline, tolerance)
    else:
        die_length = emitter["die_length_mm"]
        die_width = die_length if shape == "round" else emitter["die_width_mm"]
        min_x, max_x = -die_length / 2.0, die_length / 2.0
        min_y, max_y = -die_width / 2.0, die_width / 2.0

        if shape == "round":
            radius_sq = (die_length / 2.0) ** 2

            def inside_test(probe_x, probe_y):
                """True where a probe lies on the circular die."""
                return probe_x ** 2 + probe_y ** 2 <= radius_sq
        else:
            inside_test = None  # A rectangle fills its own bounding box.

    sample_x = np.linspace(min_x, max_x, elements_per_side)
    sample_y = np.linspace(min_y, max_y, elements_per_side)
    x_lo, x_hi = _cell_bounds(sample_x, min_x, max_x)
    y_lo, y_hi = _cell_bounds(sample_y, min_y, max_y)

    coverage = (np.ones((len(y_lo), len(x_lo))) if inside_test is None
                else _cell_coverage(x_lo, x_hi, y_lo, y_hi, inside_test, subsamples))
    weight = (np.outer(y_hi - y_lo, x_hi - x_lo) * coverage).ravel()

    grid_x, grid_y = np.meshgrid(sample_x, sample_y)
    emitting = weight > 0.0
    if not emitting.any():
        raise ValueError(
            f"The die outline covers none of the {elements_per_side} x "
            f"{elements_per_side} sample cells. Raise sim_emitter_elements, or "
            f"check the outline is in millimetres.")

    return grid_x.ravel()[emitting], grid_y.ravel()[emitting], weight[emitting]


# Azimuth steps that divide 180 exactly and are round numbers to type.
# Suggested in place of a step that would bias the beam.
TIDY_AZIMUTH_STEPS_DEG = (0.1, 0.2, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
                          4.0, 5.0, 6.0, 9.0, 10.0, 12.0, 15.0, 18.0,
                          20.0, 30.0, 36.0, 45.0, 60.0, 90.0)


def angular_sampling_warnings(config: SimulationConfig) -> List[str]:
    """Reports angular step sizes that would bias the beam.

    The azimuth sweep runs in equal steps around the full turn. Mirroring the
    beam left to right maps an azimuth to 180 degrees minus itself, so unless
    the step divides 180 the sampled directions are not a mirror image of
    themselves and a perfectly symmetric build comes out lopsided. It is not a
    small effect: a round emitter in a round reflector traced at 8 degrees
    differs by 60% between its left and right halves, where 9 degrees gives a
    difference of exactly zero.

    The sweep also has to close on itself, which needs the range to be a whole
    number of steps.

    Args:
        config: Active configuration.

    Returns:
        A list of warnings, empty when the sampling is sound.
    """
    warnings = []
    step = float(config.sim_phi_step_deg)
    span = float(config.sim_phi_max_deg) - float(config.sim_phi_min_deg)

    if step <= 0.0:
        return ["Azimuth step must be greater than zero."]

    def divides(total, by):
        """True when total is a whole number of steps of size by."""
        return abs(total / by - round(total / by)) < 1e-9

    if not divides(span, step):
        warnings.append(
            f"Azimuth range {span:g} deg is not a whole number of "
            f"{step:g} deg steps, so the sweep does not close evenly.")
    elif not divides(180.0, step):
        nearest = min(TIDY_AZIMUTH_STEPS_DEG, key=lambda tidy: abs(tidy - step))
        warnings.append(
            f"Azimuth step {step:g} deg does not divide 180, so the beam will "
            f"not be left-right symmetric even for a symmetric build. "
            f"Try {nearest:g} deg.")

    return warnings


def _build_ray_directions(config: SimulationConfig):
    """Builds the fan of ray directions and the solid angle each one carries.

    Args:
        config: Active configuration, for the angular ranges and step sizes.

    Returns:
        (vx, vy, vz, intensity, solid_angle): the unit direction of each ray,
        its relative Lambertian intensity, and the solid angle it represents.
    """
    theta, phi = np.meshgrid(
        np.radians(np.arange(config.sim_theta_min_deg,
                             config.sim_theta_max_deg, config.sim_theta_step_deg)),
        np.radians(np.arange(config.sim_phi_min_deg,
                             config.sim_phi_max_deg, config.sim_phi_step_deg)))
    theta, phi = theta.flatten(), phi.flatten()

    solid_angle = (np.sin(theta)
                   * np.radians(config.sim_theta_step_deg)
                   * np.radians(config.sim_phi_step_deg))

    return (np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
            lambertian_intensity(theta),
            solid_angle)


def simulate_wall_illuminance(geom: dict, emitter: dict, current_amps: float, finish: str,
                              config: SimulationConfig,
                              log_callback: LogCallback = None,
                              progress_callback: ProgressCallback = None,
                              is_cancelled_callback: CancelCallback = None
                              ) -> Optional[WallIllumination]:
    """Traces the full ray budget and returns the illuminance on the wall.

    Every die element fires every ray direction, so the workload is
    ``elements * directions``. Reflected light is accumulated apart from direct
    light so an orange peel finish can be blurred without smearing the spill.

    Args:
        geom: Output of get_sim_geometry.
        emitter: Emitter specs.
        current_amps: Drive current in amps.
        finish: "smooth" or "orange_peel".
        config: Active configuration.
        log_callback: Receives progress text.
        progress_callback: Receives completion percentage.
        is_cancelled_callback: Polled between chunks; returns True to stop.

    Returns:
        A WallIllumination, or None if the run was cancelled.
    """
    total_lumens = calculate_lumens(emitter, current_amps, config)

    # Normalise the Lambertian profile so it integrates to the emitter's output.
    theta_samples = np.radians(np.arange(0, 90, config.lumen_calc_step_deg))
    hemisphere_integral = np.sum(lambertian_intensity(theta_samples)
                                 * np.sin(theta_samples)
                                 * np.radians(config.lumen_calc_step_deg))
    peak_intensity = total_lumens / (2 * np.pi * hemisphere_integral)

    die_shape = spec_or_default(emitter, "emitter", "shape", config)
    element_x, element_y, element_area = _build_emitter_elements(
        emitter, config.sim_emitter_elements, die_shape,
        emitter_die_outline(emitter, die_shape))
    element_count = len(element_x)

    # Flux follows emitting area, not element count: a point on the perimeter
    # stands for half or a quarter of the area an interior point does, and a
    # cell only partly covered by the die stands for less again. Normalising
    # here keeps the emitter's total output exactly as rated whatever the die
    # shape or the grid resolution.
    element_weight = element_area / element_area.sum()

    ray_vx, ray_vy, ray_vz, ray_intensity, solid_angle = _build_ray_directions(config)
    ray_flux = peak_intensity * ray_intensity * solid_angle

    # One contiguous float64 copy of each array, shared by both back ends.
    element_x = np.ascontiguousarray(element_x, dtype=np.float64)
    element_y = np.ascontiguousarray(element_y, dtype=np.float64)
    element_weight = np.ascontiguousarray(element_weight, dtype=np.float64)
    ray_vx = np.ascontiguousarray(ray_vx, dtype=np.float64)
    ray_vy = np.ascontiguousarray(ray_vy, dtype=np.float64)
    ray_vz = np.ascontiguousarray(ray_vz, dtype=np.float64)
    ray_flux = np.ascontiguousarray(ray_flux, dtype=np.float64)

    target_z_mm = config.target_distance_m * 1000.0
    total_threads = element_count * len(ray_vx)

    use_gpu = False
    if config.use_gpu:
        use_gpu, gpu_error = probe_cuda_toolchain()
        if not use_gpu and log_callback:
            log_callback(f"[!] GPU toggled ON, but the CUDA toolchain is unusable:\n"
                         f"-> {gpu_error}\nFalling back to CPU.")

    grid_shape = (config.sim_grid_res, config.sim_grid_res)
    hotspot_grid = np.zeros(grid_shape, dtype=np.float64)
    spill_grid = np.zeros(grid_shape, dtype=np.float64)

    if use_gpu:
        try:
            if log_callback:
                log_callback(f"[CUDA FEA Engine] GPU enabled. Dispatching "
                             f"{total_threads:,} ray-element pairs "
                             f"({element_count:,} elements x {len(ray_vx):,} rays)...")

            device_hotspot = cuda.to_device(hotspot_grid)
            device_spill = cuda.to_device(spill_grid)
            args = _build_kernel_args(
                cuda.to_device(element_x), cuda.to_device(element_y),
                cuda.to_device(element_weight),
                cuda.to_device(ray_vx), cuda.to_device(ray_vy), cuda.to_device(ray_vz),
                cuda.to_device(ray_flux),
                geom, config, target_z_mm, device_hotspot, device_spill)

            execute_tracers(True, ray_trace_kernel_gpu, total_threads, args,
                            log_callback, progress_callback, is_cancelled_callback)
            if is_cancelled_callback and is_cancelled_callback():
                return None

            hotspot_grid = device_hotspot.copy_to_host()
            spill_grid = device_spill.copy_to_host()

        except Exception as gpu_error:
            # Anything that survives the probe (VRAM exhaustion, a driver reset,
            # a failed launch) costs time, not the whole job.
            if log_callback:
                log_callback(f"[!] GPU execution failed, restarting the job on CPU:\n"
                             f"-> {type(gpu_error).__name__}: {gpu_error}")
            use_gpu = False
            hotspot_grid = np.zeros(grid_shape, dtype=np.float64)
            spill_grid = np.zeros(grid_shape, dtype=np.float64)
            if progress_callback:
                progress_callback(0.0)

    if not use_gpu:
        if log_callback:
            log_callback(f"[CPU FEA Engine] Processing {total_threads:,} "
                         f"ray-element pairs on logical cores...")

        args = _build_kernel_args(element_x, element_y, element_weight,
                                  ray_vx, ray_vy, ray_vz, ray_flux,
                                  geom, config, target_z_mm, hotspot_grid, spill_grid)

        execute_tracers(False, ray_trace_kernel_cpu, total_threads, args,
                        log_callback, progress_callback, is_cancelled_callback)
        if is_cancelled_callback and is_cancelled_callback():
            return None

    if log_callback:
        log_callback("Applying spatial blur and generating final lux arrays...")

    # An orange peel finish scatters the reflected light; the blur radius scales
    # with the grid so the result is resolution independent.
    blur_sigma = 0.0
    if finish == "orange_peel":
        blur_sigma = (config.default_op_blur_strength * geom["op_multiplier"]
                      * (config.sim_grid_res / 1000.0))
    if blur_sigma > 0:
        hotspot_grid = gaussian_filter(hotspot_grid, sigma=blur_sigma)

    # Flux per pixel becomes illuminance once divided by the pixel's area.
    pixel_area_m2 = (2.0 * config.wall_radius_m / config.sim_grid_res) ** 2
    hotspot_lux = hotspot_grid / pixel_area_m2
    spill_lux = spill_grid / pixel_area_m2

    return WallIllumination(hotspot_lux + spill_lux, hotspot_lux, spill_lux, total_lumens)


# ==============================================================================
# 5. PLOTTING & EXPORT MANAGER
# ==============================================================================


class BeamMetrics(NamedTuple):
    """Angular size and physical size of each region of the beam."""

    spill_angle_deg: float
    spill_diameter_m: float
    corona_angle_deg: float
    corona_diameter_m: float
    hotspot_angle_deg: float
    hotspot_diameter_m: float
    candela_per_lumen: float


def apply_camera_exposure_and_tonemap(wall_lux: np.ndarray,
                                      config: SimulationConfig) -> np.ndarray:
    """Converts an illuminance map into a displayable 0-1 image.

    Auto exposure pins the 99.5th percentile to mid grey; manual exposure uses
    the ISO/aperture/shutter triangle. The result is tone mapped with the ACES
    filmic curve and gamma corrected, so a bright hotspot rolls off instead of
    clipping flat.

    Args:
        wall_lux: Illuminance on the wall, in lux.
        config: Active configuration, for the camera settings.

    Returns:
        Display values in the range 0-1, same shape as wall_lux.
    """
    if config.use_auto_exposure:
        reference_lux = np.percentile(wall_lux, 99.5) or 1.0
        exposed = wall_lux * (1.0 / reference_lux) * (2 ** config.auto_exposure_compensation_ev)
    else:
        exposure_value = 2 ** np.log2((config.cam_f_stop ** 2) / config.cam_shutter_speed_s)
        saturation_lux = (250.0 * exposure_value) / config.cam_iso
        exposed = (wall_lux / saturation_lux) * 0.18

    tone_mapped = ((exposed * (2.51 * exposed + 0.03))
                   / (exposed * (2.43 * exposed + 0.59) + 0.14))
    return np.power(np.clip(tone_mapped, 0.0, 1.0), 1.0 / 2.2)


def get_beam_metrics(wall_lux: np.ndarray, hotspot_lux: np.ndarray, spill_lux: np.ndarray,
                     max_cd: float, total_flux: float,
                     config: SimulationConfig) -> BeamMetrics:
    """Measures the spill, corona and hotspot of a simulated beam.

    Each region is sized by the furthest pixel from the centre that passes its
    threshold, which is then converted to an angle at the target distance.

    Args:
        wall_lux: Combined illuminance map.
        hotspot_lux: Reflected component only.
        spill_lux: Direct component only.
        max_cd: Peak intensity in candela.
        total_flux: Total emitter output in lumens.
        config: Active configuration, for the thresholds and distance.

    Returns:
        The measured BeamMetrics.
    """
    pixel_size_m = (2.0 * config.wall_radius_m) / config.sim_grid_res
    centre_idx = (config.sim_grid_res - 1) / 2.0

    def max_radius(mask: np.ndarray) -> float:
        """Distance in metres from the centre to the furthest lit pixel."""
        if not np.any(mask):
            return 0.0
        rows, cols = np.nonzero(mask)
        return np.max(np.sqrt((cols - centre_idx) ** 2
                              + (rows - centre_idx) ** 2)) * pixel_size_m

    def full_angle_deg(radius_m: float) -> float:
        """Full cone angle subtended by a radius at the target distance."""
        return 2 * np.degrees(np.arctan(radius_m / config.target_distance_m))

    spill_radius = max_radius(spill_lux > config.spill_visible_threshold_lux)
    corona_radius = max_radius(
        hotspot_lux > (np.max(hotspot_lux) * config.corona_visible_threshold))
    hotspot_radius = max_radius(
        wall_lux >= (np.max(wall_lux) * config.hotspot_fwhm_threshold))

    return BeamMetrics(
        spill_angle_deg=full_angle_deg(spill_radius),
        spill_diameter_m=2 * spill_radius,
        corona_angle_deg=full_angle_deg(corona_radius),
        corona_diameter_m=2 * corona_radius,
        hotspot_angle_deg=full_angle_deg(hotspot_radius),
        hotspot_diameter_m=2 * hotspot_radius,
        candela_per_lumen=max_cd / total_flux,
    )


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
                         total_flux: float,
                         config: SimulationConfig) -> str:
    """Builds the output table for the usual 1/10/35/100 percent drive levels.

    Intensity is scaled from the simulated maximum by the lumen ratio, since the
    beam shape does not change with drive current.
    """
    table = " Mode | Amps | Lumens |  Candela | Throw \n" + "-" * 46 + "\n"
    for fraction in (0.01, 0.10, 0.35, 1.0):
        amps = max_amps * fraction
        lumens = calculate_lumens(emitter, amps, config)
        candela = max_cd * (lumens / total_flux)
        table += (f"{int(fraction * 100):>4}% | {amps:>4.1f} | {int(lumens):>6,} | "
                  f"{int(candela):>8,} | {int(np.sqrt(candela * 4)):>4,}m\n")
    return table


def _render_wall_shot(render_data: np.ndarray, title_str: str, geometry_text: str,
                      modes_text: str, config: SimulationConfig,
                      save_path: Optional[str]):
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

    if save_path and config.export_plots:
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
        (figure, results): the wall shot figure (None when plot_wall_shot is
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
        return None, None  # Cancelled.

    # Peak intensity, and the ANSI throw distance where it falls to 0.25 lux.
    max_cd = np.max(illumination.total_lux) * (config.target_distance_m ** 2)
    throw_m = int(np.sqrt(max_cd / 0.25))

    metrics = get_beam_metrics(illumination.total_lux, illumination.hotspot_lux,
                               illumination.spill_lux, max_cd,
                               illumination.total_lumens, config)

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

    figure = None
    if config.plot_wall_shot:
        if log_callback:
            log_callback("Rendering final camera visualization...")
        figure = _render_wall_shot(
            apply_camera_exposure_and_tonemap(illumination.total_lux, config),
            title_str,
            _format_beam_geometry(metrics, config),
            _format_output_modes(emitter, max_amps, max_cd,
                                 illumination.total_lumens, config),
            config, save_path)

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
        "Spill Angle (deg)": round(metrics.spill_angle_deg, 1),
        "Corona Angle (deg)": round(metrics.corona_angle_deg, 1),
        "Hotspot Angle (deg)": round(metrics.hotspot_angle_deg, 1),
        "Cd/Lm Ratio": round(metrics.candela_per_lumen, 1),
    }


# ==============================================================================
# 6. API EXECUTION ENTRY POINT
# ==============================================================================

CSV_HEADERS = [
    "Reflector", "Emitter", "Gasket", "Finish", "Max Candela (cd)", "Throw (m)",
    "Total Lumens", "Spill Angle (deg)", "Corona Angle (deg)",
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
        (figure, results): the wall shot figure for a single render (None in
        batch mode), and every result row keyed by hardware combination.
        (None, None) if the batch was cancelled.
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
                return None, None
            if log_callback:
                log_callback(f"[{position}/{len(combinations)}] Rendering {reflector} + "
                             f"{emitter} + {gasket} ({combo_finish.upper()})...")

            _, metrics = generate_flashlight_plot(
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
        figure = None

    else:
        if log_callback:
            log_callback(f"Starting specific render: {active_reflector} + "
                         f"{active_emitter} + {active_gasket}")

        figure, metrics = generate_flashlight_plot(
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

    return figure, results
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
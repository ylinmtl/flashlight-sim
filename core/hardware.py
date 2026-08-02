"""The catalogue of reflectors, emitters and gaskets.

Part of the flashlight simulator core; see core/__init__.py for the
public surface.
"""

import copy
import math
import os
import shutil
from typing import Dict, List

from .config import SimulationConfig
from .paths import _read_json, _write_json, resource_path, user_data_path


# The three interchangeable hardware categories, and the JSON section each one
# is stored under in hardware_library.json.
HARDWARE_KINDS = ("emitter", "reflector", "gasket")
_JSON_SECTION_BY_KIND = {
    "emitter": "emitters",
    "reflector": "reflectors",
    "gasket": "gaskets",
}
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

# How the front lens scatters. "clear" passes light straight through,
# "frosted" is the glass itself etched, and "film" is a diffuser stuck
# on the outside. The two scatter alike but behave differently with
# angle, so they are kept apart.
LENS_FINISHES = ("clear", "frosted", "film")

# How the emitting surface is laid out. "monolithic" is one
# continuous die. "array" is a grid of separate dies with phosphor
# between them, as used by multi die COBs and by the larger CSP parts.
DIE_LAYOUTS = ("monolithic", "array")
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
    "reflector": {
        "thickness_diameter_mm": ("wall_thickness_mm", 0.5),
        # Both scattering specs used to be a standard deviation, one as a
        # slope error in milliradians and the other as an angle in degrees.
        # Both are now the width of the beam they produce, in degrees, so
        # 0.269843 turns a slope sigma in mrad into a beam FWHM in degrees
        # (doubled for reflection, times 2.3548 for sigma to FWHM, converted
        # from milliradians) and 2.3548 does the same for the lens.
        # Roughness was a slope error in milliradians, then briefly the beam
        # width it produced. It is now the height of the texture itself,
        # which is what a surface is actually measured in. 17.7 nm per mrad
        # of slope follows from the correlation length in optics.py.
        "surface_roughness_mrad": ("surface_roughness_nm", 17.678),
        "surface_blur_deg": ("surface_roughness_nm", 65.51),
        # Orange peel was an abstract multiplier on a post-process blur. It
        # is now the depth of the dimples, with their spacing beside it, so
        # a texture measured off a real reflector can be typed straight in.
        "OP_Factor": ("op_dimple_depth_um", 3.0),
        "lens_diffusion_deg": ("lens_diffusion_fwhm_deg", 2.354820),
    },
    "gasket": {
        "gasket_thickness_mm": ("thickness_mm", 1.0),
        "gasket_total_height_mm": ("total_height_mm", 1.0),
        "gasket_opening_mm": ("inner_diameter_mm", 1.0),
    },
}
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
        "transmissivity_lens": "default_transmissivity_lens",
        "surface_finish": "default_surface_finish",
        "surface_roughness_nm": "default_surface_roughness_nm",
        "surface_correlation_um": "default_surface_correlation_um",
        "op_dimple_pitch_mm": "default_op_dimple_pitch_mm",
        "op_dimple_depth_um": "default_op_dimple_depth_um",
        "lens_finish": "default_lens_finish",
        "lens_diffusion_fwhm_deg": "default_lens_diffusion_fwhm_deg",
        "lens_refractive_index": "default_lens_refractive_index",
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
        "die_layout": "default_die_layout",
        "die_rows": "default_die_rows",
        "die_columns": "default_die_columns",
        "die_gap_mm": "default_die_gap_mm",
        "die_gap_output": "default_die_gap_output",
    },
    "gasket": {
        "inner_diameter_mm": "default_gasket_inner_diameter_mm",
        "wall_shape": "default_gasket_wall_shape",
        "thickness_mm": "default_gasket_thickness_mm",
        "total_height_mm": "default_gasket_total_height_mm",
    },
}


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
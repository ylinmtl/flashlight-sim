"""Ray tracing engine for the flashlight beam simulator.

The core is split by responsibility, each layer depending only on the ones
above it, so there are no import cycles:

    paths        where the settings, catalogue and outputs live
    cuda_setup   puts the bundled CUDA toolkit on the search path
    config       the settings file and the values read from it
    hardware     the catalogue of reflectors, emitters and gaskets
    optics       a hardware combination turned into traceable geometry
    tracer       the ray tracing kernels and the back ends that run them
    simulation   a trace reduced to a beam landing on a wall
    photometry   that beam written out as an IES file
    report       the rendered plots, the CSV, and the job that drives it all

Everything the GUI needs is re-exported here, so callers import from `core`
rather than reaching into a submodule. Names starting with an underscore are
internal to their module and are deliberately not listed.
"""

from .config import CancelCallback, LogCallback, ProgressCallback, SimulationConfig
from .hardware import (DIE_LAYOUTS, DIE_SHAPES, GASKET_WALL_SHAPES, HARDWARE_KINDS,
                       LENS_FINISHES, OUTPUT_MODES,
                       RENAMED_SPECS, SPEC_DEFAULT_SETTINGS, SURFACE_FINISHES,
                       HardwareLibrary, spec_or_default)
from .optics import (DIE_SUBSAMPLES, die_array_extent, die_array_layout,
                     die_array_resolution,
                     die_cell_size, die_centres, NO_EMITTER_OFFSET, EmitterOffset,
                     calculate_lumens, effective_bore_diameter,
                     emitter_die_outline, emitter_footprint_diagonal,
                     forward_voltage, get_sim_geometry, lambertian_intensity)
from .paths import resource_path, user_data_path
from .photometry import beam_candela_grid, export_beam_ies, write_ies_file
from .report import (PLOT_NAMES, generate_flashlight_plot, render_plot,
                     render_wall_shot, run_simulation_job)
from .simulation import (BeamMetrics, WallIllumination, angular_sampling_warnings,
                         apply_camera_exposure_and_tonemap, get_beam_metrics,
                         simulate_wall_illuminance)
from .tracer import probe_cuda_toolchain

__all__ = [
    "BeamMetrics",
    "CancelCallback",
    "DIE_LAYOUTS",
    "DIE_SHAPES",
    "DIE_SUBSAMPLES",
    "EmitterOffset",
    "GASKET_WALL_SHAPES",
    "HARDWARE_KINDS",
    "HardwareLibrary",
    "LENS_FINISHES",
    "LogCallback",
    "NO_EMITTER_OFFSET",
    "OUTPUT_MODES",
    "PLOT_NAMES",
    "ProgressCallback",
    "RENAMED_SPECS",
    "SPEC_DEFAULT_SETTINGS",
    "SURFACE_FINISHES",
    "SimulationConfig",
    "WallIllumination",
    "angular_sampling_warnings",
    "apply_camera_exposure_and_tonemap",
    "beam_candela_grid",
    "calculate_lumens",
    "die_array_extent",
    "die_array_layout",
    "die_array_resolution",
    "die_cell_size",
    "die_centres",
    "effective_bore_diameter",
    "emitter_die_outline",
    "emitter_footprint_diagonal",
    "export_beam_ies",
    "forward_voltage",
    "generate_flashlight_plot",
    "get_beam_metrics",
    "get_sim_geometry",
    "lambertian_intensity",
    "probe_cuda_toolchain",
    "resource_path",
    "run_simulation_job",
    "simulate_wall_illuminance",
    "spec_or_default",
    "user_data_path",
    "write_ies_file",
    "render_plot",
    "render_wall_shot",
]
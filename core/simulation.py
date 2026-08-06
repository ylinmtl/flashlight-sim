"""Running a trace and reducing it to a beam on a wall.

Part of the flashlight simulator core; see core/__init__.py for the
public surface.
"""

import math
import time
from typing import Callable, List, NamedTuple, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import (CancelCallback, LogCallback,
                     ProgressCallback, SimulationConfig)
from .hardware import spec_or_default
from .optics import (_build_emitter_elements, calculate_lumens,
                     emitter_die_outline, lambertian_intensity)
from .tracer import (_build_kernel_args, execute_tracers,
                     probe_cuda_toolchain, ray_trace_kernel_cpu,
                     ray_trace_kernel_gpu)
from numba import cuda


class WallIllumination(NamedTuple):
    """Illuminance maps produced by a single trace, in lux.

    Attributes:
        total_lux: Combined illuminance on the wall.
        hotspot_lux: The reflected component alone.
        spill_lux: The direct component alone.
        total_lumens: What left the die, the emitter's rated output.
        delivered_lumens: What reached the wall, after the reflector, the
            lens and the gasket have taken their share, and after light
            leaving the head too wide to land on the canvas is lost.
    """

    total_lux: np.ndarray
    hotspot_lux: np.ndarray
    spill_lux: np.ndarray
    total_lumens: float
    delivered_lumens: float
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


# Fraction of the free memory the angular grid is allowed to claim. The rest is
# needed for the ray arrays, the projection's working copies and whatever else
# the machine is doing.
DOME_MEMORY_FRACTION = 0.5


def available_memory_mb(use_gpu: bool):
    """Free memory in megabytes, or None when it cannot be worked out.

    Args:
        use_gpu: True to ask the CUDA device, False for host memory.

    Returns:
        Free megabytes, or None if nothing could answer.
    """
    if use_gpu:
        try:
            free_bytes, _ = cuda.current_context().get_memory_info()
            return free_bytes / 2 ** 20
        except Exception:
            return None
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 20
    except Exception:
        return None


def dome_grid_shape(config: SimulationConfig, use_gpu: bool = False,
                    log_callback: LogCallback = None):
    """Returns the angular grid shape, coarsened if it will not fit.

    At the default 0.05 degrees a hemisphere is 1800 by 7200 bins, which is
    about 200 MB across the two accumulators. That is fine on most machines and
    far too much on some, and a failed allocation part way through a long trace
    is a poor way to find out. The step is therefore halved until the grid fits
    the smaller of the configured budget and what the machine actually has free,
    and the operator is told it happened.

    Args:
        config: Active configuration.
        use_gpu: True when the grid will live in device memory.
        log_callback: Receives a note if the grid had to be coarsened.

    Returns:
        (polar_bins, azimuth_bins).
    """
    polar_bins = max(1, int(round((config.dome_angle_deg / 2.0)
                                  / config.dome_polar_step_deg)))
    azimuth_bins = max(1, int(round(360.0 / config.dome_azimuth_step_deg)))

    def megabytes(polar, azimuth):
        """Bytes for both float64 accumulators, in megabytes."""
        return polar * azimuth * 8 * 2 / 2 ** 20

    budget = float(config.dome_memory_budget_mb)
    free = available_memory_mb(use_gpu)
    if free is not None:
        budget = min(budget, free * DOME_MEMORY_FRACTION)

    wanted = megabytes(polar_bins, azimuth_bins)
    steps = 0
    while megabytes(polar_bins, azimuth_bins) > budget and min(polar_bins,
                                                               azimuth_bins) > 1:
        polar_bins = max(1, polar_bins // 2)
        azimuth_bins = max(1, azimuth_bins // 2)
        steps += 1

    if steps and log_callback:
        polar_step = (config.dome_angle_deg / 2.0) / polar_bins
        azimuth_step = 360.0 / azimuth_bins
        log_callback(
            f"[!] Spherical grid needs {wanted:,.0f} MB, but only "
            f"{budget:,.0f} MB is available. Coarsened to "
            f"{polar_bins:,} x {azimuth_bins:,} bins "
            f"({polar_step:.3f} deg polar, {azimuth_step:.3f} deg azimuth, "
            f"{megabytes(polar_bins, azimuth_bins):,.0f} MB).")

    return polar_bins, azimuth_bins


def project_dome_to_wall(dome_flux: np.ndarray, config: SimulationConfig):
    """Scatters flux from the angular grid onto the flat wall.

    Each populated bin stands for a narrow cone of directions leaving the head.
    Where that cone meets the wall is pure geometry, so the bin's flux is laid
    down there. Scattering rather than sampling is what keeps the total honest:
    no bin is counted twice and none is missed, and the only light that
    disappears is what genuinely falls outside the wall.

    The landing is spread bilinearly over the four surrounding pixels, which
    is what stops the two grids beating against each other into moire rings.

    Args:
        dome_flux: Flux per angular bin, indexed [polar, azimuth].
        config: Active configuration.

    Returns:
        Flux per wall pixel, indexed [row, column].
    """
    resolution = config.sim_grid_res
    wall = np.zeros((resolution, resolution), dtype=np.float64)

    filled = np.nonzero(dome_flux)
    if not len(filled[0]):
        return wall

    polar_bins, azimuth_bins = dome_flux.shape
    polar_step = math.radians(config.dome_angle_deg / 2.0) / polar_bins
    azimuth_step = 2.0 * math.pi / azimuth_bins

    polar = (filled[0] + 0.5) * polar_step
    azimuth = (filled[1] + 0.5) * azimuth_step
    flux = dome_flux[filled]

    # Only the forward hemisphere can reach a wall in front of the head; a dome
    # wider than that simply has nothing to project from those bins.
    forward = polar < (math.pi / 2.0 - 1e-9)
    polar, azimuth, flux = polar[forward], azimuth[forward], flux[forward]

    offset = config.target_distance_m * np.tan(polar)
    cell = (2.0 * config.wall_radius_m) / resolution
    column = (offset * np.cos(azimuth) + config.wall_radius_m) / cell - 0.5
    row = (offset * np.sin(azimuth) + config.wall_radius_m) / cell - 0.5

    # Dropping each bin into the nearest pixel is what causes the moire:
    # the angular lattice and the linear one beat against each other, so
    # neighbouring pixels collect different numbers of bins and the pattern
    # shows up as rings. Splitting each bin across the four pixels it sits
    # between, in proportion to how close it is to each, removes the beat
    # without blurring anything: the weights sum to one, so not a lumen is
    # gained or lost, and a bin landing dead centre still lands whole.
    low_column = np.floor(column).astype(np.int64)
    low_row = np.floor(row).astype(np.int64)
    column_fraction = column - low_column
    row_fraction = row - low_row

    for row_offset, row_weight in ((0, 1.0 - row_fraction), (1, row_fraction)):
        for col_offset, col_weight in ((0, 1.0 - column_fraction),
                                       (1, column_fraction)):
            target_row = low_row + row_offset
            target_column = low_column + col_offset
            landed = ((target_column >= 0) & (target_column < resolution)
                      & (target_row >= 0) & (target_row < resolution))
            np.add.at(wall, (target_row[landed], target_column[landed]),
                      (flux * row_weight * col_weight)[landed])
    return wall


def _hemisphere_weight(config: SimulationConfig) -> float:
    """Total Lambertian ray weight over a full hemisphere, on the trace's grid.

    Dividing the rated output by this, then multiplying it back through the
    traced rays, returns the rated output exactly. That only holds because the
    step sizes here are the ones the trace itself uses, so the discretisation
    cancels rather than leaving a residue: integrating on a separate, finer grid
    left the emitted flux drifting from the rating as the ray step was changed.

    The sum covers the whole hemisphere even when the sweep does not, so a
    deliberately narrowed sweep carries only its share of the output instead of
    the whole of it squeezed into a cone.

    Args:
        config: Active configuration.

    Returns:
        The summed weight, in steradians times relative intensity.
    """
    theta = np.radians(np.arange(config.sim_theta_min_deg, 90.0,
                                 config.sim_theta_step_deg))
    phi_steps = len(np.arange(0.0, 360.0, config.sim_phi_step_deg))
    return float(np.sum(lambertian_intensity(theta) * np.sin(theta))
                 * np.radians(config.sim_theta_step_deg)
                 * np.radians(config.sim_phi_step_deg) * phi_steps)


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

    # Scale the Lambertian profile so the rays leaving the die carry exactly
    # the emitter's rated output. Whatever the reflector, the lens and the
    # gasket then absorb comes off that, so less than this reaches the wall.
    peak_intensity = total_lumens / _hemisphere_weight(config)

    die_shape = spec_or_default(emitter, "emitter", "shape", config)
    element_x, element_y, element_area = _build_emitter_elements(
        emitter, config.sim_emitter_elements, die_shape,
        emitter_die_outline(emitter, die_shape), config=config)
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

    # With the dome enabled the kernels bin by outgoing direction and the
    # wall is filled in afterwards, so the accumulators are angular rather
    # than spatial and carry every ray the head emits, however wide.
    if config.use_spherical_projection:
        polar_bins, azimuth_bins = dome_grid_shape(config, use_gpu, log_callback)
        grid_shape = (polar_bins, azimuth_bins)
        dome = (polar_bins,
                math.radians(config.dome_angle_deg / 2.0) / polar_bins,
                2.0 * math.pi / azimuth_bins,
                azimuth_bins)
        if log_callback:
            log_callback(f"Spherical grid: {polar_bins:,} polar x "
                         f"{azimuth_bins:,} azimuth bins over "
                         f"{config.dome_angle_deg:g} deg.")
    else:
        grid_shape = (config.sim_grid_res, config.sim_grid_res)
        dome = (0, 0.0, 0.0, 0)

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
                geom, config, target_z_mm, device_hotspot, device_spill, dome)

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
                                  geom, config, target_z_mm, hotspot_grid,
                                  spill_grid, dome)

        execute_tracers(False, ray_trace_kernel_cpu, total_threads, args,
                        log_callback, progress_callback, is_cancelled_callback)
        if is_cancelled_callback and is_cancelled_callback():
            return None

    if config.use_spherical_projection:
        if log_callback:
            log_callback("Projecting the spherical grid onto the wall...")
        hotspot_grid = project_dome_to_wall(hotspot_grid, config)
        spill_grid = project_dome_to_wall(spill_grid, config)

    if log_callback:
        log_callback("Applying spatial blur and generating final lux arrays...")

    # An orange peel finish scatters the reflected light; the blur radius scales
    # with the grid so the result is resolution independent.
    blur_sigma = 0.0
    
    # Run the classic 2D statistical blur only if the Dimple Simulation is toggled off
    if finish == "orange_peel" and not getattr(config, "use_dimple_op_simulation", False):
        blur_sigma = (getattr(config, "op_blur_strength", 1.5) * geom["op_factor"]
                      * (config.sim_grid_res / 1000.0))
        
    if blur_sigma > 0:
        hotspot_grid = gaussian_filter(hotspot_grid, sigma=blur_sigma)

    # The grids still hold flux at this point, so summing them gives what
    # actually landed: the rated output less whatever the reflector, the
    # lens and the gasket absorbed, and less anything that left the head at
    # too wide an angle to reach the canvas.
    delivered_lumens = float(hotspot_grid.sum() + spill_grid.sum())

    # Flux per pixel becomes illuminance once divided by the pixel's area.
    pixel_area_m2 = (2.0 * config.wall_radius_m / config.sim_grid_res) ** 2
    hotspot_lux = hotspot_grid / pixel_area_m2
    spill_lux = spill_grid / pixel_area_m2

    return WallIllumination(hotspot_lux + spill_lux, hotspot_lux, spill_lux,
                            total_lumens, delivered_lumens)
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
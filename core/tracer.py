"""The ray tracing kernels and the back ends that run them.

Part of the flashlight simulator core; see core/__init__.py for the
public surface.
"""

import math
import time
from typing import Optional, Tuple

import numpy as np

# Must precede the Numba import: Numba caches its toolkit path the
# first time it is asked for one, so the bundled toolkit has to be on
# the search path before then.
from . import cuda_setup  # noqa: F401  (imported for its side effect)
from numba import cuda, njit

from .config import (CancelCallback, LogCallback, ProgressCallback,
                     SimulationConfig)


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
# CUDA launch geometry. 256 threads per block suits every architecture the
# simulator targets.
_THREADS_PER_BLOCK = 256
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


# Result of probe_cuda_toolchain, cached for the life of the process. The probe
# compiles and runs a real kernel, which is slow, and the answer cannot change
# while the application is running. None means it has not been asked yet.
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

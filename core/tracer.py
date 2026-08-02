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


# How much a frosted surface flattens the Fresnel falloff. Zero would leave it
# behaving like polished glass, one would remove the angle dependence entirely;
# half reflects that the microfacets spread the incidence angle without erasing
# it. An applied film sits on an otherwise flat outer face, so it keeps the
# full Fresnel curve.
FROSTED_ANGULAR_SOFTENING = 0.5

# How the scattering cone widens off axis, as an exponent on 1/cos(incidence).
# A film is a volume: path length through it grows as 1/cos, and a random walk
# spreads as the square root of path length, giving one half. Etched glass has
# no depth, so its widening comes only from the microfacet slopes projecting
# differently off axis, which is real but weaker.
FILM_ANGULAR_SPREAD_EXPONENT = 0.5
FROSTED_ANGULAR_SPREAD_EXPONENT = 0.25


@njit
def _hash_uniform(seed):
    """A deterministic pseudo random number in (0, 1) from an integer.

    A hash rather than a stateful generator, for two reasons: the CPU and GPU
    kernels share no RNG state to keep in step, and a given ray must scatter the
    same way on every run or the same build would not reproduce.

    Args:
        seed: Any integer; nearby values give unrelated results.

    Returns:
        A float in (0, 1).
    """
    value = (seed * 1103515245 + 12345) & 0x7FFFFFFF
    value ^= value >> 13
    value = (value * 1274126177) & 0x7FFFFFFF
    return (value + 0.5) / 2147483648.0


@njit
def scatter_direction(dx, dy, dz, spread_rad, seed):
    """Tilts a direction by a small random angle drawn from a Gaussian.

    A rough mirror and a diffusing lens do the same thing to a ray: nudge it off
    course by an angle from a narrow distribution. The tilt is applied across
    the ray and the result renormalised, so only the direction changes.

    Args:
        dx, dy, dz: The direction, assumed unit length.
        spread_rad: Standard deviation of the tilt, in radians.
        seed: Chooses which tilt this ray gets.

    Returns:
        The tilted unit direction.
    """
    if spread_rad <= 0.0:
        return dx, dy, dz

    magnitude = spread_rad * math.sqrt(-2.0 * math.log(_hash_uniform(seed)))
    around = 2.0 * math.pi * _hash_uniform(seed + 7919)

    # Any vector not parallel to the ray gives a basis across it.
    if abs(dz) < 0.9:
        ax, ay, az = 0.0, 0.0, 1.0
    else:
        ax, ay, az = 1.0, 0.0, 0.0
    ux, uy, uz = dy * az - dz * ay, dz * ax - dx * az, dx * ay - dy * ax
    length = math.sqrt(ux * ux + uy * uy + uz * uz)
    ux, uy, uz = ux / length, uy / length, uz / length
    vx, vy, vz = dy * uz - dz * uy, dz * ux - dx * uz, dx * uy - dy * ux

    tilt_u = magnitude * math.cos(around)
    tilt_v = magnitude * math.sin(around)
    nx = dx + tilt_u * ux + tilt_v * vx
    ny = dy + tilt_u * uy + tilt_v * vy
    nz = dz + tilt_u * uz + tilt_v * vz
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    return nx / length, ny / length, nz / length


@njit
def fresnel_transmission(cos_incidence, index):
    """Fraction of unpolarised light passing both faces of a flat window.

    Straight from the Fresnel equations, averaged over the two polarisations
    because an LED emits no preferred one, and squared because the light meets
    an air to glass face going in and a glass to air face coming out. Total
    internal reflection cannot happen entering from air, so the only thing to
    guard is the arithmetic at grazing incidence.

    Args:
        cos_incidence: Cosine of the angle to the lens normal.
        index: Refractive index of the lens.

    Returns:
        Transmitted fraction, between 0 and 1.
    """
    cos_in = min(1.0, max(1e-6, abs(cos_incidence)))
    sin_out = math.sqrt(max(0.0, 1.0 - cos_in * cos_in)) / index
    if sin_out >= 1.0:
        return 0.0
    cos_out = math.sqrt(max(0.0, 1.0 - sin_out * sin_out))

    perpendicular = ((cos_in - index * cos_out) / (cos_in + index * cos_out)) ** 2
    parallel = ((index * cos_in - cos_out) / (index * cos_in + cos_out)) ** 2
    single_face = 1.0 - 0.5 * (perpendicular + parallel)
    return single_face * single_face


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
                       emitter_offset_x, emitter_offset_y, transmissivity_lens,
                       use_spherical, dome_polar_step_rad, dome_azimuth_step_rad,
                       dome_polar_bins, dome_azimuth_bins,
                       scatter_sigma_rad, lens_finish_code, lens_diffusion_rad,
                       lens_index, ray_seed):
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
        use_spherical: 1 to bin by outgoing direction on the dome, 0 to bin
            straight onto the wall.
        dome_polar_step_rad, dome_azimuth_step_rad: Angular bin sizes.
        dome_polar_bins, dome_azimuth_bins: Angular bin counts.
        scatter_sigma_rad: How much the bowl softens the beam, as a
            standard deviation in radians. A vapour deposited coating is
            smooth in itself, so this is really the finish of the
            substrate underneath it.
        lens_finish_code: 0 clear, 1 frosted, 2 applied film.
        lens_diffusion_rad: Scattering angle of a frosted or filmed lens.
        lens_index: Refractive index of the lens, or zero to skip the
            angle dependent Fresnel losses entirely. No thickness is
            needed: the index governs reflection at the two faces, which
            is what varies with angle, while absorption through the bulk
            is already covered by transmissivity_lens.
        ray_seed: Makes this ray scatter the same way every run.

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

            # No mirror is perfect. Aluminium laid down by vapour deposition
            # is smooth in itself, so what softens the beam is the finish of
            # the substrate beneath it. Only the bowl is treated this way:
            # the bore wall and the gasket are not optical surfaces.
            if hit_type == _HIT_PARABOLA and scatter_sigma_rad > 0.0:
                dx, dy, dz = scatter_direction(dx, dy, dz, scatter_sigma_rad,
                                               ray_seed * 4096 + bounces)
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
                    # The lens is a flat window across the mouth, so the
                    # angle to its normal is simply the ray's tilt from the
                    # axis.
                    exit_length = math.sqrt(dx * dx + dy * dy + dz * dz)
                    cos_incidence = dz / exit_length
                    remaining_flux *= transmissivity_lens

                    # An index of zero means the operator has not said what
                    # the lens is made of, so the flat figure above stands on
                    # its own and the Fresnel work is skipped. It costs two
                    # square roots and four divisions on every ray that gets
                    # out, which is worth avoiding when it buys nothing.
                    if lens_index > 1.0:
                        angular = (fresnel_transmission(cos_incidence, lens_index)
                                   / fresnel_transmission(1.0, lens_index))

                        # Etched glass presents randomly tilted microfacets,
                        # so a ray arriving off axis still meets many of them
                        # near square on. Averaging over that spread flattens
                        # the Fresnel curve, which is why a frosted lens
                        # loses less at a steep angle than a clear one.
                        if lens_finish_code == 1:
                            angular = 1.0 + FROSTED_ANGULAR_SOFTENING * (angular - 1.0)
                        remaining_flux *= angular

                    if lens_diffusion_rad > 0.0 and lens_finish_code != 0:
                        # Both kinds of diffuser scatter more widely off
                        # axis, but for different reasons and by different
                        # amounts. A film is a layer with thickness, so an
                        # oblique ray takes a longer path through it and
                        # meets more scattering centres; for a random walk
                        # the spread grows as the square root of that path.
                        # Etched glass scatters at a single rough interface
                        # with no depth to lengthen, so the broadening comes
                        # only from the microfacet slopes projecting
                        # differently, which is a weaker effect. Hence the
                        # smaller exponent rather than none at all.
                        exponent = (FILM_ANGULAR_SPREAD_EXPONENT
                                    if lens_finish_code == 2
                                    else FROSTED_ANGULAR_SPREAD_EXPONENT)
                        spread = lens_diffusion_rad * (
                            1.0 / max(cos_incidence, 0.05)) ** exponent
                        dx, dy, dz = scatter_direction(dx, dy, dz, spread,
                                                       ray_seed * 4096 + 2048)

                    if use_spherical:
                        # Bin by the direction the ray leaves in, which is
                        # what the head actually emits. Where that lands on
                        # a wall is a separate question, answered later.
                        length = math.sqrt(dx * dx + dy * dy + dz * dz)
                        polar = math.acos(min(1.0, max(-1.0, dz / length)))
                        azimuth = math.atan2(dy, dx)
                        if azimuth < 0.0:
                            azimuth += 2.0 * math.pi

                        row = int(polar / dome_polar_step_rad)
                        col = int(azimuth / dome_azimuth_step_rad)
                        if 0 <= row < dome_polar_bins and 0 <= col < dome_azimuth_bins:
                            return remaining_flux, row, col, bounces
                        return 0.0, -1, -1, -1

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
                         use_spherical, dome_polar_step_rad,
                         dome_azimuth_step_rad, dome_polar_bins,
                         dome_azimuth_bins, scatter_sigma_rad, lens_finish_code,
                         lens_diffusion_rad, lens_index,
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
        emitter_offset_x, emitter_offset_y, transmissivity_lens,
        use_spherical, dome_polar_step_rad, dome_azimuth_step_rad,
        dome_polar_bins, dome_azimuth_bins,
        scatter_sigma_rad, lens_finish_code, lens_diffusion_rad,
        lens_index, index)

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
                         use_spherical, dome_polar_step_rad,
                         dome_azimuth_step_rad, dome_polar_bins,
                         dome_azimuth_bins, scatter_sigma_rad, lens_finish_code,
                         lens_diffusion_rad, lens_index,
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
            emitter_offset_x, emitter_offset_y, transmissivity_lens,
        use_spherical, dome_polar_step_rad, dome_azimuth_step_rad,
        dome_polar_bins, dome_azimuth_bins,
        scatter_sigma_rad, lens_finish_code, lens_diffusion_rad,
        lens_index, index)

        if row != -1 and col != -1:
            if bounces > 0:
                hotspot_grid[row, col] += final_flux
            else:
                spill_grid[row, col] += final_flux


def _build_kernel_args(element_x, element_y, element_weight,
                      ray_vx, ray_vy, ray_vz, ray_flux,
                       geom: dict, config: SimulationConfig, target_z_mm: float,
                       hotspot_grid, spill_grid,
                       dome=(0, 0.0, 0.0, 0)) -> tuple:
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
        int(dome[0] > 0), float(dome[1]), float(dome[2]), int(dome[0]), int(dome[3]),
        float(geom["scatter_sigma_rad"]), int(geom["lens_finish_code"]),
        float(geom["lens_diffusion_rad"]), float(geom["lens_index"]),
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
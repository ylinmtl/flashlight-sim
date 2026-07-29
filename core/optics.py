"""Turning a hardware combination into traceable geometry.

Part of the flashlight simulator core; see core/__init__.py for the
public surface.
"""

import json
import math
from typing import NamedTuple, Optional

import numpy as np

from .config import SimulationConfig
from .hardware import DIE_SHAPES, spec_or_default


# Probes per axis used to work out how much of a boundary cell lies on the die.
# Eight gives sixty four probes per cell, which places the emitting area of a
# circle or a chamfered die within about 0.2% of its true value. The cost is a
# few milliseconds once per simulation, so accuracy is worth more than speed
# here. Raising it further converges roughly in proportion to 1 / probes.
DIE_SUBSAMPLES = 8


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

"""Builds and checks emitter die outlines for the flashlight simulator.

An emitter whose light emitting surface is neither a plain rectangle nor a
circle is modelled with ``"shape": "polygon"`` and a ``die_outline``: the
corners of the emitting area, in millimetres, measured from the centre of the
die. This module writes those corner lists, so that a shape is described by the
two or three numbers taken off a datasheet rather than by hand typed
coordinates, and the same numbers always give the same outline.

Run it to print an outline ready to paste into the Die Outline box:

    python emitter_shapes.py sft60 --size 2.0 --corner 0.35
    python emitter_shapes.py sft40 --size 2.0 --corner 0.3
    python emitter_shapes.py rectangle --size 2.0x1.2
    python emitter_shapes.py --list

Add ``--preview`` to any of those to see the outline drawn with the sample
points the tracer will actually use, which is the quickest way to catch an
outline that is the wrong size or the wrong way round.

Coordinates use the simulator's convention: +x to the right, +y up, origin at
the centre of the die, which is also the reflector's optical axis.
"""

import argparse
import json
import math
import sys

# Vertices closer together than this, in millimetres, are treated as one. It is
# far below any real feature on a die and well above floating point noise.
_MERGE_TOLERANCE_MM = 1e-6


def rectangle(width_mm, height_mm=None):
    """Builds a plain rectangle, the base every other shape starts from.

    Args:
        width_mm: Size along x.
        height_mm: Size along y. Defaults to width_mm, giving a square.

    Returns:
        Four vertices, anticlockwise from the bottom left.
    """
    height_mm = width_mm if height_mm is None else height_mm
    half_w, half_h = width_mm / 2.0, height_mm / 2.0
    return [[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]]


def chamfered_rectangle(width_mm, corner_mm, height_mm=None):
    """Builds a rectangle with the corners cut off at 45 degrees.

    This is the Luminus SFT60 family: a square die whose corners are sliced
    away, leaving an octagon.

    Args:
        width_mm: Size along x, measured across the flats.
        corner_mm: How far back along each edge the cut starts. A cut of half
            the width on a square turns it into a diamond.
        height_mm: Size along y. Defaults to width_mm.

    Returns:
        Eight vertices, anticlockwise from the bottom edge.
    """
    height_mm = width_mm if height_mm is None else height_mm
    half_w, half_h = width_mm / 2.0, height_mm / 2.0
    cut = float(corner_mm)
    return [
        [-half_w + cut, -half_h], [half_w - cut, -half_h],
        [half_w, -half_h + cut], [half_w, half_h - cut],
        [half_w - cut, half_h], [-half_w + cut, half_h],
        [-half_w, half_h - cut], [-half_w, -half_h + cut],
    ]


def notched_rectangle(width_mm, corner_mm, height_mm=None, notch_height_mm=None):
    """Builds a rectangle with a square notch bitten out of each corner.

    This is the Luminus SFT40 family: a square die with a small rectangular
    piece missing from each corner, so the outline is concave.

    Args:
        width_mm: Size along x.
        corner_mm: Width of the notch, measured along x.
        height_mm: Size along y. Defaults to width_mm.
        notch_height_mm: Depth of the notch, measured along y. Defaults to
            corner_mm, giving a square notch.

    Returns:
        Twelve vertices, anticlockwise from the bottom edge.
    """
    height_mm = width_mm if height_mm is None else height_mm
    notch_height_mm = corner_mm if notch_height_mm is None else notch_height_mm
    half_w, half_h = width_mm / 2.0, height_mm / 2.0
    cut_x, cut_y = float(corner_mm), float(notch_height_mm)
    return [
        [-half_w + cut_x, -half_h], [half_w - cut_x, -half_h],
        [half_w - cut_x, -half_h + cut_y], [half_w, -half_h + cut_y],
        [half_w, half_h - cut_y], [half_w - cut_x, half_h - cut_y],
        [half_w - cut_x, half_h], [-half_w + cut_x, half_h],
        [-half_w + cut_x, half_h - cut_y], [-half_w, half_h - cut_y],
        [-half_w, -half_h + cut_y], [-half_w + cut_x, -half_h + cut_y],
    ]


def rounded_rectangle(width_mm, corner_mm, height_mm=None, segments=6):
    """Builds a rectangle with rounded corners, approximated by short edges.

    Args:
        width_mm: Size along x.
        corner_mm: Corner radius.
        height_mm: Size along y. Defaults to width_mm.
        segments: Straight edges used per corner. Six is plenty at the
            resolutions the tracer samples a die at.

    Returns:
        4 * (segments + 1) vertices, anticlockwise.
    """
    height_mm = width_mm if height_mm is None else height_mm
    half_w, half_h = width_mm / 2.0, height_mm / 2.0
    radius = float(corner_mm)

    vertices = []
    corners = [(half_w - radius, -half_h + radius, -90.0),
               (half_w - radius, half_h - radius, 0.0),
               (-half_w + radius, half_h - radius, 90.0),
               (-half_w + radius, -half_h + radius, 180.0)]
    for centre_x, centre_y, start_deg in corners:
        for step in range(segments + 1):
            angle = math.radians(start_deg + 90.0 * step / segments)
            vertices.append([centre_x + radius * math.cos(angle),
                             centre_y + radius * math.sin(angle)])
    return vertices


def regular_polygon(across_flats_mm, sides, rotation_deg=0.0):
    """Builds a regular polygon, sized across the flats like a spanner size.

    Args:
        across_flats_mm: Distance between opposite edges.
        sides: Number of edges, at least three.
        rotation_deg: Anticlockwise rotation applied to the whole outline.

    Returns:
        One vertex per side, anticlockwise.
    """
    if sides < 3:
        raise ValueError("A polygon needs at least three sides.")

    inradius = across_flats_mm / 2.0
    circumradius = inradius / math.cos(math.pi / sides)
    return [[circumradius * math.cos(math.radians(rotation_deg)
                                     + 2.0 * math.pi * i / sides + math.pi / sides),
             circumradius * math.sin(math.radians(rotation_deg)
                                     + 2.0 * math.pi * i / sides + math.pi / sides)]
            for i in range(sides)]


def circle(diameter_mm, segments=32):
    """Builds a circular outline.

    A round die is better described with ``"shape": "round"``, which masks an
    exact circle. This is here for dies that are round but not centred, or
    round with a flat, once the result is edited.

    Args:
        diameter_mm: Diameter of the emitting area.
        segments: Straight edges approximating the circle.

    Returns:
        ``segments`` vertices, anticlockwise.
    """
    radius = diameter_mm / 2.0
    return [[radius * math.cos(2.0 * math.pi * i / segments),
             radius * math.sin(2.0 * math.pi * i / segments)]
            for i in range(segments)]


# Shapes the command line can build, as name -> (builder, help text).
SHAPE_BUILDERS = {
    "rectangle": (rectangle, "plain rectangle or square"),
    "sft60": (chamfered_rectangle, "square with the corners cut off (SFT60 style)"),
    "chamfered": (chamfered_rectangle, "same as sft60"),
    "sft40": (notched_rectangle, "square with a notch in each corner (SFT40 style)"),
    "notched": (notched_rectangle, "same as sft40"),
    "rounded": (rounded_rectangle, "rectangle with rounded corners"),
    "polygon": (regular_polygon, "regular polygon, sized across the flats"),
    "circle": (circle, "circle approximated by straight edges"),
}


def outline_area_mm2(vertices):
    """Returns the enclosed area in square millimetres, via the shoelace sum.

    Args:
        vertices: List of [x, y] pairs in order around the outline.

    Returns:
        The area. Positive when the vertices run anticlockwise.
    """
    total = 0.0
    for index, (x0, y0) in enumerate(vertices):
        x1, y1 = vertices[(index + 1) % len(vertices)]
        total += x0 * y1 - x1 * y0
    return total / 2.0


def describe(vertices):
    """Summarises an outline, for checking it against a datasheet.

    Args:
        vertices: List of [x, y] pairs in order around the outline.

    Returns:
        A dict of bounding box, area, fill fraction and centroid offset.
    """
    xs = [x for x, _ in vertices]
    ys = [y for _, y in vertices]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    area = abs(outline_area_mm2(vertices))
    return {
        "vertices": len(vertices),
        "width_mm": width,
        "height_mm": height,
        "area_mm2": area,
        "fill_fraction": area / (width * height) if width and height else 0.0,
        "centre_offset_mm": (( max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0),
    }


def validate(vertices):
    """Checks an outline for the mistakes that make a die model wrong.

    Catches the errors that still produce a runnable simulation, which are the
    dangerous ones: an outline in the wrong units, one that is not centred on
    the optical axis, or one with duplicated corners.

    Args:
        vertices: List of [x, y] pairs in order around the outline.

    Returns:
        A list of problem descriptions. Empty means the outline looks sound.
    """
    problems = []

    if not isinstance(vertices, list) or len(vertices) < 3:
        return ["Needs at least three [x, y] vertex pairs."]
    if any(not isinstance(v, (list, tuple)) or len(v) != 2 for v in vertices):
        return ["Every vertex must be a pair of numbers, [x, y]."]

    facts = describe(vertices)
    if facts["area_mm2"] <= 0.0:
        problems.append("Encloses no area; the vertices may be out of order.")

    if max(facts["width_mm"], facts["height_mm"]) > 25.0:
        problems.append(
            f"Spans {facts['width_mm']:.1f} x {facts['height_mm']:.1f}, which is "
            f"large for a die. Check the units are millimetres, not microns.")

    offset_x, offset_y = facts["centre_offset_mm"]
    if max(abs(offset_x), abs(offset_y)) > 0.05 * max(facts["width_mm"],
                                                      facts["height_mm"]):
        problems.append(
            f"Bounding box centre is at ({offset_x:.3f}, {offset_y:.3f}) rather "
            f"than the origin. The die will sit off the optical axis.")

    for index, (x0, y0) in enumerate(vertices):
        x1, y1 = vertices[(index + 1) % len(vertices)]
        if math.hypot(x1 - x0, y1 - y0) < _MERGE_TOLERANCE_MM:
            problems.append(f"Vertices {index} and {index + 1} are duplicates.")

    return problems


def to_json(vertices, decimals=4):
    """Renders an outline as the compact JSON the Die Outline box expects.

    Args:
        vertices: List of [x, y] pairs.
        decimals: Places to round to, to keep the text readable.

    Returns:
        A single line of JSON.
    """
    rounded = [[round(float(x), decimals), round(float(y), decimals)]
               for x, y in vertices]
    return json.dumps(rounded, separators=(",", ":"))


def preview(vertices, elements_per_side=16):
    """Draws the outline with the points the tracer would sample inside it.

    Args:
        vertices: List of [x, y] pairs.
        elements_per_side: Grid resolution, matching sim_emitter_elements.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Preview needs matplotlib and numpy.", file=sys.stderr)
        return

    from core.optics import _points_in_polygon

    outline = np.asarray(vertices, dtype=float)
    min_x, min_y = outline.min(axis=0)
    max_x, max_y = outline.max(axis=0)
    grid_x, grid_y = np.meshgrid(
        np.linspace(min_x, max_x, elements_per_side),
        np.linspace(min_y, max_y, elements_per_side))
    tolerance = 1e-9 * max(max_x - min_x, max_y - min_y, 1.0)
    inside = _points_in_polygon(grid_x.ravel(), grid_y.ravel(), outline, tolerance)

    closed = np.vstack([outline, outline[:1]])
    figure, axes = plt.subplots(figsize=(5, 5))
    axes.plot(closed[:, 0], closed[:, 1], "-o", color="#0072B2", markersize=3)
    axes.scatter(grid_x.ravel()[inside], grid_y.ravel()[inside], s=12,
                 color="#D55E00", label=f"{int(inside.sum())} emitting elements")
    axes.scatter(grid_x.ravel()[~inside], grid_y.ravel()[~inside], s=6,
                 color="#BBBBBB", label="masked out")
    axes.axhline(0, lw=0.5, color="#999999")
    axes.axvline(0, lw=0.5, color="#999999")
    axes.set_aspect("equal")
    axes.set_xlabel("x (mm)")
    axes.set_ylabel("y (mm)")
    axes.set_title(f"Die outline, {elements_per_side} x {elements_per_side} grid")
    axes.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    plt.show()


def _parse_size(text):
    """Parses a --size argument of "2.0" or "2.0x1.2".

    Args:
        text: The argument value.

    Returns:
        (width_mm, height_mm), where height is None for a single number.
    """
    if "x" in text.lower():
        width, height = text.lower().split("x", 1)
        return float(width), float(height)
    return float(text), None


def main(argv=None):
    """Builds one outline from the command line and prints it.

    Args:
        argv: Argument list, defaulting to sys.argv.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(
        description="Generate an emitter die outline for the flashlight simulator.")
    parser.add_argument("shape", nargs="?", choices=sorted(SHAPE_BUILDERS),
                        help="Which outline to build.")
    parser.add_argument("--size", default="2.0",
                        help="Die size in mm, as WIDTH or WIDTHxHEIGHT.")
    parser.add_argument("--corner", type=float, default=0.3,
                        help="Chamfer, notch or corner radius in mm.")
    parser.add_argument("--notch-depth", type=float, default=None,
                        help="Notch depth in mm, for a non-square notch.")
    parser.add_argument("--sides", type=int, default=6,
                        help="Sides, for the polygon shape.")
    parser.add_argument("--elements", type=int, default=16,
                        help="Grid resolution for --preview, i.e. "
                             "sim_emitter_elements.")
    parser.add_argument("--preview", action="store_true",
                        help="Draw the outline and its sample points.")
    parser.add_argument("--list", action="store_true",
                        help="List the shapes this can build.")
    args = parser.parse_args(argv)

    if args.list or not args.shape:
        print("Available shapes:\n")
        for name in sorted(SHAPE_BUILDERS):
            print(f"  {name:<12} {SHAPE_BUILDERS[name][1]}")
        print("\nExample: python emitter_shapes.py sft60 --size 2.0 --corner 0.35")
        return 0

    width, height = _parse_size(args.size)
    builder = SHAPE_BUILDERS[args.shape][0]

    if builder is rectangle:
        vertices = rectangle(width, height)
    elif builder is chamfered_rectangle:
        vertices = chamfered_rectangle(width, args.corner, height)
    elif builder is notched_rectangle:
        vertices = notched_rectangle(width, args.corner, height, args.notch_depth)
    elif builder is rounded_rectangle:
        vertices = rounded_rectangle(width, args.corner, height)
    elif builder is regular_polygon:
        vertices = regular_polygon(width, args.sides)
    else:
        vertices = circle(width)

    facts = describe(vertices)
    print(f"{args.shape}: {facts['vertices']} vertices, "
          f"{facts['width_mm']:.3f} x {facts['height_mm']:.3f} mm, "
          f"area {facts['area_mm2']:.4f} mm2 "
          f"({facts['fill_fraction'] * 100:.1f}% of the bounding box)\n")

    problems = validate(vertices)
    for problem in problems:
        print(f"  [!] {problem}")
    if problems:
        print()

    print("Paste this into the emitter's Die Outline box, and set Shape to "
          "polygon:\n")
    print(to_json(vertices))

    if args.preview:
        preview(vertices, args.elements)
    return 0


if __name__ == "__main__":
    sys.exit(main())

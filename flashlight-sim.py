"""PyQt6 desktop front end for the flashlight beam simulator.

The window is laid out in mainwindow.ui and loaded at runtime. It offers a
catalogue browser for the three hardware kinds (reflector, emitter, gasket), a
settings dialog generated from the active SimulationConfig, and an embedded
Matplotlib canvas for the rendered beam. Simulations run on a worker thread so
the interface stays responsive and remains cancellable.
"""

import json
import math
import os
import sys
import traceback

from PyQt6 import uic
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                             QSlider,
                             QFormLayout, QGroupBox, QHBoxLayout, QInputDialog,
                             QLineEdit,
                             QMainWindow, QMessageBox, QPushButton, QScrollArea,
                             QSizePolicy, QVBoxLayout, QWidget)

# Matplotlib's Qt canvas, used to embed the engine's figure in the window.
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import to_rgba_array
from matplotlib.figure import Figure
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from core import (DIE_SHAPES, GASKET_WALL_SHAPES, OUTPUT_MODES,
                    SPEC_DEFAULT_SETTINGS, SURFACE_FINISHES, EmitterOffset,
                    HardwareLibrary, SimulationConfig,
                    effective_bore_diameter, emitter_die_outline,
                    emitter_footprint_diagonal, get_sim_geometry,
                    render_wall_shot, resource_path, run_simulation_job,
                    spec_or_default)

# Specs shown for each hardware kind, in the order they appear in the form. Each
# one maps to a QLineEdit in mainwindow.ui named <prefix><spec>, for example
# reflector "diameter_mm" -> txtRef_diameter_mm.
SPEC_FIELDS = {
    "reflector": ("diameter_mm", "height_mm", "opening_diameter_mm",
                  "focus_offset_mm", "wall_thickness_mm", "thickness_height_mm",
                  "surface_finish", "reflectivity_smooth", "reflectivity_op",
                  "reflectivity_cylinder", "gasket_reflectivity", "OP_Factor",
                  "transmissivity_lens"),
    "emitter": ("max_current_amps", "output_mode", "max_lumens",
                "forward_voltage_v", "vf_turn_on_v", "vf_scale",
                "base_efficacy_lm_w", "droop_factor",
                "footprint_x_mm", "footprint_y_mm", "height_mm",
                "dome_size_mm", "refractive_index", "die_length_mm", "die_width_mm",
                "shape", "die_outline"),
    "gasket": ("outer_diameter_mm", "inner_diameter_mm", "emitter_size_mm",
               "wall_shape", "thickness_mm", "total_height_mm"),
}

FIELD_WIDGET_PREFIX = {
    "reflector": "txtRef_",
    "emitter": "txtEmi_",
    "gasket": "txtGask_",
}

# Specs that stay strings; every other field is parsed as a float.
TEXT_SPECS = frozenset({"shape"})

# Emitter palette, shared by both previews so the emitter looks identical
# whether it is shown on its own or sitting inside the reflector.
EMITTER_BODY_COLOUR = "#3F3F3F"    # sides of the package
EMITTER_BASE_COLOUR = "#C6C8CC"    # underside, the silver solder pad face
EMITTER_TOP_COLOUR = "#FFFFFF"     # top face, around the emitting surface
EMITTER_TOP_EDGE_COLOUR = "#7A7A7A"
EMITTER_DIE_COLOUR = "#FF9E1B"     # the light emitting surface itself
EMITTER_DIE_EDGE_COLOUR = "#C77B14"
EMITTER_DOME_COLOUR = "#CBE7F5"

# The reflector is drawn as a single opaque silver body, stroked along
# its own mesh so the shape reads. The gasket and emitter go on after
# it, so they stay visible without the shell having to be see through.
REFLECTOR_COLOUR = "#C6C8CC"
REFLECTOR_ALPHA = 1.0
REFLECTOR_EDGE_COLOUR = "#94969B"

# How much wider than the parts themselves the gasket preview draws. The
# seat sits proud of the wall so its edge stays visible behind it, and
# the emitter window is cut with a little clearance a side.
GASKET_SEAT_MARGIN_MM = 2.0

# Slack allowed when judging whether a gasket fits, in millimetres.
# Catalogue figures are rounded, so an exact comparison would reject a
# gasket that is the intended match by a hundredth of a millimetre.
GASKET_FIT_TOLERANCE_MM = 0.01

# Slack allowed before a dimension counts as mismatched, and how far a
# gasket may sit under the bore before it is worth pointing out.
FIT_TOLERANCE_MM = 0.01
GASKET_BORE_SLACK_MM = 0.5

# Stand ins used while a column is empty, so the reflector can still be
# drawn beside an emitter or gasket that is still being typed in. Only
# the specs the geometry reads are needed, and they contribute nothing.
BLANK_EMITTER = {"footprint_x_mm": 0.0, "footprint_y_mm": 0.0,
                 "height_mm": 0.0, "dome_size_mm": 0.0,
                 "refractive_index": 1.0}
BLANK_GASKET = {"thickness_mm": 0.0, "total_height_mm": 0.0,
                "inner_diameter_mm": 0.0}
GASKET_EMITTER_CLEARANCE_MM = 0.1

# The gasket is a rubber part, so it is drawn darker than the metal around it
# and near enough opaque to read against the reflector behind it.
GASKET_COLOUR = "#D8D5CE"
GASKET_EDGE_COLOUR = "#6B6960"

# Both previews sit on black, like the simulated beam shot does.
PREVIEW_BACKGROUND = "#000000"
PREVIEW_TEXT_COLOUR = "#FFFFFF"

# How far the exposure slider reaches, in stops either side of zero. The
# box beside it is not limited to this: a value typed there simply parks
# the slider at its end stop.
EXPOSURE_SLIDER_RANGE_EV = 10.0

# Zoom applied per mouse wheel step in a 3D preview.
PREVIEW_ZOOM_STEP = 1.15

# Specs the form offers as a drop down, mapped to the values the engine
# accepts. The box shows each value capitalised, with underscores as spaces,
# while the catalogue keeps the plain value listed here.
CHOICE_SPECS = {
    "shape": DIE_SHAPES,
    "surface_finish": SURFACE_FINISHES,
    "wall_shape": GASKET_WALL_SHAPES,
    "output_mode": OUTPUT_MODES,
}

# Specs only meaningful for certain choices, as spec -> (deciding spec,
# values that need it). The row is hidden when it does not apply, so an
# outline box is not offered for a die that has no outline.
CONDITIONAL_SPECS = {
    "die_outline": ("shape", frozenset({"polygon"})),
    "max_lumens": ("output_mode", frozenset({"simple"})),
    "forward_voltage_v": ("output_mode", frozenset({"simple"})),
    "vf_turn_on_v": ("output_mode", frozenset({"advanced"})),
    "vf_scale": ("output_mode", frozenset({"advanced"})),
    "base_efficacy_lm_w": ("output_mode", frozenset({"advanced"})),
    "droop_factor": ("output_mode", frozenset({"advanced"})),
}

# Where New should start a field somewhere other than the settings
# default. A new emitter is described the simple way, but one already
# in a catalogue predates the choice and has to be read the old way, so
# the setting itself stays on advanced for their sake.
NEW_ENTRY_OVERRIDES = {"emitter": {"output_mode": "simple"}}

# Specs held as a JSON array rather than a single value. The input box takes
# the text an outline generator produces, so a die shape can be pasted in.
LIST_SPECS = frozenset({"die_outline"})

# Reflector inputs that describe the build being simulated rather than the
# reflector itself, as (field, label). Each one is a QLineEdit in
# mainwindow.ui named <prefix><field>, exactly like a spec, but they are
# deliberately absent from SPEC_FIELDS so that saving a reflector discards
# them, and they reset to zero whenever the form is reloaded.
RUN_ONLY_REFLECTOR_FIELDS = (
    ("emitter_offset_distance_mm", "Emitter Offset Distance (mm)"),
    ("emitter_offset_angle_deg", "Emitter Offset Angle (° CW from up)"),
)

# Settings offered by the settings dialog, grouped exactly as they are stored,
# mapping each attribute of SimulationConfig to its human readable label.
SETTING_LABELS = {
    "Output & Rendering": {
        "generate_all_plots": "Generate All Plots (Batch Mode)",
        "plot_wall_shot": "Plot Wall Shot (2D Image)",
        "plot_intensity_x": "Plot Intensity Profile (X-Axis)",
        "plot_intensity_y": "Plot Intensity Profile (Y-Axis)",
        "plot_intensity_45": "Plot Intensity Profile (45° Diagonal)",
        "show_human_silhouette": "Show Human Silhouette Reference",
        "export_csv": "Export Results to CSV",
        "export_plots": "Export Plot Images",
        "batch_output_directory": "Output Directory Path",
    },
    "IES Export": {
        "export_ies": "Export IES",
        "ies_vertical_step_deg": "IES Vertical Step (deg)",
        "ies_horizontal_step_deg": "IES Horizontal Step (deg)",
        "ies_max_vertical_angle_deg": "IES Max Vertical Angle (deg)",
    },
    "Simulation Space & Constraints": {
        "use_gpu": "Use GPU Acceleration (CUDA)",
        "max_multiple_reflections": "Max Multiple Reflections (Bounces)",
        "use_reflector_opening": "Force Reflector Opening Size",
        "target_distance_m": "Target Distance (meters)",
        "canvas_fov_deg": "Canvas Field of View (degrees)",
        "plot_fov_deg": "Plot Field of View (degrees)",
    },
    "Camera Settings": {
        "use_auto_exposure": "Use Auto Exposure",
        "auto_exposure_compensation_ev": "Auto Exposure Compensation (EV)",
        "cam_iso": "Camera ISO",
        "cam_f_stop": "Camera f-stop",
        "cam_shutter_speed_s": "Camera Shutter Speed (seconds)",
    },
    "Resolution & Angular Density": {
        "sim_grid_res": "Simulation Grid Resolution (px)",
        "sim_emitter_elements": "Emitter Subdivision Elements",
        "sim_theta_step_deg": "Theta Step Size (degrees)",
        "sim_phi_step_deg": "Phi Step Size (degrees)",
        "sim_theta_min_deg": "Theta Minimum (degrees)",
        "sim_theta_max_deg": "Theta Maximum (degrees)",
        "sim_phi_min_deg": "Phi Minimum (degrees)",
        "sim_phi_max_deg": "Phi Maximum (degrees)",
    },
    "Material Defaults & Thresholds": {
        "default_reflectivity_smooth": "Default Reflectivity (Smooth)",
        "default_reflectivity_op": "Default Reflectivity (Orange Peel)",
        "default_reflectivity_cylinder": "Default Reflectivity (Cylinder)",
        "default_gasket_reflectivity": "Default Reflectivity (Gasket)",
        "default_op_blur_strength": "Orange Peel Blur Strength",
        "default_op_factor": "Default OP Factor",
        "default_transmissivity_lens": "Default Lens Transmissivity",
        "default_surface_finish": "Default Surface Finish",
        "spill_visible_threshold_lux": "Spill Visible Threshold (Lux)",
        "corona_visible_threshold": "Corona Visible Threshold",
        "hotspot_fwhm_threshold": "Hotspot FWHM Threshold",
        "default_gasket_thickness_mm": "Default Gasket Thickness (mm)",
        "default_gasket_total_height_mm": "Default Gasket Total Height (mm)",
        "default_gasket_inner_diameter_mm": "Default Gasket Inner Diameter (mm)",
        "default_gasket_wall_shape": "Default Gasket Wall Shape",
        "default_reflector_wall_thickness_mm": "Default Reflector Wall Thickness (mm)",
        "default_reflector_base_thickness_mm": "Default Reflector Base Thickness (mm)",
        "default_focus_offset_mm": "Default Focus Offset (mm)",
        "default_opening_diameter_mm": "Default Reflector Opening Diameter (mm)",
        "default_dome_size_mm": "Default Emitter Dome Size (mm)",
        "default_refractive_index": "Default Emitter Refractive Index",
        "default_emitter_shape": "Default Emitter Die Shape",
        "default_emitter_output_mode": "Default Emitter Output Mode",
        "default_max_lumens": "Default Max Lumens (lm)",
        "default_forward_voltage_v": "Default Voltage (V)",
        "default_vf_turn_on_v": "Default VF Turn On (V)",
        "default_vf_scale": "Default VF Scale",
        "default_base_efficacy_lm_w": "Default Base Efficiency (lm/W)",
        "default_droop_factor": "Default Droop Factor",
    },
}


def _choice_label(value):
    """Renders a stored choice for display.

    Args:
        value: The value as the catalogue stores it, such as "orange_peel".

    Returns:
        The caption to show, such as "Orange Peel".
    """
    return value.replace("_", " ").title()


def _polygon_normal(face):
    """Returns a polygon's unit normal, by Newell's method.

    Newell's method works for any planar polygon, convex or not, and gives
    the outward normal when the vertices run anticlockwise seen from
    outside the solid.

    Args:
        face: Sequence of (x, y, z) vertices in order around the polygon.

    Returns:
        A unit normal as a length 3 array, or zeros for a degenerate face.
    """
    vertices = np.asarray(face, dtype=float)
    following = np.roll(vertices, -1, axis=0)
    normal = np.array([
        np.sum((vertices[:, 1] - following[:, 1])
               * (vertices[:, 2] + following[:, 2])),
        np.sum((vertices[:, 2] - following[:, 2])
               * (vertices[:, 0] + following[:, 0])),
        np.sum((vertices[:, 0] - following[:, 0])
               * (vertices[:, 1] + following[:, 1]))])
    length = float(np.linalg.norm(normal))
    return normal / length if length else normal


class _SolidFaces(Poly3DCollection):
    """Polygon collection that hides the faces turned away from the viewer.

    Depth sorting cannot order the faces of a thin box reliably: the top and
    the underside of a sub-millimetre package overlap in depth at a shallow
    angle, so the die shows through from below however the sort is tuned.
    For a closed solid that question never has to be answered, because a
    face pointing away from the viewer cannot be seen. Those are blanked
    instead, which is exact rather than approximate and follows the view as
    it is rotated.
    """

    def __init__(self, faces, facecolours, edgecolours, **kwargs):
        """Builds the collection and records each face's outward normal.

        Args:
            faces: List of polygons, each a sequence of (x, y, z) vertices
                wound anticlockwise seen from outside the solid.
            facecolours: One fill colour per face.
            edgecolours: One edge colour per face.
            **kwargs: Passed to Poly3DCollection.
        """
        super().__init__(faces, **kwargs)
        self._face_normals = np.array([_polygon_normal(f) for f in faces])
        self._solid_facecolours = to_rgba_array(facecolours)
        self._solid_edgecolours = to_rgba_array(edgecolours)

    def do_3d_projection(self):
        """Blanks the back faces for the current view, then projects.

        Returns:
            The depth Matplotlib should sort this artist by.
        """
        elevation = math.radians(self.axes.elev)
        azimuth = math.radians(self.axes.azim)
        towards_viewer = np.array([
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation)])

        facing = self._face_normals @ towards_viewer > 0.0
        for colours, apply in ((self._solid_facecolours, self.set_facecolor),
                               (self._solid_edgecolours, self.set_edgecolor)):
            shown = colours.copy()
            shown[~facing, 3] = 0.0
            apply(shown)
        return super().do_3d_projection()


def _revolved_surface(radii, heights, segments=72):
    """Turns a profile in the radius/height plane into a surface of revolution.

    Args:
        radii: Radius at each point along the profile, in millimetres.
        heights: Height at each point along the profile, in millimetres.
        segments: Steps around the full turn.

    Returns:
        (x, y, z) arrays shaped for plot_surface.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, segments)[:, None]
    radii = np.asarray(radii, dtype=float)[None, :]
    heights = np.asarray(heights, dtype=float)[None, :]
    return (radii * np.cos(angle), radii * np.sin(angle),
            np.repeat(heights, segments, axis=0))


def _dome_surface(radius, centre_z, segments=32):
    """Builds the upper half of a sphere, the shape of a silicone dome.

    The tracer treats the dome as a sphere centred on the die, so only the half
    above the die is drawn: that is the part light actually crosses.

    Args:
        radius: Dome radius in millimetres.
        centre_z: Height of the die, which is the centre of the sphere.
        segments: Steps around the turn; half as many are used up the dome.

    Returns:
        (x, y, z) arrays shaped for plot_surface.
    """
    polar = np.linspace(0.0, np.pi / 2.0, max(segments // 2, 4))[None, :]
    azimuth = np.linspace(0.0, 2.0 * np.pi, segments)[:, None]
    return (radius * np.sin(polar) * np.cos(azimuth),
            radius * np.sin(polar) * np.sin(azimuth),
            centre_z + radius * np.cos(polar) + 0.0 * azimuth)


def _solid_quads(surface, outward):
    """Splits a surface grid into quads wound so their normals face outward.

    _SolidFaces decides what to hide from each face's normal, and a normal only
    means anything if the vertices run consistently. Rather than working the
    winding out by hand for every piece, each quad is checked against the
    direction that piece is known to face and reversed where it disagrees.

    The whole grid is handled at once. A preview rebuilds several thousand
    quads every time a field changes, and doing that a quad at a time in Python
    is what made the previews lag.

    Args:
        surface: (x, y, z) arrays as the surface helpers return them.
        outward: Which way the faces look: "up", "down", "out" for away from
            the axis, or "in" for towards it.

    Returns:
        A list of quads, each a (4, 3) array of vertices.
    """
    x, y, z = surface
    corners = [np.stack([x[a:b or None, c:d or None],
                         y[a:b or None, c:d or None],
                         z[a:b or None, c:d or None]], axis=-1)
               for a, b, c, d in ((0, -1, 0, -1), (1, 0, 0, -1),
                                  (1, 0, 1, 0), (0, -1, 1, 0))]
    quads = np.stack(corners, axis=2).reshape(-1, 4, 3)
    if not len(quads):
        return []

    # Newell's method, run over every quad at once.
    following = np.roll(quads, -1, axis=1)
    normals = np.stack([
        np.sum((quads[:, :, 1] - following[:, :, 1])
               * (quads[:, :, 2] + following[:, :, 2]), axis=1),
        np.sum((quads[:, :, 2] - following[:, :, 2])
               * (quads[:, :, 0] + following[:, :, 0]), axis=1),
        np.sum((quads[:, :, 0] - following[:, :, 0])
               * (quads[:, :, 1] + following[:, :, 1]), axis=1)], axis=1)

    if outward in ("up", "down"):
        wanted = np.zeros_like(normals)
        wanted[:, 2] = 1.0 if outward == "up" else -1.0
    else:
        centres = quads.mean(axis=1)
        wanted = np.stack([centres[:, 0], centres[:, 1],
                           np.zeros(len(quads))], axis=1)
        if outward == "in":
            wanted = -wanted

    facing_away = np.sum(normals * wanted, axis=1) < 0.0
    quads[facing_away] = quads[facing_away][:, ::-1]
    return list(quads)


def _aperture_radius(angle, half_size=None, radius=None):
    """Distance from the centre out to an aperture edge, at each angle.

    A square aperture is described the same way a round one is, as a radius
    that happens to vary with angle, so one piece of code builds either.

    Args:
        angle: Angles in radians, any shape.
        half_size: Half the width of a square aperture.
        radius: Radius of a round aperture. Used when half_size is None.

    Returns:
        The radius at each angle, shaped like angle.
    """
    if half_size is None:
        return np.full_like(angle, float(radius))
    return half_size / np.maximum(np.abs(np.cos(angle)), np.abs(np.sin(angle)))


def _ring_surface(inner, outer, height, segments=192):
    """Builds a flat ring at one height, between an inner and an outer radius.

    Args:
        inner: Inner radius, either a scalar or one value per angle step.
        outer: Outer radius, in the same form.
        height: Height of the plane the ring lies in.
        segments: Steps around the turn.

    Returns:
        (x, y, z) arrays shaped for plot_surface.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, segments)[:, None]
    inner = np.broadcast_to(np.asarray(inner, dtype=float).reshape(-1, 1), angle.shape)
    outer = np.broadcast_to(np.asarray(outer, dtype=float).reshape(-1, 1), angle.shape)
    radii = np.concatenate([inner, outer], axis=1)
    return (radii * np.cos(angle), radii * np.sin(angle),
            np.full_like(radii, float(height)))


def _wall_surface(radius, z_low, z_high, segments=192):
    """Builds an upright wall at a radius that may vary with angle.

    Args:
        radius: Radius, either a scalar or one value per angle step.
        z_low: Height of the bottom edge.
        z_high: Height of the top edge.
        segments: Steps around the turn.

    Returns:
        (x, y, z) arrays shaped for plot_surface.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, segments)[:, None]
    radius = np.broadcast_to(np.asarray(radius, dtype=float).reshape(-1, 1), angle.shape)
    x, y = radius * np.cos(angle), radius * np.sin(angle)
    return (np.concatenate([x, x], axis=1), np.concatenate([y, y], axis=1),
            np.tile([[float(z_low), float(z_high)]], (angle.shape[0], 1)))


def _dimpled_revolution(radii, heights, segments, amplitude, around, along):
    """Revolves a profile with a dimpled surface, the look of orange peel.

    The dimples come from a product of sines rather than random noise, for
    two reasons: a whole number of cycles around the axis closes on itself,
    so there is no seam where the revolution meets, and the same reflector
    draws identically every time instead of shimmering on each redraw.

    Args:
        radii: Radius at each point along the profile, in millimetres.
        heights: Height at each point along the profile, in millimetres.
        segments: Steps around the full turn.
        amplitude: Dimple depth as a fraction of the local radius.
        around: Whole number of dimples around the circumference.
        along: Number of dimples from the bore to the mouth.

    Returns:
        (x, y, z) arrays shaped for plot_surface.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, segments)[:, None]
    radii = np.asarray(radii, dtype=float)[None, :]
    heights = np.asarray(heights, dtype=float)[None, :]
    position = np.linspace(0.0, 1.0, radii.shape[1])[None, :]

    ripple = np.sin(around * angle) * np.sin(along * np.pi * position)
    pushed = radii * (1.0 + amplitude * ripple)
    return (pushed * np.cos(angle), pushed * np.sin(angle),
            heights + 0.0 * angle)


def _outline_area(vertices):
    """Returns the area a polygon outline encloses, via the shoelace sum.

    Args:
        vertices: (N, 2) array of corners in order around the outline.

    Returns:
        The signed area in square millimetres.
    """
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


class SquarePreview(QWidget):
    """Holds a Matplotlib canvas and stays as tall as it is wide.

    Qt has no aspect ratio constraint, so the height is driven from the width
    on every resize. The widget asks for no width of its own, which is what
    keeps a preview from widening the column it sits in: it takes whatever
    width the column already has and squares itself off against it.
    """

    def __init__(self, parent=None):
        """Builds an empty square canvas holder.

        Args:
            parent: Widget to attach to.
        """
        super().__init__(parent)
        self.figure = Figure(figsize=(2.4, 2.4), facecolor=PREVIEW_BACKGROUND)
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(0)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

    def on_scroll(self, event):
        """Zooms a 3D preview about its centre on a mouse wheel step.

        Matplotlib's 3D axes have no scroll zoom of their own, so the axis
        limits are scaled directly. All three are scaled together, which
        keeps the proportions of whatever is on screen.

        Args:
            event: Matplotlib scroll event.
        """
        axes = event.inaxes
        if axes is None and self.figure.axes:
            axes = self.figure.axes[0]
        if axes is None or not hasattr(axes, "get_zlim3d"):
            return

        scale = (1.0 / PREVIEW_ZOOM_STEP if event.button == "up"
                 else PREVIEW_ZOOM_STEP)
        for get_limits, set_limits in ((axes.get_xlim3d, axes.set_xlim3d),
                                       (axes.get_ylim3d, axes.set_ylim3d),
                                       (axes.get_zlim3d, axes.set_zlim3d)):
            low, high = get_limits()
            middle = (low + high) / 2.0
            half = (high - low) / 2.0 * scale
            set_limits(middle - half, middle + half)
        self.canvas.draw_idle()

    def resizeEvent(self, event):
        """Matches the height to the width so the preview stays square.

        Args:
            event: The resize event, passed on to the base class.
        """
        super().resizeEvent(event)
        if self.height() != self.width():
            self.setFixedHeight(self.width())

    def message(self, text):
        """Clears the preview and shows a short explanation instead.

        Args:
            text: Why there is nothing to draw.
        """
        self.figure.clear()
        self.figure.patch.set_facecolor(PREVIEW_BACKGROUND)
        axes = self.figure.add_axes([0.0, 0.0, 1.0, 1.0])
        axes.axis("off")
        axes.text(0.5, 0.5, text, ha="center", va="center", wrap=True,
                  fontsize=7, color=PREVIEW_TEXT_COLOUR)
        self.canvas.draw_idle()


class SimulationWorker(QThread):
    """Runs one simulation job off the GUI thread.

    Signals:
        progress_signal: Completion percentage, 0-100.
        log_signal: A line of engine output.
        finished_signal: (figure, results, shot) once the job ends, where
            shot carries the render inputs so the camera can redraw the
            wall shot without tracing again; (None, {}, None) if it
            was cancelled.
        error_signal: Formatted traceback if the engine raised.
    """

    progress_signal = pyqtSignal(float)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object, dict, object)
    error_signal = pyqtSignal(str)

    def __init__(self, config, library, reflector_name, emitter_name, gasket_name,
                 finish, emitter_offset):
        """Captures everything the job needs; nothing is read from the GUI later.

        Args:
            config: Active SimulationConfig.
            library: Detached HardwareLibrary copy for this run, already
                carrying any unsaved edits from the form. It is never written,
                so nothing the worker does can reach hardware_library.json.
            reflector_name: Reflector to simulate.
            emitter_name: Emitter to simulate.
            gasket_name: Gasket to simulate.
            finish: "smooth" or "orange_peel".
            emitter_offset: EmitterOffset for this run. It is captured here
                rather than stored anywhere, so it lasts exactly one job.
        """
        super().__init__()
        self.config = config
        self.library = library
        self.reflector_name = reflector_name
        self.emitter_name = emitter_name
        self.gasket_name = gasket_name
        self.finish = finish
        self.emitter_offset = emitter_offset
        self._is_cancelled = False

    def cancel(self):
        """Asks the engine to stop at its next chunk boundary."""
        self._is_cancelled = True

    def run(self):
        """Runs the job and reports the outcome through the signals."""
        try:
            figure, results, shot = run_simulation_job(
                self.config, self.library,
                self.reflector_name, self.emitter_name, self.gasket_name, self.finish,
                log_callback=self.log_signal.emit,
                progress_callback=self.progress_signal.emit,
                is_cancelled_callback=lambda: self._is_cancelled,
                emitter_offset=self.emitter_offset)

            if self._is_cancelled:
                self.log_signal.emit("\n[!] Simulation stopped by user.")
                self.finished_signal.emit(None, {}, None)
            else:
                self.finished_signal.emit(figure, results, shot)

        except Exception:
            self.error_signal.emit(traceback.format_exc())


class SettingsDialog(QDialog):
    """Editor for every simulation setting, generated from SETTING_LABELS.

    Booleans become checkboxes and everything else a text box. Edits are written
    back to the config with the type the setting already had, so a value that
    was loaded as an int stays an int.
    """

    def __init__(self, config, parent=None):
        """Builds the scrollable form for the given config.

        Args:
            config: SimulationConfig to edit in place.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Simulation Settings")
        self.resize(550, 750)
        self.config = config
        self.input_widgets = {}

        self.scroll_layout = QVBoxLayout()
        scroll_contents = QWidget()
        scroll_contents.setLayout(self.scroll_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_contents)

        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self.reset_to_defaults)
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)

        button_row = QHBoxLayout()
        button_row.addWidget(reset_button)
        button_row.addWidget(save_button)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll_area)
        layout.addLayout(button_row)

        self.populate_form()

    def populate_form(self):
        """Rebuilds every input from the current config values."""
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.input_widgets.clear()

        for category, labels in SETTING_LABELS.items():
            group_box = QGroupBox(category)
            bold_font = group_box.font()
            bold_font.setBold(True)
            group_box.setFont(bold_font)
            form = QFormLayout(group_box)

            for key, label in labels.items():
                value = getattr(self.config, key, None)
                if value is None:
                    continue  # Setting is absent from this config file.

                if isinstance(value, bool):
                    widget = QCheckBox()
                    widget.setChecked(value)
                else:
                    widget = QLineEdit(str(value))

                # Undo the group box's bold font, which children inherit.
                normal_font = widget.font()
                normal_font.setBold(False)
                widget.setFont(normal_font)

                form.addRow(label, widget)
                self.input_widgets[key] = widget

            self.scroll_layout.addWidget(group_box)

        self.scroll_layout.addStretch()

    def save_settings(self):
        """Writes every input back to the config, then to disk, and closes."""
        for key, widget in self.input_widgets.items():
            previous_value = getattr(self.config, key)

            if isinstance(widget, QCheckBox):
                setattr(self.config, key, widget.isChecked())
                continue

            text = widget.text().strip()
            try:
                # bools are excluded because in Python they are also ints.
                if isinstance(previous_value, int) and not isinstance(previous_value, bool):
                    setattr(self.config, key, int(text))
                elif isinstance(previous_value, float):
                    setattr(self.config, key, float(text))
                else:
                    setattr(self.config, key, text)
            except ValueError:
                print(f"Warning: Could not parse '{text}' for setting '{key}'. "
                      f"Keeping previous value.")

        self.config.save_settings()
        self.accept()

    def reset_to_defaults(self):
        """Reloads every tunable from the shipped template, after confirmation."""
        reply = QMessageBox.question(
            self, "Confirm Reset",
            "Are you sure you want to revert all settings to the default template?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            self.config.reset_to_defaults()
        except FileNotFoundError as error:
            QMessageBox.warning(self, "Missing Template", str(error))
            return

        self.populate_form()


class MainWindow(QMainWindow):
    """Main window: hardware catalogues, simulation controls and results."""

    def __init__(self):
        """Loads the UI, the config and the hardware library, then wires it up."""
        super().__init__()

        ui_path = resource_path("mainwindow.ui")
        if not os.path.exists(ui_path):
            QMessageBox.critical(self, "Error", f"Could not find UI file: {ui_path}")
            sys.exit(1)
        uic.loadUi(ui_path, self)

        try:
            self.config = SimulationConfig()
            self.library = HardwareLibrary()
            # A newer release may have added whole hardware entries, or added
            # specs to entries the operator already has. Both are compared
            # against the shipped copies and applied silently.
            self.imported_entries = self.library.import_new_entries()
            # Renames run before the restore, so a spec carried across to a new
            # name is not then mistaken for a missing one and overwritten.
            self.renamed_specs = self.library.rename_legacy_specs()
            self.restored_specs = self.library.restore_missing_specs(self.config)
        except Exception as error:
            QMessageBox.critical(self, "Initialization Error", str(error))
            sys.exit(1)

        self.figure_canvas = None
        self.worker = None

        self.setup_canvas()
        self.setup_previews()
        self.setup_camera_controls()
        self.setup_hardware_widgets()
        self.connect_signals()

        for kind in SPEC_FIELDS:
            self.reload_fields(kind)
        self.update_previews()

        # Both files are upgraded silently on load; say so, because the
        # operator is about to simulate with values they never chose.
        if self.config.restored_settings:
            self.log_message(
                f"Settings file upgraded: {len(self.config.restored_settings)} new "
                f"setting(s) taken from the template "
                f"({', '.join(self.config.restored_settings)}).")

        if self.imported_entries:
            added_count = sum(len(names) for names in self.imported_entries.values())
            self.log_message(
                f"Hardware library upgraded: {added_count} new entrie(s) added "
                f"from the shipped library.")
            for kind, names in sorted(self.imported_entries.items()):
                self.log_message(f"  {kind}: {', '.join(names)}")

        if self.renamed_specs:
            self.log_message(
                f"Hardware library upgraded: renamed spec(s) on "
                f"{len(self.renamed_specs)} entrie(s). Wall thickness now means "
                f"one wall, so stored values were halved.")

        if self.restored_specs:
            restored_count = sum(len(specs) for specs in self.restored_specs.values())
            self.log_message(
                f"Hardware library upgraded: {restored_count} missing spec(s) "
                f"filled in from the settings across "
                f"{len(self.restored_specs)} entrie(s).")
            for entry, specs in sorted(self.restored_specs.items()):
                self.log_message(f"  {entry}: {', '.join(specs)}")

    # --- SETUP ---

    def setup_canvas(self):
        """Prepares the plot area to receive a Matplotlib canvas."""
        self.lblPlotPlaceholder.hide()
        if self.grpPlot.layout() is None:
            self.grpPlot.setLayout(QVBoxLayout())

    def setup_previews(self):
        """Puts a square canvas inside each preview placeholder from the .ui."""
        self.reflector_preview = SquarePreview()
        self.widgetReflectorPreview.layout().addWidget(self.reflector_preview)

        self.emitter_preview = SquarePreview()
        self.widgetEmitterPreview.layout().addWidget(self.emitter_preview)

        self.gasket_preview = SquarePreview()
        self.widgetGasketPreview.layout().addWidget(self.gasket_preview)

    def current_specs(self, kind):
        """Returns the stored specs for a kind, overlaid with the form's edits.

        Args:
            kind: One of the keys of SPEC_FIELDS.

        Returns:
            A merged specs dict, or None when nothing is selected.
        """
        name = self.combo_boxes[kind].currentText()
        if not name:
            return None
        return dict(self.library.get(kind, name), **self.read_fields(kind))

    def update_previews(self):
        """Redraws both previews from whatever is currently on screen.

        Both are refreshed together because the reflector's geometry depends on
        the emitter and the gasket as well as on the reflector itself.
        """
        self.draw_reflector_preview()
        self.draw_emitter_preview()
        self.draw_gasket_preview()
        self.update_warnings()

    def draw_reflector_preview(self):
        """Draws the reflector as a full solid of revolution, emitter included.

        The profile comes from the engine's own geometry, so the preview shows
        the surface that would actually be traced rather than a second, and
        possibly divergent, idea of the same shape. The reflective bowl is
        drawn semi transparent so the emitter sitting down in the bore stays
        visible from any angle.
        """
        preview = self.reflector_preview
        reflector = self.current_specs("reflector")
        if reflector is None:
            preview.message("")
            return

        # An emitter or gasket part way through being entered should not take
        # the reflector down with it, so the geometry falls back to a blank
        # stand in and only the parts that can be drawn are drawn.
        emitter = self.current_specs("emitter")
        gasket = self.current_specs("gasket")
        specs = [reflector, emitter or BLANK_EMITTER, gasket or BLANK_GASKET]
        finish = spec_or_default(reflector, "reflector", "surface_finish", self.config)
        emitter_offset = self.read_emitter_offset()
        try:
            geom = get_sim_geometry(reflector, specs[1], specs[2], finish,
                                    self.config, emitter_offset)
            outer_radius = float(reflector["diameter_mm"]) / 2.0
        except (KeyError, ValueError, ZeroDivisionError) as error:
            preview.message(f"Cannot draw:\n{error}")
            return

        focal_length = geom["focal_length"]
        r_hole, radius_max = geom["r_hole"], geom["radius_max"]
        z_bottom, z_max_cut = geom["z_bottom"], geom["z_max_cut"]
        z_min_cut, z_hole_top = geom["z_min_cut"], geom["z_hole_top"]
        if focal_length <= 0.0 or radius_max <= r_hole:
            preview.message("Reflector dimensions\ndo not form a bowl")
            return

        preview.figure.clear()
        preview.figure.patch.set_facecolor(PREVIEW_BACKGROUND)
        # computed_zorder=False turns off mplot3d's automatic depth sorting,
        # which otherwise buries the emitter behind the translucent body no
        # matter how it is drawn. Artists then render in the order added, so
        # the emitter goes on last and stays visible from any angle.
        axes = preview.figure.add_subplot(111, projection="3d",
                                          computed_zorder=False)

        # Wall thickness is a radial figure, the same way the spec defines it,
        # so the outer surface is the same parabola pushed out by that much.
        wall = max(outer_radius - radius_max, 0.0)

        # The bowl starts where the bore or the shelf leaves off, not at the
        # bore radius, so the parabola is cut off rather than curling back
        # under the shelf.
        bowl_start = math.sqrt(max(0.0, 4.0 * focal_length * z_hole_top))
        bowl_r = np.linspace(bowl_start, radius_max, 120)

        outer_z = np.linspace(z_bottom, z_max_cut, 48)
        outer_r = np.sqrt(4.0 * focal_length * np.maximum(outer_z, 0.0)) + wall

        # The cross section closes into a solid: up the bore, out across the
        # shelf, up the bowl, over the rim, down the outer wall and back along
        # the underside. Order matters because depth sorting is off, so these
        # run roughly back to front.
        body = [
            (outer_r, outer_z),                                    # outer wall
            ([r_hole, outer_r[0]], [z_bottom, z_bottom]),          # underside
            ([r_hole, bowl_start], [z_hole_top, z_hole_top]),      # shelf
            ([r_hole, r_hole], [z_bottom, z_hole_top]),            # bore wall
            ([radius_max, outer_radius], [z_max_cut, z_max_cut]),  # rim
        ]
        for radii, heights in body[:2]:
            axes.plot_surface(*_revolved_surface(radii, heights),
                              color=REFLECTOR_COLOUR, alpha=REFLECTOR_ALPHA,
                              edgecolor=REFLECTOR_EDGE_COLOUR,
                              linewidth=0.2, antialiased=True)

        # The bowl itself, stippled when the reflector is orange peel so the
        # finish is visible rather than only being a number in the form. The
        # dimples are scaled by the reflector's own OP factor, and shading is
        # what makes them read, so the surface is left opaque enough to shade.
        bowl_z = bowl_r ** 2 / (4.0 * focal_length)
        if finish == "orange_peel":
            strength = float(spec_or_default(reflector, "reflector", "OP_Factor",
                                             self.config))
            bowl = _dimpled_revolution(bowl_r, bowl_z, 240,
                                       0.035 * max(strength, 0.1), 40, 8)
        else:
            bowl = _revolved_surface(bowl_r, bowl_z)
        # No stroke on the bowl itself: at the resolution the dimples
        # need, a mesh over the top of them hides what it is there to
        # show. Shading carries the shape, and the flats and walls
        # around it keep their edging.
        axes.plot_surface(*bowl, color=REFLECTOR_COLOUR,
                          alpha=REFLECTOR_ALPHA, linewidth=0,
                          antialiased=True)

        for radii, heights in body[2:]:
            axes.plot_surface(*_revolved_surface(radii, heights),
                              color=REFLECTOR_COLOUR, alpha=REFLECTOR_ALPHA,
                              edgecolor=REFLECTOR_EDGE_COLOUR,
                              linewidth=0.2, antialiased=True)

        # The gasket sits on the same board as the emitter, its seat filling
        # the gap up to the reflector and its wall rising into the bore. It
        # goes on before the emitter, so the emitter stays on top of it.
        lowest = []
        if gasket is not None:
            thickness = float(spec_or_default(gasket, "gasket", "thickness_mm",
                                              self.config))
            drawn = self.add_gasket(axes, gasket, z_bottom - thickness)
            if drawn is not None:
                lowest.append(drawn["base_z"])

        if emitter is not None:
            drawn = self.add_emitter(axes, emitter, geom["ez_base"],
                                     geom["emitter_offset_x"],
                                     geom["emitter_offset_y"])
            if drawn is not None:
                lowest.append(drawn["base_z"])

        emitter_low = min(lowest) if lowest else None

        # True proportions, so a deep reflector looks deep. The emitter can sit
        # below the reflector's floor once the gasket is compressed, so the
        # lower limit follows it rather than clipping it away.
        span = 2.0 * outer_radius
        z_low = z_bottom if emitter_low is None else min(z_bottom, emitter_low)
        height = max(z_max_cut - z_low, 1e-6)
        axes.set_facecolor(PREVIEW_BACKGROUND)
        axes.set_box_aspect((1.0, 1.0, min(max(height / span, 0.25), 2.5)))
        axes.set_xlim(-outer_radius, outer_radius)
        axes.set_ylim(-outer_radius, outer_radius)
        axes.set_zlim(z_low, z_max_cut)
        axes.set_axis_off()
        axes.view_init(elev=28.0, azim=-58.0)
        preview.figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        preview.canvas.draw_idle()

    def add_emitter(self, axes, emitter, die_z, offset_x=0.0, offset_y=0.0):
        """Draws the emitter package, die and dome as an opaque solid.

        Both previews call this, so the emitter looks the same on its own as it
        does sitting in the reflector. The package is opaque: dark grey sides
        and underside, a white top face with the emitting surface on it, and a
        translucent dome over the lot so the die stays readable through it.

        Every face goes into one collection, which is what makes the solid look
        solid: Matplotlib sorts polygons within a collection by depth, so the
        sides and underside hide the top face once the view swings below the
        emitter. Faces spread across several collections cannot be sorted
        against each other and the top would show through from underneath.

        Args:
            axes: 3D axes to draw on.
            emitter: Emitter specs, already merged with the form's edits.
            die_z: Height of the light emitting surface. The package hangs
                below it by the emitter's own height.
            offset_x, offset_y: Centring error of the emitter in
                millimetres, which moves the whole package off the axis.

        Returns:
            A dict of the extents drawn, for the caller to frame the view with,
            or None when the emitter specs are too incomplete to draw.
        """
        try:
            shape = spec_or_default(emitter, "emitter", "shape", self.config)
            outline = emitter_die_outline(emitter, shape)
            footprint_x = float(emitter["footprint_x_mm"])
            footprint_y = float(emitter["footprint_y_mm"])
            package_height = float(emitter["height_mm"])
            die_length = float(emitter["die_length_mm"])
            die_width = die_length if shape == "round" else float(emitter["die_width_mm"])
            dome_radius = self.dome_radius(emitter)
        except (KeyError, ValueError, TypeError):
            return None  # The reflector is still worth showing without it.

        base_z = die_z - package_height
        die_corners = self.die_corners(shape, outline, die_length, die_width)
        package = [(offset_x - footprint_x / 2.0, offset_y - footprint_y / 2.0),
                   (offset_x + footprint_x / 2.0, offset_y - footprint_y / 2.0),
                   (offset_x + footprint_x / 2.0, offset_y + footprint_y / 2.0),
                   (offset_x - footprint_x / 2.0, offset_y + footprint_y / 2.0)]
        die_corners = [(x + offset_x, y + offset_y) for x, y in die_corners]

        faces, colours, edges = [], [], []

        # Sides, one quad per edge of the footprint.
        for index, (x0, y0) in enumerate(package):
            x1, y1 = package[(index + 1) % len(package)]
            faces.append([(x0, y0, base_z), (x1, y1, base_z),
                          (x1, y1, die_z), (x0, y0, die_z)])
            colours.append(EMITTER_BODY_COLOUR)
            edges.append(EMITTER_BODY_COLOUR)

        # Reversed, so the underside's normal points down and out of the
        # solid the way every other face's does.
        faces.append([(x, y, base_z) for x, y in reversed(package)])
        colours.append(EMITTER_BASE_COLOUR)
        edges.append(EMITTER_BASE_COLOUR)

        # The white top would dissolve into the background along its far edge,
        # so it is the one face given a contrasting outline.
        faces.append([(x, y, die_z) for x, y in package])
        colours.append(EMITTER_TOP_COLOUR)
        edges.append(EMITTER_TOP_EDGE_COLOUR)

        # The die is coplanar with the top face, which leaves the depth sort to
        # pick between them arbitrarily. Lifting it by a thousandth of the
        # package height settles that without being a thickness anyone notices.
        die_draw_z = die_z + max(package_height, 0.1) * 1e-3
        faces.append([(x, y, die_draw_z) for x, y in die_corners])
        colours.append(EMITTER_DIE_COLOUR)
        edges.append(EMITTER_DIE_EDGE_COLOUR)

        axes.add_collection3d(_SolidFaces(
            faces, colours, edges, linewidths=0.8, zsort="max"))

        if dome_radius > 0.0:
            dome_x, dome_y, dome_z = _dome_surface(dome_radius, die_z)
            axes.plot_surface(dome_x + offset_x, dome_y + offset_y, dome_z,
                              color=EMITTER_DOME_COLOUR, alpha=0.42,
                              linewidth=0, antialiased=True)

        return {
            "base_z": base_z,
            "top_z": die_z + dome_radius,
            "reach": max(footprint_x, footprint_y, 2.0 * dome_radius) / 2.0,
            "die_area": abs(_outline_area(np.asarray(die_corners, dtype=float))),
            "footprint": (footprint_x, footprint_y),
        }

    def draw_emitter_preview(self):
        """Draws the emitter on its own in 3D, free to rotate and zoom.

        The die is built from the same outline the tracer samples, so a chamfer
        or a notch that is wrong in the catalogue is visible here rather than
        having to be inferred from a beam shot.
        """
        preview = self.emitter_preview
        specs = self.current_specs("emitter")
        if specs is None:
            preview.message("")
            return

        preview.figure.clear()
        preview.figure.patch.set_facecolor(PREVIEW_BACKGROUND)
        # Depth sorting is left on here, unlike the reflector preview: the
        # package is an opaque solid, so its own sides and underside should
        # hide the top face when the view is rotated below it.
        axes = preview.figure.add_subplot(111, projection="3d")
        try:
            package_height = float(specs["height_mm"])
        except (KeyError, ValueError, TypeError) as error:
            preview.message(f"Cannot draw:\n{error}")
            return

        # Drawn with the die at the package height, so the board sits at zero.
        extents = self.add_emitter(axes, specs, package_height)
        if extents is None:
            preview.message("Emitter specs are incomplete")
            return

        reach = max(extents["reach"] * 1.15, 1e-3)
        top = max(extents["top_z"], package_height * 1.2, reach * 0.6)
        footprint_x, footprint_y = extents["footprint"]
        axes.set_xlim(-reach, reach)
        axes.set_ylim(-reach, reach)
        axes.set_zlim(0.0, top)
        axes.set_facecolor(PREVIEW_BACKGROUND)
        axes.set_box_aspect((1.0, 1.0, min(max(top / (2.0 * reach), 0.3), 1.6)))
        axes.set_axis_off()
        axes.view_init(elev=24.0, azim=-58.0)
        preview.figure.text(0.5, 0.015,
                            f"{footprint_x:g} x {footprint_y:g} mm package, "
                            f"{extents['die_area']:.2f} mm\u00b2 die",
                            color=PREVIEW_TEXT_COLOUR, fontsize=7,
                            ha="center", va="bottom")
        preview.figure.subplots_adjust(left=0.0, right=1.0, bottom=0.06, top=1.0)
        preview.canvas.draw_idle()

    def dome_radius(self, emitter):
        """Resolves an emitter's dome radius the same way the engine does.

        Args:
            emitter: Emitter specs.

        Returns:
            The radius in millimetres; 0 for a flat or domeless emitter.
        """
        dome = float(spec_or_default(emitter, "emitter", "dome_size_mm", self.config))
        if dome == -1:
            # -1 means "as wide as the narrowest footprint edge".
            dome = min(float(emitter["footprint_x_mm"]),
                       float(emitter["footprint_y_mm"]))
        return max(0.0, dome) / 2.0

    @staticmethod
    def die_corners(shape, outline, die_length, die_width):
        """Returns the die outline as (x, y) pairs, whatever the shape.

        Args:
            shape: Die shape, already resolved against the settings default.
            outline: Vertices from emitter_die_outline, or None.
            die_length: Die size along x.
            die_width: Die size along y.

        Returns:
            A list of (x, y) pairs in order around the die.
        """
        if shape == "polygon":
            return [(float(x), float(y)) for x, y in outline]
        if shape == "round":
            angle = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
            return list(zip(die_length / 2.0 * np.cos(angle),
                            die_length / 2.0 * np.sin(angle)))
        return [(-die_length / 2.0, -die_width / 2.0),
                (die_length / 2.0, -die_width / 2.0),
                (die_length / 2.0, die_width / 2.0),
                (-die_length / 2.0, die_width / 2.0)]

    def add_gasket(self, axes, gasket, base_z):
        """Draws the gasket as the two stacked shapes it is made of.

        The lower shape is the seat: a disc a little wider than the gasket, with
        a square window cut through it that clears the emitter package. The
        upper shape is the wall the reflector presses onto, at the gasket's own
        diameter, with the aperture the light actually leaves through. A round
        wall makes that a hollow cylinder; a square one repeats the window from
        the seat below.

        Both previews call this, so the gasket looks the same on its own as it
        does sitting under the reflector.

        Args:
            axes: 3D axes to draw on.
            gasket: Gasket specs, already merged with the form's edits.
            base_z: Height the underside of the seat sits at.

        Returns:
            A dict of the extents drawn, or None when the specs are too
            incomplete to draw.
        """
        try:
            outer = float(spec_or_default(gasket, "gasket", "outer_diameter_mm",
                                          self.config))
            inner = float(spec_or_default(gasket, "gasket", "inner_diameter_mm",
                                          self.config))
            emitter_size = float(spec_or_default(gasket, "gasket", "emitter_size_mm",
                                                 self.config))
            wall_shape = spec_or_default(gasket, "gasket", "wall_shape", self.config)
            thickness = float(spec_or_default(gasket, "gasket", "thickness_mm",
                                              self.config))
            total_height = float(spec_or_default(gasket, "gasket", "total_height_mm",
                                                 self.config))
        except (KeyError, ValueError, TypeError):
            return None

        seat_radius = outer / 2.0 + GASKET_SEAT_MARGIN_MM
        wall_radius = outer / 2.0
        # The window always clears the package by the same margin a side, so
        # the seat never covers the emitter whatever the wall above it does.
        window_half = emitter_size / 2.0 + GASKET_EMITTER_CLEARANCE_MM

        seat_top = base_z + thickness
        wall_top = base_z + max(total_height, thickness)

        steps = 64
        angle = np.linspace(0.0, 2.0 * np.pi, steps)
        window = _aperture_radius(angle, half_size=window_half)
        aperture = (_aperture_radius(angle, radius=inner / 2.0)
                    if wall_shape == "round"
                    else _aperture_radius(angle, half_size=window_half))

        # Every face of both discs goes into one collection, which is what
        # makes them read as solid: faces spread over several collections
        # cannot be sorted against each other and the discs look hollow.
        # Each disc is closed top and bottom. The two faces of a disc never
        # appear together, because whichever one points away from the viewer
        # is culled, so the underside of the wall can share a plane with the
        # top of the seat without the two fighting over it.
        # Where the two discs meet, only the rings that are actually exposed
        # are built. Culling hides a face pointing away from the viewer, but
        # it cannot know that a face is buried inside the part, and a buried
        # face has to be sorted behind the one covering it every time or its
        # mesh shows through. Not building it is exact; sorting is not.
        seat_inner_top = np.maximum(window, aperture)
        wall_underside = np.maximum(aperture, window)

        faces = []
        for surface, direction in (
                # The seat: underside, rim, and the window through it.
                (_ring_surface(window, seat_radius, base_z, steps), "down"),
                (_wall_surface(seat_radius, base_z, seat_top, steps), "out"),
                (_wall_surface(window, base_z, seat_top, steps), "in"),
                # Its top, in the two bands the wall above does not cover.
                (_ring_surface(window, seat_inner_top, seat_top, steps), "up"),
                (_ring_surface(wall_radius, seat_radius, seat_top, steps), "up"),
                # The wall standing on it. Its underside exists only where
                # the seat has a hole beneath it to be seen through.
                (_ring_surface(aperture, wall_underside, seat_top, steps), "down"),
                (_ring_surface(aperture, wall_radius, wall_top, steps), "up"),
                (_wall_surface(wall_radius, seat_top, wall_top, steps), "out"),
                (_wall_surface(aperture, seat_top, wall_top, steps), "in")):
            faces.extend(_solid_quads(surface, direction))

        # The mesh itself is the edging: stroking each quad costs nothing to
        # build, and the strokes are culled and depth sorted with the faces
        # they belong to, so an edge behind the gasket stays behind it.
        axes.add_collection3d(_SolidFaces(
            faces, [GASKET_COLOUR] * len(faces),
            [GASKET_EDGE_COLOUR] * len(faces), linewidths=0.35, zsort="max"))

        return {"reach": seat_radius, "top_z": wall_top, "base_z": base_z}

    def draw_gasket_preview(self):
        """Draws the gasket on its own in 3D, free to rotate and zoom."""
        preview = self.gasket_preview
        specs = self.current_specs("gasket")
        if specs is None:
            preview.message("")
            return

        preview.figure.clear()
        preview.figure.patch.set_facecolor(PREVIEW_BACKGROUND)
        axes = preview.figure.add_subplot(111, projection="3d")

        extents = self.add_gasket(axes, specs, 0.0)
        if extents is None:
            preview.message("Gasket specs are incomplete")
            return

        reach = max(extents["reach"] * 1.1, 1e-3)
        height = max(extents["top_z"], reach * 0.35)
        axes.set_facecolor(PREVIEW_BACKGROUND)
        axes.set_xlim(-reach, reach)
        axes.set_ylim(-reach, reach)
        axes.set_zlim(0.0, height)
        axes.set_box_aspect((1.0, 1.0, min(max(height / (2.0 * reach), 0.18), 1.5)))
        axes.set_axis_off()
        axes.view_init(elev=26.0, azim=-58.0)
        preview.figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        preview.canvas.draw_idle()

    def new_hardware(self, kind):
        """Clears the column so a new entry can be typed in from scratch.

        The dropdown is emptied, so a Save asks for a name rather than
        overwriting whatever happened to be selected. Specs that have a default
        in the settings are filled in with it; the rest are left blank, because
        they are what actually describes the new part and there is no sensible
        stand in for them.

        Args:
            kind: One of the keys of SPEC_FIELDS.
        """
        self.combo_boxes[kind].setCurrentIndex(-1)

        overrides = NEW_ENTRY_OVERRIDES.get(kind, {})
        for field, widget in self.field_widgets[kind].items():
            setting = SPEC_DEFAULT_SETTINGS[kind].get(field)
            value = "" if setting is None else getattr(self.config, setting, "")
            value = overrides.get(field, value)
            if isinstance(widget, QComboBox):
                widget.setCurrentIndex(widget.findData(str(value)))
            else:
                widget.setText(str(value))

        for widget in self.run_only_widgets.get(kind, {}).values():
            widget.setText("0.0")

        self.apply_conditional_rows(kind)
        self.update_previews()
        self.log_message(f"New {kind}: fill in the blank fields, then press Save.")

    def best_gasket_for(self, reflector, emitter):
        """Picks the gasket that fits the current reflector and emitter.

        Ranked in the order the parts constrain each other:

        1. Outer diameter as close as possible to the reflector's bore without
           exceeding it, because a gasket wider than the bore will not seat.
        2. Emitter size as close as possible to the emitter's footprint without
           going under it, because a window smaller than the package covers the
           die.
        3. Thickness as close as possible to the emitter height, which is what
           sets how near the die sits to the focus.

        Each rule is a tie break for the one above, so a worse fit on an
        earlier rule is never traded for a better fit on a later one.

        Args:
            reflector: Reflector specs, already merged with the form's edits.
            emitter: Emitter specs, already merged with the form's edits.

        Returns:
            The name of the best gasket, or None if nothing fits or the specs
            needed for the comparison are missing.
        """
        try:
            bore = float(spec_or_default(reflector, "reflector",
                                         "opening_diameter_mm", self.config))
            footprint = max(float(emitter["footprint_x_mm"]),
                            float(emitter["footprint_y_mm"]))
            height = float(emitter["height_mm"])
        except (KeyError, ValueError, TypeError):
            return None

        if bore <= 0.0:
            return None

        ranked = []
        for name in self.library.names("gasket"):
            specs = self.library.get("gasket", name)
            try:
                outer = float(spec_or_default(specs, "gasket", "outer_diameter_mm",
                                              self.config))
                size = float(spec_or_default(specs, "gasket", "emitter_size_mm",
                                             self.config))
                thickness = float(spec_or_default(specs, "gasket", "thickness_mm",
                                                  self.config))
            except (KeyError, ValueError, TypeError):
                continue

            if outer > bore + GASKET_FIT_TOLERANCE_MM:
                continue  # Too wide for the bore, so it cannot seat at all.
            # Sorting on (too small, shortfall) puts every gasket that covers
            # the emitter ahead of every one that does not, and only then
            # prefers the snuggest of them.
            undersized = size < footprint - GASKET_FIT_TOLERANCE_MM
            ranked.append(((bore - outer, undersized, abs(size - footprint),
                            abs(thickness - height)), name))

        return min(ranked)[1] if ranked else None

    def autoselect_gasket(self):
        """Switches to the gasket that best fits the reflector and emitter.

        Called whenever the reflector or emitter changes, by selection or by
        edit, since either can change which gasket fits.
        """
        if getattr(self, "_choosing_gasket", False):
            return  # Already inside a selection; do not recurse.

        reflector = self.current_specs("reflector")
        emitter = self.current_specs("emitter")
        if reflector is None or emitter is None:
            return

        best = self.best_gasket_for(reflector, emitter)
        if not best or best == self.combo_boxes["gasket"].currentText():
            return

        self._choosing_gasket = True
        try:
            self.combo_boxes["gasket"].setCurrentText(best)
            self.reload_fields("gasket")
        finally:
            self._choosing_gasket = False
        self.log_message(f"Gasket auto-selected: {best}")

    def update_warnings(self):
        """Refreshes the fit warnings shown at the top of each column.

        Every warning is worked out from the same specs the run would use, so
        what is flagged here is what the tracer would actually do.
        """
        reflector = self.current_specs("reflector")
        emitter = self.current_specs("emitter")
        gasket = self.current_specs("gasket")

        bore = None
        if reflector is not None and emitter is not None:
            try:
                bore = effective_bore_diameter(reflector, emitter, self.config)
            except (KeyError, ValueError, TypeError):
                bore = None

        self._set_warning(self.lblReflectorWarning,
                          self._reflector_warnings(reflector, emitter, bore))
        self._set_warning(self.lblEmitterWarning,
                          self._emitter_warnings(reflector, emitter))
        self._set_warning(self.lblGasketWarning,
                          self._gasket_warnings(gasket, emitter, bore))

    @staticmethod
    def _set_warning(label, messages):
        """Shows the given warnings, or hides the label when there are none.

        Args:
            label: The QLabel to write into.
            messages: Warning lines, possibly empty.
        """
        label.setText("\n".join(messages))
        label.setVisible(bool(messages))

    def _reflector_warnings(self, reflector, emitter, bore):
        """Warns when the bore is being assumed rather than read from the entry.

        Args:
            reflector: Reflector specs, or None.
            emitter: Emitter specs, or None.
            bore: The bore the tracer will use, or None if it cannot be worked out.

        Returns:
            A list of warning lines.
        """
        if reflector is None or emitter is None or bore is None:
            return []

        try:
            diameter = float(reflector["diameter_mm"])
            height = float(reflector["height_mm"])
            opening = float(spec_or_default(reflector, "reflector",
                                            "opening_diameter_mm", self.config))
        except (KeyError, ValueError, TypeError):
            return []

        if diameter > 0.0 and height > 0.0 and opening == 0.0:
            return [f"No opening size: assuming {bore:.2f} mm "
                    f"(footprint diagonal)."]
        return []

    def _emitter_warnings(self, reflector, emitter):
        """Warns when the emitter package will not pass through the bore.

        Args:
            reflector: Reflector specs, or None.
            emitter: Emitter specs, or None.

        Returns:
            A list of warning lines.
        """
        if reflector is None or emitter is None:
            return []

        try:
            opening = float(spec_or_default(reflector, "reflector",
                                            "opening_diameter_mm", self.config))
            diagonal = emitter_footprint_diagonal(emitter)
        except (KeyError, ValueError, TypeError):
            return []

        # An opening of zero is assumed from this very diagonal, so there is
        # nothing to compare against; the reflector column says so instead.
        if opening > 0.0 and diagonal > opening + FIT_TOLERANCE_MM:
            return [f"Footprint diagonal {diagonal:.2f} mm > "
                    f"opening {opening:.2f} mm."]
        return []

    def _gasket_warnings(self, gasket, emitter, bore):
        """Warns when the gasket does not match the bore or the emitter.

        Args:
            gasket: Gasket specs, or None.
            emitter: Emitter specs, or None.
            bore: The bore the tracer will use, or None.

        Returns:
            A list of warning lines.
        """
        if gasket is None:
            return []

        messages = []
        try:
            outer = float(spec_or_default(gasket, "gasket", "outer_diameter_mm",
                                          self.config))
        except (KeyError, ValueError, TypeError):
            outer = None

        if outer is not None and bore is not None and outer > 0.0:
            if outer > bore + FIT_TOLERANCE_MM:
                messages.append(f"Outer {outer:.2f} mm > opening {bore:.2f} mm.")
            elif bore - outer > GASKET_BORE_SLACK_MM:
                messages.append(f"Outer {outer:.2f} mm is {bore - outer:.2f} mm "
                                f"under opening {bore:.2f} mm.")

        if emitter is not None:
            try:
                size = float(spec_or_default(gasket, "gasket", "emitter_size_mm",
                                             self.config))
                footprint_x = float(emitter["footprint_x_mm"])
                footprint_y = float(emitter["footprint_y_mm"])
            except (KeyError, ValueError, TypeError):
                return messages

            # Both axes are checked, but a single line reports whichever of
            # them missed, so a square package does not say the same thing
            # twice.
            missed = [(axis, value)
                      for axis, value in (("X", footprint_x), ("Y", footprint_y))
                      if abs(size - value) > FIT_TOLERANCE_MM]
            if len(missed) == 2:
                messages.append(f"Window {size:.2f} mm != footprint "
                                f"{footprint_x:.2f} x {footprint_y:.2f} mm.")
            elif missed:
                axis, value = missed[0]
                messages.append(f"Window {size:.2f} mm != footprint "
                                f"{axis} {value:.2f} mm.")
        return messages

    def setup_camera_controls(self):
        """Wires the camera bar and puts the saved settings into it.

        The camera only decides how a finished result is displayed, so these
        redraw the stored wall shot rather than starting a new trace.
        """
        self.grpCamera.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.wall_shot = None
        self.btnExportWallShot.setEnabled(False)
        self._syncing_exposure = False

        self.chkAutoExposure.setChecked(bool(self.config.use_auto_exposure))
        self.txtExposureEV.setText(f"{self.config.auto_exposure_compensation_ev:g}")
        self.sldExposureEV.setValue(
            self._exposure_to_slider(self.config.auto_exposure_compensation_ev))
        self.txtCamIso.setText(f"{self.config.cam_iso:g}")
        self.txtCamFStop.setText(f"{self.config.cam_f_stop:g}")
        self.txtCamShutter.setText(
            f"{1.0 / self.config.cam_shutter_speed_s:g}"
            if self.config.cam_shutter_speed_s else "0")

        self.chkAutoExposure.toggled.connect(self.apply_camera_settings)
        self.sldExposureEV.valueChanged.connect(self.exposure_slider_moved)
        self.sldExposureEV.sliderReleased.connect(self.apply_camera_settings)
        self.txtExposureEV.editingFinished.connect(self.exposure_text_edited)
        for widget in (self.txtCamIso, self.txtCamFStop, self.txtCamShutter):
            widget.editingFinished.connect(self.apply_camera_settings)
        self.btnSaveCameraDefaults.clicked.connect(self.save_camera_defaults)
        self.btnExportWallShot.clicked.connect(self.export_wall_shot)

        self.apply_camera_settings()

    @staticmethod
    def _exposure_to_slider(stops):
        """Converts stops of exposure into the slider's tenth-of-a-stop steps.

        Args:
            stops: Exposure compensation in stops.

        Returns:
            The slider position, clamped to its own range.
        """
        return int(round(max(-EXPOSURE_SLIDER_RANGE_EV,
                             min(EXPOSURE_SLIDER_RANGE_EV, stops)) * 10))

    def exposure_slider_moved(self, position):
        """Writes the slider's value into the box, then redraws.

        Args:
            position: Slider position, in tenths of a stop.
        """
        if self._syncing_exposure:
            return
        self._syncing_exposure = True
        self.txtExposureEV.setText(f"{position / 10.0:g}")
        self._syncing_exposure = False
        
        # Only redraw the image if the slider isn't actively being dragged
        if not self.sldExposureEV.isSliderDown():
            self.apply_camera_settings()

    def exposure_text_edited(self, *_):
        """Moves the slider to match the box, then redraws.

        The box is the authority: it may hold a value beyond the slider's range,
        in which case the slider simply sits at its end stop.
        """
        if self._syncing_exposure:
            return
        self._syncing_exposure = True
        self.sldExposureEV.setValue(self._exposure_to_slider(self.camera_values()[1]))
        self._syncing_exposure = False
        self.apply_camera_settings()

    def camera_values(self):
        """Reads the camera bar, falling back to the stored settings.

        Returns:
            (auto, stops, iso, f_stop, shutter_seconds). A box that cannot be
            read keeps its saved value rather than stopping the redraw.
        """
        def number(widget, fallback):
            """Parses one box, or returns the saved value."""
            try:
                return float(widget.text().strip())
            except ValueError:
                return fallback

        denominator = number(self.txtCamShutter,
                             1.0 / self.config.cam_shutter_speed_s
                             if self.config.cam_shutter_speed_s else 0.0)
        return (self.chkAutoExposure.isChecked(),
                number(self.txtExposureEV, self.config.auto_exposure_compensation_ev),
                number(self.txtCamIso, self.config.cam_iso),
                number(self.txtCamFStop, self.config.cam_f_stop),
                1.0 / denominator if denominator else self.config.cam_shutter_speed_s)

    def apply_camera_settings(self, *_):
        """Shows the relevant boxes and redraws the stored wall shot.

        Written to take and ignore an argument, because the signals it is
        connected to pass one: toggled sends a bool, clicked likewise.

        Auto exposure needs one number, manual needs three, so only the set in
        use is shown. Nothing here touches the trace: if there is no result yet
        the settings are simply remembered for the next one.
        """
        auto, stops, iso, f_stop, shutter = self.camera_values()

        self.txtExposureEV.setVisible(auto)
        self.sldExposureEV.setVisible(auto)
        for widget in (self.lblCamIso, self.txtCamIso, self.lblCamFStop,
                       self.txtCamFStop, self.lblCamShutter, self.txtCamShutter):
            widget.setVisible(not auto)

        self.config.use_auto_exposure = auto
        self.config.auto_exposure_compensation_ev = stops
        self.config.cam_iso = iso
        self.config.cam_f_stop = f_stop
        self.config.cam_shutter_speed_s = shutter

        if self.wall_shot is not None:
            self.show_figure(render_wall_shot(self.wall_shot, self.config))

    def save_camera_defaults(self, *_):
        """Writes the camera bar's values into the settings file."""
        self.apply_camera_settings()
        self.config.save_settings()
        if self.config.use_auto_exposure:
            detail = f"auto, {self.config.auto_exposure_compensation_ev:+g} EV"
        else:
            detail = (f"ISO {self.config.cam_iso:g}, f/{self.config.cam_f_stop:g}, "
                      f"1/{1.0 / self.config.cam_shutter_speed_s:.0f}s")
        self.log_message(f"Camera defaults saved ({detail}).")

    def export_wall_shot(self, *_):
        """Writes the wall shot to the output directory at the current exposure."""
        if self.wall_shot is None:
            self.log_message("No wall shot to export yet; run a simulation first.")
            return

        directory = self.config.resolved_output_directory
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, self.wall_shot.filename)
        render_wall_shot(self.wall_shot, self.config, path,
                         always_save=True)
        self.log_message(f"Wall shot exported: {path}")

    def setup_hardware_widgets(self):
        """Indexes the combo boxes and spec inputs by hardware kind."""
        self.combo_boxes = {
            "reflector": self.cmbReflector,
            "emitter": self.cmbEmitter,
            "gasket": self.cmbGasket,
        }
        self.field_widgets = {
            kind: {field: getattr(self, FIELD_WIDGET_PREFIX[kind] + field)
                   for field in fields}
            for kind, fields in SPEC_FIELDS.items()
        }
        # Kept apart from field_widgets so that read_fields, and therefore
        # saving, never sees them.
        self.run_only_widgets = {
            "reflector": {
                field: getattr(self, FIELD_WIDGET_PREFIX["reflector"] + field)
                for field, _ in RUN_ONLY_REFLECTOR_FIELDS
            },
        }
        for kind, combo in self.combo_boxes.items():
            combo.addItems(self.library.names(kind))

        self.populate_choice_fields()

    def populate_choice_fields(self):
        """Fills the spec drop downs from the values the engine accepts.

        The caption shown and the value stored are kept apart: the caption is
        the tidied up version, while the item carries the plain value as its
        data, which is what reaches the catalogue.
        """
        for widgets in self.field_widgets.values():
            for field, widget in widgets.items():
                if field not in CHOICE_SPECS:
                    continue
                widget.blockSignals(True)
                widget.clear()
                for value in CHOICE_SPECS[field]:
                    widget.addItem(_choice_label(value), value)
                widget.blockSignals(False)

    def connect_signals(self):
        """Connects the catalogue buttons and the bottom control bar."""
        for kind, save_button, delete_button, reset_button, new_button in (
                ("reflector", self.btnSaveReflector, self.btnDeleteReflector,
                 self.btnResetReflector, self.btnNewReflector),
                ("emitter", self.btnSaveEmitter, self.btnDeleteEmitter,
                 self.btnResetEmitter, self.btnNewEmitter),
                ("gasket", self.btnSaveGasket, self.btnDeleteGasket,
                 self.btnResetGasket, self.btnNewGasket)):
            # k=kind binds the current value; the signals' own arguments are
            # swallowed by *_ because none of these slots need them.
            self.combo_boxes[kind].currentIndexChanged.connect(
                lambda *_, k=kind: self.reload_fields(k))
            if kind != "gasket":
                self.combo_boxes[kind].currentIndexChanged.connect(
                    lambda *_: self.autoselect_gasket())
            reset_button.clicked.connect(lambda *_, k=kind: self.reload_fields(k))
            new_button.clicked.connect(lambda *_, k=kind: self.new_hardware(k))
            save_button.clicked.connect(lambda *_, k=kind: self.save_hardware(k))
            delete_button.clicked.connect(lambda *_, k=kind: self.delete_hardware(k))

        # Previews follow the form, refreshed when a box is committed rather
        # than on every keystroke, which would redraw mid-number.
        for kind, widgets in self.field_widgets.items():
            for widget in widgets.values():
                if isinstance(widget, QComboBox):
                    # A choice can decide whether another row applies, so it
                    # refreshes the form as well as the previews.
                    widget.currentIndexChanged.connect(
                        lambda *_, k=kind: self.apply_conditional_rows(k))
                    widget.currentIndexChanged.connect(
                        lambda *_: self.update_previews())
                else:
                    widget.editingFinished.connect(self.update_previews)

                if kind != "gasket":
                    # Editing a reflector bore or an emitter package can
                    # change which gasket fits, so the choice is revisited.
                    signal = (widget.currentIndexChanged
                              if isinstance(widget, QComboBox)
                              else widget.editingFinished)
                    signal.connect(lambda *_: self.autoselect_gasket())

        # The centring offset is not a spec, but it moves the emitter in the
        # reflector preview, so it refreshes it just the same.
        for widgets in self.run_only_widgets.values():
            for widget in widgets.values():
                widget.editingFinished.connect(self.update_previews)

        self.btnSettings.clicked.connect(self.open_settings)
        self.btnSimulate.clicked.connect(self.run_simulation)
        self.btnStop.clicked.connect(self.stop_simulation)

    # --- HARDWARE CATALOGUE ---

    def reload_fields(self, kind):
        """Fills the spec inputs from the catalogue entry now selected.

        Every optional spec is filled in at start up, so nothing shows blank
        unless the entry is genuinely missing a mandatory spec. The run-only
        inputs go back to zero, since the catalogue holds no value for them to
        be restored from.

        Args:
            kind: One of the keys of SPEC_FIELDS.
        """
        for widget in self.run_only_widgets.get(kind, {}).values():
            widget.setText("0.0")

        name = self.combo_boxes[kind].currentText()
        if not name:
            return

        specs = self.library.get(kind, name)
        for field, widget in self.field_widgets[kind].items():
            value = specs.get(field, "")
            if field in LIST_SPECS and value != "":
                # Compact JSON, so the box holds exactly what a generator emits
                # and can be copied back out again.
                value = json.dumps(value, separators=(",", ":"))
            if isinstance(widget, QComboBox):
                index = widget.findData(str(value))
                widget.setCurrentIndex(max(index, 0))
            else:
                widget.setText(str(value))

        self.apply_conditional_rows(kind)
        self.update_previews()

    def apply_conditional_rows(self, kind):
        """Shows or hides the spec rows that only apply to certain choices.

        The die outline is meaningless for a rectangular or circular die, so
        its row is taken out of the form rather than left there inviting a
        value that would be ignored.

        Args:
            kind: One of the keys of SPEC_FIELDS.
        """
        widgets = self.field_widgets[kind]
        for field, (deciding_field, needed_for) in CONDITIONAL_SPECS.items():
            if field not in widgets or deciding_field not in widgets:
                continue

            current = self.field_value(widgets[deciding_field])
            self.set_row_visible(widgets[field], current in needed_for)

    @staticmethod
    def set_row_visible(widget, visible):
        """Shows or hides one row of the form a widget belongs to.

        Qt 6.4 gained setRowVisible, which also collapses the space the row
        took up. Older builds have to settle for hiding the two widgets, which
        leaves a gap but keeps the row out of the way.

        Args:
            widget: The input whose row should be shown or hidden.
            visible: True to show the row.
        """
        layout = widget.parentWidget().layout() if widget.parentWidget() else None
        if isinstance(layout, QFormLayout):
            if hasattr(layout, "setRowVisible"):
                layout.setRowVisible(widget, visible)
                return
            label = layout.labelForField(widget)
            if label is not None:
                label.setVisible(visible)
        widget.setVisible(visible)

    @staticmethod
    def field_value(widget):
        """Returns what a spec input currently holds, as the catalogue sees it.

        Args:
            widget: A spec input, either a text box or a drop down.

        Returns:
            The stored value for a drop down, or the trimmed text otherwise.
        """
        if isinstance(widget, QComboBox):
            data = widget.currentData()
            return widget.currentText() if data is None else str(data)
        return widget.text().strip()

    def read_fields(self, kind):
        """Reads the spec inputs back into a specs dict.

        Blank inputs are omitted so the stored value survives. Unparseable
        numbers fall back to 0.0 and are reported in the log.

        Args:
            kind: One of the keys of SPEC_FIELDS.

        Returns:
            The specs the operator currently has on screen.
        """
        specs = {}
        for field, widget in self.field_widgets[kind].items():
            text = self.field_value(widget)
            if not text:
                continue

            if isinstance(widget, QComboBox):
                specs[field] = text
                continue

            if field in TEXT_SPECS:
                specs[field] = text
                continue

            if field in LIST_SPECS:
                try:
                    specs[field] = json.loads(text)
                except json.JSONDecodeError as error:
                    self.log_message(f"Warning: Could not parse {field} as JSON "
                                     f"({error}). Keeping the stored value.")
                continue

            try:
                specs[field] = float(text)
            except ValueError:
                self.log_message(f"Warning: Could not parse '{text}' for {field}. "
                                 f"Defaulting to 0.0")
                specs[field] = 0.0
        return specs

    def read_emitter_offset(self):
        """Reads the run-only emitter centring offset from the Reflector column.

        Returns:
            An EmitterOffset in polar form. A blank or unparseable box reads as
            zero, so a typo simulates a centred emitter rather than stopping
            the run, and is reported in the log.
        """
        values = {}
        for field, widget in self.run_only_widgets["reflector"].items():
            text = widget.text().strip()
            try:
                values[field] = float(text) if text else 0.0
            except ValueError:
                self.log_message(f"Warning: Could not parse '{text}' for {field}. "
                                 f"Defaulting to 0.0")
                values[field] = 0.0

        return EmitterOffset(values["emitter_offset_distance_mm"],
                             values["emitter_offset_angle_deg"])

    def refresh_dropdown(self, kind, select=None):
        """Reloads one dropdown from the library without firing its signals.

        Args:
            kind: One of the keys of SPEC_FIELDS.
            select: Entry to select afterwards, or None to leave the selection
                to Qt.
        """
        combo = self.combo_boxes[kind]
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(self.library.names(kind))
        if select is not None:
            combo.setCurrentText(select)
        combo.blockSignals(False)

    def save_hardware(self, kind):
        """Saves the on-screen specs to the catalogue under a chosen name.

        Args:
            kind: One of the keys of SPEC_FIELDS.
        """
        label = kind.capitalize()
        new_name, confirmed = QInputDialog.getText(
            self, f"Save {label}", f"Enter name for the {label}:",
            QLineEdit.EchoMode.Normal, self.combo_boxes[kind].currentText())

        if not confirmed or not new_name.strip():
            return
        new_name = new_name.strip()

        if new_name in self.library.names(kind):
            reply = QMessageBox.question(
                self, "Overwrite Confirm",
                f"A {label} named '{new_name}' already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return

        self.library.save(kind, new_name, self.read_fields(kind))
        self.refresh_dropdown(kind, select=new_name)
        self.log_message(f"Successfully saved {label}: {new_name}")

    def delete_hardware(self, kind):
        """Deletes the selected catalogue entry after confirmation.

        Args:
            kind: One of the keys of SPEC_FIELDS.
        """
        label = kind.capitalize()
        name = self.combo_boxes[kind].currentText()
        if not name:
            return

        reply = QMessageBox.question(
            self, "Delete Confirm",
            f"Are you sure you want to permanently delete the {label} '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.library.delete(kind, name)
        self.refresh_dropdown(kind)
        self.reload_fields(kind)
        self.log_message(f"Deleted {label}: {name}")

    # --- SIMULATION ---

    def open_settings(self):
        """Opens the settings dialog modally."""
        SettingsDialog(self.config, self).exec()

    def log_message(self, message):
        """Appends a line to the log pane and scrolls to it."""
        self.txtLogs.appendPlainText(message)
        scrollbar = self.txtLogs.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_progress(self, percent):
        """Moves the progress bar."""
        self.progressBar.setValue(int(percent))

    def set_controls_running(self, is_running):
        """Enables the stop button and disables the rest while a job runs."""
        self.btnSimulate.setEnabled(not is_running)
        self.btnSettings.setEnabled(not is_running)
        self.btnStop.setEnabled(is_running)

    def run_simulation(self):
        """Validates the selection, applies edits and starts the worker."""
        names = {kind: combo.currentText() for kind, combo in self.combo_boxes.items()}
        if not all(names.values()):
            QMessageBox.warning(self, "Missing Hardware",
                                "Please select a Reflector, Emitter, and Gasket.")
            return

        # The run works on a detached copy of the catalogue. On-screen edits are
        # overlaid onto that copy, so they apply to this simulation and to
        # nothing else: the live catalogue is untouched, and only the Save
        # button ever changes hardware_library.json.
        run_library = self.library.copy_for_run()
        for kind, name in names.items():
            run_library.apply_overrides(kind, name, self.read_fields(kind))

        # The finish is a reflector spec now, so it comes from the same place
        # as every other value the run uses.
        finish = spec_or_default(run_library.get("reflector", names["reflector"]),
                                 "reflector", "surface_finish", self.config)

        # The centring offset never reaches the catalogue at all, not even the
        # run copy; it is passed straight to the job.
        emitter_offset = self.read_emitter_offset()

        # The Run button always renders the current selection. Batch mode is
        # driven from the settings dialog instead.
        self.config.generate_all_plots = False

        self.set_controls_running(True)
        self.progressBar.setValue(0)
        self.txtLogs.clear()
        self.log_message("--- INITIALIZING SIMULATION ---")

        self.worker = SimulationWorker(self.config, run_library, names["reflector"],
                                       names["emitter"], names["gasket"], finish,
                                       emitter_offset)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.log_signal.connect(self.log_message)
        self.worker.error_signal.connect(self.handle_simulation_error)
        self.worker.finished_signal.connect(self.handle_simulation_finished)
        self.worker.start()

    def stop_simulation(self):
        """Asks a running job to stop at its next chunk boundary."""
        if self.worker is not None and self.worker.isRunning():
            self.log_message("Sending interrupt signal to engine...")
            self.worker.cancel()
            self.btnStop.setEnabled(False)

    def handle_simulation_error(self, error_traceback):
        """Restores the controls and reports an engine crash.

        Args:
            error_traceback: Formatted traceback from the worker.
        """
        self.set_controls_running(False)
        self.log_message("CRITICAL ERROR IN ENGINE:")
        self.log_message(error_traceback)
        QMessageBox.critical(self, "Simulation Error",
                             "An error occurred during simulation. Check logs.")

    def handle_simulation_finished(self, figure, results, shot):
        """Restores the controls, logs the results and shows the new plot.

        Args:
            figure: Matplotlib figure to display, or None.
            results: Headline results keyed by label; empty if cancelled.
            shot: Render inputs kept so the camera controls can redraw the
                wall shot, or None when there is nothing to redraw.
        """
        self.set_controls_running(False)
        self.progressBar.setValue(100)
        self.wall_shot = shot
        self.btnExportWallShot.setEnabled(shot is not None)

        if results:
            self.log_message("\n--- SIMULATION RESULTS ---")
            for label, value in results.items():
                self.log_message(f"{label}: {value}")

        if figure:
            self.show_figure(figure)

    def show_figure(self, figure):
        """Puts a figure in the output box, replacing whatever was there.

        Args:
            figure: The Matplotlib figure to display.
        """
        if self.figure_canvas is not None:
            self.grpPlot.layout().removeWidget(self.figure_canvas)
            self.figure_canvas.deleteLater()

        self.figure_canvas = FigureCanvas(figure)
        self.grpPlot.layout().addWidget(self.figure_canvas)
        self.figure_canvas.draw()


def main():
    """Starts the application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
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
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                             QFileDialog, QListWidgetItem,
                            
                             QFormLayout, QGroupBox, QHBoxLayout, QInputDialog,
                             QLineEdit,
                             QMainWindow, QMessageBox, QPushButton, QScrollArea,
                             QSizePolicy, QVBoxLayout, QWidget)
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QListWidgetItem

# Matplotlib's Qt canvas, used to embed the engine's figure in the window.
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import to_rgba_array
from matplotlib.text import Text
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from core import (DIE_LAYOUTS, DIE_SHAPES, GASKET_WALL_SHAPES,
                  die_array_extent, die_array_layout,
                  die_array_resolution, die_cell_size,
                  die_centres,
                  LENS_FINISHES,
                  OUTPUT_MODES,
                    SPEC_DEFAULT_SETTINGS, SURFACE_FINISHES, EmitterOffset,
                    HardwareLibrary, SimulationConfig,
                    effective_bore_diameter, emitter_die_outline,
                    emitter_footprint_diagonal, get_sim_geometry,
                    PLOT_NAMES, render_plot,
                  resource_path, run_simulation_job,
                    spec_or_default)

# Specs shown for each hardware kind, in the order they appear in the form. Each
# one maps to a QLineEdit in mainwindow.ui named <prefix><spec>, for example
# reflector "diameter_mm" -> txtRef_diameter_mm.
# Every spec the hardware forms show, in the order they appear and under the
# heading they sit below. Grouping keeps a long form readable: the reflector
# alone has seventeen specs, and hunting for one in a flat list is tedious.
# This is the single source of truth for the form; mainwindow.ui is generated
# to match, so a field added here needs a row added there under the same
# heading.
SPEC_GROUPS = {
    "reflector": (
        ("Geometry", ("diameter_mm", "height_mm", "opening_diameter_mm",
                      "focus_offset_mm", "wall_thickness_mm",
                      "thickness_height_mm")),
        ("Reflective Surface", ("surface_finish", "surface_roughness_nm",
                                "surface_correlation_um", "op_dimple_pitch_mm",
                                "op_dimple_depth_um", "op_factor", "reflectivity_smooth",
                                "reflectivity_op", "reflectivity_cylinder",
                                "gasket_reflectivity")),
        ("Front Lens", ("transmissivity_lens", "lens_finish",
                        "lens_diffusion_fwhm_deg", "lens_refractive_index")),
    ),
    "emitter": (
        ("Output", ("output_mode", "max_current_amps", "max_lumens",
                    "forward_voltage_v", "vf_turn_on_v", "vf_scale",
                    "base_efficacy_lm_w", "droop_factor")),
        ("Package", ("footprint_x_mm", "footprint_y_mm", "height_mm",
                     "dome_size_mm", "refractive_index")),
        ("Die", ("shape", "die_length_mm", "die_width_mm", "die_outline",
                 "die_layout", "die_rows", "die_columns", "die_gap_mm",
                 "die_gap_output")),
    ),
    "gasket": (
        ("Dimensions", ("outer_diameter_mm", "inner_diameter_mm",
                        "emitter_size_mm", "wall_shape")),
        ("Height", ("thickness_mm", "total_height_mm")),
    ),
}

# The specs of each kind, flattened out of the groups above. Everything that
# reads or writes a catalogue entry works from this, and never needs to know
# how the form is arranged.
SPEC_FIELDS = {
    kind: tuple(field for _, fields in groups for field in fields)
    for kind, groups in SPEC_GROUPS.items()
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

# The reflector is drawn as a single opaque silver body, stroked along its
# own mesh so the shape reads. It is the backdrop everything else sits in,
# so it is the darkest of the three: a bowl, a gasket and an emitter within
# thirteen shades of each other read as one blob however well they are
# sorted, which is what made the parts look like they were bleeding
# together. Cool silver here, warm cream for the gasket, white for the
# emitter, so the eye separates them by hue as well as by brightness.
REFLECTOR_COLOUR = "#A9AEB6"
REFLECTOR_ALPHA = 1.0
REFLECTOR_EDGE_COLOUR = "#7C818A"

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
# How much wider the seat window is than the emitter it clears, measured
# across the opening rather than per side: a 7 mm emitter looks through a
# 7.1 mm window, so the two edges are visibly separate in the picture.
GASKET_WINDOW_OFFSET_MM = 0.1

# The gasket is warm cream and fully opaque: darker than the emitter's top
# face and lighter than its base, as a silicone gasket is, and warm where
# the metal around it is cool. The four parts used to sit within thirteen
# shades of each other, which is why they looked like they were bleeding
# together however well they were sorted; they are now spaced evenly and
# separated by hue as well as brightness.
GASKET_COLOUR = "#EFE3CC"
GASKET_EDGE_COLOUR = "#8A7F68"

# Rings the gasket's flat faces are split into. They are the widest
# surfaces in the assembly, and a depth sorter places a polygon by a
# single number, so one quad spanning the whole seat can be judged
# nearer than the emitter it surrounds and drawn over the top of it.
GASKET_RADIAL_BANDS = 6

# Height the emitter is raised by in the assembly view, to keep it off
# the gasket seat's plane. A common build has an emitter exactly as
# tall as the seat thickness, which lands the two surfaces on the same
# plane; a micron of separation resolves that and cannot be seen.
ASSEMBLY_Z_NUDGE_MM = 0.001

# How much the orange peel dimples are exaggerated in the reflector
# preview, and how deep they are allowed to get. A real texture is a
# few micrometres deep on a bowl tens of millimetres across, which
# would be invisible; these turn the depth-to-pitch ratio into a
# fraction of the bowl radius that can actually be seen, with a cap so
# an extreme texture cannot turn the bowl inside out.
# Parts that occupy the same space are drawn in this colour, so an
# interference is obvious in the picture and not only in the warnings.
# Left clear around the window when it is capped to the display, so it
# does not sit flush against a taskbar or a dock.
WINDOW_SCREEN_MARGIN_PX = 48

# The size the window opens at, before the layout and the screen have
# their say. Qt will not go below what the widgets need, so a request
# narrower than the three columns simply gets the columns' width.
WINDOW_DEFAULT_WIDTH_PX = 900
WINDOW_DEFAULT_HEIGHT_PX = 1000

# Spec inputs are sized in digits rather than pixels. A hard pixel width
# is wrong on any display whose font is not the one it was chosen on:
# too wide on a small screen, too narrow to read on a scaled one.
FIELD_WIDTH_IN_DIGITS = 11
FIELD_MIN_WIDTH_PX = 64

# Room left around a column beyond its widest caption, for the form's
# own margins, the gap to the input and the scrollbar when one appears.
COLUMN_PADDING_PX = 40

# Room a drop down needs beyond its text, for the arrow and the frame.
COMBO_ARROW_ALLOWANCE_PX = 34

# Figure height the plot fonts were chosen against. Text is scaled by
# how the figure compares with this, so a panel half the size gets half
# the type instead of the same type overlapping itself.
FIGURE_REFERENCE_HEIGHT_IN = 10.0
FIGURE_MIN_FONT_SCALE = 0.35

CLASH_COLOUR = "#D2544B"
CLASH_TOLERANCE_MM = 0.01

# How the front lens looks. Clear glass shows only as a rim, since a pane
# that hides nothing should not be drawn hiding things. Frosted glass is
# milky and does obscure the bowl, and a film adds a layer over a clear
# pane. The standoff keeps whichever it is in front of the rim.
LENS_APPEARANCE = {
    "clear_rim": ("#9FC4DA", 0.55),
    "frosted": ("#E4ECF1", 0.38),
    "film": ("#BFD8E8", 0.14),
    "film_layer": ("#D6E4EC", 0.30),
}

# Where the clear lens rim starts, as a fraction of the mouth radius.
LENS_RIM_INNER_FRACTION = 0.93
LENS_STANDOFF_MM = 0.15
LENS_FILM_THICKNESS_MM = 0.12

DIMPLE_PREVIEW_EXAGGERATION = 5.8
DIMPLE_PREVIEW_MAX_AMPLITUDE = 0.08

# Both previews sit on black, like the simulated beam shot does.
PREVIEW_BACKGROUND = "#000000"
PREVIEW_TEXT_COLOUR = "#FFFFFF"

# How far the exposure slider reaches, in stops either side of zero. The
# box beside it is not limited to this: a value typed there simply parks
# the slider at its end stop.
EXPOSURE_SLIDER_RANGE_EV = 10.0

# Where the results tab sits in the bar. It is hidden until a run has
# produced something for it to hold.
RESULTS_TAB_INDEX = 2

# Zoom applied per mouse wheel step in a 3D preview.
PREVIEW_ZOOM_STEP = 1.15

# Specs the form offers as a drop down, mapped to the values the engine
# accepts. The box shows each value capitalised, with underscores as spaces,
# while the catalogue keeps the plain value listed here.
# Specs that are counts rather than measurements. They are stored as
# numbers like everything else, but "2.0 rows" reads like an invitation
# to type 2.5, so they are shown without a decimal point.
INTEGER_SPECS = frozenset({"die_rows", "die_columns"})

CHOICE_SPECS = {
    "shape": DIE_SHAPES,
    "surface_finish": SURFACE_FINISHES,
    "wall_shape": GASKET_WALL_SHAPES,
    "output_mode": OUTPUT_MODES,
    "lens_finish": LENS_FINISHES,
    "die_layout": DIE_LAYOUTS,
}

# Specs only meaningful for certain choices, as spec -> (deciding spec,
# values that need it). The row is hidden when it does not apply, so an
# outline box is not offered for a die that has no outline.
CONDITIONAL_SPECS = {
    "die_outline": ("shape", frozenset({"polygon"})),
    # A reflector is smooth or textured, never both, so each finish shows
    # only the numbers that describe it and the reflectivity it uses.
    "surface_roughness_nm": ("surface_finish", frozenset({"smooth"})),
    "surface_correlation_um": ("surface_finish", frozenset({"smooth"})),
    "reflectivity_smooth": ("surface_finish", frozenset({"smooth"})),
    "op_dimple_pitch_mm": ("surface_finish", frozenset({"orange_peel"})),
    "op_dimple_depth_um": ("surface_finish", frozenset({"orange_peel"})),
    "op_factor": ("surface_finish", frozenset({"orange_peel"})),
    "reflectivity_op": ("surface_finish", frozenset({"orange_peel"})),
    "die_rows": ("die_layout", frozenset({"array"})),
    "die_columns": ("die_layout", frozenset({"array"})),
    "die_gap_mm": ("die_layout", frozenset({"array"})),
    "die_gap_output": ("die_layout", frozenset({"array"})),
    "lens_diffusion_fwhm_deg": ("lens_finish", frozenset({"frosted", "film"})),
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
    ("emitter_offset_distance_mm", "Offset Distance (mm)"),
    ("emitter_offset_angle_deg", "Offset Angle (° CW from up)"),
)

# The heading the run-only inputs sit under. They are not specs, they are
# not saved, and they belong to the run rather than the part, so they always
# come last whatever else the reflector form gains.
RUN_ONLY_GROUP_TITLE = "Emitter Centring"

# Settings offered by the settings dialog, grouped exactly as they are stored,
# mapping each attribute of SimulationConfig to its human readable label.
SETTING_LABELS = {
    "Simulation Space & Constraints": {
        "use_gpu": "Use GPU Acceleration (CUDA)",
        "enable_lens_simulation": "Enable Front Lens Simulation",
        "use_dimple_op_simulation": "Use Dimple Bump-Map for Orange Peel",
        "max_multiple_reflections": "Max Multiple Reflections",
        "use_reflector_opening": "Force Reflector Opening Size",
        "target_distance_m": "Target Distance (meters)",
        "canvas_fov_deg": "Canvas Field of View (degrees)",
        "plot_fov_deg": "Plot Field of View (degrees)",
        "op_blur_strength": "OP Blur Base Strength",
    },
    "Resolution & Angular Density": {
        "sim_grid_res": "Simulation Grid Resolution (pixels)",
        "sim_emitter_elements": "Emitter Subdivision Elements",
        "sim_theta_step_deg": "Theta Step Size (degrees)",
        "sim_phi_step_deg": "Phi Step Size (degrees)",
        "sim_theta_min_deg": "Theta Minimum (degrees)",
        "sim_theta_max_deg": "Theta Maximum (degrees)",
        "sim_phi_min_deg": "Phi Minimum (degrees)",
        "sim_phi_max_deg": "Phi Maximum (degrees)",
    },
    "Output & Rendering": {
        "plot_scale": "Plot Scale (Distance/Angle)",
        "plot_show_primary_grid": "Show Primary Grid",
        "plot_show_secondary_grid": "Show Secondary Grid",
        "plot_simple_output_scaling": "Show % Scaling Table in Simple Mode",
        "generate_all_plots": "Generate All Plots (Batch Mode)",
        "stored_run_count": "Results Kept (runs)",
        "plot_wall_shot": "Plot Wall Shot (2D Image)",
        "plot_intensity_x": "Plot Intensity Profile (X-Axis)",
        "plot_intensity_y": "Plot Intensity Profile (Y-Axis)",
        "plot_intensity_45": "Plot Intensity Profile (45° Diagonal)",
        "show_human_silhouette": "Show Human Silhouette Reference",
        "export_csv": "Export Results to CSV",
        "export_plots": "Export Plot Images",
        "batch_output_directory": "Output Directory Path",
    },
    "Spherical Projection": {
        "use_spherical_projection": "Use Spherical Projection",
        "dome_angle_deg": "Dome Angle (degrees)",
        "dome_polar_step_deg": "Dome Polar Step (degrees)",
        "dome_azimuth_step_deg": "Dome Azimuth Step (degrees)",
        "dome_memory_budget_mb": "Dome Memory Budget (MB)",
    },
    "Camera Settings": {
        "use_auto_exposure": "Use Auto Exposure",
        "auto_exposure_compensation_ev": "Auto Exposure Compensation (EV)",
        "cam_iso": "Camera ISO",
        "cam_f_stop": "Camera f-stop",
        "cam_shutter_speed_s": "Camera Shutter Speed (seconds)",
    },
    "IES Export": {
        "export_ies": "Export IES",
        "ies_vertical_step_deg": "IES Vertical Step (degrees)",
        "ies_horizontal_step_deg": "IES Horizontal Step (degrees)",
        "ies_max_vertical_angle_deg": "IES Max Vertical Angle (degrees)",
    },
    "Material Defaults & Thresholds": {
        "default_reflectivity_smooth": "Default Reflectivity (Smooth)",
        "default_reflectivity_op": "Default Reflectivity (Orange Peel)",
        "default_reflectivity_cylinder": "Default Reflectivity (Cylinder)",
        "default_gasket_reflectivity": "Default Reflectivity (Gasket)",
        "default_transmissivity_lens": "Default Lens Transmissivity",
        "default_surface_finish": "Default Surface Finish",
        "default_surface_roughness_nm": "Default Surface Roughness (nm RMS)",
        "default_surface_correlation_um": "Default Roughness Scale (µm)",
        "default_op_dimple_pitch_mm": "Default Orange Peel Pitch (mm)",
        "default_op_dimple_depth_um": "Default Orange Peel Depth (µm)",
        "default_op_factor": "Default OP Factor (Gaussian Blur)",
        "default_lens_finish": "Default Lens Finish",
        "default_lens_diffusion_fwhm_deg": "Default Lens Diffusion (° FWHM)",
        "default_lens_refractive_index": "Default Lens Refractive Index",
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
        "default_die_layout": "Default Die Layout",
        "default_die_rows": "Default Die Rows",
        "default_die_columns": "Default Die Columns",
        "default_die_gap_mm": "Default Die Gap (mm)",
        "default_die_gap_output": "Default Gap Output (fraction)",
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


def _polygon_normals(faces):
    """Returns the unit normal of every face, by Newell's method.

    Newell's method works for any planar polygon, convex or not, and gives the
    outward normal when the vertices run anticlockwise seen from outside the
    solid.

    Doing them together rather than one at a time is what makes this cheap. A
    preview holds a couple of thousand faces and is rebuilt on every edit, and
    a NumPy call per face spends far longer in call overhead than in
    arithmetic. Faces are grouped by how many vertices they have, since only
    faces of the same length can be stacked, and in practice almost everything
    is a quad.

    Args:
        faces: Sequence of faces, each a sequence of (x, y, z) vertices in
            order around the polygon.

    Returns:
        An (n, 3) array of unit normals, zero for any degenerate face.
    """
    normals = np.zeros((len(faces), 3))
    if not len(faces):
        return normals

    by_length = {}
    for position, face in enumerate(faces):
        by_length.setdefault(len(face), []).append(position)

    for positions in by_length.values():
        block = np.asarray([faces[position] for position in positions],
                           dtype=float)
        following = np.roll(block, -1, axis=1)
        difference = block - following
        total = block + following

        # The three components of Newell's sum, taken over the vertices of
        # every face in the block at once.
        group = np.empty((len(positions), 3))
        group[:, 0] = np.sum(difference[:, :, 1] * total[:, :, 2], axis=1)
        group[:, 1] = np.sum(difference[:, :, 2] * total[:, :, 0], axis=1)
        group[:, 2] = np.sum(difference[:, :, 0] * total[:, :, 1], axis=1)

        lengths = np.linalg.norm(group, axis=1)
        usable = lengths > 0.0
        group[usable] /= lengths[usable, None]
        group[~usable] = 0.0
        normals[positions] = group

    return normals


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
        self._face_normals = _polygon_normals(faces)
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


def _ring_surface(inner, outer, height, segments=192, bands=1):
    """Builds a flat ring at one height, between an inner and an outer radius.

    Args:
        inner: Inner radius, either a scalar or one value per angle step.
        outer: Outer radius, in the same form.
        height: Height of the plane the ring lies in.
        segments: Steps around the turn.
        bands: How many rings to split the span into. One wide quad reaching
            from the hole to the rim covers a lot of depth, and a depth
            sorter has only one number to place it by, so it can end up in
            front of something it surrounds. Splitting it radially gives
            each piece a depth close to where it actually is.

    Returns:
        (x, y, z) arrays shaped for plot_surface.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, segments)[:, None]
    inner = np.broadcast_to(np.asarray(inner, dtype=float).reshape(-1, 1), angle.shape)
    outer = np.broadcast_to(np.asarray(outer, dtype=float).reshape(-1, 1), angle.shape)
    steps = np.linspace(0.0, 1.0, max(1, bands) + 1)[None, :]
    radii = inner + (outer - inner) * steps
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
        self.setup_batch_tab()
        self.setup_results_tab()
        self.setup_output_settings()
        self.setup_hardware_widgets()
        self.connect_signals()

        # Both of these need the forms to exist: one measures the inputs it
        # is resizing, the other caps a window whose layout is by then
        # settled. Fitting the screen goes last so it sees the final size.
        self.scale_input_widths()
        self.fit_to_screen()

        # Put the buttons in their idle state, which is also what hides the
        # separate stop button. Without this it sits there on first launch
        # until the first run ends and tidies it away.
        self.set_controls_running(False)

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
        # finish is visible rather than only being a pair of numbers in the
        # form. The dimples are drawn far larger and fewer than the real
        # ones: a 0.5 mm texture on a 25 mm bowl is over three hundred
        # dimples around, which no preview mesh can resolve and which would
        # alias into a mess. What is kept faithful is their steepness, since
        # depth against pitch is what decides how much the finish scatters,
        # so a coarser or deeper texture looks coarser or deeper here too.
        bowl_z = bowl_r ** 2 / (4.0 * focal_length)
        if finish == "orange_peel":
            pitch_mm = float(spec_or_default(reflector, "reflector",
                                             "op_dimple_pitch_mm", self.config))
            depth_mm = float(spec_or_default(reflector, "reflector",
                                             "op_dimple_depth_um",
                                             self.config)) / 1000.0
            aspect = depth_mm / pitch_mm if pitch_mm > 0.0 else 0.0
            amplitude = min(aspect * DIMPLE_PREVIEW_EXAGGERATION,
                            DIMPLE_PREVIEW_MAX_AMPLITUDE)
            bowl = _dimpled_revolution(bowl_r, bowl_z, 240, amplitude, 40, 8)
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
        # the gap up to the reflector and its wall rising into the bore.
        # Both go into one collection so they sort against each other: a
        # gasket wall standing in front of the emitter hides it, instead of
        # the emitter showing through as though the rubber were glass.
        clashes = self.detect_clashes(geom, reflector, emitter, gasket)
        shared_faces, shared_colours, shared_edges = [], [], []

        def collect(part_faces, part_colours, part_edges):
            """Adds one part's faces to the shared collection."""
            shared_faces.extend(part_faces)
            shared_colours.extend(part_colours)
            shared_edges.extend(part_edges)

        lowest = []
        if gasket is not None:
            thickness = float(spec_or_default(gasket, "gasket", "thickness_mm",
                                              self.config))
            drawn = self.add_gasket(
                axes, gasket, z_bottom - thickness, collect,
                CLASH_COLOUR if "gasket" in clashes else None)
            if drawn is not None:
                lowest.append(drawn["base_z"])

        if emitter is not None:
            # An emitter as tall as the gasket seat puts its top face on
            # exactly the plane of the seat around it, and two coplanar
            # surfaces have no answer to which is in front: the gasket wins
            # in patches and appears to spill over the package. Lifting the
            # emitter by a fraction of a micron settles it, and is far below
            # anything the picture can show.
            drawn = self.add_emitter(
                axes, emitter, geom["ez_base"] + ASSEMBLY_Z_NUDGE_MM,
                geom["emitter_offset_x"], geom["emitter_offset_y"], collect,
                CLASH_COLOUR if "emitter" in clashes else None)
            if drawn is not None:
                lowest.append(drawn["base_z"])

        if shared_faces:
            axes.add_collection3d(_SolidFaces(
                shared_faces, shared_colours, shared_edges,
                linewidths=0.6, zsort="max"))

        self.add_lens(axes, reflector, radius_max, outer_radius, z_max_cut)
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

    def add_emitter(self, axes, emitter, die_z, offset_x=0.0, offset_y=0.0,
                    collect=None, tint=None):
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
        dies = self.die_positions(emitter, shape, outline, die_length, die_width)
        package = [(offset_x - footprint_x / 2.0, offset_y - footprint_y / 2.0),
                   (offset_x + footprint_x / 2.0, offset_y - footprint_y / 2.0),
                   (offset_x + footprint_x / 2.0, offset_y + footprint_y / 2.0),
                   (offset_x - footprint_x / 2.0, offset_y + footprint_y / 2.0)]
        dies = [[(x + offset_x, y + offset_y) for x, y in die]
                for die in dies]

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
        for die in dies:
            faces.append([(x, y, die_draw_z) for x, y in die])
            colours.append(EMITTER_DIE_COLOUR)
            edges.append(EMITTER_DIE_EDGE_COLOUR)

        if tint is not None:
            colours = [tint] * len(faces)

        # Handing the faces back rather than drawing them lets the caller put
        # this part and its neighbours into one collection. That is what
        # makes them hide each other: Matplotlib sorts faces within a
        # collection but has no way to interleave two of them, so an emitter
        # drawn separately always floats in front of the gasket around it.
        if collect is not None:
            collect(faces, colours, edges)
        else:
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
            # Every die counts: an array of four emits four dies worth,
            # which is the figure the caption should carry.
            "die_area": sum(abs(_outline_area(np.asarray(die, dtype=float)))
                            for die in dies),
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

    def die_positions(self, emitter, shape, outline, die_length, die_width):
        """Returns every die's outline, one entry per die on the emitter.

        Die Length and Width describe the whole emitting area, so an array
        divides that area rather than repeating it: the dies shrink to make
        room for the gaps between them. The same helpers the tracer uses work
        out how big each die ends up and where it sits, so the picture and the
        simulation cannot disagree about the emitter.

        Args:
            emitter: Emitter specs.
            shape: Die shape, already resolved against the settings default.
            outline: Vertices from emitter_die_outline, or None.
            die_length: Whole emitting area along x, in millimetres.
            die_width: Whole emitting area along y, in millimetres.

        Returns:
            A list of dies, each a list of (x, y) pairs.
        """
        rows, columns, gap, _ = die_array_layout(emitter, self.config)
        if rows == 1 and columns == 1:
            return [self.die_corners(shape, outline, die_length, die_width)]

        cell_length, cell_width = die_cell_size(emitter, shape, self.config)
        corners = self.die_corners(shape, outline, cell_length, cell_width)
        across = die_centres(columns, cell_length, gap)
        down = die_centres(rows, cell_width, gap)

        return [[(x + centre_x, y + centre_y) for x, y in corners]
                for centre_y in down for centre_x in across]

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

    def detect_clashes(self, geom, reflector, emitter, gasket):
        """Finds parts that occupy the same space, so they can be flagged red.

        Only genuine interference counts. A gasket standing taller than the
        emitter is not a fault, it is the normal arrangement: the wall rises
        past the package to meet the reflector, and a domed emitter often looks
        through the aperture from below. What matters is whether two solids
        would have to be in the same place at the same time.

        Three ways that can happen:

        * The gasket is wider than the bore it seats in, so it cannot go in.
        * Its wall is wider than the bowl at the height it reaches. Standing
          proud of the bore is fine on its own, since the parabola widens as
          it climbs; being wider than the parabola is not.
        * An opening in the gasket is narrower than the emitter passing through
          it, over the height where the two overlap. The seat is checked
          against its window, and the wall against its aperture, but the wall
          only where the package actually reaches into it.

        Args:
            geom: The traced geometry, which already knows where each part sits.
            reflector: Reflector specs, or None.
            emitter: Emitter specs, or None.
            gasket: Gasket specs, or None.

        Returns:
            A dict of part name to the reason it clashes. Empty when the stack
            fits together.
        """
        clashes = {}
        if gasket is None:
            return clashes

        try:
            outer = float(spec_or_default(gasket, "gasket", "outer_diameter_mm",
                                          self.config))
            inner = float(spec_or_default(gasket, "gasket", "inner_diameter_mm",
                                          self.config))
            window = float(spec_or_default(gasket, "gasket", "emitter_size_mm",
                                           self.config))
            wall_shape = spec_or_default(gasket, "gasket", "wall_shape", self.config)
            thickness = float(spec_or_default(gasket, "gasket", "thickness_mm",
                                              self.config))
        except (KeyError, ValueError, TypeError):
            return clashes

        def clash(reason, *parts):
            """Records one interference against every part involved."""
            for part in parts:
                clashes.setdefault(part, reason)

        bore = geom["r_hole"] * 2.0
        if outer > bore + CLASH_TOLERANCE_MM:
            clash(f"gasket outer {outer:.2f} mm will not fit the {bore:.2f} mm bore",
                  "gasket")

        # What the reflector leaves room for depends on how high up you
        # look: a straight bore up to the shelf, then the parabola, which
        # only widens from there. Standing proud of the bore is therefore
        # fine on its own; being wider than whatever is there is not.
        if geom["z_gasket_top"] <= geom["z_hole_top"]:
            clearance = geom["r_hole"]
        else:
            clearance = math.sqrt(4.0 * geom["focal_length"]
                                  * geom["z_gasket_top"])
        if outer / 2.0 > clearance + CLASH_TOLERANCE_MM:
            clash(f"gasket is {outer:.2f} mm across where the reflector "
                  f"leaves {clearance * 2.0:.2f} mm", "gasket")

        if emitter is None:
            return clashes

        try:
            footprint_x = float(emitter["footprint_x_mm"])
            footprint_y = float(emitter["footprint_y_mm"])
            height = float(emitter["height_mm"])
        except (KeyError, ValueError, TypeError):
            return clashes

        widest = max(footprint_x, footprint_y)
        diagonal = math.hypot(footprint_x, footprint_y)

        if widest > window + CLASH_TOLERANCE_MM:
            clash(f"emitter {widest:.2f} mm is wider than the {window:.2f} mm "
                  f"gasket window", "gasket", "emitter")

        # The wall only matters where the package rises into it. A package
        # shorter than the seat never reaches the wall at all.
        if height > thickness + CLASH_TOLERANCE_MM:
            if wall_shape == "round":
                if diagonal > inner + CLASH_TOLERANCE_MM:
                    clash(f"emitter corners span {diagonal:.2f} mm, wider than the "
                          f"{inner:.2f} mm gasket aperture", "gasket", "emitter")
            elif widest > window + CLASH_TOLERANCE_MM:
                clash(f"emitter {widest:.2f} mm will not pass the {window:.2f} mm "
                      f"square gasket aperture", "gasket", "emitter")

        return clashes

    def add_lens(self, axes, reflector, radius_max, outer_radius, top_z):
        """Draws the front lens across the mouth of the reflector.

        A clear lens is drawn as a rim and nothing else. Filling the mouth with
        a pane, however faint, lays a wash over the whole bowl and the parts
        sitting in it, which is worse than misleading: a clear lens is invisible
        in life, so a picture that shows one is showing something that is not
        there. The rim says it is fitted without hiding anything.

        A frosted lens or an applied film does obscure what is behind it, so
        those are drawn as a pane, and a film gets its own layer above the glass.

        Args:
            axes: 3D axes to draw on.
            reflector: Reflector specs.
            radius_max: Radius of the bowl at the mouth, in millimetres.
            outer_radius: Outer radius of the reflector body, in millimetres.
            top_z: Height of the rim.
        """
        try:
            finish = spec_or_default(reflector, "reflector", "lens_finish",
                                     self.config)
        except (KeyError, ValueError, TypeError):
            return

        radius = max(outer_radius, radius_max)
        seat = top_z + LENS_STANDOFF_MM

        if finish == "clear":
            colour, alpha = LENS_APPEARANCE["clear_rim"]
            axes.plot_surface(
                *_ring_surface(radius * LENS_RIM_INNER_FRACTION, radius, seat, 96),
                color=colour, alpha=alpha, linewidth=0, antialiased=True)
            return

        colour, alpha = LENS_APPEARANCE[finish]
        axes.plot_surface(*_ring_surface(0.0, radius, seat, 96), color=colour,
                          alpha=alpha, linewidth=0, antialiased=True)

        # A film is stuck on, so it is drawn as a separate layer above the
        # glass, slightly inset the way a cut sheet sits inside the bezel.
        if finish == "film":
            film_colour, film_alpha = LENS_APPEARANCE["film_layer"]
            axes.plot_surface(
                *_ring_surface(0.0, radius * 0.97, seat + LENS_FILM_THICKNESS_MM, 96),
                color=film_colour, alpha=film_alpha, linewidth=0, antialiased=True)

    def add_gasket(self, axes, gasket, base_z, collect=None, tint=None):
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
        window_half = (emitter_size + GASKET_WINDOW_OFFSET_MM) / 2.0

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
                (_ring_surface(window, seat_radius, base_z, steps,
                               GASKET_RADIAL_BANDS), "down"),
                (_wall_surface(seat_radius, base_z, seat_top, steps), "out"),
                (_wall_surface(window, base_z, seat_top, steps), "in"),
                # Its top, in the two bands the wall above does not cover.
                (_ring_surface(window, seat_inner_top, seat_top, steps,
                               GASKET_RADIAL_BANDS), "up"),
                (_ring_surface(wall_radius, seat_radius, seat_top, steps,
                               GASKET_RADIAL_BANDS), "up"),
                # The wall standing on it. Its underside exists only where
                # the seat has a hole beneath it to be seen through.
                (_ring_surface(aperture, wall_underside, seat_top, steps,
                               GASKET_RADIAL_BANDS), "down"),
                (_ring_surface(aperture, wall_radius, wall_top, steps,
                               GASKET_RADIAL_BANDS), "up"),
                (_wall_surface(wall_radius, seat_top, wall_top, steps), "out"),
                (_wall_surface(aperture, seat_top, wall_top, steps), "in")):
            faces.extend(_solid_quads(surface, direction))

        # The mesh itself is the edging: stroking each quad costs nothing to
        # build, and the strokes are culled and depth sorted with the faces
        # they belong to, so an edge behind the gasket stays behind it.
        colours = [tint or GASKET_COLOUR] * len(faces)

        edges = [GASKET_EDGE_COLOUR] * len(faces)
        if collect is not None:
            collect(faces, colours, edges)
        else:
            axes.add_collection3d(_SolidFaces(faces, colours, edges,
                                              linewidths=0.35, zsort="max"))

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
            if field in INTEGER_SPECS and value != "":
                value = int(round(float(value)))
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
                          self._reflector_warnings(reflector, emitter, gasket, bore))
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

    def _reflector_warnings(self, reflector, emitter, gasket, bore):
        """Warns when the bore is being assumed rather than read from the entry.

        Args:
            reflector: Reflector specs, or None.
            emitter: Emitter specs, or None.
            gasket: Gasket specs, or None.
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

        messages = []
        if diameter > 0.0 and height > 0.0 and opening == 0.0:
            messages.append(f"No opening size: assuming {bore:.2f} mm "
                            f"(footprint diagonal).")

        # Whatever the preview paints red, say why in words as well.
        if gasket is not None:
            try:
                geom = get_sim_geometry(reflector, emitter, gasket, "smooth",
                                        self.config)
            except (KeyError, ValueError, TypeError, ZeroDivisionError):
                geom = None
            if geom is not None:
                for reason in dict.fromkeys(
                        self.detect_clashes(geom, reflector, emitter,
                                            gasket).values()):
                    messages.append(f"Clash: {reason}.")
        return messages

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
        messages = []
        if opening > 0.0 and diagonal > opening + FIT_TOLERANCE_MM:
            messages.append(f"Footprint diagonal {diagonal:.2f} mm > "
                            f"opening {opening:.2f} mm.")

        # The dies share the area they are given, so an array cannot grow
        # past its own specification. What it can do is share it out until
        # there is nothing left, once the gaps account for the whole span.
        shape = spec_or_default(emitter, "emitter", "shape", self.config)
        try:
            emitting_x, emitting_y = die_array_extent(emitter, self.config)
            cell_length, cell_width = die_cell_size(emitter, shape, self.config)
            footprint_x = float(emitter["footprint_x_mm"])
            footprint_y = float(emitter["footprint_y_mm"])
        except (KeyError, ValueError, TypeError):
            return messages

        # Said before the size checks below, because when this fires the
        # array is not what is being traced and its die sizes are moot.
        resolvable, minimum = die_array_resolution(emitter, self.config)
        if not resolvable:
            messages.append(f"Die gaps too fine to resolve for emitter subdivision. "
                            f"Tracing as monotonic die. "
                            f"{minimum} subdivision needed for array.")

        if min(cell_length, cell_width) <= 0.0:
            rows, columns, gap, _ = die_array_layout(emitter, self.config)
            messages.append(f"{rows} x {columns} dies with {gap:g} mm gaps "
                            f"leave no room in a {emitting_x:.2f} mm die.")
        elif (emitting_x > footprint_x + FIT_TOLERANCE_MM
              or emitting_y > footprint_y + FIT_TOLERANCE_MM):
            messages.append(f"Die is {emitting_x:.2f} x {emitting_y:.2f} mm, "
                            f"larger than the {footprint_x:.2f} x "
                            f"{footprint_y:.2f} mm package.")
        return messages

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

    def fit_to_screen(self):
        """Caps the window to the display it opens on, and centres it.

        The layout is designed around a wide desktop, and Qt will open a window
        larger than the screen without complaint: the excess simply falls off
        the edge, taking the buttons at the bottom with it. Asking the desktop
        how much room there actually is costs nothing and makes the app usable
        on a laptop panel.
        """
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return

        available = screen.availableGeometry()
        self.resize(
            min(WINDOW_DEFAULT_WIDTH_PX,
                available.width() - WINDOW_SCREEN_MARGIN_PX),
            min(WINDOW_DEFAULT_HEIGHT_PX,
                available.height() - WINDOW_SCREEN_MARGIN_PX))
        self.move(available.center() - self.rect().center())

    def scale_input_widths(self):
        """Sizes the spec inputs, and the columns holding them, from the font.

        Three things have to agree or a box ends up unreadable. The input needs
        a width in digits rather than pixels, so it suits whatever font the
        display is using. It needs a floor as well as a ceiling, because a
        maximum alone does not stop a layout squeezing it to nothing. And the
        column has to be wide enough for the longest caption beside it, since
        the caption is served first and the input lives on what is left.

        A drop down needs more room than a text box for the same content,
        because its arrow eats into the width, so it gets an allowance.
        """
        digit = self.fontMetrics().horizontalAdvance("0")
        field_width = max(FIELD_MIN_WIDTH_PX, digit * FIELD_WIDTH_IN_DIGITS)

        # A drop down is measured against the longest caption it can show,
        # not a digit count: "Monolithic" and "Orange Peel" are far wider
        # than any number typed beside them, and a guess at the difference
        # is what left them clipped.
        combo_width = field_width
        for widgets in self.field_widgets.values():
            for widget in widgets.values():
                if not isinstance(widget, QComboBox):
                    continue
                for index in range(widget.count()):
                    combo_width = max(
                        combo_width,
                        widget.fontMetrics().horizontalAdvance(
                            widget.itemText(index)) + COMBO_ARROW_ALLOWANCE_PX)

        for widgets in (list(self.field_widgets.values())
                        + list(self.run_only_widgets.values())):
            for widget in widgets.values():
                width = (combo_width if isinstance(widget, QComboBox)
                         else field_width)
                widget.setMinimumWidth(width)
                widget.setMaximumWidth(width)

        widest_caption = 0
        for layout in (self.formLayout_Reflector, self.formLayout_Emitter,
                       self.formLayout_Gasket):
            # Wrapping a row was worse than the problem it solved: the field
            # dropped below its caption and the form grew a ragged extra line
            # for every long label. Keeping every row on one line and sizing
            # the column to suit is tidier and easier to read.
            layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
            for row in range(layout.rowCount()):
                for role in (QFormLayout.ItemRole.LabelRole,
                             QFormLayout.ItemRole.SpanningRole):
                    item = layout.itemAt(row, role)
                    label = item.widget() if item is not None else None
                    if label is None or not hasattr(label, "text"):
                        continue
                    widest_caption = max(
                        widest_caption,
                        label.fontMetrics().horizontalAdvance(label.text()))

        column_width = widest_caption + combo_width + COLUMN_PADDING_PX
        for column in (self.scrollReflector, self.scrollEmitter,
                       self.scrollGasket):
            column.setMinimumWidth(column_width)

        # Equal stretch, so the three share the width evenly however wide
        # the window is. Done here rather than as a layout property in the
        # .ui, because uic turns a property into a setter call by name and
        # a box layout has no setStretch that takes one.
        for index in range(self.horizontalLayout_Columns.count()):
            self.horizontalLayout_Columns.setStretch(index, 1)

    @staticmethod
    def scale_figure_text(figure):
        """Resizes every label on a figure in proportion to the figure itself.

        Matplotlib measures type in points, which do not shrink when the panel
        does, so a plot squeezed into a small window ends up with its title
        across its axis labels. Each artist remembers the size it was given and
        is redrawn at a fraction of it, so the layout holds together at any
        size. A floor stops the text vanishing altogether on a very small panel.

        Args:
            figure: The Matplotlib figure to rescale.
        """
        scale = max(FIGURE_MIN_FONT_SCALE,
                    figure.get_size_inches()[1] / FIGURE_REFERENCE_HEIGHT_IN)
        for artist in figure.findobj(Text):
            original = getattr(artist, "_unscaled_fontsize", None)
            if original is None:
                original = artist.get_fontsize()
                artist._unscaled_fontsize = original
            artist.set_fontsize(original * scale)

    def setup_batch_tab(self):
        """Wires up the UI buttons and components for the Batch Tab."""
        if not hasattr(self, 'tabBatch'):
            return
        self.btnAddBatch.clicked.connect(self.add_to_batch)
        self.btnBatchUp.clicked.connect(self.move_batch_up)
        self.btnBatchDown.clicked.connect(self.move_batch_down)
        self.btnBatchDelete.clicked.connect(self.delete_batch_items)
        self.current_batch_index = -1
        self.populate_batch_lists()

    def populate_batch_lists(self):
        """Fills the batch selection lists from the active hardware library."""
        if not hasattr(self, 'lstBatchReflectors'):
            return
        self.lstBatchReflectors.clear()
        self.lstBatchEmitters.clear()
        self.lstBatchGaskets.clear()
        
        # Use .names() to retrieve the list of hardware components correctly
        self.lstBatchReflectors.addItems(sorted(self.library.names("reflector")))
        self.lstBatchEmitters.addItems(sorted(self.library.names("emitter")))
        self.lstBatchGaskets.addItems(sorted(self.library.names("gasket")))

    def add_to_batch(self):
        """Generates all combinations of selected components and validates physical fits."""
        reflectors = [item.text() for item in self.lstBatchReflectors.selectedItems()]
        emitters = [item.text() for item in self.lstBatchEmitters.selectedItems()]
        gaskets = [item.text() for item in self.lstBatchGaskets.selectedItems()]

        if not reflectors or not emitters or not gaskets:
            self.log_message("Please select at least one item from each column to build combinations.")
            return

        for r_name in reflectors:
            for e_name in emitters:
                for g_name in gaskets:
                    item_text = f"{r_name} | {e_name} | {g_name}"
                    list_item = QListWidgetItem(item_text)

                    # Fetch raw dictionaries from the library
                    ref = self.library.get("reflector", r_name)
                    emi = self.library.get("emitter", e_name)
                    gsk = self.library.get("gasket", g_name)

                    try:
                        bore = effective_bore_diameter(ref, emi, self.config)
                    except (KeyError, ValueError, TypeError):
                        bore = None

                    # Run the exact same geometry validations used by the Setup Tab
                    warnings = []
                    warnings.extend(self._reflector_warnings(ref, emi, gsk, bore))
                    warnings.extend(self._emitter_warnings(ref, emi))
                    warnings.extend(self._gasket_warnings(gsk, emi, bore))

                    mismatch = False
                    filtered_warnings = []
                    
                    for w in warnings:
                        # Ignore the die subdivision resolution warning
                        if "Die gaps too fine" not in w:
                            mismatch = True
                            filtered_warnings.append(w)

                    if mismatch:
                        list_item.setBackground(QColor(255, 255, 150)) # Yellow warning
                        list_item.setForeground(QColor(0, 0, 0))       # Ensure black text
                        # Add a helpful tooltip so you can hover over the yellow item to see exactly what clashes!
                        list_item.setToolTip("\n".join(filtered_warnings))

                    self.lstBatchQueue.addItem(list_item)
                    
        self.log_message(f"Added {len(reflectors) * len(emitters) * len(gaskets)} combinations to the batch.")

    def move_batch_up(self):
        row = self.lstBatchQueue.currentRow()
        if row > 0:
            item = self.lstBatchQueue.takeItem(row)
            self.lstBatchQueue.insertItem(row - 1, item)
            self.lstBatchQueue.setCurrentRow(row - 1)

    def move_batch_down(self):
        row = self.lstBatchQueue.currentRow()
        if 0 <= row < self.lstBatchQueue.count() - 1:
            item = self.lstBatchQueue.takeItem(row)
            self.lstBatchQueue.insertItem(row + 1, item)
            self.lstBatchQueue.setCurrentRow(row + 1)

    def delete_batch_items(self):
        for item in self.lstBatchQueue.selectedItems():
            self.lstBatchQueue.takeItem(self.lstBatchQueue.row(item))

    def setup_results_tab(self):
        """Wires the results tab and hides it until there is something in it."""
        self.tabMain.setTabVisible(RESULTS_TAB_INDEX, False)
        self.wall_shots = []
        self.plot_entries = []
        self.lstPlots.currentRowChanged.connect(lambda *_: self.show_selected_plot())

    def remember_result(self, shot):
        """Puts a finished run at the top of the results list.

        Earlier runs stay below it, so a build can be compared against the one
        before without tracing either again. The oldest fall off the end once
        the list is full: each run holds a wall grid, and keeping every one of
        them for a long session would quietly grow without limit.

        Args:
            shot: The render inputs from a completed run.
        """
        self.wall_shots.insert(0, shot)
        del self.wall_shots[max(1, int(self.config.stored_run_count)):]

        self.lstPlots.blockSignals(True)
        self.lstPlots.clear()
        self.plot_entries = []
        
        from PyQt6.QtCore import Qt
        
        for index, stored in enumerate(self.wall_shots):
            # Add an unselectable visual separator between runs
            if index > 0:
                sep_item = QListWidgetItem("────────────────────")
                sep_item.setFlags(sep_item.flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsEnabled)
                sep_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.lstPlots.addItem(sep_item)
                
                # Add a blank entry to keep the backend list synced with the UI rows
                self.plot_entries.append((None, None))
                
            for name in PLOT_NAMES:
                self.plot_entries.append((stored, name))
                self.lstPlots.addItem(QListWidgetItem(f"{name} - {stored.label}"))
                
        self.lstPlots.setCurrentRow(0)
        self.lstPlots.blockSignals(False)

    def selected_result(self):
        """Returns the (shot, plot name) the user is looking at.

        Returns:
            A pair, or (None, None) when there is nothing to show yet.
        """
        row = self.lstPlots.currentRow()
        if 0 <= row < len(self.plot_entries):
            return self.plot_entries[row]
        return None, None

    def show_selected_plot(self):
        """Draws whichever plot is chosen, and shows the camera bar for the shot.

        Only the wall shot is a photograph, so only it has an exposure. Leaving
        the camera controls on show beside a line graph would suggest they did
        something there.
        """
        shot, name = self.selected_result()
        if shot is None:
            return

        self.grpCamera.setVisible(name == PLOT_NAMES[0])
        self.show_figure(render_plot(shot, name, self.config))

    def save_plots(self):
        """Writes the chosen plots of the selected run to a specified file base."""
        shot, _ = self.selected_result()
        if shot is None:
            self.log_message("No plots to save yet; run a simulation first.")
            return

        wanted = []
        if self.chkSaveWallShot.isChecked(): wanted.append(PLOT_NAMES[0])
        if self.chkSaveXAxis.isChecked(): wanted.append(PLOT_NAMES[1])
        if self.chkSaveYAxis.isChecked(): wanted.append(PLOT_NAMES[2])
        if self.chkSave45Deg.isChecked(): wanted.append(PLOT_NAMES[3])

        if not wanted:
            self.log_message("No plots selected to save. Check the boxes in Output Settings.")
            return

        # Pre-fill the Save As dialog with the auto-generated filename
        default_path = os.path.join(self.config.resolved_output_directory, shot.filename)
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Plots As (Base Name)", 
            default_path, 
            "PNG Images (*.png)"
        )
        
        if not file_path:
            return

        import csv
        base, extension = os.path.splitext(file_path)
        export_csv = hasattr(self, 'chkExportPlotsCSV') and self.chkExportPlotsCSV.isChecked()

        # Isolate the exact data arrays used for the intensity plots
        centre = shot.wall_lux.shape[0] // 2
        slices = {
            "X-Axis": (shot.wall_lux[centre, :], shot.axis_distance),
            "Y-Axis": (shot.wall_lux[:, centre], shot.axis_distance),
            "45-Deg": (np.diagonal(shot.wall_lux), shot.diagonal_distance)
        }

        # The render engine will automatically append suffixes to this base path
        for name in wanted:
            # Save the PNG Image
            plt.close(render_plot(shot, name, self.config, file_path, always_save=True))
            
            # Save the CSV if checked and it is an intensity plot
            if export_csv and name in slices:
                csv_path = f"{base}_{name}.csv"
                values_lux, distances = slices[name]
                
                # Convert Lux to Candela using the frozen geometry
                values_cd = values_lux * (shot.shot_config.target_distance_m ** 2)
                
                # Calculate angles dynamically so the CSV always has both metrics
                angles_deg = np.degrees(np.arctan(distances / shot.shot_config.target_distance_m))
                
                with open(csv_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Distance_m", "Angle_deg", "Intensity_cd"])
                    for d, a, cd in zip(distances, angles_deg, values_cd):
                        writer.writerow([f"{d:.4f}", f"{a:.4f}", f"{cd:.2f}"])
            
        self.log_message(f"Saved {len(wanted)} plot(s) of {shot.label} based on {file_path}")
        if export_csv:
            self.log_message("-> Associated CSV data files exported successfully.")

    def setup_camera_controls(self):
        """Wires the camera bar and puts the saved settings into it.

        The camera only decides how a finished result is displayed, so these
        redraw the chosen wall shot rather than starting a new trace.
        """
        # Force the layout to keep the spacing even when the widget is hidden
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        policy.setRetainSizeWhenHidden(True)
        self.grpCamera.setSizePolicy(policy)
        
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
                       self.txtCamFStop, self.lblCamShutter, self.txtCamShutter,
                       self.lblCamShutterUnit):
            widget.setVisible(not auto)

        self.config.use_auto_exposure = auto
        self.config.auto_exposure_compensation_ev = stops
        self.config.cam_iso = iso
        self.config.cam_f_stop = f_stop
        self.config.cam_shutter_speed_s = shutter

        self.show_selected_plot()

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

    def setup_output_settings(self):
        """Wires up the output settings panel."""
        # Initial states
        # The saved value is free text and the combo entries are
        # capitalised, so a file holding "distance" would not select
        # anything and the box would show whatever happened to be first.
        scale = str(getattr(self.config, "plot_scale", "Distance")).strip()
        for index in range(self.cmbPlotScale.count()):
            if self.cmbPlotScale.itemText(index).lower() == scale.lower():
                self.cmbPlotScale.setCurrentIndex(index)
                break
        self.chkShowPrimaryGrid.setChecked(getattr(self.config, "plot_show_primary_grid", True))
        self.chkShowSecondaryGrid.setChecked(getattr(self.config, "plot_show_secondary_grid", False))

        self.chkSaveWallShot.setChecked(self.config.plot_wall_shot)
        self.chkSaveXAxis.setChecked(self.config.plot_intensity_x)
        self.chkSaveYAxis.setChecked(self.config.plot_intensity_y)
        self.chkSave45Deg.setChecked(self.config.plot_intensity_45)

        # Signals
        self.cmbPlotScale.currentIndexChanged.connect(self.on_plot_scale_changed)
        self.chkShowPrimaryGrid.toggled.connect(lambda v: self.update_plot_setting("plot_show_primary_grid", v))
        self.chkShowSecondaryGrid.toggled.connect(lambda v: self.update_plot_setting("plot_show_secondary_grid", v))
        self.chkSaveWallShot.toggled.connect(lambda v: self.update_plot_setting("plot_wall_shot", v))
        self.chkSaveXAxis.toggled.connect(lambda v: self.update_plot_setting("plot_intensity_x", v))
        self.chkSaveYAxis.toggled.connect(lambda v: self.update_plot_setting("plot_intensity_y", v))
        self.chkSave45Deg.toggled.connect(lambda v: self.update_plot_setting("plot_intensity_45", v))

        self.btnSavePlots.clicked.connect(self.save_plots)

    def on_plot_scale_changed(self):
        self.config.plot_scale = self.cmbPlotScale.currentText()
        self.config.save_settings()
        self.show_selected_plot()

    def update_plot_setting(self, key, value):
        setattr(self.config, key, value)
        self.config.save_settings()
        # Instantly redraw the plot to show/hide the grids
        if key in ("plot_show_primary_grid", "plot_show_secondary_grid"):
            self.show_selected_plot()

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
        self.btnSimulate.clicked.connect(lambda *_: self.run_or_stop())
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
            if field in INTEGER_SPECS and value != "":
                value = int(round(float(value)))
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
        for field, widget in widgets.items():
            visible = True

            # 1. Base check: Drop down dependencies (e.g., OP vs Smooth)
            if field in CONDITIONAL_SPECS:
                deciding_field, needed_for = CONDITIONAL_SPECS[field]
                if deciding_field in widgets:
                    current = self.field_value(widgets[deciding_field])
                    if current not in needed_for:
                        visible = False

            # 2. Global config overrides for the Reflector column
            if kind == "reflector":
                # Front Lens Toggle
                if field in ("lens_finish", "lens_diffusion_fwhm_deg", "lens_refractive_index"):
                    if not getattr(self.config, "enable_lens_simulation", True):
                        visible = False
                        
                # Dimple vs Gaussian OP Toggle
                if field in ("op_dimple_pitch_mm", "op_dimple_depth_um"):
                    if not getattr(self.config, "use_dimple_op_simulation", False):
                        visible = False
                        
                if field == "op_factor":
                    if getattr(self.config, "use_dimple_op_simulation", False):
                        visible = False

            self.set_row_visible(widget, visible)

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
        
        # Apply conditionals again in case global settings (like dimple OP or lens sim) changed
        for kind in SPEC_FIELDS:
            self.apply_conditional_rows(kind)
            
        self.update_previews()

    def log_message(self, message):
        """Appends a message to the UI logs."""
        if hasattr(self, 'txtLogs'):
            self.txtLogs.appendPlainText(message)
            scrollbar = self.txtLogs.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
        if hasattr(self, 'txtBatchLogs'):
            self.txtBatchLogs.appendPlainText(message)
            scrollbar = self.txtBatchLogs.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

    def update_progress(self, percent):
        """Moves the progress bar."""
        self.progressBar.setValue(int(percent))

    def set_controls_running(self, is_running):
        """Turns the run button into a stop button, and back again.

        One button rather than two: the two are never both useful, and a
        greyed out Stop beside a live Run tells the operator nothing they
        cannot see from the progress bar.

        Args:
            is_running: True while a job is in progress.
        """
        self.btnSimulate.setText("Stop Simulation" if is_running
                                 else "Run FEA Simulation")
        self.btnSettings.setEnabled(not is_running)
        self.btnStop.setVisible(False)

    def run_or_stop(self):
        """Starts a run, or stops the one already going."""
        if self.worker is not None and self.worker.isRunning():
            self.stop_simulation()
        else:
            self.run_simulation()

    def _dispatch_simulation(self):
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

    def run_simulation(self):
        """Intercepts the 'Run' button. Runs a batch loop if there are unprocessed items, otherwise runs normally."""
        if hasattr(self, 'lstBatchQueue') and self.lstBatchQueue.count() > 0:
            # Find the first item that hasn't been completed (not Green and not Red)
            first_unprocessed = -1
            for i in range(self.lstBatchQueue.count()):
                bg_rgb = self.lstBatchQueue.item(i).background().color().getRgb()[:3]
                if bg_rgb not in [(150, 255, 150), (255, 150, 150)]:
                    first_unprocessed = i
                    break
            
            if first_unprocessed == -1:
                self.log_message("All items in the batch queue have already been processed.")
                return

            if hasattr(self, 'txtLogs'): self.txtLogs.clear()
            if hasattr(self, 'txtBatchLogs'): self.txtBatchLogs.clear()
            
            # Automatically switch the UI to the Batch tab so you can watch it run
            if hasattr(self, 'tabMain') and hasattr(self, 'tabBatch'):
                self.tabMain.setCurrentWidget(self.tabBatch)
            
            self.current_batch_index = first_unprocessed
            self._start_next_batch_item()
        else:
            self.current_batch_index = -1
            self._dispatch_simulation()
            
    def _start_next_batch_item(self):
        if self.current_batch_index >= self.lstBatchQueue.count():
            self.log_message("\n=== BATCH COMPLETE ===")
            self.btnSimulate.setEnabled(True)
            self.current_batch_index = -1
            return

        item = self.lstBatchQueue.item(self.current_batch_index)
        item.setBackground(QColor(150, 220, 255)) # Cyan (Running status)
        item.setForeground(QColor(0, 0, 0))
        self.lstBatchQueue.scrollToItem(item)

        parts = item.text().split(" | ")
        if len(parts) == 3:
            self.combo_boxes["reflector"].setCurrentText(parts[0].strip())
            self.combo_boxes["emitter"].setCurrentText(parts[1].strip())
            self.combo_boxes["gasket"].setCurrentText(parts[2].strip())

        self.log_message(f"\n--- BATCH JOB {self.current_batch_index + 1} OF {self.lstBatchQueue.count()} ---")
        self._dispatch_simulation()

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

        if getattr(self, 'current_batch_index', -1) >= 0:
            item = self.lstBatchQueue.item(self.current_batch_index)
            item.setBackground(QColor(255, 150, 150)) # Red (Error)
            self.current_batch_index += 1
            self._start_next_batch_item()
        else:
            self.set_controls_running(False)

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
        
        # Results only exist once something has been traced, so the tab
        # appears with them and the view moves to it.
        if shot is not None:
            self.remember_result(shot)
            self.tabMain.setTabVisible(RESULTS_TAB_INDEX, True)
            
            # Only yank the view to the Results tab if we are NOT running a batch queue
            if getattr(self, 'current_batch_index', -1) < 0:
                self.tabMain.setCurrentIndex(RESULTS_TAB_INDEX)
                
            self.show_selected_plot()

        if results:
            self.log_message("\n--- SIMULATION RESULTS ---")
            for label, value in results.items():
                self.log_message(f"{label}: {value}")

        # Batch Loop Continuer
        if getattr(self, 'current_batch_index', -1) >= 0:
            item = self.lstBatchQueue.item(self.current_batch_index)
            
            if shot is not None:
                item.setBackground(QColor(150, 255, 150)) # Green (Success)
                self.current_batch_index += 1
                self._start_next_batch_item()
            else:
                # Job was cancelled. Halt the batch loop.
                self.log_message("\n[!] Batch processing halted by user.")
                self.current_batch_index = -1


    def show_figure(self, figure):
        """Puts a figure in the output box, replacing whatever was there.

        Args:
            figure: The Matplotlib figure to display.
        """
        if self.figure_canvas is not None:
            self.grpPlot.layout().removeWidget(self.figure_canvas)
            self.figure_canvas.deleteLater()
            plt.close(self.figure_canvas.figure)

        self.figure_canvas = FigureCanvas(figure)
        self.grpPlot.layout().addWidget(self.figure_canvas)

        # Rescale on every resize, not just once: the panel changes size
        # whenever the window does, and the type has to follow it.
        self.figure_canvas.mpl_connect(
            "resize_event", lambda *_: self.scale_figure_text(figure))
        self.scale_figure_text(figure)
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
import os
import sys
import math
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
from numba import cuda, njit

# ==============================================================================
# 0. QUICK-SET ACTIVE HARDWARE SELECTION & VISUALIZATION
# ==============================================================================
active_emitter_name = "SFT60_6500K"
active_reflector_name = "S2_S6_S8_T6"
reflector_finish = "orange_peel"  # Options are "smooth" or "orange_peel".

max_multiple_reflections = 10  # 0 means only the initial bounce is tracked. 1 means up to 1 extra bounce, etc.
use_reflector_opening = True  # TRUE = Use reflector opening diameter, FALSE = Use emitter footprint diagonal if larger than reflector opening diameter

# --- SIMULATION SPACE (AUTO-SCALING FOV) ---
target_distance_m = 10  # Distance from the flashlight to the wall along the Z-axis.
canvas_fov_deg = 120.0  # Field of view of the underlying simulation that catches the rays.
plot_fov_deg = 90.0  # Field of view for the generated plot image, effectively zooming the camera.

# Automatically calculate the physical canvas sizes based on distance and viewing angles.
wall_radius_m = target_distance_m * math.tan(math.radians(canvas_fov_deg / 2.0))
plot_radius_m = target_distance_m * math.tan(math.radians(plot_fov_deg / 2.0))

# --- VISUALIZATION TOGGLES ---
generate_all_plots = False        # TRUE = Batch render all combinations to disk. FALSE = Render active selection only.
show_human_silhouette = False    # TRUE = Draw a 1.75m scale human silhouette centered in the plot for reference.
plot_wall_shot = True            # TRUE = Generate the standard 2D wall projection image.
plot_intensity_x = False         # TRUE = Generate a 2D line graph of intensity across the X axis.
plot_intensity_y = False         # TRUE = Generate a 2D line graph of intensity across the Y axis.
plot_intensity_45 = False        # TRUE = Generate a 2D line graph of intensity across the 45-degree diagonal.
batch_output_directory = "D:/Documents/Reflectors"

# --- PHOTOREALISTIC CAMERA SIMULATION SETTINGS ---
use_auto_exposure = False  # True normalizes peak brightness per light. False locks to the manual camera settings below.
auto_exposure_compensation_ev = 1.0  # EV compensation when auto-exposure is enabled (+1.0 is 2x brighter, -1.0 is half as bright).

# Absolute camera settings (used only when use_auto_exposure is False).
cam_iso = 200  # Sensor sensitivity.
cam_f_stop = 8  # Aperture (e.g., 2.8, 4.0, 5.6, 8.0).
cam_shutter_speed_s = 1.0 / 3.0  # Shutter speed in seconds (e.g., 1.0/60.0).

# --- SIMULATION RESOLUTION & ANGULAR DENSITY ---
sim_grid_res = 1024  # Target plane virtual resolution representing width and height in pixels.
sim_emitter_elements = 192  # LED die subdivision (e.g., 128x128 = 16,384 discrete emission points).
sim_theta_step_deg = 0.05  # Theta angular resolution (elevation from dead center).
sim_phi_step_deg = 0.05  # Phi angular resolution (rotation around the die).

# ==============================================================================
# 1. SIMULATION FALLBACKS & THRESHOLDS
# ==============================================================================

# --- ANGULAR RANGES (LED Emission Bounds) ---
sim_theta_min_deg = 0.1  # Avoid pure 0 to prevent divide-by-zero anomalies during ray generation.
sim_theta_max_deg = 90.0  # Stop at 90° since LEDs sit flat and light doesn't travel backward.
sim_phi_min_deg = 0.0  # Start of rotation.
sim_phi_max_deg = 360.0  # End of full circle rotation.

lumen_calc_step_deg = 0.05  # Step size used for integrating the Lambertian curve to calibrate theoretical lumens.

# --- REFLECTIVITY & OPTICAL DEFAULTS ---
default_reflectivity_smooth = 0.85  # Percentage of light conserved when bouncing off a smooth mirror finish.
default_reflectivity_op = 0.80  # Percentage of light conserved off an orange peel finish.
default_reflectivity_cylinder = 0.8  # Percentage of light conserved off the cylindrical hole wall.
default_reflectivity_gasket = 0.1  # Percentage of light conserved off the white plastic gasket.
default_op_blur_strength = 6.0  # Base Gaussian blur sigma applied to orange peel hotspots.

# --- BEAM GEOMETRY MEASUREMENT THRESHOLDS ---
spill_visible_threshold_lux = 1e-4  # Absolute minimum lux required to define the outer edge of the direct spill.
corona_visible_threshold = 0.01  # Defines the visible edge of the blurred corona as 1% of the peak hotspot intensity.
hotspot_fwhm_threshold = 0.5  # Defines the true hotspot angle as 50% of the peak intensity (industry standard FWHM).

# --- HARDWARE DIMENSION FALLBACKS (If missing from the hardware dictionary) ---
default_gasket_thickness_mm = 0.95  # Standard thickness for LED centering gaskets.
default_gasket_total_height_mm = 0.95  # Total height of the gasket. 
default_gasket_opening_mm = 0.0  # Inner diameter of the gasket opening. 0 defaults to the rectangular emitter footprint.
default_reflector_wall_thickness_mm = 2.0  # Subtracts from the outer diameter to find the actual internal reflector width.
default_reflector_base_thickness_mm = 0.5  # Subtracts from the total height to find the actual internal reflector depth.
default_focus_offset_mm = 0.0  # Assumes perfect focal height if not specified.

# ==============================================================================
# 2. HARDWARE LIBRARIES
# ==============================================================================
EMITTERS = {
    "219F_5700K":       {"die_length_mm": 2.0,  "die_width_mm": 2.0,  "shape": "square", "footprint_x_mm": 3.5, "footprint_y_mm": 3.5, "height_mm": 0.72, "max_current_amps": 2.4,  "vf_turn_on_v": 2.569, "vf_scale": 0.622, "base_efficacy_lm_w": 213.3, "droop_factor": 0.319, "dome_size_mm": 3.1, "refractive_index": 1.41},
#   "519A_4000K":       {"die_length_mm": 2.0,  "die_width_mm": 2.0,  "shape": "square", "footprint_x_mm": 3.5, "footprint_y_mm": 3.5, "height_mm": 0.77, "max_current_amps": 5.0,  "vf_turn_on_v": 2.739, "vf_scale": 0.377, "base_efficacy_lm_w": 116.3, "droop_factor": 0.147, "dome_size_mm": 3.1, "refractive_index": 1.41},
    "519A_5000K":       {"die_length_mm": 2.0,  "die_width_mm": 2.0,  "shape": "square", "footprint_x_mm": 3.5, "footprint_y_mm": 3.5, "height_mm": 0.77, "max_current_amps": 5.0,  "vf_turn_on_v": 2.739, "vf_scale": 0.377, "base_efficacy_lm_w": 136.3, "droop_factor": 0.142, "dome_size_mm": 3.1, "refractive_index": 1.41},
    "719A_5000K":       {"die_length_mm": 2.0,  "die_width_mm": 2.0,  "shape": "square", "footprint_x_mm": 3.5, "footprint_y_mm": 3.5, "height_mm": 0.88, "max_current_amps": 1.8,  "vf_turn_on_v": 6.009, "vf_scale": 0.962, "base_efficacy_lm_w": 106.1, "droop_factor": 0.184, "dome_size_mm": 0.0, "refractive_index": 1.0},
    "B35AM_5700K":      {"die_length_mm": 3.12, "die_width_mm": 3.12, "shape": "square", "footprint_x_mm": 3.65,"footprint_y_mm": 3.65,"height_mm": 0.73, "max_current_amps": 2.4,  "vf_turn_on_v": 5.278, "vf_scale": 0.754, "base_efficacy_lm_w": 133.3, "droop_factor": 0.125, "dome_size_mm": 0.0, "refractive_index": 1.0},
    "GT_FC40_5000K":    {"die_length_mm": 5.55, "die_width_mm": 5.55, "shape": "square", "footprint_x_mm": 7.0, "footprint_y_mm": 7.0, "height_mm": 1.0,  "max_current_amps": 2.5,  "vf_turn_on_v": 9.523, "vf_scale": 1.787, "base_efficacy_lm_w": 107.9, "droop_factor": 0.0845, "dome_size_mm": 0.0, "refractive_index": 1.0},
    "KW_CULPM1.TG":     {"die_length_mm": 1.59, "die_width_mm": 1.25, "shape": "square", "footprint_x_mm": 4.0, "footprint_y_mm": 4.0, "height_mm": 0.732,  "max_current_amps": 8.0,  "vf_turn_on_v": 2.600, "vf_scale": 0.364, "base_efficacy_lm_w": 94.2,  "droop_factor": 0.058, "dome_size_mm": 0.0, "refractive_index": 1.0},
#   "LHP531_6500K":     {"die_length_mm": 3.67, "die_width_mm": 3.67, "shape": "square", "footprint_x_mm": 5.0, "footprint_y_mm": 5.0, "height_mm": 0.8,  "max_current_amps": 9.0, "vf_turn_on_v": 1.737, "vf_scale": 0.535, "base_efficacy_lm_w": 161.0, "droop_factor": 0.0329, "dome_size_mm": 0.0, "refractive_index": 1.0},
    "LHP531_5000K":     {"die_length_mm": 3.67, "die_width_mm": 3.67, "shape": "square", "footprint_x_mm": 5.0, "footprint_y_mm": 5.0, "height_mm": 0.8,  "max_current_amps": 5.0, "vf_turn_on_v": 1.832, "vf_scale": 0.487, "base_efficacy_lm_w": 156.2, "droop_factor": 0.0321, "dome_size_mm": 0.0, "refractive_index": 1.0},
#   "LHP531_4000K":     {"die_length_mm": 3.67, "die_width_mm": 3.67, "shape": "square", "footprint_x_mm": 5.0, "footprint_y_mm": 5.0, "height_mm": 0.8,  "max_current_amps": 9.0, "vf_turn_on_v": 1.813, "vf_scale": 0.495, "base_efficacy_lm_w": 148.8, "droop_factor": 0.0302, "dome_size_mm": 0.0, "refractive_index": 1.0},
#   "LHP531_3000K":     {"die_length_mm": 3.67, "die_width_mm": 3.67, "shape": "square", "footprint_x_mm": 5.0, "footprint_y_mm": 5.0, "height_mm": 0.8,  "max_current_amps": 9.0, "vf_turn_on_v": 1.863, "vf_scale": 0.495, "base_efficacy_lm_w": 148.3, "droop_factor": 0.0346, "dome_size_mm": 0.0, "refractive_index": 1.0},
#   "LHP531_1800K":     {"die_length_mm": 3.78, "die_width_mm": 3.78, "shape": "square", "footprint_x_mm": 5.0, "footprint_y_mm": 5.0, "height_mm": 0.8,  "max_current_amps": 9.0, "vf_turn_on_v": 1.837, "vf_scale": 0.464, "base_efficacy_lm_w": 137.2, "droop_factor": 0.0482, "dome_size_mm": 0.0, "refractive_index": 1.0},
#   "LHP73B_5000K":     {"die_length_mm": 5.0,  "die_width_mm": 5.0,  "shape": "square", "footprint_x_mm": 7.0, "footprint_y_mm": 7.0, "height_mm": 0.7,  "max_current_amps": 20.5, "vf_turn_on_v": 2.025, "vf_scale": 0.407, "base_efficacy_lm_w": 152.9, "droop_factor": 0.0188, "dome_size_mm": 0.0, "refractive_index": 1.0},
    "SBT90.2_6500K":    {"die_length_mm": 3.0,  "die_width_mm": 3.0,  "shape": "square", "footprint_x_mm": 10.0, "footprint_y_mm": 11.0, "height_mm": 1.54, "max_current_amps": 20.5, "vf_turn_on_v": 2.000, "vf_scale": 0.455, "base_efficacy_lm_w": 151.2, "droop_factor": 0.0335, "dome_size_mm": 0.0, "refractive_index": 1.0},
    "SFT12_6500K":      {"die_length_mm": 1.26, "die_width_mm": 1.26, "shape": "round", "footprint_x_mm": 3.45, "footprint_y_mm": 3.45, "height_mm": 0.91, "max_current_amps": 5.0, "vf_turn_on_v": 2.508, "vf_scale": 0.709, "base_efficacy_lm_w": 154.6, "droop_factor": 0.1995, "dome_size_mm": 0.0, "refractive_index": 1.0},
#   "SFT25R_5700K":     {"die_length_mm": 1.7,  "die_width_mm": 1.7,  "shape": "round", "footprint_x_mm": 5.0, "footprint_y_mm": 5.0, "height_mm": 1.0, "max_current_amps": 7.5, "vf_turn_on_v": 2.300, "vf_scale": 0.683, "base_efficacy_lm_w": 114.1, "droop_factor": 0.1098, "dome_size_mm": 0.0, "refractive_index": 1.0},
    "SFT25R_6500K":     {"die_length_mm": 1.7,  "die_width_mm": 1.7,  "shape": "round", "footprint_x_mm": 5.0, "footprint_y_mm": 5.0, "height_mm": 1.0, "max_current_amps": 8.0, "vf_turn_on_v": 2.485, "vf_scale": 0.598, "base_efficacy_lm_w": 135.1, "droop_factor": 0.0900, "dome_size_mm": 0.0, "refractive_index": 1.0},
#   "SFT40_3000K":      {"die_length_mm": 1.97, "die_width_mm": 1.97, "shape": "square", "footprint_x_mm": 5.0, "footprint_y_mm": 5.0, "height_mm": 1.0, "max_current_amps": 8.0, "vf_turn_on_v": 1.882, "vf_scale": 0.819, "base_efficacy_lm_w": 96.3, "droop_factor": 0.0842, "dome_size_mm": 0.0, "refractive_index": 1.0},
    "SFT40_6500K":      {"die_length_mm": 1.97, "die_width_mm": 1.97, "shape": "square", "footprint_x_mm": 5.0, "footprint_y_mm": 5.0, "height_mm": 1.0, "max_current_amps": 5.0, "vf_turn_on_v": 1.734, "vf_scale": 0.818, "base_efficacy_lm_w": 162.8, "droop_factor": 0.0738, "dome_size_mm": 0.0, "refractive_index": 1.0},
#   "SFT42R_6500K":     {"die_length_mm": 2.32, "die_width_mm": 2.32, "shape": "round", "footprint_x_mm": 5.0, "footprint_y_mm": 5.0, "height_mm": 1.0, "max_current_amps": 13.0, "vf_turn_on_v": 2.336, "vf_scale": 0.479, "base_efficacy_lm_w": 164.9, "droop_factor": 0.0642, "dome_size_mm": 0.0, "refractive_index": 1.0},
    "SFT60_6500K":      {"die_length_mm": 2.55, "die_width_mm": 2.55, "shape": "square", "footprint_x_mm": 5.0, "footprint_y_mm": 5.0, "height_mm": 0.7, "max_current_amps": 5.0, "vf_turn_on_v": 2.465, "vf_scale": 0.366, "base_efficacy_lm_w": 149.7, "droop_factor": 0.0390, "dome_size_mm": 0.0, "refractive_index": 1.0},
#   "SFT70_3000K":      {"die_length_mm": 2.38, "die_width_mm": 2.38, "shape": "square", "footprint_x_mm": 5.0, "footprint_y_mm": 5.0, "height_mm": 1.0, "max_current_amps": 8.0, "vf_turn_on_v": 4.800, "vf_scale": 1.092, "base_efficacy_lm_w": 92.7, "droop_factor": 0.1046, "dome_size_mm": 0.0, "refractive_index": 1.0},
    "SFT70_6500K":      {"die_length_mm": 2.38, "die_width_mm": 2.38, "shape": "square", "footprint_x_mm": 5.0, "footprint_y_mm": 5.0, "height_mm": 1.0, "max_current_amps": 5.0, "vf_turn_on_v": 4.591, "vf_scale": 1.233, "base_efficacy_lm_w": 140.5, "droop_factor": 0.0977, "dome_size_mm": 0.0, "refractive_index": 1.0},
#   "SFT90_6500K":      {"die_length_mm": 3.0,  "die_width_mm": 3.0,  "shape": "square", "footprint_x_mm": 10.0, "footprint_y_mm": 11.0, "height_mm": 1.0, "max_current_amps": 20.5, "vf_turn_on_v": 1.667, "vf_scale": 0.744, "base_efficacy_lm_w": 171.4, "droop_factor": 0.0502, "dome_size_mm": 0.0, "refractive_index": 1.0},
#   "W5050SQ5_6500K":   {"die_length_mm": 1.7,  "die_width_mm": 1.7,  "shape": "round", "footprint_x_mm": 5.0, "footprint_y_mm": 5.0, "height_mm": 1.0, "max_current_amps": 7.5, "vf_turn_on_v": 2.222, "vf_scale": 0.708, "base_efficacy_lm_w": 124.7, "droop_factor": 0.1376, "dome_size_mm": 0.0, "refractive_index": 1.0},
#   "XHP50.3_HI_4500K": {"die_length_mm": 2.95, "die_width_mm": 2.95, "shape": "square", "footprint_x_mm": 5.0, "footprint_y_mm": 5.0, "height_mm": 1.0, "max_current_amps": 10.0, "vf_turn_on_v": 4.286, "vf_scale": 1.292, "base_efficacy_lm_w": 124.3, "droop_factor": 0.0908, "dome_size_mm": 0.0, "refractive_index": 1.0},
#   "XHP70.3_HI_4000K": {"die_length_mm": 4.0,  "die_width_mm": 4.0,  "shape": "square", "footprint_x_mm": 7.0, "footprint_y_mm": 7.0, "height_mm": 1.0, "max_current_amps": 8.0, "vf_turn_on_v": 4.821, "vf_scale": 0.714, "base_efficacy_lm_w": 156.0, "droop_factor": 0.0450, "dome_size_mm": 0.0, "refractive_index": 1.0},
}

REFLECTORS = {
#   "T2_T3":            {"diameter_mm": 17.9, "height_mm": 12.2, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 5.1, "gasket_thickness_mm": 0.5, "focus_offset_mm": 0.0, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.0},
    "S2plus_S3_T4":     {"diameter_mm": 20.0, "height_mm": 12.0, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 7.1, "gasket_thickness_mm": 1.0, "focus_offset_mm": 0.0, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.0},
#   "FC11C":            {"diameter_mm": 20.0, "height_mm": 10.0, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 0.0, "gasket_thickness_mm": 0.5, "focus_offset_mm": 0.27, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.0},
    "S2_S6_S8_T6":      {"diameter_mm": 21.0, "height_mm": 20.0, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 7.0, "gasket_thickness_mm": 0.95, "gasket_total_height_mm": 1.6, "gasket_opening_mm": 5.3, "focus_offset_mm": 0.0, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.0},
#   "S2_S6_S8_T6":      {"diameter_mm": 21.0, "height_mm": 20.0, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 5.2, "gasket_thickness_mm": 1.1, "gasket_total_height_mm": 1.1, "gasket_opening_mm": 0.0, "focus_offset_mm": 0.0, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.0},
    "S21A_B_E":         {"diameter_mm": 23.1, "height_mm": 11.6, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 7.1, "gasket_thickness_mm": 0.7, "focus_offset_mm": 0.0, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.0},
    "S16":              {"diameter_mm": 28.0, "height_mm": 20.0, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 7.1, "gasket_thickness_mm": 0.7, "focus_offset_mm": 0.0, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.0},
    "S11":              {"diameter_mm": 28.5, "height_mm": 20.7, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 7.1, "gasket_thickness_mm": 0.7, "focus_offset_mm": 0.0, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.0},
#   "M2":               {"diameter_mm": 28.9, "height_mm": 22.8, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 7.1, "gasket_thickness_mm": 1.0, "focus_offset_mm": 0.0, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.0},
#   "TS28":             {"diameter_mm": 30.5, "height_mm": 21.2, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 0.0, "gasket_thickness_mm": 0.5, "focus_offset_mm": 0.0, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.0},
    "M1_M21B":          {"diameter_mm": 31.8, "height_mm": 23.2, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 7.1, "gasket_thickness_mm": 1.0, "focus_offset_mm": 0.0, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.0},
#   "M21F":             {"diameter_mm": 33.1, "height_mm": 25.1, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 9.1, "gasket_thickness_mm": 1.0, "focus_offset_mm": 0.0, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.0},
#   "TS23":             {"diameter_mm": 36.4, "height_mm": 23.4, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 0.0, "gasket_thickness_mm": 1.0, "focus_offset_mm": 0.0, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.0},
    "C8_M21A_E":        {"diameter_mm": 42.0, "height_mm": 31.5, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 7.1, "gasket_thickness_mm": 1.0, "focus_offset_mm": 0.0, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.0},
#   "C8L":              {"diameter_mm": 43.6, "height_mm": 32.3, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 0.0, "gasket_thickness_mm": 1.0, "focus_offset_mm": 0.0, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.0},
#   "M3":               {"diameter_mm": 43.9, "height_mm": 30.8, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 11.1, "gasket_thickness_mm": 1.0, "focus_offset_mm": 0.0, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.0},
    "M21C_D_G_M26C":    {"diameter_mm": 45.9, "height_mm": 39.7, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 9.1, "gasket_thickness_mm": 1.0, "focus_offset_mm": 0.0, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.0},
    "L21A_B":           {"diameter_mm": 58.0, "height_mm": 50.2, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 9.1, "gasket_thickness_mm": 2.0, "focus_offset_mm": 0.0, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.0},
#   "4x18":             {"diameter_mm": 85.5, "height_mm": 60.8, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 11.1, "gasket_thickness_mm": 2.0, "focus_offset_mm": 0.0, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.0},
    "L6":               {"diameter_mm": 67.6, "height_mm": 47.8, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 11.1, "gasket_thickness_mm": 1.0, "focus_offset_mm": 0.0, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.1},
    "L7":               {"diameter_mm": 67.6, "height_mm": 47.8, "thickness_diameter_mm": 2.0, "thickness_height_mm": 0.1, "opening_diameter_mm": 13.1, "gasket_thickness_mm": 2.0, "focus_offset_mm": -0.5, "reflectivity_smooth": 0.85, "reflectivity_op": 0.80, "OP_Factor": 1.1}

}

# ==============================================================================
# 3. STANDARDIZED OPTICAL PROFILE INTERPOLATOR & LUMEN CALC
# ==============================================================================
def get_standard_emitter_intensity_vec(theta_rad):
    """Calculates the relative angular intensity of an emitter.

    Uses a standard Lambertian (cosine) emission curve to determine how bright
    the emitter appears from a given viewing angle.

    Args:
        theta_rad (np.ndarray): The viewing angles in radians.

    Returns:
        np.ndarray: The relative intensity at each provided angle.
    """
    abs_angle = np.abs(np.degrees(theta_rad))
    intensity = np.cos(theta_rad)
    intensity[abs_angle > 90.0] = 0.0
    return intensity

def calculate_lumens(emitter, current_amps):
    """Calculates the true theoretical lumen output using diode physics.

    Voltage is modeled as a logarithmic curve to physically guarantee safe 
    scaling during extreme overdrive scenarios:
    V(I) = V_turn_on + V_scale * ln(I + 1)

    Args:
        emitter (dict): The dictionary containing the emitter's specifications.
        current_amps (float): The drive current in amperes.

    Returns:
        float: The calculated total lumen output.
    """
    voltage = emitter["vf_turn_on_v"] + (emitter["vf_scale"] * np.log(current_amps + 1.0))
    power_watts = current_amps * voltage
    
    # Apply efficiency droop adjustments to the base efficacy.
    efficiency = emitter["base_efficacy_lm_w"] * np.exp(-emitter["droop_factor"] * current_amps)
    
    return power_watts * efficiency

# ==============================================================================
# 4. FULL FINITE ELEMENT ANALYSIS (FEA) ENGINE
# ==============================================================================

@njit
def process_single_ray(ex, ey, ez_base, vx, vy, vz, flux,
                       focal_length, z_bottom, z_min_cut, z_hole_top, z_max_cut,
                       radius_max, r_hole, target_z_mm, grid_res, wall_radius_m,
                       reflectivity_parabola, reflectivity_cylinder, reflectivity_gasket,
                       dome_radius, refractive_index, max_multiple_reflections,
                       z_gasket_top, r_gasket, gasket_x_half, gasket_y_half, is_cylindrical_gasket):
    """Device-agnostic math: Processes exactly ONE ray and returns its final landing spot.
    
    Returns:
        tuple: (final_flux, row, col, bounce_count) 
        If the ray is absorbed or trapped, returns (0.0, -1, -1, -1).
    """
    ez_current = ez_base
    blocked = False

    # Calculate Snell's law for dome refraction.
    if dome_radius > 0.0:
        # Define local space where the center of the die base is at (0, 0, 0).
        lx = ex
        ly = ey
        
        # Perform ray-sphere intersection.
        P_sq = lx**2 + ly**2
        c = P_sq - dome_radius**2
        b = 2.0 * (lx * vx + ly * vy)
        a = 1.0  # Normalized vector magnitude is 1.0.
        
        discriminant = b**2 - 4.0 * a * c
        if discriminant >= 0.0:
            t = (-b + math.sqrt(discriminant)) / 2.0
            if t > 0.0:
                # Find intersection point on the dome surface.
                hx = lx + t * vx
                hy = ly + t * vy
                hz = t * vz
                
                # Determine normal vector pointing outward from the sphere.
                nx = hx / dome_radius
                ny = hy / dome_radius
                nz = hz / dome_radius
                
                # Apply Snell's law in vector form.
                c1 = vx * nx + vy * ny + vz * nz  # Cosine of incident angle.
                r = refractive_index / 1.0  # Ratio of n_silicone to n_air.
                
                # Check for total internal reflection (TIR).
                tir_check = 1.0 - r**2 * (1.0 - c1**2)
                
                if tir_check >= 0.0:
                    c2 = math.sqrt(tir_check)
                    # Calculate bent ray direction.
                    vx = r * vx - (r * c1 - c2) * nx
                    vy = r * vy - (r * c1 - c2) * ny
                    vz = r * vz - (r * c1 - c2) * nz
                    
                    # Re-normalize to fix floating point drift.
                    mag = math.sqrt(vx**2 + vy**2 + vz**2)
                    vx /= mag
                    vy /= mag
                    vz /= mag
                    
                    # Update origin point to the dome surface.
                    ex = hx
                    ey = hy
                    ez_current += hz
                else:
                    # The ray is trapped by total internal reflection.
                    blocked = True

    # Since the parabola is constructed dynamically matching the focus offset,
    # ez_base is correctly placed in absolute space relative to the parabola vertex.
    current_ex = ex
    current_ey = ey
    current_ez = ez_current
    current_vx = vx
    current_vy = vy
    current_vz = vz
    current_flux = flux
    
    bounce_count = 0
    bin_size = (2.0 * wall_radius_m) / grid_res
    
    while not blocked:
        hit_type = 0  # 0: none, 1: parabola, 2: cylinder walls, 3: horizontal plane, 4: gasket walls
        t_hit = 1e9
        
        # 1. Check Cylinder Intersection (Walls of the hole)
        a_cyl = current_vx**2 + current_vy**2
        if a_cyl > 1e-8:
            b_cyl = 2.0 * (current_ex * current_vx + current_ey * current_vy)
            c_cyl = current_ex**2 + current_ey**2 - r_hole**2
            disc_cyl = b_cyl**2 - 4.0 * a_cyl * c_cyl
            
            if disc_cyl >= 0.0:
                sqrt_disc_cyl = math.sqrt(disc_cyl)
                t_c1 = (-b_cyl - sqrt_disc_cyl) / (2.0 * a_cyl)
                t_c2 = (-b_cyl + sqrt_disc_cyl) / (2.0 * a_cyl)
                
                if 1e-4 < t_c1 < t_hit:
                    z_c1 = current_ez + t_c1 * current_vz
                    if z_bottom <= z_c1 <= z_hole_top:
                        t_hit = t_c1
                        hit_type = 2
                        
                if 1e-4 < t_c2 < t_hit:
                    z_c2 = current_ez + t_c2 * current_vz
                    if z_bottom <= z_c2 <= z_hole_top:
                        t_hit = t_c2
                        hit_type = 2

        # 2. Check Horizontal Plane Intersections (Absorbing surfaces)
        if current_vz < 0.0:
            # Check up-facing shelf first if it exists
            if z_hole_top == z_min_cut and current_ez > z_min_cut:
                t_plane = (z_min_cut - current_ez) / current_vz
                if 1e-4 < t_plane < t_hit:
                    rx_plane = current_ex + t_plane * current_vx
                    ry_plane = current_ey + t_plane * current_vy
                    if (rx_plane**2 + ry_plane**2) > (r_hole**2):
                        t_hit = t_plane
                        hit_type = 3
                        
            # Check top of gasket
            if z_gasket_top > z_bottom and current_ez > z_gasket_top:
                t_plane = (z_gasket_top - current_ez) / current_vz
                if 1e-4 < t_plane < t_hit:
                    rx_plane = current_ex + t_plane * current_vx
                    ry_plane = current_ey + t_plane * current_vy
                    if is_cylindrical_gasket == 1:
                        if (rx_plane**2 + ry_plane**2) >= (r_gasket**2):
                            t_hit = t_plane
                            hit_type = 4
                    else:
                        if abs(rx_plane) >= gasket_x_half or abs(ry_plane) >= gasket_y_half:
                            t_hit = t_plane
                            hit_type = 4
            
            # Check bottom plane (PCB/Gasket) inside or outside the hole
            if current_ez > z_bottom:
                t_plane = (z_bottom - current_ez) / current_vz
                if 1e-4 < t_plane < t_hit:
                    t_hit = t_plane
                    hit_type = 3

        elif current_vz > 0.0 and current_ez < z_bottom:
            # Upward moving rays hitting bottom of reflector outside hole
            t_plane = (z_bottom - current_ez) / current_vz
            if 1e-4 < t_plane < t_hit:
                rx_plane = current_ex + t_plane * current_vx
                ry_plane = current_ey + t_plane * current_vy
                if (rx_plane**2 + ry_plane**2) > (r_hole**2):
                    t_hit = t_plane
                    hit_type = 3
                
        # 3. Check Paraboloid Intersection
        a_par = current_vx**2 + current_vy**2
        b_par = 2.0 * (current_ex * current_vx + current_ey * current_vy) - 4.0 * focal_length * current_vz
        c_par = current_ex**2 + current_ey**2 - 4.0 * focal_length * current_ez
        disc_par = b_par**2 - 4.0 * a_par * c_par
        
        if a_par > 0.0 and disc_par >= 0.0:
            sqrt_disc_par = math.sqrt(disc_par)
            t_p1 = (-b_par - sqrt_disc_par) / (2.0 * a_par)
            t_p2 = (-b_par + sqrt_disc_par) / (2.0 * a_par)
            
            if 1e-4 < t_p1 < t_hit:
                z_p1 = current_ez + t_p1 * current_vz
                if z_hole_top <= z_p1 <= z_max_cut:
                    t_hit = t_p1
                    hit_type = 1
                    
            if 1e-4 < t_p2 < t_hit:
                z_p2 = current_ez + t_p2 * current_vz
                if z_hole_top <= z_p2 <= z_max_cut:
                    t_hit = t_p2
                    hit_type = 1
                    
        # 4. Check Gasket Wall Intersections
        if z_gasket_top > z_bottom:
            if is_cylindrical_gasket == 1:
                a_g = current_vx**2 + current_vy**2
                if a_g > 1e-8:
                    b_g = 2.0 * (current_ex * current_vx + current_ey * current_vy)
                    c_g = current_ex**2 + current_ey**2 - r_gasket**2
                    disc_g = b_g**2 - 4.0 * a_g * c_g
                    
                    if disc_g >= 0.0:
                        t_g = (-b_g + math.sqrt(disc_g)) / (2.0 * a_g)
                        if 1e-4 < t_g < t_hit:
                            z_g = current_ez + t_g * current_vz
                            if z_g <= z_gasket_top:
                                t_hit = t_g
                                hit_type = 4
            else:
                if abs(current_vx) > 1e-8:
                    sign_x = 1.0 if current_vx > 0.0 else -1.0
                    t_x = (gasket_x_half * sign_x - current_ex) / current_vx
                    if 1e-4 < t_x < t_hit:
                        z_x = current_ez + t_x * current_vz
                        y_x = current_ey + t_x * current_vy
                        if z_x <= z_gasket_top and -gasket_y_half <= y_x <= gasket_y_half:
                            t_hit = t_x
                            hit_type = 4
                if abs(current_vy) > 1e-8:
                    sign_y = 1.0 if current_vy > 0.0 else -1.0
                    t_y = (gasket_y_half * sign_y - current_ey) / current_vy
                    if 1e-4 < t_y < t_hit:
                        z_y = current_ez + t_y * current_vz
                        x_y = current_ex + t_y * current_vx
                        if z_y <= z_gasket_top and -gasket_x_half <= x_y <= gasket_x_half:
                            t_hit = t_y
                            hit_type = 4

        # Process the confirmed collision if any
        if hit_type == 1 or hit_type == 2 or hit_type == 4:
            if bounce_count >= max_multiple_reflections + 1:
                return 0.0, -1, -1, -1
                
            bounce_count += 1
            
            rx = current_ex + t_hit * current_vx
            ry = current_ey + t_hit * current_vy
            rz = current_ez + t_hit * current_vz
            
            if hit_type == 1:
                nx = -rx
                ny = -ry
                nz = 2.0 * focal_length
                current_refl = reflectivity_parabola
            elif hit_type == 2:
                nx = -rx
                ny = -ry
                nz = 0.0
                current_refl = reflectivity_cylinder
            elif hit_type == 4:
                if abs(rz - z_gasket_top) < 1e-4 and current_vz < 0.0:
                    nx = 0.0
                    ny = 0.0
                    nz = 1.0
                else:
                    if is_cylindrical_gasket == 1:
                        nx = -rx
                        ny = -ry
                        nz = 0.0
                    else:
                        if abs(abs(rx) - gasket_x_half) < 1e-4:
                            nx = -1.0 if rx > 0.0 else 1.0
                            ny = 0.0
                            nz = 0.0
                        else:
                            nx = 0.0
                            ny = -1.0 if ry > 0.0 else 1.0
                            nz = 0.0
                current_refl = reflectivity_gasket
                
            mag = math.sqrt(nx**2 + ny**2 + nz**2)
            nx /= mag
            ny /= mag
            nz /= mag
            
            dot = current_vx * nx + current_vy * ny + current_vz * nz
            current_vx = current_vx - 2.0 * dot * nx
            current_vy = current_vy - 2.0 * dot * ny
            current_vz = current_vz - 2.0 * dot * nz
            
            current_ex = rx
            current_ey = ry
            current_ez = rz
            current_flux *= current_refl
            
        elif hit_type == 3:
            return 0.0, -1, -1, -1
            
        else:
            if current_vz > 0.0:
                dist_z = z_max_cut - current_ez
                esc_x = current_ex + (dist_z / current_vz) * current_vx
                esc_y = current_ey + (dist_z / current_vz) * current_vy
                
                if math.sqrt(esc_x**2 + esc_y**2) <= radius_max + 1e-4:
                    s = (target_z_mm - current_ez) / current_vz
                    hx = (current_ex + s * current_vx) / 1000.0
                    hy = (current_ey + s * current_vy) / 1000.0
                    
                    col = int((hx + wall_radius_m) / bin_size)
                    row = int((hy + wall_radius_m) / bin_size)
                    
                    if 0 <= col < grid_res and 0 <= row < grid_res:
                        return current_flux, row, col, bounce_count
            return 0.0, -1, -1, -1

    return 0.0, -1, -1, -1

@cuda.jit
def ray_trace_kernel_gpu(ex_arr, ey_arr, vx_arr, vy_arr, vz_arr, flux_arr,
                         focal_length, ez_base, z_bottom, z_min_cut, z_hole_top, z_max_cut, radius_max, r_hole,
                         target_z_mm, grid_res, wall_radius_m, reflectivity_parabola, reflectivity_cylinder, reflectivity_gasket,
                         dome_radius, refractive_index, max_multiple_reflections,
                         z_gasket_top, r_gasket, gasket_x_half, gasket_y_half, is_cylindrical_gasket,
                         hotspot_grid, spill_grid, start_idx, end_idx):
    """GPU Wrapper: Handles threading, offsets, and atomic memory writes."""
    idx = cuda.grid(1) + start_idx
    
    if idx >= end_idx:
        return
        
    total_rays = vx_arr.shape[0]
    element_idx = idx // total_rays
    ray_idx = idx % total_rays

    final_flux, row, col, bounces = process_single_ray(
        ex_arr[element_idx], ey_arr[element_idx], ez_base,
        vx_arr[ray_idx], vy_arr[ray_idx], vz_arr[ray_idx], flux_arr[ray_idx],
        focal_length, z_bottom, z_min_cut, z_hole_top, z_max_cut,
        radius_max, r_hole, target_z_mm, grid_res, wall_radius_m,
        reflectivity_parabola, reflectivity_cylinder, reflectivity_gasket,
        dome_radius, refractive_index, max_multiple_reflections,
        z_gasket_top, r_gasket, gasket_x_half, gasket_y_half, is_cylindrical_gasket
    )

    if row != -1 and col != -1:
        if bounces > 0:
            cuda.atomic.add(hotspot_grid, (row, col), final_flux)
        else:
            cuda.atomic.add(spill_grid, (row, col), final_flux)

@njit
def ray_trace_kernel_cpu(ex_arr, ey_arr, vx_arr, vy_arr, vz_arr, flux_arr,
                         focal_length, ez_base, z_bottom, z_min_cut, z_hole_top, z_max_cut, radius_max, r_hole,
                         target_z_mm, grid_res, wall_radius_m, reflectivity_parabola, reflectivity_cylinder, reflectivity_gasket,
                         dome_radius, refractive_index, max_multiple_reflections,
                         z_gasket_top, r_gasket, gasket_x_half, gasket_y_half, is_cylindrical_gasket,
                         hotspot_grid, spill_grid, start_idx, end_idx):
    """CPU Wrapper: Handles standard serial looping and offsets."""
    total_rays = vx_arr.shape[0]
    
    for idx in range(start_idx, end_idx):
        element_idx = idx // total_rays
        ray_idx = idx % total_rays
        
        final_flux, row, col, bounces = process_single_ray(
            ex_arr[element_idx], ey_arr[element_idx], ez_base,
            vx_arr[ray_idx], vy_arr[ray_idx], vz_arr[ray_idx], flux_arr[ray_idx],
            focal_length, z_bottom, z_min_cut, z_hole_top, z_max_cut,
            radius_max, r_hole, target_z_mm, grid_res, wall_radius_m,
            reflectivity_parabola, reflectivity_cylinder, reflectivity_gasket,
            dome_radius, refractive_index, max_multiple_reflections,
            z_gasket_top, r_gasket, gasket_x_half, gasket_y_half, is_cylindrical_gasket
        )

        if row != -1 and col != -1:
            if bounces > 0:
                hotspot_grid[row, col] += final_flux
            else:
                spill_grid[row, col] += final_flux


def run_pure_fea_sim_vectorized(reflector, emitter, current_amps, target_distance_m=5.0, wall_radius_m=5.0, finish="smooth"):
    """Runs the Numba accelerated finite element analysis engine.

    Args:
        reflector (dict): The selected reflector's hardware specifications.
        emitter (dict): The selected emitter's hardware specifications.
        current_amps (float): The active drive current.
        target_distance_m (float): Distance to the simulated wall in meters.
        wall_radius_m (float): The physical radius of the simulated wall capture plane.
        finish (str): The reflector finish, either "smooth" or "orange_peel".

    Returns:
        tuple: Contains final_lux_grid, processed_hotspot_lux, spill_lux, and total_lumens.
    """
    D = reflector["diameter_mm"] - reflector.get("thickness_diameter_mm", default_reflector_wall_thickness_mm)
    H_total = reflector["height_mm"]  # Do NOT subtract base thickness here.
    
    R = D / 2.0
    
    footprint_diag = math.sqrt(emitter["footprint_x_mm"]**2 + emitter["footprint_y_mm"]**2)
    d_hole_input = reflector.get("opening_diameter_mm", 0.0)
    
    focus_offset_mm = reflector.get("focus_offset_mm", default_focus_offset_mm)
    thickness_height_mm = reflector.get("thickness_height_mm", default_reflector_base_thickness_mm)
    
    gasket_thickness_mm = reflector.get("gasket_thickness_mm", default_gasket_thickness_mm)
    gasket_total_height_mm = reflector.get("gasket_total_height_mm", default_gasket_total_height_mm)
    gasket_opening_mm = reflector.get("gasket_opening_mm", default_gasket_opening_mm)

    # Calculate focal length for the FULL parabola constrained by top diameter and total height.
    H_eff = H_total - focus_offset_mm
    focal_length = (-H_eff + math.sqrt(H_eff**2 + R**2)) / 2.0
    
    # Establish absolute Z-space physical planes.
    z_bottom = focal_length - focus_offset_mm      # Absolute bottom of the physical reflector.
    z_min_cut = z_bottom + thickness_height_mm     # Where the flat shelf is located.
    z_max_cut = z_bottom + H_total                 # Top of the reflector.
    radius_max = R

    # Setup Gasket Parameters
    h_gasket_ext = max(0.0, gasket_total_height_mm - gasket_thickness_mm)
    z_gasket_top = z_bottom + h_gasket_ext
    
    if gasket_opening_mm > 0.0:
        r_gasket = gasket_opening_mm / 2.0
        gasket_x_half = 0.0
        gasket_y_half = 0.0
        is_cylindrical_gasket = 1
    else:
        r_gasket = 0.0
        gasket_x_half = emitter["footprint_x_mm"] / 2.0
        gasket_y_half = emitter["footprint_y_mm"] / 2.0
        is_cylindrical_gasket = 0

    # Apply physical constraints to the opening size.
    if use_reflector_opening:
        effective_d_hole = d_hole_input
    else:
        effective_d_hole = max(d_hole_input, footprint_diag)
            
    r_hole = effective_d_hole / 2.0

    # Calculate where the hole cylinder mathematically meets the parabola.
    z_intersect = (r_hole**2) / (4.0 * focal_length)

    # If the parabola is smaller than the hole at the shelf line, the cylinder cuts into the parabola
    # meaning there is no shelf and the cylinder walls extend up to z_intersect.
    if z_intersect > z_min_cut:
        z_hole_top = float(z_intersect)
    else:
        z_hole_top = float(z_min_cut)
    
    # Extract dimensions along the X and Y axes.
    die_length = emitter["die_length_mm"]
    if emitter.get("shape") == "round":
        die_width = die_length
    else:
        die_width = emitter["die_width_mm"]
    
    # Map the physical stack-up to math coordinates relative to the bottom of the reflector.
    ez_base = z_bottom + (emitter["height_mm"] - gasket_thickness_mm)
    
    total_lumens = calculate_lumens(emitter, current_amps)
    
    theta_int = np.radians(np.arange(0, 90, lumen_calc_step_deg))
    intensity_vec = get_standard_emitter_intensity_vec(theta_int)
    N_integral = np.sum(intensity_vec * np.sin(theta_int) * np.radians(lumen_calc_step_deg))
    I_peak_base = total_lumens / (2 * np.pi * N_integral)
    
    grid_res = sim_grid_res
    pixel_area_m2 = (2.0 * wall_radius_m / grid_res) ** 2
    
    emitter_elements = sim_emitter_elements
    
    # Calculate dome parameters.
    dome_input = emitter.get("dome_size_mm", 0.0)
    if dome_input == -1:
        # Default to the size of the shortest side of the footprint.
        dome_diameter = min(emitter["footprint_x_mm"], emitter["footprint_y_mm"])
    else:
        dome_diameter = max(0.0, dome_input)
        
    dome_radius = dome_diameter / 2.0
    refractive_index = emitter.get("refractive_index", 1.0)
    
    # Create spatial arrays natively supporting rectangles.
    ex_1d = np.linspace(-die_length/2, die_length/2, emitter_elements)
    ey_1d = np.linspace(-die_width/2, die_width/2, emitter_elements)
    EX, EY = np.meshgrid(ex_1d, ey_1d)
    
    # Filter coordinates based on the emitter shape.
    if emitter.get("shape") == "round":
        mask = (EX**2 + EY**2) <= (die_length / 2.0)**2
        ex_flat = np.ascontiguousarray(EX[mask], dtype=np.float64)
        ey_flat = np.ascontiguousarray(EY[mask], dtype=np.float64)
    else:
        ex_flat = np.ascontiguousarray(EX.flatten(), dtype=np.float64)
        ey_flat = np.ascontiguousarray(EY.flatten(), dtype=np.float64)
        
    actual_elements = len(ex_flat)
    
    # Create angular arrays bounded by new global fallbacks.
    theta_1d = np.radians(np.arange(sim_theta_min_deg, sim_theta_max_deg, sim_theta_step_deg))
    phi_1d = np.radians(np.arange(sim_phi_min_deg, sim_phi_max_deg, sim_phi_step_deg))
    THETA, PHI = np.meshgrid(theta_1d, phi_1d)
    THETA_flat = THETA.flatten()
    PHI_flat = PHI.flatten()
    
    angular_intensity = get_standard_emitter_intensity_vec(THETA_flat)
    solid_angle = np.sin(THETA_flat) * np.radians(sim_theta_step_deg) * np.radians(sim_phi_step_deg)
    
    # Distribute the total theoretical flux evenly across the actual emitting area.
    ray_flux = np.ascontiguousarray((I_peak_base * angular_intensity * solid_angle) / actual_elements, dtype=np.float64)
    
    vx = np.ascontiguousarray(np.sin(THETA_flat) * np.cos(PHI_flat), dtype=np.float64)
    vy = np.ascontiguousarray(np.sin(THETA_flat) * np.sin(PHI_flat), dtype=np.float64)
    vz = np.ascontiguousarray(np.cos(THETA_flat), dtype=np.float64)

    # Use the appropriate global reflectivity fallback based on finish.
    reflectivity_parabola = reflector.get("reflectivity_op", default_reflectivity_op) if finish == "orange_peel" else reflector.get("reflectivity_smooth", default_reflectivity_smooth)
    reflectivity_cylinder = reflector.get("reflectivity_cylinder", default_reflectivity_cylinder)
    reflectivity_gasket = reflector.get("reflectivity_gasket", default_reflectivity_gasket)

    target_z_mm = target_distance_m * 1000.0

    total_threads = len(ex_flat) * len(vx)
    has_gpu = cuda.is_available()

    if has_gpu:
        device = cuda.get_current_device()
        print(f"\n[CUDA FEA Engine] GPU Detected: {device.name.decode('utf-8')}")
        print(f"[CUDA FEA Engine] Pushing {total_threads:,} rays to VRAM...")

        d_ex = cuda.to_device(ex_flat)
        d_ey = cuda.to_device(ey_flat)
        d_vx = cuda.to_device(vx)
        d_vy = cuda.to_device(vy)
        d_vz = cuda.to_device(vz)
        d_flux = cuda.to_device(ray_flux)
        
        d_hotspot = cuda.to_device(np.zeros((grid_res, grid_res), dtype=np.float64))
        d_spill = cuda.to_device(np.zeros((grid_res, grid_res), dtype=np.float64))

        threads_per_block = 256
        
        if total_threads > 1:
            # 1. WARMUP: Process exactly 1 ray to force Numba to JIT compile the kernel.
            print("[CUDA FEA Engine] Compiling kernel (Warmup)...", end="", flush=True)
            ray_trace_kernel_gpu[1, 1](
                d_ex, d_ey, d_vx, d_vy, d_vz, d_flux,
                float(focal_length), float(ez_base), float(z_bottom), float(z_min_cut), float(z_hole_top), float(z_max_cut), 
                float(radius_max), float(r_hole), float(target_z_mm), 
                int(grid_res), float(wall_radius_m), float(reflectivity_parabola), float(reflectivity_cylinder), float(reflectivity_gasket),
                float(dome_radius), float(refractive_index), int(max_multiple_reflections),
                float(z_gasket_top), float(r_gasket), float(gasket_x_half), float(gasket_y_half), int(is_cylindrical_gasket),
                d_hotspot, d_spill, 0, 1
            )
            cuda.synchronize()
            print(" Done.")
            
            # 2. CALIBRATION: Process ~2% of the rays to benchmark actual hardware speed.
            cal_size = min(max(int(total_threads * 0.02), 250_000), total_threads - 1)
            blocks_cal = (cal_size + (threads_per_block - 1)) // threads_per_block
            
            print(f"[CUDA FEA Engine] Calibrating hardware performance...", end="", flush=True)
            t0 = time.time()
            ray_trace_kernel_gpu[blocks_cal, threads_per_block](
                d_ex, d_ey, d_vx, d_vy, d_vz, d_flux,
                float(focal_length), float(ez_base), float(z_bottom), float(z_min_cut), float(z_hole_top), float(z_max_cut), 
                float(radius_max), float(r_hole), float(target_z_mm), 
                int(grid_res), float(wall_radius_m), float(reflectivity_parabola), float(reflectivity_cylinder), float(reflectivity_gasket),
                float(dome_radius), float(refractive_index), int(max_multiple_reflections),
                float(z_gasket_top), float(r_gasket), float(gasket_x_half), float(gasket_y_half), int(is_cylindrical_gasket),
                d_hotspot, d_spill, 1, 1 + cal_size
            )
            cuda.synchronize()
            t1 = time.time()
            
            cal_time = t1 - t0
            rays_per_sec = cal_size / cal_time if cal_time > 0 else 1
            remaining_threads = total_threads - (1 + cal_size)
            est_time = remaining_threads / rays_per_sec
            
            print(f" Done. ({rays_per_sec:,.0f} rays/sec)")
            print(f"[CUDA FEA Engine] Predicted remaining time: ~{est_time:.2f} seconds")
            
            # 3. MAIN EXECUTION: Process the rest of the rays.
            if remaining_threads > 0:
                blocks_main = (remaining_threads + (threads_per_block - 1)) // threads_per_block
                ray_trace_kernel_gpu[blocks_main, threads_per_block](
                    d_ex, d_ey, d_vx, d_vy, d_vz, d_flux,
                    float(focal_length), float(ez_base), float(z_bottom), float(z_min_cut), float(z_hole_top), float(z_max_cut), 
                    float(radius_max), float(r_hole), float(target_z_mm), 
                    int(grid_res), float(wall_radius_m), float(reflectivity_parabola), float(reflectivity_cylinder), float(reflectivity_gasket),
                    float(dome_radius), float(refractive_index), int(max_multiple_reflections),
                    float(z_gasket_top), float(r_gasket), float(gasket_x_half), float(gasket_y_half), int(is_cylindrical_gasket),
                    d_hotspot, d_spill, 1 + cal_size, total_threads
                )
                cuda.synchronize()
        else:
            # Fallback if grid is trivially small
            ray_trace_kernel_gpu[1, 1](
                d_ex, d_ey, d_vx, d_vy, d_vz, d_flux,
                float(focal_length), float(ez_base), float(z_bottom), float(z_min_cut), float(z_hole_top), float(z_max_cut), 
                float(radius_max), float(r_hole), float(target_z_mm), 
                int(grid_res), float(wall_radius_m), float(reflectivity_parabola), float(reflectivity_cylinder), float(reflectivity_gasket),
                float(dome_radius), float(refractive_index), int(max_multiple_reflections),
                float(z_gasket_top), float(r_gasket), float(gasket_x_half), float(gasket_y_half), int(is_cylindrical_gasket),
                d_hotspot, d_spill, 0, total_threads
            )
            cuda.synchronize()

        hotspot_grid = d_hotspot.copy_to_host()
        spill_grid = d_spill.copy_to_host()
        print("[CUDA FEA Engine] Ray tracing complete. Applying spatial blur...\n")

    else:
        print(f"\n[CPU FEA Engine] No GPU Detected. Falling back to CPU ({os.cpu_count()} logical cores)...")

        hotspot_grid = np.zeros((grid_res, grid_res), dtype=np.float64)
        spill_grid = np.zeros((grid_res, grid_res), dtype=np.float64)

        if total_threads > 1:
            # 1. WARMUP
            print("[CPU FEA Engine] Compiling kernel (Warmup)...", end="", flush=True)
            ray_trace_kernel_cpu(
                ex_flat, ey_flat, vx, vy, vz, ray_flux,
                float(focal_length), float(ez_base), float(z_bottom), float(z_min_cut), float(z_hole_top), float(z_max_cut), 
                float(radius_max), float(r_hole), float(target_z_mm), 
                int(grid_res), float(wall_radius_m), float(reflectivity_parabola), float(reflectivity_cylinder), float(reflectivity_gasket),
                float(dome_radius), float(refractive_index), int(max_multiple_reflections),
                float(z_gasket_top), float(r_gasket), float(gasket_x_half), float(gasket_y_half), int(is_cylindrical_gasket),
                hotspot_grid, spill_grid, 0, 1
            )
            print(" Done.")
            
            # 2. CALIBRATION
            cal_size = min(max(int(total_threads * 0.02), 25_000), total_threads - 1)
            
            print(f"[CPU FEA Engine] Calibrating hardware performance...", end="", flush=True)
            t0 = time.time()
            ray_trace_kernel_cpu(
                ex_flat, ey_flat, vx, vy, vz, ray_flux,
                float(focal_length), float(ez_base), float(z_bottom), float(z_min_cut), float(z_hole_top), float(z_max_cut), 
                float(radius_max), float(r_hole), float(target_z_mm), 
                int(grid_res), float(wall_radius_m), float(reflectivity_parabola), float(reflectivity_cylinder), float(reflectivity_gasket),
                float(dome_radius), float(refractive_index), int(max_multiple_reflections),
                float(z_gasket_top), float(r_gasket), float(gasket_x_half), float(gasket_y_half), int(is_cylindrical_gasket),
                hotspot_grid, spill_grid, 1, 1 + cal_size
            )
            t1 = time.time()
            
            cal_time = t1 - t0
            rays_per_sec = cal_size / cal_time if cal_time > 0 else 1
            remaining_threads = total_threads - (1 + cal_size)
            est_time = remaining_threads / rays_per_sec
            
            print(f" Done. ({rays_per_sec:,.0f} rays/sec)")
            print(f"[CPU FEA Engine] Predicted remaining time: ~{est_time:.2f} seconds")
            
            # 3. MAIN EXECUTION
            if remaining_threads > 0:
                ray_trace_kernel_cpu(
                    ex_flat, ey_flat, vx, vy, vz, ray_flux,
                    float(focal_length), float(ez_base), float(z_bottom), float(z_min_cut), float(z_hole_top), float(z_max_cut), 
                    float(radius_max), float(r_hole), float(target_z_mm), 
                    int(grid_res), float(wall_radius_m), float(reflectivity_parabola), float(reflectivity_cylinder), float(reflectivity_gasket),
                    float(dome_radius), float(refractive_index), int(max_multiple_reflections),
                    float(z_gasket_top), float(r_gasket), float(gasket_x_half), float(gasket_y_half), int(is_cylindrical_gasket),
                    hotspot_grid, spill_grid, 1 + cal_size, total_threads
                )
        else:
            ray_trace_kernel_cpu(
                ex_flat, ey_flat, vx, vy, vz, ray_flux,
                float(focal_length), float(ez_base), float(z_bottom), float(z_min_cut), float(z_hole_top), float(z_max_cut), 
                float(radius_max), float(r_hole), float(target_z_mm), 
                int(grid_res), float(wall_radius_m), float(reflectivity_parabola), float(reflectivity_cylinder), float(reflectivity_gasket),
                float(dome_radius), float(refractive_index), int(max_multiple_reflections),
                float(z_gasket_top), float(r_gasket), float(gasket_x_half), float(gasket_y_half), int(is_cylindrical_gasket),
                hotspot_grid, spill_grid, 0, total_threads
            )

        print("[CPU FEA Engine] Ray tracing complete. Applying spatial blur...\n")

    op_multiplier = reflector.get("OP_Factor", 1.0)
    blur_strength = (default_op_blur_strength * op_multiplier) if finish == "orange_peel" else 0.0
    
    # The gaussian filter works best if it scales proportionally to resolution.
    scaled_blur = blur_strength * (grid_res / 1000.0)
    if scaled_blur > 0:
        processed_hotspot = gaussian_filter(hotspot_grid, sigma=scaled_blur)
    else:
        processed_hotspot = hotspot_grid
        
    # Scale grids individually so we can analyze them later.
    processed_hotspot_lux = processed_hotspot / pixel_area_m2
    spill_lux = spill_grid / pixel_area_m2
    final_lux_grid = processed_hotspot_lux + spill_lux
    
    return final_lux_grid, processed_hotspot_lux, spill_lux, total_lumens

# ==============================================================================
# 5. PLOTTING & BATCH EXPORT MANAGER
# ==============================================================================
def generate_flashlight_plot(emitter_name, reflector_name, finish_type, save_path=None):
    """Generates the requested plot images for a hardware combination and extracts metrics.

    Args:
        emitter_name (str): Key matching the chosen emitter in the EMITTERS dict.
        reflector_name (str): Key matching the chosen reflector in the REFLECTORS dict.
        finish_type (str): The reflector finish type ("smooth" or "orange_peel").
        save_path (str, optional): The file path to save the generated plot. Defaults to None.

    Returns:
        dict: A dictionary containing the simulation metrics logged for CSV output.
    """
    selected_reflector = REFLECTORS[reflector_name]
    selected_emitter = EMITTERS[emitter_name]
    amps = selected_emitter["max_current_amps"]

    # Run the core calculation engine.
    wall_lux, hotspot_lux, spill_lux, total_flux = run_pure_fea_sim_vectorized(
        selected_reflector, 
        selected_emitter, 
        amps, 
        target_distance_m,
        wall_radius_m, 
        finish_type
    )

    # Extract core metrics directly from the FEA output.
    max_cd = np.max(wall_lux) * (target_distance_m**2)
    throw_m = int(np.sqrt(max_cd / 0.25))

    # --- VISUALIZATION PROCESSING PIPELINE (PHOTOREALISTIC CAMERA SIMULATION) ---
    if use_auto_exposure:
        # Define "proper exposure" based on the 99.5th percentile to ignore extreme peak pixels.
        auto_exposure_target = np.percentile(wall_lux, 99.5)
        if auto_exposure_target == 0:
            auto_exposure_target = 1.0  # Prevent division by zero.
            
        exposed_lux = wall_lux * (1.0 / auto_exposure_target) * (2 ** auto_exposure_compensation_ev)
    else:
        # Calculate standard photographic EV (Exposure Value) for the chosen settings.
        camera_ev = np.log2((cam_f_stop**2) / cam_shutter_speed_s)
        
        # Incident light meter calibration constant (C=250 for flat sensors).
        incident_meter_constant = 250.0 
        
        # Calculate the scene illuminance (Lux) that results in a perfectly exposed middle-gray.
        lux_for_proper_exposure = (incident_meter_constant * (2 ** camera_ev)) / cam_iso
        
        # Scale the simulated wall lux by the camera's sensitivity threshold.
        # Multiply by 0.18 because standard ACES maps 1.0 to white, and mid-gray is 18% reflectance.
        exposed_lux = (wall_lux / lux_for_proper_exposure) * 0.18
        
    # Apply ACES filmic tone mapping curve.
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    
    mapped_data = (exposed_lux * (a * exposed_lux + b)) / (exposed_lux * (c * exposed_lux + d) + e)
    
    # Apply monitor gamma correction and clamp bounds.
    clamped_data = np.clip(mapped_data, 0.0, 1.0)
    render_data = np.power(clamped_data, 1.0 / 2.2) 

    # Heads up display (HUD) calculations for hardware parameters.
    D = selected_reflector["diameter_mm"] - selected_reflector.get("thickness_diameter_mm", default_reflector_wall_thickness_mm)
    H_total = selected_reflector["height_mm"]
    thickness_height_mm = selected_reflector.get("thickness_height_mm", default_reflector_base_thickness_mm)
    R = D / 2.0
    
    d_hole_input = selected_reflector.get("opening_diameter_mm", 0.0)
    footprint_diag = math.sqrt(selected_emitter["footprint_x_mm"]**2 + selected_emitter["footprint_y_mm"]**2)

    # Apply physical constraints to the opening size.
    if use_reflector_opening:
        effective_d_hole = d_hole_input
    else:
        effective_d_hole = max(d_hole_input, footprint_diag)

    focus_offset_mm = selected_reflector.get("focus_offset_mm", default_focus_offset_mm)
    H_eff = H_total - focus_offset_mm
    focal_length = (-H_eff + math.sqrt(H_eff**2 + R**2)) / 2.0
    
    z_bottom = focal_length - focus_offset_mm
    
    physical_emitter_height = selected_emitter["height_mm"] - selected_reflector.get("gasket_thickness_mm", default_gasket_thickness_mm)
    actual_ez = z_bottom + physical_emitter_height
    focus_delta = actual_ez - focal_length

    # Extract beam geometry from the ray-traced grids.
    true_center_idx = (sim_grid_res - 1) / 2.0
    pixel_size_m = (2.0 * wall_radius_m) / sim_grid_res
    pixel_size_mm = pixel_size_m * 1000.0

    def get_max_radius(mask):
        """Finds the physical maximum radius from the center for a given boolean mask.
        
        Args:
            mask (np.ndarray): Boolean mask of pixels satisfying a condition.
            
        Returns:
            float: The maximum physical radius in meters.
        """
        if not np.any(mask): 
            return 0.0
        y_idx, x_idx = np.nonzero(mask)
        distances_px = np.sqrt((x_idx - true_center_idx)**2 + (y_idx - true_center_idx)**2)
        return np.max(distances_px) * pixel_size_m

    # Find the geometric edge where direct unreflected light (spill) hits the wall.
    spill_radius_m = get_max_radius(spill_lux > spill_visible_threshold_lux)
    spill_angle = 2 * np.degrees(np.arctan(spill_radius_m / target_distance_m))
    spill_size_m = 2 * spill_radius_m

    # Find the outer visible bounds of the reflected beam (corona).
    peak_hotspot = np.max(hotspot_lux)
    corona_radius_m = get_max_radius(hotspot_lux > (peak_hotspot * corona_visible_threshold))
    corona_angle = 2 * np.degrees(np.arctan(corona_radius_m / target_distance_m))
    corona_size_m = 2 * corona_radius_m

    # Find the true hotspot extent using standard full width at half maximum (FWHM).
    peak_lux = np.max(wall_lux)
    hotspot_radius_m = get_max_radius(wall_lux >= (peak_lux * hotspot_fwhm_threshold))
    hotspot_angle = 2 * np.degrees(np.arctan(hotspot_radius_m / target_distance_m))
    hotspot_size_m = 2 * hotspot_radius_m

    cd_lm_ratio = max_cd / total_flux

    # Determine the active camera label for the UI footer.
    if use_auto_exposure:
        cam_text = f"Exposure: Auto (EV {auto_exposure_compensation_ev:+.1f})"
    else:
        shutter_str = f"1/{round(1.0/cam_shutter_speed_s)}" if cam_shutter_speed_s < 1.0 else f"{cam_shutter_speed_s}"
        cam_text = f"Exposure: ISO {cam_iso} | f/{cam_f_stop} | {shutter_str}s"

    geo_text = (
        f"Spill Angle: {spill_angle:.1f}°\n"
        f"Spill Ø @ {target_distance_m}m: {spill_size_m:.2f}m\n"
        f"Corona Angle: {corona_angle:.1f}°\n"
        f"Corona Ø @ {target_distance_m}m: {corona_size_m:.2f}m\n"
        f"Hotspot Angle: {hotspot_angle:.1f}°\n"
        f"Hotspot Ø @ {target_distance_m}m: {hotspot_size_m:.2f}m\n"
        f"Cd/Lm Ratio: {cd_lm_ratio:.1f} cd/lm\n"
    )

    # Calculate the bottom-right performance table.
    table_str = " Mode | Amps | Lumens |  Candela | Throw \n" + "-"*46 + "\n"
    for pct in [0.01, 0.10, 0.35, 1.0]:
        amp_val = amps * pct
        lm_mode = calculate_lumens(selected_emitter, amp_val)
        cd_mode = max_cd * (lm_mode / total_flux)
        throw_mode = np.sqrt(cd_mode / 0.25)
        table_str += f"{int(pct*100):>4}% | {amp_val:>4.1f} | {int(lm_mode):>6,} | {int(cd_mode):>8,} | {int(throw_mode):>4,}m\n"

    # Format shared output title for all requested plots.
    title_str = (
        f"Hardware: {emitter_name} | Reflector: {reflector_name} ({finish_type.upper()})\n"
        f"Opening: {effective_d_hole:.1f}mm | Focus Delta: {focus_delta:+.2f}mm | Max Intensity: {int(max_cd):,} cd | Throw: {throw_m:,}m"
    )
    
    footer_text = f"Canvas FOV: {canvas_fov_deg}° | Plot FOV: {plot_fov_deg}° | Grid Res: {pixel_size_mm:.1f} mm/px | [{cam_text}]"

    # -------------------------------------------------------------------------
    # RENDER SELECTED PLOTS
    # -------------------------------------------------------------------------

    if plot_wall_shot:
        fig_wall = plt.figure(figsize=(10, 10), facecolor='black')
        ax_wall = fig_wall.add_subplot(111, facecolor='black')

        # The image data physically spans the full canvas FOV.
        extent_bounds = [-wall_radius_m, wall_radius_m, -wall_radius_m, wall_radius_m]
        ax_wall.imshow(render_data, extent=extent_bounds, cmap='gray', origin='lower', vmin=0, vmax=1)
        
        # Zoom the camera in to exactly match plot_fov_deg.
        ax_wall.set_xlim(-plot_radius_m, plot_radius_m)
        ax_wall.set_ylim(-plot_radius_m, plot_radius_m)

        ax_wall.set_xlabel("Horizontal Distance (m)", color='#CCCCCC', fontsize=11, labelpad=10)
        ax_wall.set_ylabel("Vertical Distance (m)", color='#CCCCCC', fontsize=11, labelpad=10)
        ax_wall.tick_params(colors='#CCCCCC', labelsize=10)
        ax_wall.grid(False)

        for spine in ax_wall.spines.values():
            spine.set_color('#555555')

        if show_human_silhouette:
            # Add a human silhouette for scale reference.
            person_height_m = 1.75 
            person_x = 0.0
            person_y_bottom = -person_height_m * 0.65 

            # Establish proportional anatomical dimensions based on total height.
            h_rad = person_height_m * 0.08
            t_width = person_height_m * 0.25
            t_height = person_height_m * 0.35
            l_width = person_height_m * 0.08
            l_height = person_height_m * 0.45
            a_width = person_height_m * 0.06
            a_height = person_height_m * 0.40

            sil_color = '#FFFF00' 
            sil_alpha = 0.4
            sil_lw = 1.0

            # Build the geometric silhouette.
            ax_wall.add_patch(patches.Circle((person_x, person_y_bottom + l_height + t_height + h_rad), 
                                        h_rad, ec=sil_color, fc='none', alpha=sil_alpha, lw=sil_lw, ls='--'))
            ax_wall.add_patch(patches.Rectangle((person_x - t_width/2, person_y_bottom + l_height), 
                                           t_width, t_height, ec=sil_color, fc='none', alpha=sil_alpha, lw=sil_lw, ls='--'))
            ax_wall.add_patch(patches.Rectangle((person_x - t_width/2, person_y_bottom), 
                                           l_width, l_height, ec=sil_color, fc='none', alpha=sil_alpha, lw=sil_lw, ls='--'))
            ax_wall.add_patch(patches.Rectangle((person_x + t_width/2 - l_width, person_y_bottom), 
                                           l_width, l_height, ec=sil_color, fc='none', alpha=sil_alpha, lw=sil_lw, ls='--'))
            ax_wall.add_patch(patches.Rectangle((person_x - t_width/2 - a_width, person_y_bottom + l_height + t_height - a_height), 
                                           a_width, a_height, ec=sil_color, fc='none', alpha=sil_alpha, lw=sil_lw, ls='--'))
            ax_wall.add_patch(patches.Rectangle((person_x + t_width/2, person_y_bottom + l_height + t_height - a_height), 
                                           a_width, a_height, ec=sil_color, fc='none', alpha=sil_alpha, lw=sil_lw, ls='--'))

        # Add HUD elements inside the plot.
        ax_wall.text(0.02, 0.02, geo_text.strip(), transform=ax_wall.transAxes, color='#CCCCCC', 
                fontsize=10, verticalalignment='bottom', 
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=6))

        ax_wall.text(0.98, 0.02, table_str.strip(), transform=ax_wall.transAxes, color='#CCCCCC', 
                fontsize=10, family='monospace', horizontalalignment='right', verticalalignment='bottom', 
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=6))

        plt.figure(fig_wall.number) # Set current figure
        plt.figtext(0.5, 0.015, footer_text, color='#CCCCCC', fontsize=10, ha='center', va='bottom', 
                    bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=4))

        plt.title(title_str, color='#CCCCCC', pad=15)
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        if save_path:
            plt.savefig(save_path, facecolor='black', edgecolor='none', dpi=150, bbox_inches='tight')
            print(f"Saved wall plot to: {save_path}")

    # Helper function to generate 1D intensity line graphs.
    def render_intensity_profile(slice_lux, dist_array, suffix_name):
        slice_cd = slice_lux * (target_distance_m**2)
        angles = np.degrees(np.arctan(dist_array / target_distance_m))
        
        fig_line = plt.figure(figsize=(10, 5), facecolor='black')
        ax_line = fig_line.add_subplot(111, facecolor='black')
        
        ax_line.plot(angles, slice_cd, color='#FFFF00', linewidth=1.5)
        ax_line.fill_between(angles, slice_cd, color='#FFFF00', alpha=0.1)
        
        ax_line.set_xlim(-plot_fov_deg/2.0, plot_fov_deg/2.0)
        ax_line.set_ylim(0, max(np.max(slice_cd) * 1.05, 1))
        
        ax_line.set_xlabel("Angle (Degrees)", color='#CCCCCC', fontsize=11, labelpad=10)
        ax_line.set_ylabel("Intensity (Candela)", color='#CCCCCC', fontsize=11, labelpad=10)
        ax_line.tick_params(colors='#CCCCCC', labelsize=10)
        ax_line.grid(True, color='#333333', linestyle='--', alpha=0.5)
        
        for spine in ax_line.spines.values():
            spine.set_color('#555555')
            
        plt.title(title_str + f"\n[Intensity Profile: {suffix_name}]", color='#CCCCCC', pad=15)
        plt.tight_layout()
        
        if save_path:
            base, ext = os.path.splitext(save_path)
            out_path = f"{base}_{suffix_name}{ext}"
            plt.savefig(out_path, facecolor='black', edgecolor='none', dpi=150, bbox_inches='tight')
            print(f"Saved intensity plot to: {out_path}")

    # Extract 1D structural grids for our line charts mapping across the wall
    x_dist = np.linspace(-wall_radius_m, wall_radius_m, sim_grid_res)
    
    if plot_intensity_x:
        render_intensity_profile(wall_lux[int(true_center_idx), :], x_dist, "X-Axis")
        
    if plot_intensity_y:
        render_intensity_profile(wall_lux[:, int(true_center_idx)], x_dist, "Y-Axis")
        
    if plot_intensity_45:
        # Distance mapping for a diagonal slice is longer by sqrt(2)
        diag_dist = np.linspace(-wall_radius_m * math.sqrt(2), wall_radius_m * math.sqrt(2), sim_grid_res)
        render_intensity_profile(np.diagonal(wall_lux), diag_dist, "45-Deg")

    # If running a single simulation, plt.show() will pop open all unclosed figures in separate windows.
    if not generate_all_plots:
        plt.show()
    else:
        # Close all figures from this generation step to prevent memory leaks during batch loops.
        plt.close('all')

    # Return the core simulation metrics as a dictionary for the CSV logger.
    metrics = {
        "Reflector": reflector_name,
        "Emitter": emitter_name,
        "Finish": finish_type.upper(),
        "Max Candela (cd)": int(max_cd),
        "Throw (m)": int(throw_m),
        "Total Lumens": int(total_flux),
        "Spill Angle (deg)": round(spill_angle, 1),
        "Corona Angle (deg)": round(corona_angle, 1),
        "Hotspot Angle (deg)": round(hotspot_angle, 1),
        "Cd/Lm Ratio": round(cd_lm_ratio, 1)
    }
    return metrics

# ==============================================================================
# 6. EXECUTION ROUTING
# ==============================================================================
if __name__ == '__main__':
    os.makedirs(batch_output_directory, exist_ok=True)
    
    # Generate a safe filename identified by distance and exposure settings.
    if use_auto_exposure:
        exposure_id = f"Auto_EV_{auto_exposure_compensation_ev:+.1f}"
    else:
        shutter_str = f"1_{int(round(1.0/cam_shutter_speed_s))}" if cam_shutter_speed_s < 1.0 else f"{cam_shutter_speed_s}"
        exposure_id = f"ISO{cam_iso}_f{cam_f_stop}_{shutter_str}s"
        
    csv_filename = f"sim_results_{target_distance_m}m_{exposure_id}.csv"
    csv_filepath = os.path.join(batch_output_directory, csv_filename)
    
    csv_headers = [
        "Reflector", "Emitter", "Finish", "Max Candela (cd)", "Throw (m)",
        "Total Lumens", "Spill Angle (deg)", "Corona Angle (deg)", 
        "Hotspot Angle (deg)", "Cd/Lm Ratio"
    ]

    # Load existing CSV data into memory to allow replacing duplicates over time.
    existing_data = {}
    if os.path.exists(csv_filepath):
        with open(csv_filepath, mode='r', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                # Use the hardware combo as a unique composite key.
                key = (row["Reflector"], row["Emitter"], row["Finish"])
                existing_data[key] = row
                
    # Trigger simulation pipeline.
    if generate_all_plots:
        print(f"Batch generation enabled. Outputting to: {batch_output_directory}")
        
        # Pre-calculate valid combinations based on geometric fit before entering the primary loop.
        valid_combinations = []
        for r_name, r_data in REFLECTORS.items():
            for e_name, e_data in EMITTERS.items():
                fp_diag = np.sqrt(e_data["footprint_x_mm"]**2 + e_data["footprint_y_mm"]**2)
                if fp_diag <= (r_data["diameter_mm"] / 3.0):
                    for fin in ["smooth", "orange_peel"]:
                        valid_combinations.append((r_name, e_name, fin))
        
        total_plots = len(valid_combinations)
        
        for i, (r_name, e_name, fin) in enumerate(valid_combinations, 1):
            print(f"\n[{i}/{total_plots}] Rendering {r_name} + {e_name} ({fin.upper()})...")
            fin_abbr = "OP" if fin == "orange_peel" else "SMO"
            filepath = os.path.join(batch_output_directory, f"{r_name}_{e_name}_{fin_abbr}.png")
            
            sim_metrics = generate_flashlight_plot(e_name, r_name, fin, save_path=filepath)
            
            # Update the local dictionary with the new simulation data.
            key = (sim_metrics["Reflector"], sim_metrics["Emitter"], sim_metrics["Finish"])
            existing_data[key] = sim_metrics
            
        print(f"\nBatch generation complete!")
        
    else:
        # Run a single active profile render to screen.
        fin_abbr = "OP" if reflector_finish == "orange_peel" else "SMO"
        filepath = os.path.join(batch_output_directory, f"{active_reflector_name}_{active_emitter_name}_{fin_abbr}.png")
        
        print(f"\nRendering {active_reflector_name} + {active_emitter_name} ({reflector_finish.upper()})...")
        sim_metrics = generate_flashlight_plot(active_emitter_name, active_reflector_name, reflector_finish, save_path=filepath)
        
        # Update the local dictionary with the new simulation data.
        key = (sim_metrics["Reflector"], sim_metrics["Emitter"], sim_metrics["Finish"])
        existing_data[key] = sim_metrics
        
        print(f"\nSingle generation complete!")

    # Dump the updated memory block back into the CSV.
    with open(csv_filepath, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
        writer.writeheader()
        
        # Write all stored rows (combining unmodified historical rows and freshly generated rows).
        for key in existing_data:
            writer.writerow(existing_data[key])
            
    print(f"Results successfully saved to: {csv_filepath}")

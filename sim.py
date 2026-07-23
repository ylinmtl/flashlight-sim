"""Finite Element Analysis (FEA) engine for flashlight optical simulation.

This module simulates ray tracing from an LED emitter, calculating reflections 
off a parabolic housing and refraction through silicone domes to determine the 
resulting illuminance pattern on a target wall. It supports GPU acceleration 
via Numba (CUDA) with an automatic fallback to multithreaded CPU processing.
"""

import os
import math
import csv
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
from numba import cuda, njit

# ==============================================================================
# 1. CONFIGURATION & DATA MANAGEMENT
# ==============================================================================

class SimulationConfig:
    """Encapsulates all simulation constraints, thresholds, and camera settings via external JSON."""
    
    # --- Expected Attributes & Original Descriptions ---
    
    # String matching a key in the EMITTERS dictionary to select the active LED.
    active_emitter_name: str
    # String matching a key in the REFLECTORS dictionary to select the active housing.
    active_reflector_name: str
    # String matching a key in the GASKETS dictionary to select the active gasket.
    active_gasket_name: str
    # Surface finish of the reflector. Options are "smooth" or "orange_peel".
    reflector_finish: str
    
    # Integer determining how many bounces to track. 0 = direct light only, 1 = up to 1 extra bounce.
    max_multiple_reflections: int
    # Boolean. True = force reflector opening size; False = use emitter footprint diagonal if it's larger.
    use_reflector_opening: bool
    
    # Float distance from the flashlight to the simulated wall along the Z-axis in meters.
    target_distance_m: float
    # Float field of view (degrees) of the internal simulation grid capturing the rays.
    canvas_fov_deg: float
    # Float field of view (degrees) for the generated output plot (zooms the camera in/out).
    plot_fov_deg: float
    
    # Boolean. True = batch render all valid hardware combinations to disk. False = render active selection only.
    generate_all_plots: bool
    # Boolean. True = draw a 1.75m scale human silhouette centered in the plot for reference.
    show_human_silhouette: bool
    # Boolean. True = generate the standard 2D wall projection image.
    plot_wall_shot: bool
    # Boolean. True = generate a 2D line graph of intensity across the X-axis.
    plot_intensity_x: bool
    # Boolean. True = generate a 2D line graph of intensity across the Y-axis.
    plot_intensity_y: bool
    # Boolean. True = generate a 2D line graph of intensity across a 45-degree diagonal axis.
    plot_intensity_45: bool
    # String directory path where generated CSVs and plot images will be saved.
    batch_output_directory: str
    
    # Boolean. True = normalize brightness per light automatically. False = lock to manual camera settings.
    use_auto_exposure: bool
    # Float EV compensation when auto-exposure is enabled (+1.0 = 2x brighter, -1.0 = half as bright).
    auto_exposure_compensation_ev: float
    # Integer camera sensor sensitivity (ISO). Used only when use_auto_exposure is False.
    cam_iso: int
    # Integer/Float camera aperture size (e.g., 2.8, 4.0, 5.6, 8.0). Used only when use_auto_exposure is False.
    cam_f_stop: float
    # Float camera shutter speed in seconds. Used only when use_auto_exposure is False.
    cam_shutter_speed_s: float
    
    # Integer resolution (width and height in pixels) of the square target plane array.
    sim_grid_res: int
    # Integer subdivision count for the LED die (e.g., 10 generates a 10x10 grid of emission points).
    sim_emitter_elements: int
    # Float angular resolution for elevation from dead center (determines ray count).
    sim_theta_step_deg: float
    # Float angular resolution for rotation around the die (determines ray count).
    sim_phi_step_deg: float
    
    # Float angular limits (elevation). 0.1 avoids divide-by-zero at dead center.
    sim_theta_min_deg: float
    sim_theta_max_deg: float
    # Float angular limits (rotation). 0 to 360 for full sphere coverage.
    sim_phi_min_deg: float
    sim_phi_max_deg: float
    # Float step size used when integrating the Lambertian curve for theoretical lumen calibration.
    lumen_calc_step_deg: float
    
    # Float proportion of light conserved off a smooth mirror finish.
    default_reflectivity_smooth: float
    # Float proportion of light conserved off an orange peel (textured) finish.
    default_reflectivity_op: float
    # Float proportion of light conserved off the inner cylindrical hole wall.
    default_reflectivity_cylinder: float
    # Float proportion of light conserved off the white plastic centering gasket.
    default_reflectivity_gasket: float
    # Float base Gaussian blur sigma applied to simulate scattering from orange peel hotspots.
    default_op_blur_strength: float
    
    # Float minimum absolute lux required to define the outer edge of direct spill.
    spill_visible_threshold_lux: float
    # Float proportion (1%) of peak hotspot intensity defining the visible edge of the corona.
    corona_visible_threshold: float
    # Float proportion (50%) of peak intensity defining the true hotspot (FWHM standard).
    hotspot_fwhm_threshold: float
    
    # Float standard thickness for LED centering gaskets in mm.
    default_gasket_thickness_mm: float
    # Float total structural height of the gasket in mm.
    default_gasket_total_height_mm: float
    # Float inner diameter of the gasket opening in mm (0 defaults to the emitter footprint).
    default_gasket_opening_mm: float
    # Float thickness subtracted from outer diameter to find internal width in mm.
    default_reflector_wall_thickness_mm: float
    # Float thickness subtracted from total height to find internal depth in mm.
    default_reflector_base_thickness_mm: float
    # Float offset in mm assuming perfect focal alignment if not explicitly specified.
    default_focus_offset_mm: float

    def __init__(self, filepath="simulation_settings.json", default_filepath="default_settings.json"):
        self.filepath = filepath
        self.default_filepath = default_filepath
        self.load_settings()

    def load_settings(self):
        """Loads simulation configurations from disk, falling back to defaults if missing."""
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                data = json.load(f)
        elif os.path.exists(self.default_filepath):
            print(f"[{self.filepath}] not found. Generating a new profile from [{self.default_filepath}]...")
            with open(self.default_filepath, 'r') as f:
                data = json.load(f)
            
            # Create the missing simulation_settings.json for the user
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=4)
        else:
            raise FileNotFoundError(
                f"CRITICAL: Missing both '{self.filepath}' and '{self.default_filepath}'. "
                "Cannot initialize the simulation settings."
            )
            
        # Bind the loaded dictionary keys as class attributes
        for key, value in data.items():
            setattr(self, key, value)

    def save_settings(self):
        """Writes the active configuration state back to the active JSON file."""
        # Filter out properties and internal file path variables
        data = {k: v for k, v in self.__dict__.items() if not k.startswith('_') and k not in ('filepath', 'default_filepath')}
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=4)

    @property
    def wall_radius_m(self) -> float:
        """Automatically calculate the physical canvas size based on distance and viewing angle."""
        return self.target_distance_m * math.tan(math.radians(self.canvas_fov_deg / 2.0))

    @property
    def plot_radius_m(self) -> float:
        """Automatically calculate the final rendered plot radius in meters."""
        return self.target_distance_m * math.tan(math.radians(self.plot_fov_deg / 2.0))


class HardwareLibrary:
    """Manages the serialization and retrieval of hardware data from a JSON file."""
    
    def __init__(self, filepath="hardware_library.json"):
        self.filepath = filepath
        self._emitters = {}
        self._reflectors = {}
        self._gaskets = {}
        self.load_database()

    def load_database(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                self._emitters = data.get("emitters", {})
                self._reflectors = data.get("reflectors", {})
                self._gaskets = data.get("gaskets", {})
        else:
            raise FileNotFoundError(f"Could not find {self.filepath}. Please ensure the JSON file is in the directory.")

    def save_database(self):
        data = {
            "emitters": self._emitters,
            "reflectors": self._reflectors,
            "gaskets": self._gaskets
        }
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=4)

    def get_emitter(self, name: str) -> dict:
        return self._emitters[name]
        
    def list_emitters(self) -> list:
        return list(self._emitters.keys())

    def get_reflector(self, name: str) -> dict:
        return self._reflectors[name]
        
    def list_reflectors(self) -> list:
        return list(self._reflectors.keys())

    def get_gasket(self, name: str) -> dict:
        return self._gaskets[name]
        
    def list_gaskets(self) -> list:
        return list(self._gaskets.keys())


# ==============================================================================
# 2. HELPERS & HARDWARE INTERPOLATION
# ==============================================================================

def get_standard_emitter_intensity_vec(theta_rad):
    """Calculates the relative angular intensity of an emitter.
    
    Uses a standard Lambertian (cosine) emission curve to determine how bright 
    the emitter appears from a given viewing angle. 
    
    Args: 
        theta_rad: A numpy array of viewing angles in radians. 
        
    Returns: 
        A numpy array containing the relative intensity at each provided angle. 
    """
    abs_angle = np.abs(np.degrees(theta_rad))
    intensity = np.cos(theta_rad)
    intensity[abs_angle > 90.0] = 0.0
    return intensity

def calculate_lumens(emitter, current_amps):
    """Calculates the theoretical lumen output using diode physics. 
    
    Voltage is modeled as a logarithmic curve. Efficacy is adjusted 
    based on the current droop factor for the specific emitter. 
    
    Args: 
        emitter: A dictionary containing the hardware specifications of the LED. 
        current_amps: A float representing the drive current in amperes. 
        
    Returns: 
        A float representing the calculated total lumen output. 
    """
    voltage = emitter["vf_turn_on_v"] + (emitter["vf_scale"] * np.log(current_amps + 1.0))
    power_watts = current_amps * voltage
    efficiency = emitter["base_efficacy_lm_w"] * np.exp(-emitter["droop_factor"] * current_amps)
    return power_watts * efficiency

def get_sim_geometry(reflector, emitter, gasket, finish, config: SimulationConfig):
    """Parses hardware profiles and computes derived 3D spatial boundaries. 
    
    Args: 
        reflector: A dictionary containing the hardware specifications of the reflector. 
        emitter: A dictionary containing the hardware specifications of the LED. 
        gasket: A dictionary containing the hardware specifications of the gasket.
        finish: A string denoting the reflector finish ("smooth" or "orange_peel"). 
        config: The globally loaded settings object.
        
    Returns: 
        A dictionary mapping structural identifiers to their calculated geometric 
        boundaries, limits, offsets, and surface reflectivities. 
    """
    D = reflector["diameter_mm"] - reflector.get("thickness_diameter_mm", config.default_reflector_wall_thickness_mm)
    H_total = reflector["height_mm"]
    R = D / 2.0
    
    d_hole_input = reflector.get("opening_diameter_mm", 0.0)
    focus_offset_mm = reflector.get("focus_offset_mm", config.default_focus_offset_mm)
    thickness_height_mm = reflector.get("thickness_height_mm", config.default_reflector_base_thickness_mm)
    
    gasket_thickness_mm = gasket.get("gasket_thickness_mm", config.default_gasket_thickness_mm)
    gasket_total_height_mm = gasket.get("gasket_total_height_mm", config.default_gasket_total_height_mm)
    gasket_opening_mm = gasket.get("gasket_opening_mm", config.default_gasket_opening_mm)

    footprint_diag = math.sqrt(emitter["footprint_x_mm"]**2 + emitter["footprint_y_mm"]**2)
    effective_d_hole = d_hole_input if config.use_reflector_opening else max(d_hole_input, footprint_diag)
    r_hole = effective_d_hole / 2.0

    H_eff = H_total - focus_offset_mm
    focal_length = (-H_eff + math.sqrt(H_eff**2 + R**2)) / 2.0
    
    z_bottom = focal_length - focus_offset_mm
    z_min_cut = z_bottom + thickness_height_mm
    z_max_cut = z_bottom + H_total
    
    h_gasket_ext = max(0.0, gasket_total_height_mm - gasket_thickness_mm)
    z_gasket_top = z_bottom + h_gasket_ext
    
    if gasket_opening_mm > 0.0:
        r_gasket = gasket_opening_mm / 2.0
        gasket_x_half, gasket_y_half = 0.0, 0.0
        is_cylindrical_gasket = 1
    else:
        r_gasket = 0.0
        gasket_x_half = emitter["footprint_x_mm"] / 2.0
        gasket_y_half = emitter["footprint_y_mm"] / 2.0
        is_cylindrical_gasket = 0

    z_intersect = (r_hole**2) / (4.0 * focal_length)
    z_hole_top = float(max(z_intersect, z_min_cut))
    ez_base = z_bottom + (emitter["height_mm"] - gasket_thickness_mm)
    
    dome_input = emitter.get("dome_size_mm", 0.0)
    dome_diameter = min(emitter["footprint_x_mm"], emitter["footprint_y_mm"]) if dome_input == -1 else max(0.0, dome_input)
    
    refl_para = reflector.get("reflectivity_op", config.default_reflectivity_op) if finish == "orange_peel" else reflector.get("reflectivity_smooth", config.default_reflectivity_smooth)
    
    # Values useful for HUD and plotting.
    physical_emitter_height = emitter["height_mm"] - gasket_thickness_mm
    actual_ez = z_bottom + physical_emitter_height
    
    return {
        "focal_length": focal_length, "z_bottom": z_bottom, "z_min_cut": z_min_cut, 
        "z_hole_top": z_hole_top, "z_max_cut": z_max_cut, "radius_max": R, "r_hole": r_hole,
        "z_gasket_top": z_gasket_top, "r_gasket": r_gasket, "gasket_x_half": gasket_x_half, 
        "gasket_y_half": gasket_y_half, "is_cylindrical_gasket": is_cylindrical_gasket,
        "ez_base": ez_base, "dome_radius": dome_diameter / 2.0, "refractive_index": emitter.get("refractive_index", 1.0),
        "refl_para": refl_para, "refl_cyl": reflector.get("reflectivity_cylinder", config.default_reflectivity_cylinder), 
        "refl_gask": reflector.get("reflectivity_gasket", config.default_reflectivity_gasket),
        "effective_d_hole": effective_d_hole, "focus_delta": actual_ez - focal_length,
        "op_multiplier": reflector.get("OP_Factor", 1.0)
    }

# ==============================================================================
# 3. MATH & FINITE ELEMENT ANALYSIS (FEA) ENGINE
# ==============================================================================

@njit
def solve_quadratic(a, b, c):
    """Solves a quadratic equation of the form ax^2 + bx + c = 0. 
    
    Args: 
        a: Float coefficient for the squared term. 
        b: Float coefficient for the linear term. 
        c: Float constant term. 
        
    Returns: 
        A tuple of floats (t1, t2) representing the two roots. 
        Returns (1e9, 1e9) if there are no real roots or if 'a' is effectively zero. 
    """
    if a < 1e-8:
        return 1e9, 1e9
    disc = b**2 - 4.0 * a * c
    if disc < 0.0:
        return 1e9, 1e9
    sqrt_disc = math.sqrt(disc)
    return (-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)

@njit
def apply_dome_refraction(ex, ey, ez, vx, vy, vz, dome_radius, refractive_index):
    """Calculates Snell's law for total internal reflection inside an LED dome. 
    
    Args: 
        ex: Float X-coordinate of the ray origin. 
        ey: Float Y-coordinate of the ray origin. 
        ez: Float Z-coordinate of the ray origin. 
        vx: Float X-component of the ray direction vector. 
        vy: Float Y-component of the ray direction vector. 
        vz: Float Z-component of the ray direction vector. 
        dome_radius: Float radius of the silicone dome in mm. 
        refractive_index: Float refractive index ratio (n_silicone / n_air). 
        
    Returns: 
        A tuple containing (blocked, ex, ey, ez, vx, vy, vz) where blocked is a 
        boolean indicating if the ray was trapped by total internal reflection. 
        If False, returns updated origin and normalized trajectory vectors. 
    """
    P_sq = ex**2 + ey**2
    c = P_sq - dome_radius**2
    b = 2.0 * (ex * vx + ey * vy)
    
    discriminant = b**2 - 4.0 * c
    if discriminant >= 0.0:
        t = (-b + math.sqrt(discriminant)) / 2.0
        if t > 0.0:
            hx, hy, hz = ex + t * vx, ey + t * vy, t * vz
            nx, ny, nz = hx / dome_radius, hy / dome_radius, hz / dome_radius
            
            c1 = vx * nx + vy * ny + vz * nz
            r = refractive_index
            tir_check = 1.0 - r**2 * (1.0 - c1**2)
            
            if tir_check >= 0.0:
                c2 = math.sqrt(tir_check)
                vx, vy, vz = r * vx - (r * c1 - c2) * nx, r * vy - (r * c1 - c2) * ny, r * vz - (r * c1 - c2) * nz
                
                mag = math.sqrt(vx**2 + vy**2 + vz**2)
                return False, hx, hy, ez + hz, vx / mag, vy / mag, vz / mag
                
    return True, ex, ey, ez, vx, vy, vz

@njit
def process_single_ray(ex, ey, ez_base, vx, vy, vz, flux,
                       focal_length, z_bottom, z_min_cut, z_hole_top, z_max_cut,
                       radius_max, r_hole, target_z_mm, grid_res, wall_radius_m,
                       reflectivity_parabola, reflectivity_cylinder, reflectivity_gasket,
                       dome_radius, refractive_index, max_multiple_reflections,
                       z_gasket_top, r_gasket, gasket_x_half, gasket_y_half, is_cylindrical_gasket):
    """Traces a single light ray through the optical system to its final impact point. 
    
    Evaluates sequential collision intersections against internal cylindrical walls, 
    horizontal planes, the primary paraboloid, and internal gasket geometry. 
    
    Args: 
        ex: Float X-coordinate origin of the ray. 
        ey: Float Y-coordinate origin of the ray. 
        ez_base: Float Z-coordinate origin of the ray. 
        vx: Float X-direction of the ray. 
        vy: Float Y-direction of the ray. 
        vz: Float Z-direction of the ray. 
        flux: Float initial lumen flux represented by this ray. 
        focal_length: Float paraboloid focal length. 
        z_bottom: Float Z-coordinate bottom of the physical housing. 
        z_min_cut: Float Z-coordinate flat shelf line. 
        z_hole_top: Float Z-coordinate where cylinder meets the paraboloid. 
        z_max_cut: Float Z-coordinate top boundary of the housing. 
        radius_max: Float maximum radius of the housing at the top. 
        r_hole: Float radius of the center emitter opening. 
        target_z_mm: Float Z-distance to the capture plane. 
        grid_res: Integer resolution of the capture array. 
        wall_radius_m: Float physical boundary radius of the capture array. 
        reflectivity_parabola: Float efficiency coefficient for the main dish. 
        reflectivity_cylinder: Float efficiency coefficient for the inner walls. 
        reflectivity_gasket: Float efficiency coefficient for the base gasket. 
        dome_radius: Float radius of the LED silicone dome. 
        refractive_index: Float refractive index of the dome material. 
        max_multiple_reflections: Integer maximum number of collision bounces. 
        z_gasket_top: Float Z-coordinate height of the gasket. 
        r_gasket: Float outer radius of a cylindrical gasket. 
        gasket_x_half: Float half-width of a rectangular gasket. 
        gasket_y_half: Float half-length of a rectangular gasket. 
        is_cylindrical_gasket: Integer (0 or 1) indicating gasket shape. 
        
    Returns: 
        A tuple of (final_flux, row, col, bounce_count). 
        If the ray is absorbed or trapped internally, returns (0.0, -1, -1, -1). 
    """
    blocked = False
    if dome_radius > 0.0:
        blocked, ex, ey, ez_base, vx, vy, vz = apply_dome_refraction(ex, ey, ez_base, vx, vy, vz, dome_radius, refractive_index)

    current_ex, current_ey, current_ez = ex, ey, ez_base
    current_vx, current_vy, current_vz = vx, vy, vz
    current_flux = flux
    
    bounce_count = 0
    bin_size = (2.0 * wall_radius_m) / grid_res
    
    while not blocked:
        hit_type, t_hit = 0, 1e9
        
        # 1. Cylinder Intersections (Inner walls)
        a_cyl = current_vx**2 + current_vy**2
        b_cyl = 2.0 * (current_ex * current_vx + current_ey * current_vy)
        c_cyl = current_ex**2 + current_ey**2 - r_hole**2
        t_c1, t_c2 = solve_quadratic(a_cyl, b_cyl, c_cyl)
        
        for t_c in (t_c1, t_c2):
            if 1e-4 < t_c < t_hit and z_bottom <= (current_ez + t_c * current_vz) <= z_hole_top:
                t_hit, hit_type = t_c, 2

        # 2. Horizontal Plane Intersections (Absorbing surfaces)
        if current_vz < 0.0:
            if z_hole_top == z_min_cut and current_ez > z_min_cut:
                t_plane = (z_min_cut - current_ez) / current_vz
                if 1e-4 < t_plane < t_hit and (current_ex + t_plane * current_vx)**2 + (current_ey + t_plane * current_vy)**2 > r_hole**2:
                    t_hit, hit_type = t_plane, 3
                        
            if z_gasket_top > z_bottom and current_ez > z_gasket_top:
                t_plane = (z_gasket_top - current_ez) / current_vz
                if 1e-4 < t_plane < t_hit:
                    rx, ry = current_ex + t_plane * current_vx, current_ey + t_plane * current_vy
                    if (is_cylindrical_gasket == 1 and rx**2 + ry**2 >= r_gasket**2) or \
                       (is_cylindrical_gasket == 0 and (abs(rx) >= gasket_x_half or abs(ry) >= gasket_y_half)):
                        t_hit, hit_type = t_plane, 4
            
            if current_ez > z_bottom:
                t_plane = (z_bottom - current_ez) / current_vz
                if 1e-4 < t_plane < t_hit:
                    t_hit, hit_type = t_plane, 3

        elif current_vz > 0.0 and current_ez < z_bottom:
            t_plane = (z_bottom - current_ez) / current_vz
            if 1e-4 < t_plane < t_hit and (current_ex + t_plane * current_vx)**2 + (current_ey + t_plane * current_vy)**2 > r_hole**2:
                t_hit, hit_type = t_plane, 3
                
        # 3. Paraboloid Intersections (Main reflector dish)
        a_par = current_vx**2 + current_vy**2
        b_par = 2.0 * (current_ex * current_vx + current_ey * current_vy) - 4.0 * focal_length * current_vz
        c_par = current_ex**2 + current_ey**2 - 4.0 * focal_length * current_ez
        t_p1, t_p2 = solve_quadratic(a_par, b_par, c_par)
        
        for t_p in (t_p1, t_p2):
            if 1e-4 < t_p < t_hit and z_hole_top <= (current_ez + t_p * current_vz) <= z_max_cut:
                t_hit, hit_type = t_p, 1
                    
        # 4. Gasket Wall Intersections (Lateral blockages)
        if z_gasket_top > z_bottom:
            if is_cylindrical_gasket == 1:
                b_g = 2.0 * (current_ex * current_vx + current_ey * current_vy)
                c_g = current_ex**2 + current_ey**2 - r_gasket**2
                t_g1, t_g2 = solve_quadratic(current_vx**2 + current_vy**2, b_g, c_g)
                if 1e-4 < t_g2 < t_hit and (current_ez + t_g2 * current_vz) <= z_gasket_top:
                    t_hit, hit_type = t_g2, 4
            else:
                if abs(current_vx) > 1e-8:
                    t_x = (gasket_x_half * (1.0 if current_vx > 0.0 else -1.0) - current_ex) / current_vx
                    if 1e-4 < t_x < t_hit and (current_ez + t_x * current_vz) <= z_gasket_top and -gasket_y_half <= (current_ey + t_x * current_vy) <= gasket_y_half:
                        t_hit, hit_type = t_x, 4
                if abs(current_vy) > 1e-8:
                    t_y = (gasket_y_half * (1.0 if current_vy > 0.0 else -1.0) - current_ey) / current_vy
                    if 1e-4 < t_y < t_hit and (current_ez + t_y * current_vz) <= z_gasket_top and -gasket_x_half <= (current_ex + t_y * current_vx) <= gasket_x_half:
                        t_hit, hit_type = t_y, 4

        # Process the confirmed collision if any occurred
        if hit_type in (1, 2, 4):
            if bounce_count >= max_multiple_reflections + 1:
                return 0.0, -1, -1, -1
            bounce_count += 1
            
            rx, ry, rz = current_ex + t_hit * current_vx, current_ey + t_hit * current_vy, current_ez + t_hit * current_vz
            
            if hit_type == 1:
                nx, ny, nz = -rx, -ry, 2.0 * focal_length
                current_refl = reflectivity_parabola
            elif hit_type == 2:
                nx, ny, nz = -rx, -ry, 0.0
                current_refl = reflectivity_cylinder
            elif hit_type == 4:
                if abs(rz - z_gasket_top) < 1e-4 and current_vz < 0.0:
                    nx, ny, nz = 0.0, 0.0, 1.0
                else:
                    if is_cylindrical_gasket == 1:
                        nx, ny, nz = -rx, -ry, 0.0
                    else:
                        if abs(abs(rx) - gasket_x_half) < 1e-4:
                            nx, ny, nz = (-1.0 if rx > 0.0 else 1.0), 0.0, 0.0
                        else:
                            nx, ny, nz = 0.0, (-1.0 if ry > 0.0 else 1.0), 0.0
                current_refl = reflectivity_gasket
                
            mag = math.sqrt(nx**2 + ny**2 + nz**2)
            nx, ny, nz = nx / mag, ny / mag, nz / mag
            
            dot = current_vx * nx + current_vy * ny + current_vz * nz
            current_vx, current_vy, current_vz = current_vx - 2.0 * dot * nx, current_vy - 2.0 * dot * ny, current_vz - 2.0 * dot * nz
            current_ex, current_ey, current_ez = rx, ry, rz
            current_flux *= current_refl
            
        elif hit_type == 3:
            return 0.0, -1, -1, -1
            
        else:
            if current_vz > 0.0:
                esc_x = current_ex + ((z_max_cut - current_ez) / current_vz) * current_vx
                esc_y = current_ey + ((z_max_cut - current_ez) / current_vz) * current_vy
                
                if math.sqrt(esc_x**2 + esc_y**2) <= radius_max + 1e-4:
                    s = (target_z_mm - current_ez) / current_vz
                    col = int((((current_ex + s * current_vx) / 1000.0) + wall_radius_m) / bin_size)
                    row = int((((current_ey + s * current_vy) / 1000.0) + wall_radius_m) / bin_size)
                    
                    if 0 <= col < grid_res and 0 <= row < grid_res:
                        return current_flux, row, col, bounce_count
            return 0.0, -1, -1, -1

    return 0.0, -1, -1, -1

@cuda.jit
def ray_trace_kernel_gpu(args, start_idx, end_idx):
    """CUDA kernel for parallel processing of light rays on the GPU. 
    
    Args: 
        args: A tuple containing all array bounds, initial direction vectors, 
              geometry constants, and the mutable output grids. 
        start_idx: Integer starting index for this thread batch. 
        end_idx: Integer ending index for this thread batch. 
    """
    idx = cuda.grid(1) + start_idx
    if idx >= end_idx: return
    
    total_rays = args[2].shape[0]
    element_idx, ray_idx = idx // total_rays, idx % total_rays

    final_flux, row, col, bounces = process_single_ray(
        args[0][element_idx], args[1][element_idx], args[7], args[2][ray_idx], args[3][ray_idx], args[4][ray_idx], args[5][ray_idx],
        args[6], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], 
        args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27]
    )

    if row != -1 and col != -1:
        cuda.atomic.add(args[28] if bounces > 0 else args[29], (row, col), final_flux)

@njit
def ray_trace_kernel_cpu(args, start_idx, end_idx):
    """Numba-compiled JIT kernel for parallel processing of light rays on the CPU. 
    
    Args: 
        args: A tuple containing all array bounds, initial direction vectors, 
              geometry constants, and the mutable output grids. 
        start_idx: Integer starting index for this thread batch. 
        end_idx: Integer ending index for this thread batch. 
    """
    total_rays = args[2].shape[0]
    for idx in range(start_idx, end_idx):
        element_idx, ray_idx = idx // total_rays, idx % total_rays
        final_flux, row, col, bounces = process_single_ray(
            args[0][element_idx], args[1][element_idx], args[7], args[2][ray_idx], args[3][ray_idx], args[4][ray_idx], args[5][ray_idx],
            args[6], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], 
            args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27]
        )

        if row != -1 and col != -1:
            if bounces > 0: args[28][row, col] += final_flux
            else: args[29][row, col] += final_flux

def execute_tracers(is_gpu, kernel, total_threads, args):
    """Manages the execution flow of the ray tracing kernels. 
    
    Runs a warmup batch to trigger JIT compilation, followed by a small 
    calibration batch to estimate remaining run time, then pushes the 
    remainder of the workload queue. 
    
    Args: 
        is_gpu: Boolean flag indicating if CUDA should be used. 
        kernel: The compiled Numba target kernel function to call. 
        total_threads: Integer count of total rays to process. 
        args: A tuple block of all simulation variables required by the kernel. 
    """
    cal_size = min(max(int(total_threads * 0.02), 250_000), total_threads - 1)
    
    print(f"[{'CUDA' if is_gpu else 'CPU'} FEA Engine] Compiling & Calibrating...", end="", flush=True)
    t0 = time.time()
    
    if is_gpu:
        kernel[1, 1](args, 0, 1); cuda.synchronize()
        blocks = (cal_size + 255) // 256
        kernel[blocks, 256](args, 1, 1 + cal_size); cuda.synchronize()
    else:
        kernel(args, 0, 1)
        kernel(args, 1, 1 + cal_size)
        
    t1 = time.time()
    cal_time = t1 - t0
    rays_per_sec = cal_size / cal_time if cal_time > 0 else 1
    rem = total_threads - (1 + cal_size)
    
    print(f" Done. ({rays_per_sec:,.0f} rays/sec)")
    print(f"[{'CUDA' if is_gpu else 'CPU'} FEA Engine] Predicted remaining time: ~{rem / rays_per_sec:.2f} s")
    
    if rem > 0:
        if is_gpu:
            kernel[(rem + 255) // 256, 256](args, 1 + cal_size, total_threads)
            cuda.synchronize()
        else:
            kernel(args, 1 + cal_size, total_threads)

def run_pure_fea_sim_vectorized(geom, emitter, current_amps, finish, config: SimulationConfig):
    """Initializes and runs the core finite element analysis physics loop. 
    
    Calculates theoretical ray flux, prepares vector geometry boundaries, 
    allocates necessary memory structures, and executes the kernels. 
    
    Args: 
        geom: A dictionary of physical hardware parameters and limits. 
        emitter: A dictionary of specific LED physical/electrical attributes. 
        current_amps: Float continuous current feed to the LED. 
        finish: String literal defining the reflector texture. 
        config: Loaded instance of SimulationConfig.
        
    Returns: 
        A tuple of four grids/values (final_lux_grid, processed_hotspot_lux, 
        spill_lux, total_lumens) representing the processed illumination data. 
    """
    total_lumens = calculate_lumens(emitter, current_amps)
    
    theta_int = np.radians(np.arange(0, 90, config.lumen_calc_step_deg))
    N_integral = np.sum(get_standard_emitter_intensity_vec(theta_int) * np.sin(theta_int) * np.radians(config.lumen_calc_step_deg))
    I_peak_base = total_lumens / (2 * np.pi * N_integral)
    
    pixel_area_m2 = (2.0 * config.wall_radius_m / config.sim_grid_res) ** 2
    
    die_len = emitter["die_length_mm"]
    die_wid = die_len if emitter.get("shape") == "round" else emitter["die_width_mm"]
    
    EX, EY = np.meshgrid(np.linspace(-die_len/2, die_len/2, config.sim_emitter_elements), 
                         np.linspace(-die_wid/2, die_wid/2, config.sim_emitter_elements))
    if emitter.get("shape") == "round":
        mask = (EX**2 + EY**2) <= (die_len / 2.0)**2
        ex_flat, ey_flat = EX[mask], EY[mask]
    else:
        ex_flat, ey_flat = EX.flatten(), EY.flatten()
        
    actual_elements = len(ex_flat)
    
    THETA, PHI = np.meshgrid(np.radians(np.arange(config.sim_theta_min_deg, config.sim_theta_max_deg, config.sim_theta_step_deg)), 
                             np.radians(np.arange(config.sim_phi_min_deg, config.sim_phi_max_deg, config.sim_phi_step_deg)))
    THETA_flat, PHI_flat = THETA.flatten(), PHI.flatten()
    
    solid_angle = np.sin(THETA_flat) * np.radians(config.sim_theta_step_deg) * np.radians(config.sim_phi_step_deg)
    ray_flux = np.ascontiguousarray((I_peak_base * get_standard_emitter_intensity_vec(THETA_flat) * solid_angle) / actual_elements, dtype=np.float64)
    vx, vy, vz = np.sin(THETA_flat) * np.cos(PHI_flat), np.sin(THETA_flat) * np.sin(PHI_flat), np.cos(THETA_flat)

    target_z_mm = config.target_distance_m * 1000.0
    total_threads = actual_elements * len(vx)
    has_gpu = cuda.is_available()

    hotspot_grid = np.zeros((config.sim_grid_res, config.sim_grid_res), dtype=np.float64)
    spill_grid = np.zeros((config.sim_grid_res, config.sim_grid_res), dtype=np.float64)

    if has_gpu:
        print(f"\n[CUDA FEA Engine] GPU Detected. Pushing {total_threads:,} rays to VRAM...")
        d_ex, d_ey = cuda.to_device(np.ascontiguousarray(ex_flat, dtype=np.float64)), cuda.to_device(np.ascontiguousarray(ey_flat, dtype=np.float64))
        d_vx, d_vy, d_vz = cuda.to_device(np.ascontiguousarray(vx, dtype=np.float64)), cuda.to_device(np.ascontiguousarray(vy, dtype=np.float64)), cuda.to_device(np.ascontiguousarray(vz, dtype=np.float64))
        d_flux = cuda.to_device(ray_flux)
        d_hotspot, d_spill = cuda.to_device(hotspot_grid), cuda.to_device(spill_grid)
        
        args = (d_ex, d_ey, d_vx, d_vy, d_vz, d_flux, float(geom['focal_length']), float(geom['ez_base']), float(geom['z_bottom']),
                float(geom['z_min_cut']), float(geom['z_hole_top']), float(geom['z_max_cut']), float(geom['radius_max']), float(geom['r_hole']),
                float(target_z_mm), int(config.sim_grid_res), float(config.wall_radius_m), float(geom['refl_para']), float(geom['refl_cyl']), float(geom['refl_gask']),
                float(geom['dome_radius']), float(geom['refractive_index']), int(config.max_multiple_reflections), float(geom['z_gasket_top']), float(geom['r_gasket']),
                float(geom['gasket_x_half']), float(geom['gasket_y_half']), int(geom['is_cylindrical_gasket']), d_hotspot, d_spill)
        
        execute_tracers(True, ray_trace_kernel_gpu, total_threads, args)
        hotspot_grid, spill_grid = d_hotspot.copy_to_host(), d_spill.copy_to_host()
        
    else:
        print(f"\n[CPU FEA Engine] Using {os.cpu_count()} logical cores...")
        args = (np.ascontiguousarray(ex_flat, dtype=np.float64), np.ascontiguousarray(ey_flat, dtype=np.float64),
                np.ascontiguousarray(vx, dtype=np.float64), np.ascontiguousarray(vy, dtype=np.float64), np.ascontiguousarray(vz, dtype=np.float64), ray_flux,
                float(geom['focal_length']), float(geom['ez_base']), float(geom['z_bottom']), float(geom['z_min_cut']), float(geom['z_hole_top']), 
                float(geom['z_max_cut']), float(geom['radius_max']), float(geom['r_hole']), float(target_z_mm), int(config.sim_grid_res), float(config.wall_radius_m), 
                float(geom['refl_para']), float(geom['refl_cyl']), float(geom['refl_gask']), float(geom['dome_radius']), float(geom['refractive_index']), 
                int(config.max_multiple_reflections), float(geom['z_gasket_top']), float(geom['r_gasket']), float(geom['gasket_x_half']), float(geom['gasket_y_half']), 
                int(geom['is_cylindrical_gasket']), hotspot_grid, spill_grid)
        
        execute_tracers(False, ray_trace_kernel_cpu, total_threads, args)

    print(f"[{'CUDA' if has_gpu else 'CPU'} FEA Engine] Ray tracing complete. Applying spatial blur...\n")
    
    scaled_blur = (config.default_op_blur_strength * geom["op_multiplier"] * (config.sim_grid_res / 1000.0)) if finish == "orange_peel" else 0.0
    processed_hotspot = gaussian_filter(hotspot_grid, sigma=scaled_blur) if scaled_blur > 0 else hotspot_grid
        
    processed_hotspot_lux, spill_lux = processed_hotspot / pixel_area_m2, spill_grid / pixel_area_m2
    return processed_hotspot_lux + spill_lux, processed_hotspot_lux, spill_lux, total_lumens

# ==============================================================================
# 4. PLOTTING & EXPORT MANAGER
# ==============================================================================

def apply_camera_exposure_and_tonemap(wall_lux, config: SimulationConfig):
    """Applies photorealistic camera exposure and tone mapping to raw lux grids. 
    
    Args: 
        wall_lux: A 2D numpy array representing the simulated incident lux levels. 
        config: Loaded instance of SimulationConfig.
        
    Returns: 
        A 2D numpy array of image pixel values scaled [0.0, 1.0] representing 
        how the pattern would appear to a standard camera sensor. 
    """
    if config.use_auto_exposure:
        auto_target = np.percentile(wall_lux, 99.5) or 1.0
        exposed_lux = wall_lux * (1.0 / auto_target) * (2 ** config.auto_exposure_compensation_ev)
    else:
        lux_for_exposure = (250.0 * (2 ** np.log2((config.cam_f_stop**2) / config.cam_shutter_speed_s))) / config.cam_iso
        exposed_lux = (wall_lux / lux_for_exposure) * 0.18
        
    # Apply ACES filmic tone mapping curve.
    mapped = (exposed_lux * (2.51 * exposed_lux + 0.03)) / (exposed_lux * (2.43 * exposed_lux + 0.59) + 0.14)
    return np.power(np.clip(mapped, 0.0, 1.0), 1.0 / 2.2)

def get_beam_metrics(wall_lux, hotspot_lux, spill_lux, max_cd, total_flux, config: SimulationConfig):
    """Extracts geometric dimensions of the flashlight beam from the simulated grid. 
    
    Calculates the spatial extent (angles and physical diameters) of the direct 
    spill, secondary corona, and primary hotspot bounds. 
    
    Args: 
        wall_lux: 2D numpy array of total absolute illuminance (lux). 
        hotspot_lux: 2D numpy array of reflected-only illuminance (lux). 
        spill_lux: 2D numpy array of unreflected direct emission illuminance (lux). 
        max_cd: Float maximum registered intensity in candela. 
        total_flux: Float total emitted lumen count. 
        config: Loaded instance of SimulationConfig.
        
    Returns: 
        A tuple of metric values: (spill_angle, spill_size, corona_angle, 
        corona_size, hotspot_angle, hotspot_size, candela_per_lumen_ratio). 
    """
    pixel_size_m = (2.0 * config.wall_radius_m) / config.sim_grid_res
    center_idx = (config.sim_grid_res - 1) / 2.0

    def get_max_radius(mask):
        if not np.any(mask): return 0.0
        y_idx, x_idx = np.nonzero(mask)
        return np.max(np.sqrt((x_idx - center_idx)**2 + (y_idx - center_idx)**2)) * pixel_size_m

    spill_rad = get_max_radius(spill_lux > config.spill_visible_threshold_lux)
    corona_rad = get_max_radius(hotspot_lux > (np.max(hotspot_lux) * config.corona_visible_threshold))
    hotspot_rad = get_max_radius(wall_lux >= (np.max(wall_lux) * config.hotspot_fwhm_threshold))

    return (
        2 * np.degrees(np.arctan(spill_rad / config.target_distance_m)), 2 * spill_rad,
        2 * np.degrees(np.arctan(corona_rad / config.target_distance_m)), 2 * corona_rad,
        2 * np.degrees(np.arctan(hotspot_rad / config.target_distance_m)), 2 * hotspot_rad,
        max_cd / total_flux
    )

def draw_human_silhouette(ax, person_x, person_y_bottom, person_height_m):
    """Draws a 2D scale representation of a human figure on a matplotlib axis. 
    
    Args: 
        ax: A matplotlib Axes object to draw the patches on. 
        person_x: Float horizontal center X-coordinate of the figure. 
        person_y_bottom: Float vertical base Y-coordinate (feet placement). 
        person_height_m: Float total height of the figure in meters. 
    """
    h_rad, t_w, t_h, l_w, l_h, a_w, a_h = (person_height_m * v for v in (0.08, 0.25, 0.35, 0.08, 0.45, 0.06, 0.40))
    opts = dict(ec='#FFFF00', fc='none', alpha=0.4, lw=1.0, ls='--')
    
    ax.add_patch(patches.Circle((person_x, person_y_bottom + l_h + t_h + h_rad), h_rad, **opts))
    ax.add_patch(patches.Rectangle((person_x - t_w/2, person_y_bottom + l_h), t_w, t_h, **opts))
    ax.add_patch(patches.Rectangle((person_x - t_w/2, person_y_bottom), l_w, l_h, **opts))
    ax.add_patch(patches.Rectangle((person_x + t_w/2 - l_w, person_y_bottom), l_w, l_h, **opts))
    ax.add_patch(patches.Rectangle((person_x - t_w/2 - a_w, person_y_bottom + l_h + t_h - a_h), a_w, a_h, **opts))
    ax.add_patch(patches.Rectangle((person_x + t_w/2, person_y_bottom + l_h + t_h - a_h), a_w, a_h, **opts))

def render_intensity_profile(slice_lux, dist_array, suffix_name, title_str, save_path, config: SimulationConfig):
    """Generates and optionally saves a 1D line plot of beam intensity. 
    
    Args: 
        slice_lux: A 1D numpy array representing a slice of the lux array. 
        dist_array: A 1D numpy array mapping index position to physical distances. 
        suffix_name: String label to append to the saved filename (e.g., 'X-Axis'). 
        title_str: String global title block describing the simulation parameters. 
        save_path: String directory path to output the generated plot image. 
        config: Loaded instance of SimulationConfig.
    """
    slice_cd = slice_lux * (config.target_distance_m**2)
    angles = np.degrees(np.arctan(dist_array / config.target_distance_m))
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='black')
    ax.set_facecolor('black')
    
    ax.plot(angles, slice_cd, color='#FFFF00', linewidth=1.5)
    ax.fill_between(angles, slice_cd, color='#FFFF00', alpha=0.1)
    
    ax.set_xlim(-config.plot_fov_deg/2.0, config.plot_fov_deg/2.0)
    ax.set_ylim(0, max(np.max(slice_cd) * 1.05, 1))
    
    ax.set_xlabel("Angle (Degrees)", color='#CCCCCC', fontsize=11, labelpad=10)
    ax.set_ylabel("Intensity (Candela)", color='#CCCCCC', fontsize=11, labelpad=10)
    ax.tick_params(colors='#CCCCCC', labelsize=10)
    ax.grid(True, color='#333333', linestyle='--', alpha=0.5)
    for spine in ax.spines.values(): spine.set_color('#555555')
        
    plt.title(f"{title_str}\n[Intensity Profile: {suffix_name}]", color='#CCCCCC', pad=15)
    plt.tight_layout()
    
    if save_path:
        base, ext = os.path.splitext(save_path)
        out = f"{base}_{suffix_name}{ext}"
        plt.savefig(out, facecolor='black', edgecolor='none', dpi=150, bbox_inches='tight')
        print(f"Saved intensity plot to: {out}")

def generate_flashlight_plot(emitter_name, reflector_name, gasket_name, finish_type, config: SimulationConfig, library: HardwareLibrary, save_path=None):
    """Orchestrates simulation execution, data extraction, and plot rendering. 
    
    Args: 
        emitter_name: String identifying the desired LED hardware profile. 
        reflector_name: String identifying the desired reflector hardware profile.
        gasket_name: String identifying the desired gasket hardware profile.
        finish_type: String describing the reflector surface finish. 
        config: Loaded instance of SimulationConfig.
        library: Loaded instance of HardwareLibrary.
        save_path: String optional path indicating where to save the generated image. 
        
    Returns: 
        A dictionary containing the compiled quantitative metrics extracted from 
        the simulated hardware combination. 
    """
    selected_reflector = library.get_reflector(reflector_name)
    selected_emitter = library.get_emitter(emitter_name)
    selected_gasket = library.get_gasket(gasket_name)
    amps = selected_emitter["max_current_amps"]
    
    geom = get_sim_geometry(selected_reflector, selected_emitter, selected_gasket, finish_type, config)
    wall_lux, hotspot_lux, spill_lux, total_flux = run_pure_fea_sim_vectorized(geom, selected_emitter, amps, finish_type, config)

    max_cd = np.max(wall_lux) * (config.target_distance_m**2)
    throw_m = int(np.sqrt(max_cd / 0.25))
    render_data = apply_camera_exposure_and_tonemap(wall_lux, config)
    
    sp_ang, sp_sz, cor_ang, cor_sz, hot_ang, hot_sz, cd_lm = get_beam_metrics(wall_lux, hotspot_lux, spill_lux, max_cd, total_flux, config)

    cam_text = f"Exposure: Auto (EV {config.auto_exposure_compensation_ev:+.1f})" if config.use_auto_exposure else \
               f"Exposure: ISO {config.cam_iso} | f/{config.cam_f_stop} | {'1/'+str(int(1.0/config.cam_shutter_speed_s)) if config.cam_shutter_speed_s < 1.0 else config.cam_shutter_speed_s}s"

    geo_text = (f"Spill Angle: {sp_ang:.1f}°\nSpill Ø @ {config.target_distance_m}m: {sp_sz:.2f}m\n"
                f"Corona Angle: {cor_ang:.1f}°\nCorona Ø @ {config.target_distance_m}m: {cor_sz:.2f}m\n"
                f"Hotspot Angle: {hot_ang:.1f}°\nHotspot Ø @ {config.target_distance_m}m: {hot_sz:.2f}m\n"
                f"Cd/Lm Ratio: {cd_lm:.1f} cd/lm\n")

    table_str = " Mode | Amps | Lumens |  Candela | Throw \n" + "-"*46 + "\n"
    for pct in [0.01, 0.10, 0.35, 1.0]:
        amp_val = amps * pct
        lm_mode = calculate_lumens(selected_emitter, amp_val)
        cd_mode = max_cd * (lm_mode / total_flux)
        table_str += f"{int(pct*100):>4}% | {amp_val:>4.1f} | {int(lm_mode):>6,} | {int(cd_mode):>8,} | {int(np.sqrt(cd_mode * 4)):>4,}m\n"

    title_str = (f"Hardware: {emitter_name} | Reflector: {reflector_name} ({finish_type.upper()}) | Gasket: {gasket_name}\n"
                 f"Opening: {geom['effective_d_hole']:.1f}mm | Focus Delta: {geom['focus_delta']:+.2f}mm | Max Intensity: {int(max_cd):,} cd | Throw: {throw_m:,}m")

    # --- Plot Rendering ---
    if config.plot_wall_shot:
        fig_wall, ax_wall = plt.subplots(figsize=(10, 10), facecolor='black')
        ax_wall.set_facecolor('black')
        ax_wall.imshow(render_data, extent=[-config.wall_radius_m, config.wall_radius_m, -config.wall_radius_m, config.wall_radius_m], cmap='gray', origin='lower', vmin=0, vmax=1)
        ax_wall.set(xlim=(-config.plot_radius_m, config.plot_radius_m), ylim=(-config.plot_radius_m, config.plot_radius_m))
        ax_wall.set_xlabel("Horizontal Distance (m)", color='#CCCCCC', fontsize=11, labelpad=10)
        ax_wall.set_ylabel("Vertical Distance (m)", color='#CCCCCC', fontsize=11, labelpad=10)
        ax_wall.tick_params(colors='#CCCCCC', labelsize=10)
        for spine in ax_wall.spines.values(): spine.set_color('#555555')

        if config.show_human_silhouette: draw_human_silhouette(ax_wall, 0.0, -1.75 * 0.65, 1.75)

        ax_wall.text(0.02, 0.02, geo_text.strip(), transform=ax_wall.transAxes, color='#CCCCCC', fontsize=10, va='bottom', bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=6))
        ax_wall.text(0.98, 0.02, table_str.strip(), transform=ax_wall.transAxes, color='#CCCCCC', fontsize=10, family='monospace', ha='right', va='bottom', bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=6))

        plt.figtext(0.5, 0.015, f"Canvas FOV: {config.canvas_fov_deg}° | Plot FOV: {config.plot_fov_deg}° | Grid Res: {((2.0 * config.wall_radius_m) / config.sim_grid_res)*1000.0:.1f} mm/px | [{cam_text}]", color='#CCCCCC', fontsize=10, ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=4))
        plt.title(title_str, color='#CCCCCC', pad=15)
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        if save_path:
            plt.savefig(save_path, facecolor='black', edgecolor='none', dpi=150, bbox_inches='tight')
            print(f"Saved wall plot to: {save_path}")

    # --- Structural 1D Profiles ---
    x_dist = np.linspace(-config.wall_radius_m, config.wall_radius_m, config.sim_grid_res)
    center = int((config.sim_grid_res - 1) / 2.0)
    
    if config.plot_intensity_x: render_intensity_profile(wall_lux[center, :], x_dist, "X-Axis", title_str, save_path, config)
    if config.plot_intensity_y: render_intensity_profile(wall_lux[:, center], x_dist, "Y-Axis", title_str, save_path, config)
    if config.plot_intensity_45: render_intensity_profile(np.diagonal(wall_lux), np.linspace(-config.wall_radius_m * math.sqrt(2), config.wall_radius_m * math.sqrt(2), config.sim_grid_res), "45-Deg", title_str, save_path, config)

    if not config.generate_all_plots: plt.show()
    else: plt.close('all')

    return {
        "Reflector": reflector_name, "Emitter": emitter_name, "Gasket": gasket_name, "Finish": finish_type.upper(),
        "Max Candela (cd)": int(max_cd), "Throw (m)": int(throw_m), "Total Lumens": int(total_flux),
        "Spill Angle (deg)": round(sp_ang, 1), "Corona Angle (deg)": round(cor_ang, 1),
        "Hotspot Angle (deg)": round(hot_ang, 1), "Cd/Lm Ratio": round(cd_lm, 1)
    }

# ==============================================================================
# 5. EXECUTION ROUTING
# ==============================================================================

if __name__ == '__main__':
    # Initialize the core classes
    config = SimulationConfig(filepath="simulation_settings.json", default_filepath="default_settings.json")
    library = HardwareLibrary(filepath="hardware_library.json")
    
    # Initialize the output directory to dump results.
    os.makedirs(config.batch_output_directory, exist_ok=True)
    
    # Establish string identifiers to track iteration history.
    exposure_id = f"Auto_EV_{config.auto_exposure_compensation_ev:+.1f}" if config.use_auto_exposure else f"ISO{config.cam_iso}_f{config.cam_f_stop}_{'1_'+str(int(1.0/config.cam_shutter_speed_s)) if config.cam_shutter_speed_s < 1.0 else config.cam_shutter_speed_s}s"
    csv_filepath = os.path.join(config.batch_output_directory, f"sim_results_{config.target_distance_m}m_{exposure_id}.csv")
    
    # Added Gasket to the CSV headers
    csv_headers = ["Reflector", "Emitter", "Gasket", "Finish", "Max Candela (cd)", "Throw (m)", "Total Lumens", "Spill Angle (deg)", "Corona Angle (deg)", "Hotspot Angle (deg)", "Cd/Lm Ratio"]

    existing_data = {}
    if os.path.exists(csv_filepath):
        # Load historical metrics memory to support incremental generation.
        with open(csv_filepath, mode='r', newline='') as f:
            for row in csv.DictReader(f): existing_data[(row["Reflector"], row["Emitter"], row["Gasket"], row["Finish"])] = row

    if config.generate_all_plots:
        print(f"Batch generation enabled. Outputting to: {config.batch_output_directory}")
        # Verify combinatorial bounds. Discard permutations that are physically impossible.
        valid_combos = []
        for r_name in library.list_reflectors():
            rd = library.get_reflector(r_name)
            for e_name in library.list_emitters():
                ed = library.get_emitter(e_name)
                for g_name in library.list_gaskets():
                    for f in ["smooth", "orange_peel"]:
                        if np.sqrt(ed["footprint_x_mm"]**2 + ed["footprint_y_mm"]**2) <= (rd["diameter_mm"] / 3.0):
                            valid_combos.append((r_name, e_name, g_name, f))
        
        for i, (r_name, e_name, g_name, fin) in enumerate(valid_combos, 1):
            print(f"\n[{i}/{len(valid_combos)}] Rendering {r_name} + {e_name} + {g_name} ({fin.upper()})...")
            metrics = generate_flashlight_plot(e_name, r_name, g_name, fin, config, library, os.path.join(config.batch_output_directory, f"{r_name}_{e_name}_{g_name}_{'OP' if fin == 'orange_peel' else 'SMO'}.png"))
            existing_data[(metrics["Reflector"], metrics["Emitter"], metrics["Gasket"], metrics["Finish"])] = metrics
        print("\nBatch generation complete!")
    else:
        # Pass a single simulation workload through the system.
        print(f"\nRendering {config.active_reflector_name} + {config.active_emitter_name} + {config.active_gasket_name} ({config.reflector_finish.upper()})...")
        metrics = generate_flashlight_plot(config.active_emitter_name, config.active_reflector_name, config.active_gasket_name, config.reflector_finish, config, library, os.path.join(config.batch_output_directory, f"{config.active_reflector_name}_{config.active_emitter_name}_{config.active_gasket_name}_{'OP' if config.reflector_finish == 'orange_peel' else 'SMO'}.png"))
        existing_data[(metrics["Reflector"], metrics["Emitter"], metrics["Gasket"], metrics["Finish"])] = metrics
        print("\nSingle generation complete!")

    # Serialize memory back to disk.
    with open(csv_filepath, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for row in existing_data.values(): writer.writerow(row)
            
    print(f"Results successfully saved to: {csv_filepath}")

# Flashlight FEA Ray-Tracing Simulator

A hardware-accelerated Python finite element analysis (FEA) engine for simulating flashlight beam profiles. This tool calculates ray-tracing intersections against parabolic reflectors to generate photorealistic beam shots and spatial intensity profiles.

## Features

* **Hardware Modeling:** Built-in libraries for popular emitters (Nichia, Luminus, Cree, Osram) and reflectors (Convoy series).
* **Diode Physics:** Calculates theoretical lumens incorporating $V_f$ curves and efficiency droop: $V(I) = V_{turn\_on} + V_{scale} \ln(I + 1)$.
* **GPU Acceleration:** Uses Numba and CUDA to process rays in parallel, with an automatic fallback to CPU vectorization.
* **Photorealistic Rendering:** Uses ACES filmic tone mapping and adjustable camera EV settings.
* **Batching:** Generates matrix comparisons of multiple hardware combinations and exports the geometric data (throw, spill angle, candela/lumen ratio) to a CSV log.

## Installation & Setup (Windows)

This project relies on strict mathematical libraries and GPU compute toolkits. The easiest way to get it running on Windows is using Anaconda and Visual Studio Code.

### 1. Install Anaconda

1. Download the installer from [Anaconda.com](https://www.anaconda.com/download) and run it.
2. Once installed, open the **Anaconda Prompt** from your Windows Start Menu.

### 2. Create the Conda Environment

In the Anaconda Prompt, create a dedicated environment and install the required packages (including the CUDA toolkit required for Numba's GPU acceleration):

```bash
conda create -n flashlight-sim python=3.11
conda activate flashlight-sim
conda install numpy scipy matplotlib numba cudatoolkit
```

### 3. Configure Visual Studio Code

Install Visual Studio Code.

Open VS Code and install the official Python extension from the Extensions marketplace (`Ctrl+Shift+X`).

Open the folder containing this simulator code.

Press `Ctrl+Shift+P` to open the Command Palette, type `Python: Select Interpreter`, and select it.

Find the environment named `flashlight-sim` in the list (it should be marked with "conda") and select it.

Open the main `.py` script and hit the Play button in the top right to run the simulation.

## How the Code Works

The simulator is divided into several discrete pipelines:

### Hardware Libraries

The `EMITTERS` and `REFLECTORS` dictionaries act as physical data sheets. They provide the die sizes, forward voltage characteristics, and mechanical dimensions (like reflector hole diameter and focus offset) required to build the virtual 3D space.

### Emission & Lumen Math

The engine integrates a standard Lambertian (cosine) emission curve over the solid angle of a sphere. Based on the selected amperage, it scales the total flux by accounting for electrical efficiency droop, distributing rays evenly across the discretized LED die footprint.

If the emitter has a silicone dome, it calculates refraction angles using Snell's Law before the ray leaves the virtual LED.

## The FEA Ray-Tracer

### GPU/CPU Kernels

The engine uses `@cuda.jit` and `@njit` to process the workload on a CUDA-enabled GPU if available, with an automatic fallback to the CPU.

### Intersection Math

Each ray checks for intersections against:

* Planes (the flat gasket and shelves)
* A cylinder (the hole at the base of the reflector)
* A paraboloid (the main reflective surface)

### Rendering

Escaped rays land on a virtual target plane. The raw lux data is passed through a simulated camera sensor (factoring in ISO, f-stop, and shutter speed) and an ACES tone-mapping curve to generate a photorealistic 2D array, which is plotted using Matplotlib.

## Usage

You can control the simulation entirely through the variables at the top of the script under `QUICK-SET ACTIVE HARDWARE SELECTION & VISUALIZATION`.

```python
active_emitter_name = "SFT60_6500K"
active_reflector_name = "S2_S6_S8_T6"
reflector_finish = "orange_peel"
```

To run a batch calculation of all viable hardware combinations, set:

```python
generate_all_plots = True
```

The outputs will be dumped into your designated output folder alongside a CSV metric log.

## License

GNU GPL v3 License. See the `LICENSE` file for details.

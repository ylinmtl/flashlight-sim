"""Packages the flashlight simulator into a portable Windows folder.

Numba compiles CUDA kernels at runtime, so the frozen application needs a real
CUDA toolkit next to it: libNVVM, libdevice, the CUDA runtime and their
dependencies. PyInstaller cannot work that out on its own, so this script pulls
the pieces out of the active Conda environment, arranges them in the layout
Numba expects, and hands the result to PyInstaller as data files.

Run it from the project folder with the target Conda environment active:

    python build.py

The result is dist/flashlight-sim, which can be copied to any Windows machine
with an NVIDIA driver. Without a usable GPU the application falls back to the
CPU tracer on its own.
"""

import fnmatch
import os
import shutil
import subprocess
import sys

# Where the staged toolkit is assembled before PyInstaller collects it.
STAGING_DIR = "cuda_bundle"

# Application files that must be inside the bundle, as (source, destination).
APP_DATA_FILES = [
    ("mainwindow.ui", "."),
    ("hardware_library.json", "."),
    ("default_settings.json", "."),
]

# Packages whose data files and hidden imports PyInstaller cannot infer.
COLLECT_ALL_PACKAGES = ["numba", "llvmlite", "scipy"]

# Environment sub-directories that hold copies, stubs or caches rather than the
# libraries actually in use.
SKIP_DIRS = frozenset({"pkgs", "stubs", "info", ".conda", "conda-meta"})

# Minimum size of a real CUDA binary, used to reject placeholder files.
MIN_BINARY_BYTES = 10_000


def find_environment_file(pattern, env_path):
    """Finds the largest file matching a pattern in the active environment.

    Several copies of a CUDA library can exist in one environment. The largest
    is the real one; import stubs and placeholders are much smaller.

    Args:
        pattern: Case-insensitive glob, for example "nvvm64*.dll".
        env_path: Root of the environment to search.

    Returns:
        Absolute path of the best match, or None if there is no match.
    """
    matches = []
    for root, dirs, files in os.walk(env_path):
        dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS]
        for filename in files:
            if not fnmatch.fnmatch(filename.lower(), pattern.lower()):
                continue
            path = os.path.join(root, filename)
            if os.path.isfile(path) and os.path.getsize(path) > MIN_BINARY_BYTES:
                matches.append(path)

    if not matches:
        return None
    return max(matches, key=os.path.getsize)


def locate_cuda_components(env_path):
    """Locates every CUDA file the frozen application needs.

    Args:
        env_path: Root of the environment to search.

    Returns:
        A dict of component name to path. Optional components map to None when
        they are not installed.
    """
    return {
        "nvvm": (find_environment_file("nvvm64*.dll", env_path)
                 or find_environment_file("nvvm*.dll", env_path)),
        "cudart": find_environment_file("cudart64*.dll", env_path),
        "ptxas": find_environment_file("ptxas.exe", env_path),
        "libdevice": find_environment_file("libdevice*.bc", env_path),
        "zlib": (find_environment_file("zlibwapi.dll", env_path)
                 or find_environment_file("zlib.dll", env_path)),
        "msvcp": find_environment_file("msvcp140.dll", env_path),
        "vcruntime": find_environment_file("vcruntime140.dll", env_path),
    }


def check_required_components(components):
    """Exits with instructions if a component the GPU tracer needs is missing.

    Args:
        components: Output of locate_cuda_components.
    """
    required = {
        "nvvm": "nvvm64*.dll",
        "cudart": "cudart64*.dll",
        "ptxas": "ptxas.exe",
        "libdevice": "libdevice*.bc",
    }
    missing = [pattern for key, pattern in required.items() if not components[key]]

    if missing:
        print("\n[!] ERROR: Missing core CUDA files. "
              "Run: conda install -c conda-forge cudatoolkit=11.8")
        for pattern in missing:
            print(f"  - {pattern}")
        sys.exit(1)

    if not components["zlib"]:
        print("\n[!] WARNING: no zlib/zlibwapi DLL found. CUDA 11.x libNVVM needs "
              "zlibwapi.dll.")
        print("    If the GPU path falls back to CPU at runtime, run: "
              "conda install -c conda-forge zlib-wapi")


def stage_cuda_toolkit(components):
    """Copies the CUDA components into the layout Numba looks for.

    At runtime fea_engine points CUDA_HOME at the bundle root, and Numba then
    expects a genuine toolkit underneath it:

        bin/            cudart, ptxas and their dependencies
        nvvm/bin/       libNVVM, which is where Numba looks on Windows
        nvvm/libdevice/ the bitcode library libNVVM links against

    libNVVM's own dependencies are staged beside it as well as in bin/, because
    Windows resolves a DLL's imports from the directory it was loaded from.

    Args:
        components: Output of locate_cuda_components.
    """
    if os.path.exists(STAGING_DIR):
        shutil.rmtree(STAGING_DIR)

    def stage(source, *destination_parts, rename=None):
        """Copies one file into the staging tree, if it was found."""
        if not source:
            return
        destination_dir = os.path.join(STAGING_DIR, *destination_parts)
        os.makedirs(destination_dir, exist_ok=True)
        destination = os.path.join(destination_dir, rename or os.path.basename(source))
        shutil.copy2(source, destination)
        print(f"   {os.path.relpath(destination, STAGING_DIR)}  <-  {source}")

    stage(components["nvvm"], "nvvm", "bin")
    stage(components["libdevice"], "nvvm", "libdevice")
    stage(components["cudart"], "bin")
    stage(components["ptxas"], "bin")

    for dependency in (components["zlib"], components["msvcp"], components["vcruntime"]):
        stage(dependency, "bin")
        stage(dependency, "nvvm", "bin")

    # Conda sometimes ships only zlib.dll, but libNVVM imports it by the name
    # zlibwapi.dll. On x64 there is one calling convention, so a copy works.
    zlib_path = components["zlib"]
    if zlib_path and os.path.basename(zlib_path).lower() != "zlibwapi.dll":
        stage(zlib_path, "bin", rename="zlibwapi.dll")
        stage(zlib_path, "nvvm", "bin", rename="zlibwapi.dll")


def build_add_data_arguments():
    """Builds one --add-data argument per staged file.

    Directory sources changed meaning between PyInstaller 5 and 6, so every file
    is listed individually with its own destination directory, which both
    versions interpret the same way.

    Returns:
        A flat list of command line arguments.
    """
    arguments = []
    for source, destination in APP_DATA_FILES:
        arguments += ["--add-data", f"{source};{destination}"]

    for root, _, files in os.walk(STAGING_DIR):
        relative_dir = os.path.relpath(root, STAGING_DIR)
        destination = "." if relative_dir == "." else relative_dir
        for filename in files:
            arguments += ["--add-data", f"{os.path.join(root, filename)};{destination}"]

    return arguments


def run_pyinstaller():
    """Runs PyInstaller in the current interpreter's environment.

    Raises:
        SystemExit: If PyInstaller reports a failure.
    """
    command = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--noconsole",
        "--onedir",
        "--name", "flashlight-sim",
    ]
    command += build_add_data_arguments()
    for package in COLLECT_ALL_PACKAGES:
        command += ["--collect-all", package]
    command.append("flashlight-sim.py")

    if subprocess.run(command).returncode != 0:
        sys.exit("PyInstaller failed.")


def verify_bundle():
    """Reports whether the CUDA toolkit survived into the finished bundle.

    PyInstaller 6 nests data files under _internal, version 5 does not, so both
    locations are checked.
    """
    root = os.path.join("dist", "flashlight-sim")
    internal = os.path.join(root, "_internal")
    target = internal if os.path.isdir(internal) else root

    print(f"\nVerifying bundle at {target}:")
    for relative_dir in (os.path.join("nvvm", "bin"),
                         os.path.join("nvvm", "libdevice"),
                         "bin"):
        path = os.path.join(target, relative_dir)
        if os.path.isdir(path):
            print(f"  [OK] {relative_dir}  {os.listdir(path)}")
        else:
            print(f"  [MISSING] {relative_dir}")


def main():
    """Locates the CUDA toolkit, stages it and builds the application."""
    env_path = sys.prefix
    print(f"Scanning active Conda environment in: {env_path}...")

    components = locate_cuda_components(env_path)
    check_required_components(components)

    print("\n -> Building CUDA_HOME-shaped toolkit...")
    stage_cuda_toolkit(components)

    print("\nBundling into executable...")
    run_pyinstaller()
    verify_bundle()

    print("\nBuild Complete! Check the 'dist/flashlight-sim' folder.")


if __name__ == "__main__":
    main()

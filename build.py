import os
import sys
import shutil
import subprocess
import fnmatch

env_path = sys.prefix
STAGING = "cuda_bundle"
SKIP_DIRS = {'pkgs', 'stubs', 'info', '.conda', 'conda-meta'}

def find_active_cuda_file(pattern, min_size=10_000):
    """Search the active env, blocking cache and stub folders."""
    candidates = []
    for root, dirs, files in os.walk(env_path):
        dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS]
        for f in files:
            if fnmatch.fnmatch(f.lower(), pattern.lower()):
                p = os.path.join(root, f)
                if os.path.isfile(p) and os.path.getsize(p) > min_size:
                    candidates.append(p)
    return sorted(candidates, key=os.path.getsize, reverse=True)[0] if candidates else None

print(f"Scanning active Conda environment in: {env_path}...")

nvvm_src      = find_active_cuda_file("nvvm64*.dll") or find_active_cuda_file("nvvm*.dll")
cudart_src    = find_active_cuda_file("cudart64*.dll")
ptxas_src     = find_active_cuda_file("ptxas.exe")
libdevice_src = find_active_cuda_file("libdevice*.bc")
zlib_src      = find_active_cuda_file("zlibwapi.dll") or find_active_cuda_file("zlib.dll")
msvcp_src     = find_active_cuda_file("msvcp140.dll")
vcruntime_src = find_active_cuda_file("vcruntime140.dll")

missing = [n for n, v in [("nvvm64*.dll", nvvm_src), ("cudart64*.dll", cudart_src),
                          ("ptxas.exe", ptxas_src), ("libdevice*.bc", libdevice_src)] if not v]
if missing:
    print("\n[!] ERROR: Missing core CUDA files. Run: conda install -c conda-forge cudatoolkit=11.8")
    for m in missing:
        print(f"  - {m}")
    sys.exit(1)

if not zlib_src:
    print("\n[!] WARNING: no zlib/zlibwapi DLL found. CUDA 11.x libNVVM needs zlibwapi.dll.")
    print("    If the GPU path falls back to CPU at runtime, run: conda install -c conda-forge zlib-wapi")

if os.path.exists(STAGING):
    shutil.rmtree(STAGING)

def stage(src, *dest_parts, rename=None):
    if not src:
        return
    dest_dir = os.path.join(STAGING, *dest_parts)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, rename or os.path.basename(src))
    shutil.copy2(src, dest)
    print(f"   {os.path.relpath(dest, STAGING)}  <-  {src}")

print("\n -> Building CUDA_HOME-shaped toolkit...")

# THE KEY CHANGE: libNVVM lives in nvvm/bin on Windows, NOT in bin/.
stage(nvvm_src, "nvvm", "bin")
stage(libdevice_src, "nvvm", "libdevice")
stage(cudart_src, "bin")
stage(ptxas_src, "bin")

# nvvm's own dependencies must sit beside nvvm64_*.dll as well as in bin/.
for dep in (zlib_src, msvcp_src, vcruntime_src):
    stage(dep, "bin")
    stage(dep, "nvvm", "bin")

# conda sometimes ships zlib.dll only; libNVVM imports it as zlibwapi.dll.
if zlib_src and os.path.basename(zlib_src).lower() != "zlibwapi.dll":
    stage(zlib_src, "bin", rename="zlibwapi.dll")
    stage(zlib_src, "nvvm", "bin", rename="zlibwapi.dll")

# Emit one --add-data per FILE. Directory sources changed semantics between
# PyInstaller 5 and 6; per-file destinations are unambiguous in both.
add_data_args = []
for root, _, files in os.walk(STAGING):
    rel = os.path.relpath(root, STAGING)
    dest = "." if rel == "." else rel
    for f in files:
        add_data_args += ["--add-data", f"{os.path.join(root, f)};{dest}"]

print("\nBundling into executable...")

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm", "--clean",
    "--noconsole",
    "--onedir",
    "--name", "flashlight-sim",
    "--add-data", "mainwindow.ui;.",
    "--add-data", "hardware_library.json;.",
    "--add-data", "default_settings.json;.",
    *add_data_args,
    "--collect-all", "numba",
    "--collect-all", "llvmlite",
    "--collect-all", "scipy",
    "flashlight-sim.py",
]

if subprocess.run(cmd).returncode != 0:
    sys.exit("PyInstaller failed.")

# Verify the layout actually landed (PyInstaller 6 nests data under _internal).
root = os.path.join("dist", "flashlight-sim")
target = os.path.join(root, "_internal") if os.path.isdir(os.path.join(root, "_internal")) else root
print(f"\nVerifying bundle at {target}:")
for rel in (os.path.join("nvvm", "bin"), os.path.join("nvvm", "libdevice"), "bin"):
    p = os.path.join(target, rel)
    print(f"  [{'OK' if os.path.isdir(p) else 'MISSING'}] {rel}" +
          (f"  {os.listdir(p)}" if os.path.isdir(p) else ""))

print("\nBuild Complete! Check the 'dist/flashlight-sim' folder.")
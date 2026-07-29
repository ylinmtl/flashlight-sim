"""Pointing Numba at the CUDA toolkit bundled with the app.

Part of the flashlight simulator core; see core/__init__.py for the
public surface.
"""

import os
import sys


def _bootstrap_bundled_cuda() -> list:
    """Points Numba at the CUDA toolkit bundled inside a frozen build.

    Numba locates the toolkit through CUDA_HOME (falling back to CUDA_PATH) and
    expects a genuine toolkit layout underneath it:

        <CUDA_HOME>/nvvm/bin/nvvm64_*.dll
        <CUDA_HOME>/nvvm/libdevice/libdevice.*.bc
        <CUDA_HOME>/bin/cudart64_*.dll  (plus ptxas.exe and zlibwapi.dll)

    When that lookup fails Numba falls back to the bare name "nvvm.dll", which a
    frozen build cannot resolve, and the first kernel launch dies with
    NvvmSupportError. Python 3.8 and later ignore PATH when loading DLLs, so the
    directories must also be registered with os.add_dll_directory.

    Returns:
        The directory handles returned by os.add_dll_directory. They are kept
        alive for the lifetime of the process; dropping them un-registers the
        directories.
    """
    handles = []
    if not (getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")):
        return handles

    bundle_dir = sys._MEIPASS
    cuda_bin = os.path.join(bundle_dir, "bin")
    nvvm_bin = os.path.join(bundle_dir, "nvvm", "bin")

    if os.path.isdir(nvvm_bin):
        os.environ["CUDA_HOME"] = bundle_dir
        os.environ["CUDA_PATH"] = bundle_dir

    for directory in (cuda_bin, nvvm_bin):
        if not os.path.isdir(directory):
            continue
        # PATH still matters for child processes such as ptxas.exe.
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            try:
                handles.append(os.add_dll_directory(directory))
            except OSError:
                pass
    return handles
# Must run before Numba is imported: Numba caches its toolkit path lookup
# the first time it is asked for it. The handles are kept in a module
# level name on purpose. Nothing reads them, but dropping the reference
# would let the loaded DLLs be released again.
_CUDA_DLL_DIRECTORIES = _bootstrap_bundled_cuda()

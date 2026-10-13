"""Platform helpers: peak memory of the current process, keeping the machine awake, and
a description of the machine for the result files."""
import os
import platform
import subprocess
import sys


def peak_mem_mb():
    """Peak resident memory of this process so far, in MB (macOS, Linux, Windows)."""
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / 2 ** 20) if sys.platform == "darwin" else round(r / 2 ** 10)   # bytes on macOS, KB on Linux
    except ImportError:                                  # Windows
        import psutil
        return round(psutil.Process().memory_info().peak_wset / 2 ** 20)


def keep_awake():
    """Prevent system sleep while this process runs (Windows, macOS); no-op elsewhere."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    elif sys.platform == "darwin":
        try:
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        except OSError:
            pass


def machine():
    """Short description of the machine and the software versions."""
    import numpy
    import scipy
    try:
        import ortools
        ort = ortools.__version__
    except Exception:  # noqa: BLE001
        ort = "?"
    try:
        import psutil
        mem = f"{psutil.virtual_memory().total / 2 ** 30:.0f} GB"
        cores = f"{psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads"
    except ImportError:
        mem, cores = "?", f"{os.cpu_count()} threads"
    cpu = platform.processor() or platform.machine()
    if sys.platform == "win32":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return {"platform": platform.platform(), "cpu": cpu, "cores": cores, "memory": mem,
            "python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "ortools": ort}

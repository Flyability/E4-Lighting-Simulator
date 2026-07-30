"""
Build script: creates a standalone .exe desktop app using PyInstaller.

Usage:
    python build_exe.py

Output:
    dist/LightingSim/LightingSim.exe   (folder mode — includes all DLLs)
"""
import subprocess
import sys
import os

# Ensure PyInstaller is installed
try:
    import PyInstaller
except ImportError:
    print("Installing PyInstaller...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

# Data directories to bundle (configs, custom_groups_templates, exports)
script_dir = os.path.dirname(os.path.abspath(__file__))

datas = [
    (os.path.join(script_dir, "configs"), "configs"),
    (os.path.join(script_dir, "custom_groups_templates"), "custom_groups_templates"),
]
exports = os.path.join(script_dir, "exports")
if os.path.isdir(exports):
    datas.append((exports, "exports"))

# Also bundle _mesh_worker.py (used by ProcessPoolExecutor)
mesh_worker = os.path.join(script_dir, "_mesh_worker.py")
if os.path.isfile(mesh_worker):
    datas.append((mesh_worker, "."))

# Also bundle gpu_raytrace.py
gpu_rt = os.path.join(script_dir, "gpu_raytrace.py")
if os.path.isfile(gpu_rt):
    datas.append((gpu_rt, "."))

# CSV files
for f in os.listdir(script_dir):
    if f.endswith(".csv"):
        datas.append((os.path.join(script_dir, f), "."))

# Build --add-data arguments
add_data_args = []
for src, dst in datas:
    add_data_args.extend(["--add-data", f"{src};{dst}"])

# Hidden imports that PyInstaller might miss
hidden_imports = [
    "viser",
    "viser.transforms",
    "viser._scene_api",
    "trimesh",
    "trimesh.ray",
    "trimesh.ray.ray_pyembree",
    "trimesh.visual",
    "trimesh.visual.color",
    "embreex",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL.ImageFont",
    "numpy",
    "json",
    "hashlib",
    "multiprocessing",
    "concurrent.futures",
    "socket",
    "webbrowser",
]

hidden_args = []
for h in hidden_imports:
    hidden_args.extend(["--hidden-import", h])

# Try to add cupy if available
try:
    import cupy
    hidden_args.extend(["--hidden-import", "cupy"])
except ImportError:
    pass

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--name", "LightingSim",
    "--noconfirm",
    # "--onefile",  # Use folder mode for faster startup and smaller RAM usage
    "--console",  # Keep console for diagnostic output
    "--icon", "NONE",
    *add_data_args,
    *hidden_args,
    "--collect-all", "viser",
    "--collect-all", "trimesh",
    os.path.join(script_dir, "interactive_lighting.py"),
]

print("Building .exe with PyInstaller...")
print(f"Command: {' '.join(cmd)}\n")
subprocess.check_call(cmd)

print("\n" + "=" * 60)
print("  BUILD COMPLETE!")
print("=" * 60)
print(f"\n  Executable: dist/LightingSim/LightingSim.exe")
print(f"  To distribute: zip the entire dist/LightingSim/ folder")
print("=" * 60)

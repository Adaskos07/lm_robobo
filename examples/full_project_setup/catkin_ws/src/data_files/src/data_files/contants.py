from pathlib import Path

# This only works because I'm assuming docker is started with run.ps1/run.sh
RESULT_DIR = Path("/root/results")
MODELS_DIR = Path("/root/models")

if not RESULT_DIR.is_dir():
    raise ImportError("Could not resolve location for results dir, or it is a file")

if not MODELS_DIR.is_dir():
    raise ImportError("Could not resolve location for models dir, or it is a file")

FIGURES_DIR = RESULT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

if not FIGURES_DIR.is_dir():
    raise ImportError("Could not resolve location for figures dir, or it is a file")

__all__ = ("RESULT_DIR", "FIGURES_DIR", "MODELS_DIR")

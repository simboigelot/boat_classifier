"""
Central definition of the project folder layout.

    <project root>/
        input/    data read by the code (training images, downloads, test images)
        src/      code
        output/   everything produced by the code (model, plots, contact sheets)

Paths are resolved from this file's location, so the scripts work regardless of
the current working directory. output/ is created automatically on import.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Inputs
INPUT_DIR = ROOT / "input"
DATA_DIR = INPUT_DIR / "data"                    # training images, one sub-folder per class
REJECTED_DIR = INPUT_DIR / "data_rejected"       # images removed after manual review
TEST_IMAGES_DIR = INPUT_DIR / "test_images"      # ad-hoc images for --predict

# Outputs
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

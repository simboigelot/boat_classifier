"""
Boat image classifier: motor boats vs sail boats (fastai + pretrained ResNet).

Pipeline
--------
1. Download    : query DuckDuckGo image search for each class (several queries per
                 class for visual diversity) and save images to input/data/<class>/.
2. Clean       : delete files that cannot be opened as images (fastai verify_images).
3. DataLoaders : random 80/20 train/validation split (seeded), images resized to
                 IMG_SIZE x IMG_SIZE, standard fastai data augmentation on train set.
4. Train       : transfer learning - fine-tune an ImageNet-pretrained ResNet.
5. Evaluate    : error rate on validation set, confusion matrix, top losses (saved as PNG).
6. Export      : learner exported to output/boat_classifier.pkl for later inference.

Folder layout (see paths.py): inputs in input/, code in src/, all outputs in output/.

Usage (from the project root)
-----
    python src/boat_classifier.py                      # download + train
    python src/boat_classifier.py --skip-download      # retrain on existing images
    python src/boat_classifier.py --predict test1.jpeg # inference (file name looked up
                                                       # in input/test_images/, or any path)
"""

import argparse
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: save figures to file instead of opening windows
import matplotlib.pyplot as plt

from fastai.vision.all import (
    CategoryBlock, DataBlock, ImageBlock, ClassificationInterpretation,
    RandomSplitter, Resize, aug_transforms, download_images, error_rate,
    get_image_files, load_learner, parent_label, resnet34, set_seed,
    verify_images, vision_learner, PILImage,
)

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
from paths import DATA_DIR, OUTPUT_DIR, TEST_IMAGES_DIR

MODEL_PATH = OUTPUT_DIR / "boat_classifier.pkl"


# Several search queries per class -> more varied images (angles, sizes, settings).
# Folder name = class label.
CLASSES = {
    "motor_boat": ["motor boat", "speedboat", "motor yacht", "fishing motor boat"],
    "sail_boat":  ["sailboat", "sailing yacht", "sailboat at sea", "small sailing dinghy"],
}

IMAGES_PER_QUERY = 50   # ~200 raw images per class before cleaning
VALID_PCT = 0.2         # 20% of images held out for validation
IMG_SIZE = 224          # ResNet's native ImageNet resolution
BATCH_SIZE = 32
EPOCHS = 4              # fine-tuning epochs after 1 frozen epoch
SEED = 42               # reproducibility of split, augmentation, and init


# --------------------------------------------------------------------------- #
# 1. Download                                                                 #
# --------------------------------------------------------------------------- #
def search_image_urls(query: str, max_images: int) -> list[str]:
    """Return image URLs from DuckDuckGo image search."""
    from ddgs import DDGS  # imported here so --predict does not require it
    results = DDGS().images(query, max_results=max_images)
    return [r["image"] for r in results if "image" in r]


def download_dataset() -> None:
    for label, queries in CLASSES.items():
        dest = DATA_DIR / label
        dest.mkdir(parents=True, exist_ok=True)
        for q in queries:
            try:
                urls = search_image_urls(q, IMAGES_PER_QUERY)
            except Exception as e:  # rate limiting / network errors
                print(f"  [warn] search failed for '{q}': {e}")
                continue
            print(f"[{label}] '{q}': {len(urls)} URLs found, downloading...")
            download_images(dest, urls=urls, max_pics=IMAGES_PER_QUERY, timeout=10)
            time.sleep(2)  # be polite to the search endpoint, reduces rate-limit risk


# --------------------------------------------------------------------------- #
# 2. Clean                                                                    #
# --------------------------------------------------------------------------- #
def clean_dataset() -> None:
    files = get_image_files(DATA_DIR)
    failed = verify_images(files)
    failed.map(Path.unlink)
    print(f"Removed {len(failed)} corrupt/unreadable images.")
    for label in CLASSES:
        n = len(get_image_files(DATA_DIR / label))
        print(f"  {label}: {n} images")


# --------------------------------------------------------------------------- #
# 3. DataLoaders                                                              #
# --------------------------------------------------------------------------- #
def build_dataloaders():
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=VALID_PCT, seed=SEED),
        get_y=parent_label,                      # label = folder name
        item_tfms=Resize(IMG_SIZE, method="squish"),
        batch_tfms=aug_transforms(),             # flips, rotations, zoom, lighting
    )
    # num_workers=0 avoids multiprocessing issues on Windows/WSL file systems
    return dblock.dataloaders(DATA_DIR, bs=BATCH_SIZE, num_workers=0)


# --------------------------------------------------------------------------- #
# 4-6. Train, evaluate, export                                                #
# --------------------------------------------------------------------------- #
def train_and_evaluate():
    set_seed(SEED, reproducible=True)
    random.seed(SEED)

    dls = build_dataloaders()
    print(f"Classes: {dls.vocab} | train: {len(dls.train_ds)} | valid: {len(dls.valid_ds)}")
    dls.show_batch(max_n=8)
    plt.savefig(OUTPUT_DIR / "sample_batch.png", bbox_inches="tight"); plt.close("all")

    learn = vision_learner(dls, resnet34, metrics=error_rate)
    learn.fine_tune(EPOCHS)

    # Evaluation on the validation set
    loss, err = learn.validate()
    print("Evaluation of ResNet34 with 4 epochs of fine-tuning")
    print(f"Validation loss: {loss}  |  error rate: {err}  |  accuracy: {1 - err}")

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", bbox_inches="tight"); plt.close("all")
    interp.plot_top_losses(9, nrows=3)
    plt.savefig(OUTPUT_DIR / "top_losses.png", bbox_inches="tight"); plt.close("all")
    print(f"Saved sample_batch.png, confusion_matrix.png, top_losses.png in {OUTPUT_DIR}")

    learn.export(MODEL_PATH)
    print(f"Model exported to {MODEL_PATH}")

def evaluate_no_fine_tune():
    set_seed(SEED, reproducible=True)
    random.seed(SEED)

    dls = build_dataloaders()
    print(f"Classes: {dls.vocab} | train: {len(dls.train_ds)} | valid: {len(dls.valid_ds)}")
    
    learn = vision_learner(dls, resnet34, metrics=error_rate)

    # Evaluation on the validation set
    loss, err = learn.validate()
    print("Evaluation of ResNet34 without fine-tuning:")
    print(f"Validation loss: {loss}  |  error rate: {err}  |  accuracy: {1 - err}")


def predict(img_path: str) -> None:
    # Accept either a real path or a bare file name located in input/test_images/
    path = Path(img_path)
    if not path.exists() and (TEST_IMAGES_DIR / img_path).exists():
        path = TEST_IMAGES_DIR / img_path
    learn = load_learner(MODEL_PATH)
    label, _, probs = learn.predict(PILImage.create(path))
    print(f"Prediction: {label}")
    for cls, p in zip(learn.dls.vocab, probs):
        print(f"  P({cls}) = {p.item()}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motor boat vs sail boat classifier")
    parser.add_argument("--skip-download", action="store_true",
                        help="reuse images already in input/data/")
    parser.add_argument("--predict", metavar="IMAGE",
                        help="classify a single image with the exported model")
    args = parser.parse_args()

    if args.predict:
        predict(args.predict)
    else:
        if not args.skip_download:
            download_dataset()
        clean_dataset()
        train_and_evaluate()
        evaluate_no_fine_tune()

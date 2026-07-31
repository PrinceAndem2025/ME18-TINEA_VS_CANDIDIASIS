"""
ME18: Tinea vs Candidiasis
--------------------------------------------
STEP 0 - Data preparation script.

BEFORE RUNNING THIS SCRIPT:
1. Download the "Skin Disease Dataset" from Kaggle:
   https://www.kaggle.com/datasets/pacificrm/skindiseasedataset
   (free Kaggle account required to download)
2. Extract it.
3. Find the "Tinea" and "Candidiasis" class folders inside the extracted
   dataset (out of its 22 total categories) and copy ONLY those two into
   this project's `dataset/` folder, named exactly like this:

   dataset/
   ├── Tinea/          <- put all Tinea images here
   └── Candidiasis/    <- put all Candidiasis images here

   If the original folder names inside the Kaggle download differ
   slightly (e.g. spacing, capitalization), just rename them to match
   exactly, or edit the CLASSES list below to match the original names.

WHAT THIS SCRIPT DOES:
- Reads images from dataset/Tinea and dataset/Candidiasis
- Splits them 70% train / 15% val / 15% test (stratified per class)
- Copies them into me18_train/, me18_val/, me18_test/ folders in the
  correct structure for Keras' image_dataset_from_directory
"""

import os
import shutil
import random

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
RAW_DATASET_DIR = "dataset"
CLASSES = ["Tinea", "Candidiasis"]

TRAIN_DIR = "me18_train"
VAL_DIR = "me18_val"
TEST_DIR = "me18_test"

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15   # (must sum to 1.0 with the two above)

SEED = 42
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def rebuild_dir(path):
    """Delete a folder if it exists, then recreate it empty."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def list_images(folder):
    return [
        f for f in os.listdir(folder)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    random.seed(SEED)

    print("Rebuilding train/val/test folders...")
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        rebuild_dir(split_dir)
        for cls in CLASSES:
            os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

    total_counts = {}

    for cls in CLASSES:
        src_folder = os.path.join(RAW_DATASET_DIR, cls)
        if not os.path.isdir(src_folder):
            raise FileNotFoundError(
                f"Could not find '{src_folder}'. Make sure you copied your "
                f"images into dataset/{cls}/ before running this script."
            )

        images = list_images(src_folder)
        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)
        n_test = n - n_train - n_val  # remainder, so all images are used

        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        for fname in train_files:
            shutil.copy2(os.path.join(src_folder, fname), os.path.join(TRAIN_DIR, cls, fname))
        for fname in val_files:
            shutil.copy2(os.path.join(src_folder, fname), os.path.join(VAL_DIR, cls, fname))
        for fname in test_files:
            shutil.copy2(os.path.join(src_folder, fname), os.path.join(TEST_DIR, cls, fname))

        total_counts[cls] = n
        print(f"{cls}: {n} images -> train={n_train}, val={n_val}, test={n_test}")

    total = sum(total_counts.values())
    print(f"\nTotal images found: {total}")
    for cls, count in total_counts.items():
        pct = 100 * count / total
        print(f"  {cls} makes up {pct:.1f}% of the dataset")

    print("\nDone. Folders me18_train/, me18_val/, me18_test/ are ready.")
    print("Next: run `python model.py` to train the classifier.")


if __name__ == "__main__":
    main()

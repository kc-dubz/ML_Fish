import os
import shutil
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
TRAIN_DIR = BASE_DIR / "data" / "train"
VAL_DIR = BASE_DIR / "data" / "val"
TEST_DIR = BASE_DIR / "data" / "test"

SPLIT_RATIO = (0.7, 0.15, 0.15)

def create_dirs():
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        split_dir.mkdir(parents=True, exist_ok=True)

def split_data():
    create_dirs()

    for class_folder in RAW_DIR.iterdir():
        if class_folder.is_dir():
            images = list(class_folder.glob("*"))
            random.shuffle(images)

            train_end = int(len(images) * SPLIT_RATIO[0])
            val_end = train_end + int(len(images) * SPLIT_RATIO[1])

            splits = {
                TRAIN_DIR: images[:train_end],
                VAL_DIR: images[train_end:val_end],
                TEST_DIR: images[val_end:]
            }

            for split_dir, split_images in splits.items():
                class_split_dir = split_dir / class_folder.name
                class_split_dir.mkdir(parents=True, exist_ok=True)

                for img in split_images:
                    shutil.copy(img, class_split_dir)

if __name__ == "__main__":
    split_data()
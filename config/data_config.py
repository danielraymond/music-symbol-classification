from pathlib import Path

INPUT_SIZE = (40, 40)
INPUT_SHAPE = (3,) + INPUT_SIZE

base_dir = Path(f"data/b-59-850")

images_dir = base_dir / "images"
labels_dir = base_dir / "json"
image_extn = ".JPG"
label_extn = ".json"

output_dir = base_dir / "experiments"
patches_dir = base_dir / "patches"
cross_fold_dir = base_dir / "cross-fold"


def set_data_dirs(ds_name: str):
    global base_dir
    global images_dir
    global labels_dir
    global output_dir
    global patches_dir
    global image_extn
    global label_extn

    base_dir = Path(f"data/{ds_name}")
    images_dir = base_dir / "images"
    labels_dir = base_dir / "json"
    image_extn = ".JPG"
    label_extn = ".json"
    output_dir = base_dir / "experiments"
    patches_dir = base_dir / "patches"

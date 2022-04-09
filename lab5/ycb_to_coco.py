import PIL.Image
from typing import Dict, Generator, List
import time
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import json

sys.path.append("C:/Users/Adam - Student/Documents/AGH/DS/Semestr_8/uczenie-maszyn/")
from ycb_video import CONFIG, dataset


def my_uuid_gen() -> Generator[int, None, None]:
    uuid = 0
    while True:
        uuid += 1
        yield uuid


my_uuid = my_uuid_gen()


def initiate_json(json_file: Dict) -> None:
    """
    Initiate json file: one for training phase and another one for validation.
    """
    json_file["info"] = dict(url="",
                             date_created=time.strftime(
                                 "%a, %d %b %Y %H:%M:%S +0000", time.localtime()),
                             description="Conversion of YCB-Video dataset into MS-COCO format")
    json_file["categories"] = [dict(name='',  # Category name
                                         id=1,  # Id of category
                                         # Skeleton connections (check constants.py)
                                         skeleton=[],
                                         supercategory='',  # Same as category if no supercategory
                                         keypoints=[])]  # Keypoint names
    json_file["images"] = []  # Empty for initialization
    json_file["annotations"] = []  # Empty for initialization


def process_dataset(json_file: Dict, img_path: str, img_ds: List[str], kp_ds: np.ndarray) -> None:
    for img_name, kps in zip(img_ds, kp_ds):
        # Process image
        width, height = PIL.Image.open(f"{img_path}/{img_name}").size
        process_image(json_file, img_name, width, height)
        # Process keypoints
        n_kps = kps.shape[0]
        flags = np.ones((n_kps, 1))
        kps_flag = np.append(kps, flags, axis=1)
        kps_flag.shape = (-1)
        process_annotation(json_file, n_kps, kps_flag.tolist())


def process_image(json_file: Dict, img_name: str, width: int, height: int) -> None:
    """
    Update image field in json file
    """
    json_file["images"].append({
        'coco_url': "unknown",
        'file_name': img_name,  # Image name
        'id': next(my_uuid),  # Image id
        'license': 1,  # License type
        'date_captured': "unknown",
        'width': width,  # Image width (pixels)
        'height': height})  # Image height (pixels)


def process_annotation(json_file: Dict, n_kps: int, kps: np.array) -> None:
    """
    Process and include in the json file a single annotation (instance) from a given image
    """

    json_file["annotations"].append({
        'image_id': 0,  # Image id
        'category_id': 1,  # Id of the category (like car or person)
        'iscrowd': 0,  # 1 to mask crowd regions, 0 if the annotation is not a crowd annotation
        'id': 0,  # Id of the annotations
        'area': 0,  # Bounding box area of the annotation (width*height)
                    # Bounding box  coordinates (x0, y0, width, heigth), where x0, y0 are the left corner
        'bbox': [],
        'num_keypoints': n_kps,  # number of keypoints
        # Flattened list of keypoints [x, y, visibility, x, y, visibility, .. ]
        'keypoints': kps,
        'segmentation': []})  # To add a segmentation of the annotation, empty otherwise


def main():
    # Load dataset
    img_path = CONFIG['images_path']
    img_ds, kp_ds, _ = dataset.load()
    img_train, img_test, kp_train, kp_test = train_test_split(
        img_ds, kp_ds, test_size=0.25)
    # Process dataset
    train_name = "coco_train.json"
    train_json = dict()
    initiate_json(train_json)
    process_dataset(train_json, img_path, img_train, kp_train)
    test_name = "coco_test.json"
    test_json = dict()
    initiate_json(test_json)
    process_dataset(test_json, img_path, img_test, kp_test)
    # Save JSON files
    with open(train_name, 'w') as f:
        json.dump(train_json, f)
    with open(test_name, 'w') as f:
        json.dump(test_json, f)


if __name__ == "__main__":
    main()

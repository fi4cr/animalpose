

import torch
import pandas as pd

from typing import Any
import numpy as np
import math
import cv2
import openpifpaf


import os
import hashlib
import PIL

import tqdm
import glob


openpifpaf.show.Canvas.show = True
openpifpaf.show.Canvas.image_min_dpi = 100

predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30-animalpose')
predictor.long_edge = 512
predictor.preprocess = predictor._preprocess_factory()

meta = openpifpaf.plugins.animalpose.animal_kp.AnimalKp().head_metas[0] 

zero_based_skeleton = [ (a-1, b-1) for a,b in meta.draw_skeleton]


# Body joint color map. #BGR
_JOINT_CMAP = {
    0: [0, 0, 255],
    1: [255, 208, 0],
    2: [255, 161, 0],
    3: [255, 114, 0],
    4: [0, 189, 255],
    5: [0, 236, 255],
    6: [0, 255, 226],
    7: [255, 0, 76],
    8: [0, 255, 131],
    9: [255, 0, 171],
    10: [0, 255, 37],
    11: [244, 0, 253],
    12: [57, 255, 0],
    13: [151, 0, 255],
    14: [151, 255, 0],
    15: [57, 0, 255],
    16: [245, 255, 0],
    17: [0, 39, 255],
    18: [255, 169, 0],
    19: [0, 133, 255],
}


# Connection color map. #BGR
# These colors probably matter a little, and it's not clear
# exactly how to choose them
# For instance - should they be similar to the keypoint colours or not?
# I've just taken them from the example notebook

_CONNECTION_CMAP = {
 (0, 1): [127, 104, 127],
 (0, 2): [0, 94, 255],
 (0, 5): [255, 184, 0],
 (1, 3): [255, 137, 0],
 (2, 4): [255, 57, 38],
 (1, 2): [0, 212, 255],
 (5, 7): [0, 245, 240],
 (5, 8): [0, 255, 178],
 (5, 9): [127, 127, 104],
 (6, 7): [150, 127, 126],
 (6, 10): [197, 0, 254],
 (6, 11): [122, 127, 221],
 (9, 13): [104, 255, 0],
 (13, 17): [156, 127, 56],
 (8, 12): [104, 0, 255],
 (12, 16): [198, 255, 0],
 (11, 15): [28, 19, 255],
 (15, 19): [28, 66, 255],
 (10, 14): [28, 114, 255],
 (14, 18): [250, 212, 0],
}


def draw_pose(
    image: np.ndarray,
    landmark_list: Any,
    connections: Any,
    overlay: bool = True,
):# -> tuple[np.ndarray, dict[str, list[float]]]:
    """Draws the landmarks and the connections on the image.

    Args:
      image: A three channel BGR image represented as numpy ndarray.
      landmark_list: A normalized landmark list proto message to be annotated on
        the image.
      connections: A list of landmark index tuples that specifies how landmarks to
        be connected in the drawing.
      overlay: Whether to overlay on the input image.

    Returns:
      (image, dictionary).

    Raises:
      ValueError: If one of the following:
        a) If the input image is not three channel BGR.
        b) If any connetions contain invalid landmark index.
    """
    if image.shape[2] != 3:
        raise ValueError("Input image must contain three channel bgr data.")
    image_rows, image_cols, _ = image.shape
    min_length = min(image_rows, image_cols)
    draw_line_width = math.floor(min_length * 0.01)
    draw_circle_radius = math.floor(min_length * 0.015)
    idx_to_coordinates = {}
    
    for idx, landmark in enumerate(landmark_list):
        if landmark[2]>0.0:
            idx_to_coordinates[idx] = (int(landmark[0]), int(landmark[1]))
            
    if overlay:
        output_image = image.copy()
    else:
        output_image = np.zeros(
            list(image.shape[:2])
            + [
                3,
            ],
            dtype=np.uint8,
        )

    if connections:
        num_landmarks = len(landmark_list)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    "Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                cv2.line(
                    output_image,
                    pt1=idx_to_coordinates[start_idx],
                    pt2=idx_to_coordinates[end_idx],
                    color=_CONNECTION_CMAP[(start_idx, end_idx)],
                    thickness=draw_line_width,
                )

    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.
    for idx, landmark_px in idx_to_coordinates.items():
        # Fill color into the circle
        cv2.circle(
            output_image,
            center=landmark_px,
            radius=draw_circle_radius,
            color=_JOINT_CMAP[idx],
            thickness=-1,
        )

    return output_image

def determine_pose(image):
    """Estimates pose and creates an image with just the pose body points.

    The image consisting the pose body points serves as the conditioning image
    for ControlNet training.

    Args:
        image: A three channel RGB image represented as a tf.Tensor.

    Returns:
        A tuple consisting of the original image (`image`), an image where
        the original image is overlaid with the pose keypoints, and an image
        with just the pose keypoints.
    """
    image = np.asarray(image)
    
    predictions, gt_anns, image_meta = predictor.numpy_image(image)
    if not predictions:
        data = np.zeros((20,3))
    else:
        data = predictions[0].data

    # Draw pose landmarks on a copy of the input image.
    annotated_image = draw_pose(
        image, data, zero_based_skeleton
    )

        # Draw pose landmarks on a blank image.
    blank_image = draw_pose(
        image, data, zero_based_skeleton, False
    )

    return image, annotated_image, blank_image

def save_image(
        image_path: str,
        overlaid_annotation: np.ndarray,
        blank_annotation: np.ndarray,
    ):
    base_path = os.path.splitext(image_path)[0]
    PIL.Image.fromarray(overlaid_annotation).save(
        os.path.join(f"{base_path}.overlaid.jpg")
    )
    PIL.Image.fromarray(blank_annotation).save(
        os.path.join(f"{base_path}.condition.png")
    )

        
dataset_path = '/mnt/disks/persist/dataset'

# Get the list of all image paths in the dataset
image_paths = glob.glob(f'{dataset_path}/**/*.jpg', recursive=True)

for img_path in tqdm.tqdm(image_paths):
    image = cv2.imread(img_path)
    
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    orig_image, overlaid_annotation, blank_annotation = determine_pose(image)

    
    # Save the images
    save_image(img_path, overlaid_annotation, blank_annotation)


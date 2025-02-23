import os
import json
import cv2
import numpy as np


def load_data_from_folder(image_folder, ann_folder, target_size=(224, 224)):
    """
    Function to load images and annotations (for both lane detection and object detection) from folder
    """
    images = []
    labels = []

    for annotation_file in os.listdir(ann_folder):
        if annotation_file.endswith(".json"):
            # Load annotation file
            annotation_path = os.path.join(ann_folder, annotation_file)
            with open(annotation_path, "r") as f:
                annotation = json.load(f)

            # Load corresponding image
            image_file = annotation["image"]
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, target_size) / 255.0  # Normalize image
            images.append(image)

            # Process annotations (for both lane and object detection)
            lane_mask = np.zeros(target_size, dtype=np.float32)
            if "lanes" in annotation:
                for lane in annotation["lanes"]:
                    for i in range(len(lane["x"]) - 1):
                        cv2.line(
                            lane_mask,
                            (int(lane["x"][i]), int(lane["y"][i])),
                            (int(lane["x"][i + 1]), int(lane["y"][i + 1])),
                            1,
                            thickness=3,
                        )
                labels.append(lane_mask)
            else:
                boxes = []
                labels_text = []
                for obj in annotation["objects"]:
                    boxes.append(obj["bbox"])  # [x_min, y_min, x_max, y_max]
                    labels_text.append(obj["category"])
                labels.append({"boxes": np.array(boxes), "labels": labels_text})

    return np.array(images), labels


# Example usage
train_image_folder = "data/train/img"
train_ann_folder = "data/train/ann"
train_images, train_labels = load_data_from_folder(train_image_folder, train_ann_folder)

val_image_folder = "data/val/img"
val_ann_folder = "data/val/ann"
val_images, val_labels = load_data_from_folder(val_image_folder, val_ann_folder)

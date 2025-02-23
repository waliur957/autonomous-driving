import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_data(image_folder, ann_folder, target_size=(224, 224)):
    images = []
    annotations = []
    # Loop through annotation files
    for annotation_file in os.listdir(ann_folder):
        if annotation_file.endswith(".json"):
  
            annotation_path = os.path.join(ann_folder, annotation_file)
            with open(annotation_path, "r") as f:
                annotation = json.load(f)
                image_file_name = annotation_file.split(".")[0] + ".jpg"
                image_path = os.path.join(image_folder, image_file_name)

                if not os.path.exists(image_path):
                    continue

                # Load image and get original size
                image = cv2.imread(image_path)
                orig_height, orig_width = image.shape[:2]

      
                image = cv2.resize(image, target_size) / 255.0
                images.append(image)

                object_annotations = []
                for obj in annotation.get("objects", []):
                    points = np.array(obj["points"]["exterior"], dtype=np.float32)

                    # 
                    points[:, 0] = points[:, 0] * (
                        target_size[0] / orig_width
                    )  
                    points[:, 1] = points[:, 1] * (
                        target_size[1] / orig_height
                    ) 
                    object_annotations.append(
                        points.astype(np.int32)
                    )  

                annotations.append(object_annotations)

    return np.array(images), annotations


def visualize_sample(image, annotations):
    """
    Function to visualize an image and its corresponding annotations (bounding box or polygon).

    Args:
        image (np.array): Image array.
        annotations (list): List of annotation polygons.
    """
    image_copy = (image * 255).astype(np.uint8) 

    for annotation in annotations:
        if len(annotation) > 1:
            
            pts = annotation.reshape((-1, 1, 2))
            cv2.polylines(
                image_copy, [pts], isClosed=True, color=(0, 0, 255), thickness=2
            )

    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


image_folder = "data/train/img"
ann_folder = "data/train/ann"

train_images, train_annotations = load_data(image_folder, ann_folder)

random_index = np.random.randint(0, len(train_images))
image = train_images[random_index]
annotations = train_annotations[random_index]

visualize_sample(image, annotations)

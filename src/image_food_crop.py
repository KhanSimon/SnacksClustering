from preparation import resize
import numpy as np
import cv2
import os

import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch


"""
This function loads images in a directory and crops the main item for each image and overwrite the image.
"""
def load_centered_images_and_labels(data_dir):
    images = []  # List to store all images
    labels = []  # List to store labels for each image
    
    # Get the names of the subdirectories (apple, banana, candy)
    categories = ['apple', 'banana', 'cake', 'candy', 'carrot', 'cookie', 'doughnut', 'grape', 'hot dog', 'ice cream','juice','muffin','orange','pineapple','popcorn','pretzel','salad','strawberry','waffle','watermelon']
    
    for label, category in enumerate(categories):
        category_dir = os.path.join(data_dir, category)
        
        # Check if the subdirectory exists
        if not os.path.isdir(category_dir):
            print(f"Directory {category_dir} not found.")
            continue
        
        # Iterate through the images in each subdirectory
        for filename in os.listdir(category_dir):
            image_path = os.path.join(category_dir, filename)
            
            # Check if it's a valid image file (optional check for specific file types)
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Read the image
                image = cv2.imread(image_path)
                image = detect_and_crop_food(image)
                image = resize(image)
                # Append the image and its corresponding label
                images.append(image)
                labels.append(label)  # label is 0 for apple, 1 for banana, 2 for candy

                cv2.imwrite(image_path, image)
                print(len(images))
    
    # Convert images and labels to numpy arrays
    images_array = np.array(images)
    labels_array = np.array(labels)
    
    return images_array, labels_array










# Load Faster R-CNN model for food detection
food_detector = fasterrcnn_resnet50_fpn(pretrained=True)
food_detector.eval()

def detect_and_crop_food(image, confidence_threshold=0.8, min_size=30, aspect_ratio_threshold=2.5):
    """Detect food in an image and crop it, trying to avoid incorrect detections like people."""
    
    # Convert image to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        detections = food_detector(image_tensor)

    # Extract bounding boxes and confidence scores
    boxes = detections[0]['boxes'].cpu().numpy()
    scores = detections[0]['scores'].cpu().numpy()

    valid_boxes = []
    
    image_height, image_width, _ = image.shape
    center_x, center_y = image_width // 2, image_height // 2  # Middle of the image

    for box, score in zip(boxes, scores):
        score = round(float(score), 2)  # Limit score to 2 decimal places
        if score < confidence_threshold:
            continue  # Skip low-confidence detections
        
        x_min, y_min, x_max, y_max = map(int, box)
        width = x_max - x_min
        height = y_max - y_min

        # Filter out objects that are too small
        if width < min_size or height < min_size:
            continue

        # Filter out objects that are too tall (likely a person)
        if height / width > aspect_ratio_threshold:
            continue

        # Compute distance from the center of the image
        box_center_x = (x_min + x_max) / 2
        box_center_y = (y_min + y_max) / 2
        distance_from_center = ((box_center_x - center_x) ** 2 + (box_center_y - center_y) ** 2) ** 0.5
        
        valid_boxes.append((box, score, distance_from_center))

    if not valid_boxes:
        return image  # Return original image if no valid food is detected

    # Sort by score first (higher is better), then by distance from center (lower is better)
    valid_boxes.sort(key=lambda x: (-x[1], x[2]))

    best_box = valid_boxes[0][0]  # Pick the best box

    x_min, y_min, x_max, y_max = map(int, best_box)
    
    # Crop the detected food
    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image







if __name__ == "__main__":
    load_centered_images_and_labels("../data/val")

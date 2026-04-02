# Documentation used:
# https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes
# https://docs.ultralytics.com/tasks/detect/ 
import os
import cv2
import random
import matplotlib.pyplot as plt
from ultralytics import YOLO
import argparse

# Testing: Load the best trained model, visualize detections on 5 random validation images, and evaluate the model's performance on the entire validation set

if __name__ == "__main__":
    # Argument parser to take file paths as command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained YOLOv8 model weights")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="Path to the validation images directory")

    # Parse the command line arguments
    args = parser.parse_args()

    # Load the best trained model
    model = YOLO(args.model_path)
    #print(model.yaml) # DEBUG: prints which model architecture is being used 

    # Path to validation set images
    val_dir = args.val_dir

    # Get all image files
    image_files = [f for f in os.listdir(val_dir) if f.endswith((".jpg", ".png"))]

    # Randomly pick 5 images
    sample_files = random.sample(image_files, 5)

    # Plot the 5 sample images with detected bounding boxes
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))

    for ax, filename in zip(axs, sample_files):
        image_path = os.path.join(val_dir, filename)
        
        # Run YOLOv8 detection with a confidence threshold of 0.5
        results = model(image_path, conf=0.5)
        
        # Get the image with bounding boxes drawn
        img_with_boxes = results[0].plot()  # returns image as numpy array
        
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        
        ax.imshow(img_rgb)
        ax.axis('off')
        ax.set_title(filename)

    plt.tight_layout()
    plt.show()

    # Evaluate the model's performance on the validation set
    # Metrics will be saved in the "runs/detect/license_plate_model/val" directory
    metrics = model.val()
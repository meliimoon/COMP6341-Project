# Documentation used:
# https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes
# https://docs.ultralytics.com/tasks/detect/ 
import os
import cv2
import random
import matplotlib.pyplot as plt
from ultralytics import YOLO

# DEBUG just to check if GPU is available and what GPU is being used
#import torch
#print(torch.cuda.is_available())   # should be True
#print(torch.cuda.device_count())   # should be >= 1
#print(torch.cuda.get_device_name(0))  # prints GPU name

if __name__ == "__main__":
    # Fine-tuning stage: Load a pretrained YOLO8n model and train it on the Kaggle License Plate Detection Dataset for 100 epochs
    
    # Load a pretrained YOLO8n model
    # There are other YOLOv8 model variants, can test others to see if there is a difference in performance
    #   YOLOv8n (nano), YOLOv8s (small), YOLOv8m (medium), YOLOv8l (large), YOLOv8x (extra large)
    model = YOLO("yolov8n.pt")

    # Train the model on the Kaggle License Plate Detection Dataset dataset for 100 epochs
    train_results = model.train(
        data="licenseplatedataset.yaml",  # Path to dataset configuration file
        epochs=100,  # Number of training epochs
        imgsz=960,  # Image size for training
        device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
        workers=0,
        name="license_plate_model"
    )

    # Validation stage: Load the best trained model, visualize detections on 5 random validation images, and evaluate the model's performance on the entire validation set
    # Load the best trained model
    model = YOLO(r"C:\Users\Meliimoon\Documents\Concordia\Grad\Winter 2026\COMP6341\project\ProjectCode\runs\detect\license_plate_model\weights\best.pt")
    print(model.yaml) # DEBUG: prints which model architecture is being used 

    # Path to validation set images
    val_dir = r"C:\Users\Meliimoon\Documents\Concordia\Grad\Winter 2026\COMP6341\project\ProjectCode\dataset\images\val"

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
    print(metrics)
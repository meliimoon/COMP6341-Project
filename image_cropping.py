import os
import cv2
from ultralytics import YOLO

# Perform license plate detection and cropping using our best trained YOLOv8 model, and save the cropped images to specified directories for both training and validation sets.

# Load in the trained model 
# File path created automatically by the training process of YOLOv8
model = YOLO(r"C:\Users\Meliimoon\Documents\Concordia\Grad\Winter 2026\COMP6341\project\ProjectCode\runs\detect\license_plate_model\weights\best.pt")

input_directory_val = r"C:\Users\Meliimoon\Documents\Concordia\Grad\Winter 2026\COMP6341\project\ProjectCode\dataset\images\val"
input_directory_train = r"C:\Users\Meliimoon\Documents\Concordia\Grad\Winter 2026\COMP6341\project\ProjectCode\dataset\images\train"

output_directory_val = r"C:\Users\Meliimoon\Documents\Concordia\Grad\Winter 2026\COMP6341\project\ProjectCode\cropped_plates\val"
output_directory_train = r"C:\Users\Meliimoon\Documents\Concordia\Grad\Winter 2026\COMP6341\project\ProjectCode\cropped_plates\train"

os.makedirs(output_directory_val, exist_ok=True)
os.makedirs(output_directory_train, exist_ok=True)

directory_IOs = [
    (input_directory_val, output_directory_val),
    (input_directory_train, output_directory_train)
]

# Loop through images in the train and validation set directories, perform detection, and save cropped license plate images
for input_directory, output_directory in directory_IOs:
    for filename in os.listdir(input_directory):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(input_directory, filename)
            img = cv2.imread(image_path)

            results = model(image_path)
            
            # (Side note) Alternatively can do:
            #    results = model(image_path, conf=0.5) # directly sets confidence threshold for detection, so only detections with conf >= 0.5 will be returned
            # Then can remove the confidence check in the loop below
            #    for i, box in enumerate(results[0].boxes.xyxy):
            # Then remove the if statement

            # Loop through detected bounding boxes and crop the license plate regions
            for i, (box, conf) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf)):

                # Only save cropped images for detections with high confidence scores
                if conf < 0.5:
                    continue

                x_min, y_min, x_max, y_max = map(int, box.tolist())

                # Pad the bounding box in case the detected box is too tight around the license plate
                h, w, _ = img.shape
                pad = 10

                x_min = max(0, x_min - pad)
                y_min = max(0, y_min - pad)
                x_max = min(w, x_max + pad)
                y_max = min(h, y_max + pad)

                # Crop the image using the bounding box coordinates
                cropped = img[y_min:y_max, x_min:x_max]

                save_path = os.path.join(output_directory, f"{filename}_plate_{i}.jpg")
                cv2.imwrite(save_path, cropped)

                print(f"Saved: {save_path}")
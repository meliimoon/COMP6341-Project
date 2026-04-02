import os
import cv2
import argparse
from ultralytics import YOLO

if __name__ == "__main__": # Only run the code below if this script is executed directly (not imported as a module)

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", 
                        type=str,
                        default="",

                        help="Input directory containing images to be cropped")
    parser.add_argument("--output_dir", 
                        help="Output directory to save cropped images")

    args = parser.parse_args()

    # Load in the trained model 
    # File path created automatically by the training process of YOLOv8
    model = YOLO(r"C:\Users\Meliimoon\Documents\Concordia\Grad\Winter 2026\COMP6341\project\ProjectCode\runs\detect\license_plate_model\weights\best.pt")

    input_directory = r"C:\Users\Meliimoon\Documents\Concordia\Grad\Winter 2026\COMP6341\project\ProjectCode\dataset\images\val"
    output_directory = r"C:\Users\Meliimoon\Documents\Concordia\Grad\Winter 2026\COMP6341\project\ProjectCode\cropped_plates"

    os.makedirs(output_directory, exist_ok=True)

    # Loop through images in the input dataset directory, perform detection, and save cropped license plate images
    for filename in os.listdir(input_directory):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(input_directory, filename)
            img = cv2.imread(image_path)

            results = model(image_path)
            # Or can do:
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
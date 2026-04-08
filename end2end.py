import argparse
from ultralytics import YOLO
import os
import cv2
from super_image import EdsrModel
import torch
from torchvision import transforms
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer


# This script must be run from the folder location as it is downloaded from the GitHub repo, as it relies on relative file paths to the YOLOv8 model and the fast-plate-OCR model.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_img",
        type=str,
        dest = "input_img_path",
        help="the file path for the input image",
    )
    parser.add_argument(
        "-sr",
        "--use_sr",
        action='store_true',
        help="whether to perform super resolution on the cropped license plate image",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        dest = "output_file",
        default=os.getcwd(),
        help="the file path for saving fast-plate-OCR results",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # TestYOLOv8
    yolov8_model_path = os.path.join(current_dir, "yolov8", "best.pt")
    yolo_model = YOLO(yolov8_model_path).to(device)

    car_img = cv2.imread(args.input_img_path)
    results = yolo_model(car_img)

    # Crop image
    if results[0].boxes is None or len(results[0].boxes) == 0:
        print("No plates detected. Shutting down.")
        exit()

    best_idx = results[0].boxes.conf.argmax()
    box = results[0].boxes.xyxy[best_idx]
    conf = results[0].boxes.conf[best_idx]

    if conf < 0.5:
        print("No plates detected. Shutting down.")
        exit()

    x_min, y_min, x_max, y_max = box.int().tolist()

    h, w, _ = car_img.shape
    pad = 10

    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(w, x_max + pad)
    y_max = min(h, y_max + pad)

    cropped_plate = car_img[y_min:y_max, x_min:x_max]

    cropped_plate_save_path = os.path.join(args.output_file, f"Cropped_plate.jpg")
    cv2.imwrite(cropped_plate_save_path, cropped_plate)
    print(f"Cropped license plate saved to: {cropped_plate_save_path}")

    # Optional SR step
    if args.use_sr:
        edsr2x_model_path = os.path.join(current_dir, "edsr_2x_50epoch", "2x_edsr_model_weights.pth")

        edsr_model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=2).to(device) 
        edsr_model.load_state_dict(torch.load(edsr2x_model_path, map_location=device))
        edsr_model.eval()

        # Convert OpenCV image (BGR → RGB)
        img = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)

        # Convert to tensor and add batch dimension
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

        # Super-resolve
        with torch.no_grad():
            sr_tensor = edsr_model(img_tensor)

        # Convert back to image
        sr_img = sr_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        sr_img = np.clip(sr_img * 255.0, 0, 255).astype(np.uint8)

        # Convert RGB → BGR for saving
        sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)

        # Save
        cropped_plate_sr_save_path = os.path.join(args.output_file, "Cropped_plate_SR.jpg")
        cv2.imwrite(cropped_plate_sr_save_path, sr_img_bgr)

        print(f"Super-resolved license plate saved to: {cropped_plate_sr_save_path}")
        ocr_input = cropped_plate_sr_save_path
    else:
        ocr_input = cropped_plate_save_path
    
    # Perform OCR
    ocr_model = LicensePlateRecognizer('cct-s-v2-global-model')
    pred = ocr_model.run(ocr_input)[0].plate

    # Print and save
    print(f"Detected license plate: {pred}")
    output_txt_path = os.path.join(args.output_file, "Cropped_plate_OCR_result.txt")

    with open(output_txt_path, "w") as f:
        if args.use_sr:
            f.write(f"Detected license plate WITH Super-Resolution: {pred}\n")
        else:
            f.write(f"Detected license plate WITHOUT Super-Resolution: {pred}\n")

    print(f"OCR result saved to: {output_txt_path}")

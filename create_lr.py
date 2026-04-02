import argparse
import os
from tqdm import tqdm
import cv2

if __name__ == "__main__":
    # -----------------------
    # parameters settings
    # -----------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        dest = "input_dir",
        help="the file path of HR input images",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        dest="output_dir",
        help="the file path to save LR images",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=2.0,
        dest="scale",
        help="the down-sampling scale (if not specified, will use 2x scale)",
    )
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    scale = args.scale

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{scale}xLR_{filename}")
            img = cv2.imread(image_path)

            h, w, c = img.shape

            print(f"Processing {filename} with original size: {img.shape}")
            lr_img = cv2.resize(img, (w // int(scale), h // int(scale)), interpolation=cv2.INTER_CUBIC) # Down-sample the image using bicubic interpolation

            cv2.imwrite(output_path, lr_img)
            print(f"Saved LR image to {output_path} with size: {lr_img.shape}")
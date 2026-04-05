from super_image import EdsrModel
import cv2
import torch
from torchvision import transforms
import argparse
import os
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        dest = "input_dir",
        help="the file path of input images",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        dest="output_dir",
        help="the file path to the SR images",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        default=2,
        dest="scale",
        help="the up-sampling scale for super-resolution (if not specified, will use 2x scale)",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        dest="pretrained_model_path",
        help="the file path to the pretrained EDSR model weights"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load pretrained EDSR model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the EDSR model architecture and load pretrained weights
    model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=args.scale).to(device) 
    model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))
    model.eval()

    # Loop through all images in input_dir
    for filename in tqdm(os.listdir(args.input_dir), desc="Processing images"):
        if not filename.endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(args.input_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to tensor and add batch dimension
        lr_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

        # Super-resolve
        with torch.no_grad():
            sr_tensor = model(lr_tensor)

        # Convert back to image
        sr_img = sr_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        sr_img = np.clip(sr_img * 255.0, 0, 255).astype(np.uint8)

        # Save the SR image
        output_path = os.path.join(args.output_dir, f"SR_{filename}")
        sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, sr_img_bgr)

    print("All images have been super-resolved!")
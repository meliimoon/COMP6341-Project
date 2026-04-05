from super_image import EdsrModel
import cv2
import torch
from torch.utils import data
from torchvision import transforms
import argparse
import os
from tqdm import tqdm
import time
import numpy as np
import json
import matplotlib.pyplot as plt

class EdsrDataset(data.Dataset):
    def __init__(self, hr_imgs, scale, patch_size):
        self.hr_imgs = hr_imgs
        self.scale = scale
        self.patch_size = patch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.hr_imgs)

    def __getitem__(self, idx):
        img_path = self.hr_imgs[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB format

        h, w, _ = img.shape

        # Pad if smaller than PATCH_SIZE
        pad_h = max(self.patch_size - h, 0)
        pad_w = max(self.patch_size - w, 0)
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            h, w, _ = img.shape

        # Random crop HR patch
        top = np.random.randint(0, h - self.patch_size + 1)
        left = np.random.randint(0, w - self.patch_size + 1)
        hr_patch = img[top:top + self.patch_size, left:left + self.patch_size]

        # Downsample to create LR patch on the fly using bicubic interpolation
        lr_patch = cv2.resize(hr_patch, (self.patch_size // self.scale, self.patch_size // self.scale), interpolation=cv2.INTER_CUBIC)
        
        # Transform to tensors
        hr_tensor = self.transform(hr_patch)
        lr_tensor = self.transform(lr_patch)
        
        return lr_tensor, hr_tensor
    
if __name__ == "__main__":
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
        dest = "output_dir",
        help="the file path for the model weights",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-4,
        dest = "learning_rate",
        help="the learning rate for training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        dest = "num_epochs",
        help="the number of epochs for training",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        default=2,
        dest="scale",
        help="the super-resolution scale as an int (if not specified, will use 2x scale)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=64,
        dest="patch_size",
        help="the size of the HR image patches to be used for training (if not specified, will use 64x64 patches)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_imgs = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith((".jpg", ".png"))]



    # Create dataset and dataloader for training
    train_dataset = EdsrDataset(input_imgs, scale=args.scale, patch_size=args.patch_size)
    train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Load pretrained EDSR model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=args.scale).to(device)

    # Training parameters
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.L1Loss()

    losses = []

    start = time.perf_counter()

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = 0.0
        for lr_imgs, hr_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}")

    end = time.perf_counter()
    print(f"Training time: {end - start:.2f} seconds")

    # Save loss history to JSON
    json_path = os.path.join(args.output_dir, f"{args.scale}x_edsr_loss_history.json")
    with open(json_path, "w") as f:
        json.dump(losses, f, indent=4)
    print(f"Loss history saved to {json_path}")
        
    # Save the trained model weights
    model_save_path = os.path.join(args.output_dir, f"{args.scale}x_edsr_model_weights.pth")
    torch.save(model.state_dict(), model_save_path)

    # Plot training curve
    plt.figure()
    plt.plot(range(1, num_epochs+1), losses, marker='o')
    plt.title("EDSR Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, f"{args.scale}x_edsr_training_curve.png"))
    plt.close()
    print(f"Training curve saved to {args.output_dir}")
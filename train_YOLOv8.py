# Documentation used:
# https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes
# https://docs.ultralytics.com/tasks/detect/ 

import argparse
from ultralytics import YOLO

# DEBUG just to check if GPU is available and what GPU is being used
#import torch
#print(torch.cuda.is_available())   # should be True
#print(torch.cuda.device_count())   # should be >= 1
#print(torch.cuda.get_device_name(0))  # prints GPU name

# Fine-tune YOLOv8 for license plate detection with command-line arguments

if __name__ == "__main__":
    # Argument parser to take file paths as command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Pretrained YOLOv8 model (e.g., yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the dataset YAML file")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="Training image size")
    parser.add_argument("--device", type=str, default="0",
                        help="Device to train on (e.g., 'cpu', '0', '0,1')")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of data loader workers")
    parser.add_argument("--name", type=str, default="license_plate_model",
                        help="Experiment name (folder to save weights and results)")

    # Parse the command line arguments
    args = parser.parse_args()

    # Load the pretrained YOLOv8 model
    model = YOLO(args.model)

    # Train the model
    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        name=args.name
    )

    print("Training complete!")
    print(f"Best weights saved at: runs/detect/{args.name}/weights/best.pt")
# COMP6341-Project
## Project Overview
*[WIP]* \
This project incorporates three distinct models into a single cohesive pipeline that outputs the characters present in a license plate, given an input image of a vehicle with a plate visible. Our pipeline may also provide insight into the benefits of using super-resolution on low-resolution images for the task of character recognition, by toggling the super-resolution component.
First, the YOLOv8 model is used to automatically detect the bounding boxes of license plates given an input image of a vehicle with a plate visible. Then, the pipeline crops the input image to the detected bounding box of the license plate. 
The cropped images are then downsampled and passed to the *[SR model]* [...]

## Requirements
pip install ultralytics \
pip install super-image \
pip install opencv-python \
pip install matplotlib \
pip install tqdm \
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 \
or (depending on CUDA version) \
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130 \
[...]

## How to Use
1. **Download the Dataset**:
   - Download the License Plate Detection dataset from [Kaggle](https://www.kaggle.com/datasets/fareselmenshawii/license-plate-dataset).
   - Extract the dataset and organize it as follows: \
     &emsp; **Note**: you may rename the extracted folder "archive" to "dataset", as we have
     ```
     dataset/
        images/
           train/
           val/
        labels/
           train/
           val/
     ```
   - Ensure the dataset/ folder is placed in the working directory of the project
2. **Create a .yaml file**:
   - Create a .yaml file in the following format to pass the dataset to the YOLOv8 model for fine-tuning:
     ```
     path: "path\\to\\dataset" # path should reach the parent folder "dataset" we extracted in the previous step

	   train: images/train
	   val: images/val

	   names:
  	    0: license_plate
     ```
   - Ensure the .yaml file is placed in the working directory of the project
3. **Fine-tune the YOLOv8 model**:
   - Run the training script with the necessary command line arguments \
     &emsp; List of command line arguments & their default values (how our experiments were set up): \
      	&emsp; &emsp; --model, type=str, default="yolov8n.pt" \
      	&emsp; &emsp; --data, type=str, required=True \
      	&emsp; &emsp; --epochs, type=int, default=100 \
      	&emsp; &emsp; --imgsz, type=int, default=960 \
      	&emsp; &emsp; --device, type=str, default="0" \
      	&emsp; &emsp; --workers, type=int, default=0 \
      	&emsp; &emsp; --name, type=str, default="license_plate_model"
     
    ```bash
	  python train_YOLOv8.py [command line args]
    ```
4. **Evaluate the model**:
   - Run the testing script with the necessary command line arguments \
     &emsp; List of command line arguments: \
      	&emsp; &emsp; --model_path, type=str, required=True \
      	&emsp; &emsp; --val_dir, type=str, required=True
     
   ```bash   
   python test_YOLOv8.py [command line args]
   ```
5. **Crop the images to their bounding box of license plates**:
   - Run the image cropping script with the necessary command line arguments \
     &emsp; List of command line arguments: \
      	&emsp; &emsp; --input_dir, type=str \
      	&emsp; &emsp; --output_dir, type=str \
      	&emsp; &emsp; --model_path, type=str
     
   ```bash
	 python image_cropping.py [command line args]
   ```
6. **Create low resolution versions of the cropped images**:
   - Run the downsampling script with the necessary command line arguments \
     &emsp; List of command line arguments & their default values (how our experiments were set up): \
      	&emsp; &emsp; -i OR --input_dir, type=str \
      	&emsp; &emsp; -o OR --output_dir, type=str \
      	&emsp; &emsp; -s OR --scale, type=float, default=2.0
     
   ```bash
   python create_lr.py [command line args]
   ```
7. **SR step**:
   - [WIP]  
8. **OCR step**:
   - [WIP]
  
## File Descriptions:
***best.pt***: This file containing our fine-tuned YOLOv8n model's best training weights. 

***licenseplatedataset.yaml***: This file is an example of how the dataset's .yaml file should be set up for fine-tuning the YOLOv8 model. 

***train_YOLOv8.py***: This file fine-tunes a YOLOv8 model using the License Plate Detection dataset. The default parameters used in our experiments were the YOLOv8n (the smallest and fastest variant) pretrained weights, 100 epochs, image size 960, GPU device, and 0 number of workers. The script saves the best model weights to the file path of "runs/detect/{args.name}/weights/best.pt". \
&emsp;*Note*: It took 6.2 hours to finish training on an NVIDIA 3060 12GB GPU. 

***test_YOLOv8.py***: This file evaluates the saved model on the inputed validation dataset. The script saves the metrics to the file path of "runs/detect/{args.name}/val". The script also outputs 5 randomly chosen samples of the dataset with the detected bounding boxes for qualitative analysis. 

***image_cropping.py***: This file uses the best saved model to detect the bounding boxes of the dataset provided to the script and then crops the images to the bounding box area. The cropped license plate images get saved to the chosen output directory. 

***create_lr.py***: 

[...]

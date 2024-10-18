# Plant Disease Diagnosis with Pix2Pix Model

## Project Overview
This repository contains the implementation of a generative AI model for diagnosing plant leaf diseases. The approach is based on the work of [Katafuchi and Tokunaga (2020)](https://arxiv.org/pdf/2011.14306), utilizing a pix2pix model for generating healthy plant leaves and detecting anomalies by comparing color differences between healthy and diseased leaves. The goal is to create a system that can automatically diagnose diseases in plants using an unsupervised anomaly detection method.

This project builds upon the neural networks and algorithms provided in the [PyTorch CycleGAN and Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master) repository by Jun-Yan Zhu and collaborators. Their work provided essential resources for implementing the pix2pix model used in this project.

## Methodology
**Data Collection**
* **Training Data:** 50 images of healthy leaves.
* **Test Data:**
  *50 images of healthy leaves.
  *100 images of diseased leaves.

## Data Preparation
The data preparation process includes converting images to grayscale and organizing them into appropriate folders for training the **pix2pix** model.

### Image Conversion
The images are converted from RGB to grayscale using the following Python script, included in `convert_to_grayscale.py`:
```
import os
from PIL import Image

def convert_to_grayscale(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(input_dir, filename)).convert('L')
            img.save(os.path.join(output_dir, filename))
```
### Dataset Organization
The dataset is structured into paired images for training the **pix2pix** model. The images are divided into training, validation, and test sets, with grayscale and color images placed in separate folders.

## Model Training
We use the **pix2pix** model to convert grayscale images (representing leaf anomalies) back into RGB color space. The model was trained with the following command:
```
python train.py --dataroot ./datasets/Leafs --name Leafs_pix2pix --model pix2pix --direction BtoA
```

## Model Testing
The trained model generates colored images from grayscale images. To test the model, use the following command:
```
python test.py --dataroot ./datasets/Leafs --name Leafs_pix2pix --model pix2pix --netG unet_256 --direction BtoA --dataset_mode aligned --norm batch
```

## Evaluation Metrics
To assess the quality of generated images and anomaly detection, we implemented a **color difference index** using the **CIEDE2000** color space. Heatmaps are generated to highlight areas of discrepancies between healthy and diseased leaves.

### Performance Evaluation Code
```
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics():
    # Example metrics calculation (precision, recall, F1 score)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1
```

## Results
* **Precision:** ~1.00
* **Recall:** ~0.996
* **F1 Score:** ~0.998
    
The model demonstrated strong performance in generating realistic images of healthy leaves and identifying anomalies in diseased leaves. Heatmaps were useful in visualizing the differences between healthy and unhealthy leaves.

## Future Work
* Explore using **CycleGAN** for improved anomaly detection across more complex datasets.
* Expand the dataset to include more plant varieties and disease types.

## Dependencies
* Python 3.6+
* PyTorch
* PIL (Python Imaging Library)
* OpenCV
* colorspacious
* Scikit-learn

To install all dependencies, run:
```
pip install -r requirements.txt
```
## How to Run
1. Clone this repository.
2. Prepare the dataset as described above.
3. Run the training script:
```
python train.py --dataroot ./datasets/Leafs --name Leafs_pix2pix --model pix2pix --direction BtoA
```
4. Test the model using the test script:
```
python test.py --dataroot ./datasets/Leafs --name Leafs_pix2pix --model pix2pix --direction BtoA
```

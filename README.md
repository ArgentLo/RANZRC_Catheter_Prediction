# Deep Learning for X-Rays Catheters Position Classification

This is a X-Rays Catheters Position Classification framework based on few state-of-the-art Convolutional Neural Networks.


<p align="center">
    <img align="center" src="https://github.com/ArgentLo/X-Rays-Lines-and-Catheters-Classification/blob/main/imgs/Endotracheal.jpg" width="250" height="178">
    <img align="center" src="https://github.com/ArgentLo/X-Rays-Lines-and-Catheters-Classification/blob/main/imgs/nasogastric.png" width="250" height="210">
</p>


## Introduction to Catheters Position Recognition

From **X-Ray images**, radiologists can check and diagnose **malpositioned lines and tubes** in patients, which can cause serious complications. 
Doctors and nurses frequently use checklists for placement of lifesaving equipment to ensure they follow protocol in managing patients. 
Yet, these steps can be time consuming and are still prone to human error, especially in stressful situations when hospitals are at capacity.

The gold standard for the confirmation of line and tube positions are **chest radiographs**(X-Ray images). 
However, a physician or radiologist must manually check these chest x-rays to verify that the lines and tubes are in the optimal position. 
Not only does this leave room for **human error**, but delays are also common as radiologists can be busy reporting other scans. 

- **Deep learning algorithms** may be able to **automatically detect** malpositioned catheters and lines.
- Once alerted, clinicians can reposition or remove them to avoid life-threatening complications.
- This is a X-Rays Catheters Position Classification framework based on few **state-of-the-art** Convolutional Neural Networks.
    - **ReXNet**: ["Rethinking Channel Dimensions for Efficient Model Design"](https://arxiv.org/abs/2007.00992)
    - **SeResNet**: ["Squeeze-and-Excitation Networks"](https://arxiv.org/abs/1709.01507)
    - **EfficientNet**: ["EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"](https://arxiv.org/abs/1905.11946)
    - **ResNeSt**: ["ResNeSt: Split-Attention Networks"](https://arxiv.org/abs/2004.08955)

----

## Quick Start

- **Environment Requirement**

The code has been tested running under Python 3.7. The required packages are as follows:

```
timm                # Collection of PyTorch Image Models 
albumentations      # Image Augmentation Package for Model Training
pytorch_pfn_extras  # Model Manangement Tool for PyTorch
warmup_scheduler    # Optimizer Scheduler with Warmup Epochs 
pytorch == 1.3
fastprogress
tqdm
```

- **Configuration**

All parameters, such as `backbone_model` and `loss_func`, can be adjust to fix your need in `config.py` and `multi_config.py`.

- **Preprocess Image Data**

Run the preprocessing before training or evaluation.

```python
python preproc_npy.py
```

- **Training** 

```pyton
# for training Single-Head model
python train.py

# for training Multi-Head model (more GPU memory required)
python multi_train.py
```

### Dataset

In fact, X-Rays datasets are rather "rare", some widely-used **Chest X-Rays Public Datasets** can be found in the following links:

- Sample Datasets 1: [NIH Chest X-Rays Dataset](https://www.kaggle.com/nih-chest-xrays/data)
- Sample Datasets 2: [RANZCR CLiP Dataset](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/data)

----

## Chest X-Rays Visualization

- **Raw Chest X-Ray**
    - A chest x-ray produces images of the heart, lungs, airways, blood vessels and the bones of the spine and chest. 
    - An x-ray (radiograph) is a noninvasive medical test that helps physicians diagnose and treat medical conditions.
    
    <p align="center">
        <img src="https://github.com/ArgentLo/X-Rays-Lines-and-Catheters-Classification/blob/main/imgs/raw0.png" width="467.6" height="294">
    </p>

- **Chest X-Ray with Catheter and Line Position**
    - A catheter, in medicine, is a **thin tube** made from medical grade materials serving a broad range of functions.
    
    <p align="center">
        <img src="https://github.com/ArgentLo/X-Rays-Lines-and-Catheters-Classification/blob/main/imgs/draw1.png" width="750" height="336">
    </p>

    <p align="center">
        <img src="https://github.com/ArgentLo/X-Rays-Lines-and-Catheters-Classification/blob/main/imgs/draw2.png" width="750" height="332">
    </p>









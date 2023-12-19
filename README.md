<<<<<<< HEAD

# README for ECE 570 Project: Random Erasing Data Augmentation Exploration and Extension

## Author: Jonathan Kao

### Introduction
This repository contains the code and resources for the ECE 570 project titled "Exploring and Extending Random Erasing Data Augmentation". It includes Python files, Jupyter notebooks, and trained model files for the project.

### Environment Setup
1. **Platform**:
   - Kaggle Notebooks ([Kaggle: Your Home for Data Science](https://www.kaggle.com/))

2. **Kaggle Notebook Setup:**
   - **Accelerator**: GPU T4 x2
   - **Language**: Python
   - **Persistence**: No persistence
   - **Environment**: Pin to original environment (2023-07-10)
   - **Internet**: Internet on

### File Descriptions
1. **training-and-testing.ipynb**: 
   - The main Jupyter notebook to execute the experiments.
   - Contains training and testing code adapted from Idelbayev, Y.'s ResNet implementation for CIFAR10/CIFAR100 (Source: [Proper ResNet implementation for CIFAR10/CIFAR100 in PyTorch](https://github.com/akamaster/pytorch_resnet_cifar10)).
   - Includes modified transforms to train models with various data augmentation techniques.
   - Includes modifications to the learning rate scheduler.
   - Includes an added section for testing models with various augmented datasets.
   - Test error rates for the final paper are derived from this notebook.

2. **resnetgray.py**:
   - A Python file to be imported into the main notebook.
   - Contains the architecture of ResNets from Idelbayev, Y.'s ResNet implementation for CIFAR10/CIFAR100 (Source: [Proper ResNet implementation for CIFAR10/CIFAR100 in PyTorch](https://github.com/akamaster/pytorch_resnet_cifar10)).
   - Includes an extended 1-channel ResNet-20 for the Fashion-MNIST dataset.

3. **new_retransform.py**:
   - A Python file to be imported into the main notebook.
   - Implements the Random Erasing transformation function with extended features.
   - Refers to the standard implementation of Random Erasing in PyTorch (Source: [PyTorch RandomErasing](https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomErasing)).

4. **trained_models/**:
   - Folder containing all trained models.
   - Test error rates of these models should match those in the results table of the final paper, assuming correct execution of the testing code.

5. **extended-features-showcase.ipynb**: 
   - This Jupyter notebook is not utilized for experimental purposes.
   - It serves to showcase the enhancements made to the standard Random Erasing.
   - The notebook includes code that displays the augmented data for visual inspection.

### Training Settings
- **Code Base**: PyTorch
- **Dataset**: Fashion-MNIST from PyTorch (Source: [PyTorch FASHIONMNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html#fashionmnist))
- **Architecture**: 1-channel input ResNet-20
- **Epochs**: 300
- **Batch Size**: 128
- **Number of Workers**: 2
- **Learning Rate Schedule**:
  - Starts at 0.1.
  - Decays to 0.01 after 50% of epochs.
  - Further decays to 0.001 after 75% of epochs.
- **Transforms**:
  - Extended Random Erasing
  - Random Horizontal Flip
  - Random Cropping with 4px padding
  - Normalization (`transforms.Normalize(mean=[0.5], std=[0.5])`)

### Additional Notes
- The script in the notebook is configured to train one model at a time.
- Users may need to adjust the transform settings for different data augmentation techniques.
- Renaming output files (checkpoints and model files) according to the data augmentation and training settings is recommended for better organization.
- Detailed instructions and explanations are provided within the notebook file.
=======
# Exploring-and-Extending-Random-Erasing-Data-Augmentation
>>>>>>> origin/main

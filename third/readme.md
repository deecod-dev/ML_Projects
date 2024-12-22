# Machine Learning and Deep Learning Implementation Project

## Overview
This project demonstrates the implementation of machine learning and deep learning algorithms from scratch and using PyTorch. It includes Jupyter notebooks for implementations, pre-trained model files, and a detailed report of the results.

## File Structure
- `ass4q2.ipynb`: KMeans clustering algorithm implementation from scratch
- `ass4q3.ipynb`: CIFAR-10 dataset processing, CNN and MLP implementation using PyTorch
- `cnn_model.pth`: Saved PyTorch model weights for CNN
- `mlp_model.pth`: Saved PyTorch model weights for MLP
- `report.pdf`: Detailed implementation report and analysis
- `readme.md`: This file

## KMeans Clustering Implementation (ass4q2.ipynb)

### Implementation Details
1. **Initialization**
   - Used provided centroids as starting points

2. **Algorithm Steps**
   - Assignment: Points assigned to nearest centroid using Euclidean distance
   - Update: Centroids recalculated after each assignment
   - Convergence: Algorithm terminates when centroids stabilize or after 100 iterations
   - Convergence threshold: 1×10⁻⁴

3. **Analysis**
   - Comparison between provided centroids and random initialization
   - Optimal cluster determination using Elbow method
   - WCSS plotting and clustering result visualization

## CIFAR-10 Analysis with PyTorch (ass4q3.ipynb)

### Data Preparation
- Dataset loading using PyTorch
- Stratified random splits (80:20 train-validation ratio)
- Custom Dataset class and data loader implementation
- Visualization of 5 sample images per class from training and validation sets

### CNN Model
#### Architecture
- **First Convolutional Layer**
  - Kernel size: 5x5
  - Channels: 16
  - Stride: 1
  - Padding: 1
  - Max-pooling: 3x3 kernel, stride 2

- **Second Convolutional Layer**
  - Kernel size: 3x3
  - Channels: 32
  - Stride: 1
  - No padding
  - Max-pooling: 3x3 kernel, stride 3

- **Fully Connected Layer**
  - Hidden layer: 16 neurons
  - ReLU activation

### MLP Model
#### Architecture
- **First Layer**
  - 64 neurons
  - ReLU activation
- **Second Layer**
  - Classification head

### Training and Evaluation
- 15 epochs for both models
- Cross-Entropy Loss
- Adam optimizer
- Training and validation metrics logging
- Performance evaluation:
  - Accuracy and F1-score measurement
  - Confusion matrices for train, validation, and test sets
  - Model performance comparison

### Model Outputs
- Trained model weights saved as:
  - `cnn_model.pth`
  - `mlp_model.pth`
- Visualization plots and evaluation metrics in notebook

## Usage Instructions

### For KMeans Clustering (ass4q2.ipynb)
1. Open the notebook
2. Execute cells sequentially for:
   - Clustering implementation
   - Result visualization
   - Elbow method analysis

### For CIFAR-10 Analysis (ass4q3.ipynb)
1. Open the notebook
2. Run cells sequentially for:
   - Data preparation and visualization
   - Model training (CNN and MLP)
   - Performance evaluation and comparison

## Dependencies
- Python 3.8+
- Required libraries:
  - NumPy
  - Matplotlib
  - PyTorch
  - torchvision

For comprehensive details and analysis, please refer to `report.pdf`.
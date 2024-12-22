# Neural Network Implementation and Analysis

## Overview
This project explores neural network implementation through scratch-based development and package-based implementation using sklearn. The project includes comprehensive implementations, model training, and detailed analysis.

## Project Structure
- `ass3q2.ipynb`: Neural Network implementation from scratch
- `ass3q3.ipynb`: Neural Network implementation using sklearn
- `NN_models-20241111T154534Z-001.zip`: Archive containing 12 trained neural network models (.pkl files)
- `report.pdf`: Detailed report of findings, observations, and results
- `readme.md`: This file

## Neural Network Implementation from Scratch (ass3q2.ipynb)

### Neural Network Class Implementation
The custom `NeuralNetwork` class features:
- Configurable layers, activation functions, weight initialization
- Customizable learning rates, epochs, batch sizes

#### Core Methods
- `fit(X, Y)`: Trains the model
- `predict(X)`: Generates predictions
- `predict_proba(X)`: Outputs class-wise probabilities
- `score(X, Y)`: Computes accuracy

### Implementation Components
1. **Activation Functions with Gradients**
   - sigmoid
   - tanh
   - ReLU
   - Leaky ReLU
   - softmax

2. **Weight Initialization Techniques**
   - Zero Initialization
   - Random Initialization
   - Normal Initialization (scaled appropriately)

### MNIST Training Configuration
- Architecture: Hidden layers [256, 128, 64, 32]
- Data Split: 80% Training, 10% Validation, 10% Testing
- Training Parameters:
  - Learning rate: 2e-5
  - Batch size: 128
  - Epochs: 100
- Outputs: Training/validation loss plots and 12 saved models

## sklearn Neural Network Implementation (ass3q3.ipynb)

### Data Processing
- Fashion-MNIST dataset preparation
- Test dataset visualization (10 samples)

### MLP Classifier Implementation
- Architecture: [128, 64, 32]
- Activation Functions:
  - logistic
  - tanh
  - relu
  - identity
- Training Configuration:
  - Learning rate: 2e-5
  - Batch size: 128
  - Epochs: 100

### Advanced Features
1. **Hyperparameter Optimization**
   - Grid search for activation functions
   - Parameter tuning:
     - Solver selection
     - Learning rate optimization
     - Batch size adjustment

2. **MLP Regressor Development**
   - 5-layer architecture: [c, b, a, b, c] (c > b > a)
   - Training specifications:
     - Activations: relu and identity
     - Solver: adam
     - Learning rate: 2e-5
   - Analysis outputs:
     - Loss comparisons
     - Test sample regeneration
     - Performance documentation

## Installation and Usage

### Setup Requirements
- Python environment with Jupyter support
- Required libraries:
  - numpy
  - matplotlib
  - sklearn
  - pandas

### Project Execution
1. Clone the repository
2. Install dependencies
3. Extract model archive: `NN_models-20241111T154534Z-001.zip`
4. Execute notebooks:
   - `ass3q2.ipynb` for scratch implementation
   - `ass3q3.ipynb` for sklearn implementation

### Model Assessment
- Evaluation code provided in notebooks
- Comprehensive results in `report.pdf`

For detailed analysis, methodology, and complete results, refer to the project report.
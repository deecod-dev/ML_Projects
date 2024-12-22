# Machine Learning Implementation Project

## Overview
This project comprises two main sections:
1. Logistic Regression Implementation from Scratch using the Heart Disease dataset
2. Algorithm Implementation using Libraries on the Electricity Bill dataset

Each section showcases a detailed and methodical approach to solving the problem, including exploratory data analysis, model implementation, evaluation, and feature engineering.

## Project Structure
- `ass1q2.ipynb`: Contains the implementation of Logistic Regression from scratch using the Heart Disease dataset
- `ass1q3.ipynb`: Implements various regression techniques and analyses on the Electricity Bill dataset using Python libraries
- `htd1.csv`: Dataset file for Logistic Regression implementation
- `htd2.csv`: Dataset file for regression tasks
- `report.pdf`: Comprehensive documentation of the project, including methodologies, results, and insights

## Section 1: Logistic Regression Implementation (From Scratch)

### Features
- **Batch Gradient Descent**
  - Implementation of Logistic Regression with cross-entropy loss
  - Training/validation loss and accuracy plots over iterations
  - Model convergence analysis

- **Feature Scaling Comparison**
  - Investigation of Min-Max scaling vs. no scaling effects
  - Comparative analysis of loss vs. iteration plots

- **Performance Metrics**
  - Confusion matrix computation
  - Precision, recall, F1 score, and ROC-AUC calculations for validation set
  - Detailed metric analysis

- **Optimization Algorithms**
  - Implementation of Stochastic Gradient Descent and Mini-Batch Gradient Descent
  - Comparative analysis of loss and accuracy across different batch sizes

- **k-Fold Cross-Validation**
  - 5-fold cross-validation for model robustness evaluation
  - Statistical analysis of accuracy, precision, recall, and F1 score

- **Regularization and Early Stopping**
  - L1/L2 regularization experiments
  - Early stopping implementation for overfitting prevention
  - Performance analysis with visualization

## Section 2: Regression Using Libraries

### Features
- **Exploratory Data Analysis**
  - Comprehensive visualization including pair plots, box plots, violin plots
  - Categorical feature analysis with count plots
  - Correlation analysis via heatmap

- **Dimensionality Reduction**
  - UMAP implementation for 2D visualization
  - Scatter plot analysis for clustering and separability

- **Linear Regression**
  - Data preprocessing pipeline
  - Performance metrics: MSE, RMSE, R², Adjusted R², MAE for both training and test sets

- **Feature Selection**
  - RFE/Correlation analysis for top feature selection
  - Performance comparison between full and selected feature sets

- **Advanced Regression Techniques**
  - One-Hot Encoding implementation
  - Ridge Regression analysis
  - Independent Component Analysis (ICA) with varying components
  - ElasticNet Regularization with parameter optimization
  - Gradient Boosting Regression implementation and comparison

## Usage Instructions
1. Clone the repository
2. Ensure all dependencies are installed
3. Open the Jupyter Notebooks:
   - `ass1q2.ipynb` for Logistic Regression implementation
   - `ass1q3.ipynb` for regression analysis using libraries
4. Place the datasets (`htd1.csv` and `htd2.csv`) in the project directory
5. Execute the notebook cells sequentially

## Key Insights
The project demonstrates:
- The importance of understanding machine learning fundamentals through from-scratch implementation
- The practical benefits of using established libraries for complex regression tasks
- The value of comprehensive analysis and visualization in understanding model behavior

For detailed results, analyses, and visualizations, please refer to `report.pdf`.
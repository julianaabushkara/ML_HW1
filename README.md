# Machine Learning Homework 1: ANN & KNN on SIFT Features

## Project Overview

This project implements and compares **Approximate Nearest Neighbor (ANN)** and **K-Nearest Neighbor (KNN)** models for image keypoint matching and classification using SIFT (Scale-Invariant Feature Transform) features.

## Objectives

- Build and implement ANN and KNN models for image keypoint matching
- Compare performance between different nearest neighbor approaches
- Perform place recognition using keypoint-based classification
- Evaluate models on runtime and accuracy metrics

## Dataset

### SIFT Features Format
Each CSV dataset contains keypoints with the following structure:
```
Y, X, Scale, Angle, Response, Feature_1, Feature_2, ..., Feature_128
```

### Datasets Included
- **Migdal Images**: `migdal_1_sift_dataset.csv`, `migdal_2_sift_dataset.csv`
- **Landmark Images**: 11 landmarks with paired training/testing datasets
  - Format: `[landmark_name]_1_sift_dataset.csv` (training)
  - Format: `[landmark_name]_2_sift_dataset.csv` (testing)

## Project Structure

### Part A: Model Implementation (35 pts)
- **KNN**: Linear search implementation
- **ANNBase**: Abstract class with `fit()` and `kneighbors()` methods
- **LSH_ANN**: Locality Sensitive Hashing implementation
- **RKDT_ANN**: Random KD-Tree implementation

### Part B: KNN Nearest Neighbor Search (10 pts)
- Keypoint matching between migdal images
- Runtime measurement and visualization
- Top 10 best matches visualization

### Part C: ANN Models + Grid Search (30 pts)
- Grid search over hyperparameters:
  - **LSH**: K (cuts) and L (hash tables)
  - **RKDT**: N0 (max points per leaf) and L0 (number of trees)
- Ratio test implementation (threshold < 0.8)
- Performance comparison with KNN baseline

### Part D: sklearn Comparison (5 pts)
- Compare with `sklearn.neighbors.NearestNeighbors` (KDTree backend)
- Runtime and accuracy comparison

### Part E: Place Recognition (30 pts)
- Multi-class classification using keypoint voting
- Confidence score computation
- Evaluation metrics: accuracy, macro F1-score, confusion matrix

## Key Algorithms

### Ratio Test
```python
nearest_n, second_nearest = kneighbors(sample, k=2)
ratio = nearest_n_distance / second_nearest_distance
if ratio < 0.8:
    return nearest_n, nearest_n_distance
return None, nearest_n_distance
```

### Distance Ratio Error
```
ε = (1/m) * Σ |dANN(Pi) - dNN(Pi)| / dNN(Pi)
```
Where:
- `dNN(Pi)`: exact nearest neighbor distance
- `dANN(Pi)`: ANN-estimated neighbor distance

## Requirements

- Python 3.x
- Jupyter Notebook
- Required libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `sklearn` (Part D only)
  - `PIL/Pillow` (for visualizations)

## Installation & Setup

1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn pillow jupyter
   ```
3. Place SIFT datasets in appropriate folders
4. Run the Jupyter notebook

## Usage

1. **Data Exploration**: Load and visualize migdal dataset
2. **Model Training**: Implement and train KNN, LSH_ANN, RKDT_ANN
3. **Hyperparameter Tuning**: Grid search for optimal parameters
4. **Evaluation**: Compare models on accuracy and runtime
5. **Classification**: Perform place recognition on landmark datasets

## Evaluation Metrics

- **Runtime**: Time for `kneighbors()` execution
- **Accuracy**: Average distance ratio error
- **Classification**: Accuracy, Macro F1-score, Confusion matrix
- **Confidence**: Prediction confidence scores

## Submission Guidelines

- **Format**: Jupyter Notebook (.ipynb and .html)
- **Collaboration**: Pairs only (contact TAs for exceptions)
- **Documentation**: Clear markdown explanations and code comments
- **Visualizations**: Must include axis labels, titles, and legends
- **Libraries**: Use only specified libraries (sklearn for Part D only)

## Bonus Opportunity

Top 30 performing groups in Part E (place recognition) will receive bonus points:
- Top 10: +5 pts ---> our project was top10 
- Next 10: +3 pts  
- Next 10: +1 pt



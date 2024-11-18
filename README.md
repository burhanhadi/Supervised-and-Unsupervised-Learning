# Supervised-and-Unsupervised-Learning
Repository Overview
This repository contains code, datasets, and results for two comprehensive data mining projects as part of Assignment 2 for the Data Mining (CSC 452) course at Bahria University, Lahore Campus. The projects involve applying data preprocessing, unsupervised learning, feature selection, and supervised learning techniques to two different datasets: one for breast cancer prediction and another for bank loan prediction.

Author
Name: Burhan Hadi Butt
Enrollment: 03-134211-008
Class: BSCS - 8A
Email: burhanhadibutt1@gmail.com

Project Descriptions

Breast Cancer Prediction:
Objective: Apply various data mining techniques to identify malignant and benign cases in breast cancer data.

Techniques Used:
Data preprocessing and feature scaling.
Unsupervised learning (K-Means and Agglomerative Clustering).
Feature selection (Variance Threshold and Recursive Feature Elimination).
Supervised learning classifiers (Decision Tree, Random Forest, SVM, KNN, Gradient Boosting).
Performance Metrics: Accuracy, precision, recall, F1-score, classification reports, and confusion matrices.

Bank Loan Prediction:
Objective: Predict loan approvals using customer data to determine factors contributing to loan acceptance.

Techniques Used:
Data preprocessing and manual feature extraction.
Supervised learning classifiers applied to manually and automatically selected features.
Comparative analysis of feature selection methods (manual and automated).
Performance Metrics: Similar to the breast cancer project, including accuracy and confusion matrices.
Contents of the Repository

Datasets:
breast_cancer_data.csv: Dataset for breast cancer prediction.
bank_loan_data.csv: Dataset for bank loan prediction.

Code Files:
DM_Assignment_2_Part_1.ipynb: This part of the assignment applies unsupervised learning techniques, specifically K-Means and Agglomerative Clustering, to classify breast cancer data into benign and malignant clusters. The analysis includes data preprocessing, feature scaling, cluster validation through metrics such as silhouette scores, and performance evaluation using confusion matrices and adjusted Rand indices.
DM_Assignment_2_Part_2.ipynb: This part of the assignment involves supervised learning to predict breast cancer outcomes using various classifiers such as Decision Tree, Random Forest, SVM, KNN, and Gradient Boosting. The models were evaluated based on metrics like accuracy, precision, recall, F1-score, and confusion matrices, with SVM achieving the highest accuracy of 0.97, closely followed by Random Forest and Gradient Boosting, showcasing strong predictive performance.
DM_Assignment_2_Part_3.ipynb: This part of the assignment focuses on bank loan prediction using supervised learning with three feature selection methods: manual feature selection, Principal Component Analysis (PCA), and Recursive Feature Elimination (RFE). Multiple classifiers including Decision Tree, Random Forest, SVM, KNN, and Gradient Boosting were trained and evaluated on the selected features, with manual selection achieving the highest accuracy, RFE showing robust performance with automated feature selection, and PCA providing efficient dimensionality reduction with competitive results.

Results:

Performance comparison of various classifiers.
Confusion matrices and classification reports for different models.
Visual plots illustrating clustering, feature importance, and other insights.

Project Structure
Supervised-and-Unsupervised-Learning
│
├── datasets/
│   ├── breast_cancer_data.csv
│   └── bankloan.csv
│
├── notebooks/
│   ├── breast_cancer_analysis.ipynb
│   ├── loan_prediction_analysis.ipynb
│   ├── feature_selection_methods.ipynb
│   └── model_evaluation.ipynb
│
├── README.md

Key Steps and Techniques Applied

Data Preprocessing:
Data cleaning, missing value handling, encoding, and scaling.
Visual exploration for data distribution and correlations.

Unsupervised Learning (Breast Cancer Dataset):
K-Means Clustering and Agglomerative Clustering for identifying patterns.
Cluster validation using silhouette scores.

Feature Selection:
Manual feature extraction for the bank loan dataset to maximize classifier accuracy.
Automated feature extraction methods, such as RFE and PCA, to optimize the feature set.

Supervised Learning:
Applied Decision Tree, Random Forest, SVM, KNN, and Gradient Boosting classifiers.
Performance evaluated through accuracy, precision, recall, F1-score, and detailed confusion matrices.

Performance Evaluation:
Comprehensive analysis of model performance, with visualizations for comparative insights.

How to Run the Code
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/data-mining-projects-breast-cancer-loan-prediction.git
Open Jupyter Notebooks or Colab links provided for each project.
Follow the step-by-step execution as outlined in the notebooks.
Tools and Technologies Used
Programming Language: Python

Libraries:
Data Manipulation: pandas, numpy
Visualization: matplotlib, seaborn
Machine Learning: scikit-learn

Environment: Jupyter Notebook, Google Colab, PyCharm

Acknowledgements
This work was completed as part of Assignment 2 for the Data Mining (CSC 452) course under the Department of Computer Science, Bahria University, Lahore Campus.

Note: This repository adheres to academic integrity policies and is for educational purposes only.

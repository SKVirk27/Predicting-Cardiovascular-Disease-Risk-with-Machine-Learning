Predicting Cardiovascular Disease Risk with Machine Learning
Project Overview
This project aims to analyze indirect gene-environment interactions related to cardiovascular disease (CVD) using data from the National Health and Nutrition Examination Survey (NHANES). The primary focus is on environmental factors and their association with CVD risk. We employed various machine learning techniques to uncover the relationships between lifestyle choices and cardiovascular health.

Methodology
Data Collection and Preprocessing
Libraries Required:
For R: RNHANES, sqldf, plyr, dplyr, haven, mice
For Python: numpy, pandas, seaborn, matplotlib.pyplot, sklearn, xgboost
Platforms:
RStudio Version 2023.03.0+386
Google Colab for Machine Learning Modeling
The NHANES datasets from multiple years were merged and imputed using the mice package in R. The final imputed dataset was then used for machine learning modeling in Python.

Feature Selection
Three feature selection methods were employed to identify the top features influencing CVD risk:

Extra Trees Classifier
Mutual Information
Chi-Square
Machine Learning Models
We used the following machine learning models to predict cardiovascular disease risk:

Random Forest Classifier
XGBoost Classifier
Stacking Classifier (Combination of Random Forest, XGBoost, and Logistic Regression)
Results
Random Forest Classifier: Achieved an accuracy of 90%.
XGBoost Classifier: Achieved an optimized accuracy of 92%.
Stacking Classifier: Combined multiple models to achieve a balanced accuracy.
Important Files
Data Collection and Imputation Script: NHANES_Data_Imputation.R
Machine Learning Modeling Script: machine_modeling.py
Usage
To run the project, ensure you have the necessary libraries installed in your R and Python environments. Follow these steps:

Data Collection and Imputation:

Open RStudio and load the NHANES_Data_Imputation.R script.
Execute the script to merge and impute the NHANES data.
Save the imputed dataset as imputed_dataset_1.csv.
Machine Learning Modeling:

Open Google Colab or your preferred Python environment.
Load the machine_modeling.py script.
Ensure the imputed_dataset_1.csv file is accessible.
Run the script to train and evaluate the machine learning models.
Key Insights
The study revealed significant relationships between environmental factors and cardiovascular disease.
The combination of multiple models using the stacking classifier provided robust predictive capabilities.
Conclusion
Understanding the impact of environmental factors on cardiovascular disease can inform public health interventions and personal lifestyle choices. While genetic predispositions are critical, addressing modifiable environmental factors is equally important in preventing and managing cardiovascular diseases.

For further details and to access the scripts, visit the GitHub repositories linked above.








## Project Title: Predicting User Churn for E-commerce Promotions

## Introduction:

This project focuses on analyzing churned users of an e-commerce company to identify patterns in their behavior and develop strategies to reduce churn. The dataset provided (churn_predict.csv) will be utilized to understand user behavior, build a machine learning model for predicting churn, and segment churned users for targeted promotions.

## Objectives:

### 1. Analyze Churned Users:
   - Identify patterns and behaviors of churned users.
   - Provide suggestions to the company for reducing churn.

### 2. Build a Machine Learning Model:
   - Develop a predictive model for churned users, including fine-tuning for optimal performance.

### 3. Segment Churned Users:
   - Classify churned users into distinct groups based on their behaviors.
   - Analyze differences between these groups to inform promotional strategies.

## Analysis Focus:

### 1. Patterns and Behavior of Churned Users:
- **Data Exploration:**
  - Analyze key features such as purchase frequency, average order value, and engagement metrics.
  - Investigate the distance from the warehouse to the customer’s home, focusing on users with a distance less than or equal to 36.

- **Suggestions to Reduce Churn:**
  - Implement targeted retention strategies based on identified behaviors.
  - Consider enhancing delivery services for customers living closer to warehouses.
  - Offer personalized promotions or discounts to encourage repeat purchases.

### 2. Machine Learning Model for Predicting Churn:
- **Data Preparation:**
  - Clean and preprocess the dataset, handling missing values and encoding categorical variables.

- **Model Selection:**
  - Choose appropriate algorithms (e.g., Logistic Regression, Random Forest, XGBoost) for churn prediction.

- **Model Training and Fine-tuning:**
  - Split the dataset into training and testing sets.
  - Use cross-validation and hyperparameter tuning to optimize model performance.

- **Evaluation:**
  - Assess model performance using metrics such as accuracy, precision, recall, and F1-score.

### 3. Segmentation of Churned Users:
- **Clustering Analysis:**
  - Apply clustering techniques ( K-means, Hierarchical Clustering,... ) to segment churned users based on behavioral features.

- **Group Analysis:**
  - Analyze differences between segments, such as demographics, purchasing habits, and engagement levels.

- **Promotional Strategies:**
  - Develop tailored promotions for each segment based on their unique characteristics and needs.

## Tools and Techniques Used:

- Python
- Data Analysis Libraries ( Pandas, NumPy )
- Machine Learning Libraries
- Data Visualization Libraries ( Matplotlib, Seaborn )

## Execution Process:
- Data Collection: Load the churn_predict.csv dataset.
- Data Cleaning: Process the data for analysis and modeling.
- Segmentation: Classify churned customers into groups and analyze the differences.
- Exploratory Data Analysis: Identify patterns and behaviors of churned customers.
- Feature Engineering: Create and calculate necessary features for analysis and evaluation.
- Model Development: Build and fine-tune machine learning models to predict churn.
- Model Application and Evaluation: Apply machine learning models such as K-means, Hierarchical Clustering, and F1 Score to evaluate the model on the training set and apply it to the test set.
- Recommendations: Provide useful insights and promotional strategies based on the analysis.

## Results Achieved:
- Identified significant patterns in the behavior of churned users.
- Developed a robust machine learning model for predicting churn.
- Successfully segmented churned users into distinct groups, informing targeted promotional strategies.

## Project processing workflow
### Step 1: Check Dataset and import others Library
```Python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_excel('/content/drive/MyDrive/Final_project_ML/churn_prediction.xlsx')
df_null = df.isnull().sum()
df_dup = df.duplicated().sum()
df.dtypes
df.shape
df.info()
print(df_null)
print(df_dup)

import missingno as msno
msno.matrix(df)
```

- Result:

![{F08E3459-3CDB-44C2-810B-0B9758E03426}](https://github.com/user-attachments/assets/74cfc060-3746-499d-b662-e99cb7179f5f)

![{31F06F87-74E0-499D-B193-320722F94D9D}](https://github.com/user-attachments/assets/81be79a0-8c30-449b-a10d-19dfb2d523b9)


   - The dataset consists of 5,630 rows and 20 columns.
   - The data has quite a few missing values in the columns Tenure, WarehouseToHome, HourSpendOnApp, OrderAmountHikeFromLastYear, CouponUsed, OrderCount, DaySinceLastOrder, with the highest being 307 null values in one column and the lowest being 251.
   - There are no duplicated entries in the data.
   - Most data types are correctly formatted.
### Step 2: Handle values
- The data has quite a few missing values in the columns Tenure, WarehouseToHome, HourSpendOnApp, OrderAmountHikeFromLastYear, CouponUsed, OrderCount, DaySinceLastOrder, with the highest being 307 null values in one column and the lowest being 251.
- Removing the null data will result in a loss of detail in the dataset, so using the median to fill in the values and analyze is recommended.

```Python
df['Tenure'] = df['Tenure'].fillna(0)

median_WarehouseToHome = df['WarehouseToHome'].median()
df['WarehouseToHome'].fillna(median_WarehouseToHome, inplace=True)

median_HourSpendOnApp = df['HourSpendOnApp'].median()
df['HourSpendOnApp'].fillna(median_HourSpendOnApp, inplace=True)

median_OrderAmountHikeFromlastYear = df['OrderAmountHikeFromlastYear'].median()
df['OrderAmountHikeFromlastYear'].fillna(median_OrderAmountHikeFromlastYear, inplace=True)

median_CouponUsed = df['CouponUsed'].median()
df['CouponUsed'].fillna(median_CouponUsed, inplace=True)

median_OrderCount = df['OrderCount'].median()
df['OrderCount'].fillna(median_OrderCount, inplace=True)

median_DaySinceLastOrder = df['DaySinceLastOrder'].median()
df['DaySinceLastOrder'].fillna(median_DaySinceLastOrder, inplace=True)


#df = df.dropna()
#df_not_null = df.isnull().sum()
#print(df_not_null)

import missingno as msno
msno.matrix(df)
```

- Result:

![{E9977BDB-7410-4303-9F26-4544E71DF3B2}](https://github.com/user-attachments/assets/7a69b880-66c4-4ded-ba61-4c3674ce521a)






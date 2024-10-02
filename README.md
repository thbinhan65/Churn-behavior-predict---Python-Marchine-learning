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

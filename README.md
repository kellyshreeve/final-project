## Final Project: Predicting Telecom Customer Churn
<p align="center">
  <img src="images/phone_clipart.png"
  width="300"
  height="300"
  alt="Phone clip art">
</p>

### Project Overview
**Background**: Telecom operator Interconnect would like to forecast the churn of their clients. If the customer is likely to leave, they will be sent promotions and special plan offers.  

**Purpose**: Fit an imbalanced classification model that accurately predicts which customers are likely to leave the company.  

**Techniques**: CatBoost, LGBM, XGBoost, AdaBoost, pipelines, GridSearchCV, class balancing.  

### Installation and Setup

#### Codes and Resources Used

  - <b>Editor Used</b>: Visual Studio Code
  - <b>Python Version</b>: 3.10.9

#### Python Packages Used

  - <b>General Purpose</b>: ```numpy, time```  
  - <b>Data Manipulation</b>: ```pandas```  
  - <b>Data Visualization</b>: ```matplotlib, seaborn```  
  - <b>Machine Learning</b>: ```sklearn, imblearn```  
  - <b>Gradient Boosting</b>: ```catboost, lightgbm, xgboost```

### Data

*contract_df.csv*, *internet_df.csv*, *personal_df.csv*, *phone_df.csv*

**Target**:   
* *churned*: 0 = has not churned, 1 = has churned

**Features**:   
* *begin_year*: year of contract start date
* *type*: type of contract  
* *paperless_billing*: customer has paperless billing or not  
* *payment_method*: type of payment method  
* *monthly_charges*: amount customer is charged per month  
* *total_charges*: total payments over life of plan
* *internet_service*: type of internet service  
* *gender*: male or female  
* *senior_citizen*: whether the customer is a senior citizen  
* *partner*: whether the customer has a partnmer  
* *dependents*: whether the customer has dependents  
* *multiple_lines*: whether a customer has multiple phone lines; 'no_plan' for customers who do not have phone  
* *has_protection*: 0 = no proteciton services, 1 = at least one protection service  
* *has_streaming*: 0 = no streaming services, 1 = at least one streaming service
 
#### Data Acquisition

The data were provided by TripleTen's Data Science bootcamp. The full dataset is loaded into the notebook but is proprietary information and cannot be shared online.

#### Data Preprocessing

Data were checked for missing values and duplicates. The 4 missing values in total charges were filled with the median of total charges. No other duplicates or missing values were found.  

The four datasets were merged on customer id to make a comprehensive dataset for all variables.  

Four additional features were created:  
1. begin_year: begin date of service  
2. has_protection: whether a customer has any of three internet protection services  
3. has_streaming: whether a customer has either streaming service  
4. total_internet_services: total number of extra internet services on a customer's subscription  

### Code Structure
```
  ├── LICENSE
  ├── README.md          
  │
  ├── images
  │   └── churn_over_time.png
  │   └── class_imbalance.png 
  │   └── correlation_heatmap.png 
  │   └── histograms.png 
  │   └── test_results.png 
  │   └── training_results.png     
  │
  └── notebooks  
      └── project_code.ipynb
      └── project_plan.ipynb
```

### Results and Evaluation

#### Exploratory Analysis
 
<p align="left">
  <img src="/images/class_imbalance.png"
  width="500"
  height="500"
  alt="Bar plot of target variable, showing class imbalance">
</p>

There are fewer customers who churned than did not churn. This is an imbalanced classificaiton problem.

<p align="left">
  <img src="/images/churn_over_time.png" 
  width="650"
  height="250"
  alt="Correlation heatmap">
</p>

* Customers who began their contracts in 2014 - 2018 are almost all still with the company.  
* About 50% of customers who began their contracts in 2019 - 2020 have already churned.  
* New customers are more likely to leave than old customers.  
 
<p align="left">
  <img src="/images/histograms.png" 
  width="650"
  height="250"
  alt="Correlation heatmap">
</p>

* The distribution of monthly charges has three peaks at $20, $50, and $80 per month.  
* Total charges is highly right skewed, with most people paying close to $0 total and only a few people paying over $6000 over the life of their plan.  
* Contract length is bi-modal, with many people having contracts less than 100 months or more than 2000 months.  

<p align="left">
  <img src="/images/correlation_heatmap.png" 
  width="650"
  height="250"
  alt="Correlation heatmap">
</p>

* The correlation heatmap shows high correlations between numeric features, representing multicollinearity, and a violation of the assumption of non-multicollinearity. Some features will need to be removed from the model.
* Total charges, while highly correlated with begin year (r = -0.82), shares only a moderate correlation with monthly charges (r = 0.65). Tree models are not highly affected by slight multicollinearity. Total charges will be kept in the model.
* Begin year and monthly charges have a low correlation with each other and will be kept in the model (r = -0.26)
* Contract length and total internet services will be removed from the model.

#### Train Results

<p align="left">
  <img src="/images/train_results.png"
  width="450"
  height="250"
  alt="Train results">
</p>

* The best model was the LightGBM trained on SMOTE upsampled data.
* This model achieved the highest scores roc-auc and accuracy (ROC-AUC = 0.88, accuracy = 0.81), though had lower scores on precision, recall, and f1 (precision = 0.61, recall = 0.78, f1 = 0.69)
* The LightGBM Model will be tested on the test set.

#### Test Results

<p align="left">
  <img src="/images/test_results.png"
  width="250"
  height="100"
  alt="Test results">
</p>

* The LighGBM Classifier, fit on SMOTE upsampled training data, achieved a lower ROC-AUC on the test set (ROC-AUC = 0.80).
* This model is likely slightly overfit but still achieve a reasonable training score.

### Conclusions and Business Application

#### Conclusions

LightGBM GBDT achieved the best model fit (RMSE test = 1663.83). Predictions from this model will offer customers the predicted value of their car within $1,663.83 on average. The most important features were predicting price were power, registration year, postal code, and mileage.  

#### Business Application 

Rusty Bargain will be able to implement this model in their app and be confident that customers will receive accurate predictions in about 1 second. 

#### Future Research 

With additional time, more hyperparameters and trees/iterations could be performed to improve model accuracy. Additionally, further data cleaning may improve the accuracy of the results.

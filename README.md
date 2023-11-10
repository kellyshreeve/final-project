## Final Project: Predicting Telecom Customer Churn
<p align="center">
  <img src="images/phone_clipart.png"
  width="400"
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

#### Source Data

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

Variables missing data were all missing less than 15% of observations. Categorical missing values were filled with 'unknown' and quantitative missing values were imputed with medians. Duplicates were cleaned from the dataset.

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
  alt="sns pair plot of numeric variables">
</p>

There are no clear associations between the dependent variable price and registriation_year, power, mileage, or registration_month. There is also a possible violation of linearity between price and power.

<p align="left">
  <img src="/images/churn_over_time.png" 
  width="650"
  height="250"
  alt="Correlation heatmap">
</p>

Price has a moderate, positive correlation with registration year (r = 0.37) and power (r = 0.40). Price has a moderate, negative correlation with mileage(r = -0.33). Price is only weakly related to registration month (r = 0.11). The features registration year, power, and mileage are very weakly correlated with each other. Multicollinearity is not an issue.

<p align="left">
  <img src="/images/histograms.png" 
  width="650"
  height="250"
  alt="Correlation heatmap">
</p>

<p align="left">
  <img src="/images/correlation_heatmap.png" 
  width="650"
  height="250"
  alt="Correlation heatmap">
</p>

#### Train Results

<p align="left">
  <img src="/images/train_results.png"
  width="450"
  height="250"
  alt="Train results">
</p>

LightGBM achieved the lowest RMSE (RMSE = 1739.38) and highest R^2 value (R^2 = 0.85).  LightGBM took the longest to tune, but this was due to the large number of hyperparameters entered into the grid. LightGBM was able to tune more hyperparameters options than Random Forest and CatBoost in a similar amount of time. Both standard and ridge regression had very quick computations, but they were over $1000 less accurate in their predictions than  LightGBM GBDT. Considering both model score and time, LightGBM GBDT is the best model.

#### Test Results

<p align="left">
  <img src="/images/test_results.png"
  width="250"
  height="100"
  alt="Test results">
</p>

LightGBM GBDT achieved a lower RMSE and higher R^2 on the test set (RMSE = 1663.85, R^2 = 0.86). The model is likely not overfit. It was able to make predictions in less than one second.

### Conclusions and Business Application

#### Conclusions

LightGBM GBDT achieved the best model fit (RMSE test = 1663.83). Predictions from this model will offer customers the predicted value of their car within $1,663.83 on average. The most important features were predicting price were power, registration year, postal code, and mileage.  

#### Business Application 

Rusty Bargain will be able to implement this model in their app and be confident that customers will receive accurate predictions in about 1 second. 

#### Future Research 

With additional time, more hyperparameters and trees/iterations could be performed to improve model accuracy. Additionally, further data cleaning may improve the accuracy of the results.

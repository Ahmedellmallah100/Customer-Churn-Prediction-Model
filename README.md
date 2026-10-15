# Customer Churn Prediction

## Project Overview

Customer churn is a major challenge for telecommunications companies. Understanding which customers are likely to leave can help businesses improve customer retention and make better decisions.

This project applies machine learning classification techniques to predict customer churn using customer demographics, subscription information, service usage, and billing details.

The main objective is to compare different classification models, evaluate their performance, and identify an effective approach for detecting customers who are likely to leave.

## Dataset

The dataset contains **7,043 customer records and 21 columns**, including customer information, subscribed services, contract details, payment methods, and the target variable `Churn`.

### Target Variable

* `0`: No Churn
* `1`: Churn

### Main Features

* **Customer Information:** Gender, Senior Citizen, Partner, and Dependents.
* **Account Information:** Tenure and Contract.
* **Services:** Internet Service, Phone Service, Online Security, Tech Support, and Streaming Services.
* **Billing Information:** Monthly Charges, Total Charges, and Payment Method.

## Technologies Used

* Python
* Pandas and NumPy
* Matplotlib and Seaborn
* Scikit-learn
* Jupyter Notebook

## Project Workflow

### 1. Data Exploration

* Inspected the dataset structure and data types.
* Analyzed numerical and categorical features.
* Examined the distribution of the target variable.
* Checked for missing values and duplicate records.

### 2. Data Cleaning

* Removed `customerID` because it is an identifier rather than a useful predictive feature.
* Converted `TotalCharges` to a numerical data type.
* Handled missing numerical values using a median imputer.
* Encoded categorical features for machine learning.

### 3. Data Preprocessing

A preprocessing pipeline was built using Scikit-learn.

* **SimpleImputer:** Handles missing numerical and categorical values.
* **StandardScaler:** Scales numerical features.
* **OneHotEncoder:** Converts categorical features into numerical representations.
* **ColumnTransformer:** Applies the appropriate transformations to different feature groups.
* **Pipeline:** Combines preprocessing and model training into a consistent workflow.

The dataset was split into training and testing sets using an 80/20 split, with stratification to preserve the target class distribution.

### 4. Model Training and Comparison

Five classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. AdaBoost
5. K-Nearest Neighbors (KNN)

The models were compared using accuracy, precision, recall, and F1-score, with particular attention to the positive class (customers who churn).

### 5. Hyperparameter Tuning

`GridSearchCV` with five-fold cross-validation was used to search for better hyperparameter configurations for KNN and Gradient Boosting.

The search used F1-score as its evaluation metric.

## Model Performance

The following results were obtained on the test set before hyperparameter tuning:

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |    80.6% |     65.7% |  55.9% |    60.4% |
| Decision Tree       |    72.9% |     49.0% |  50.5% |    49.7% |
| Random Forest       |    77.8% |     60.3% |  47.6% |    53.2% |
| AdaBoost            |    80.2% |     65.2% |  54.5% |    59.4% |
| K-Nearest Neighbors |    76.4% |     55.3% |  57.5% |    56.4% |

*Note: Precision, recall, and F1-score refer to the churn class.*

### Hyperparameter Tuning Results

**K-Nearest Neighbors (KNN)**

The best configuration found by GridSearchCV was:

* `n_neighbors = 15`
* `metric = 'manhattan'`
* `weights = 'uniform'`

Test performance:

* Accuracy: 76.8% (approximately 77%)
* F1-Score: 56.0%

**Gradient Boosting**

The best configuration found by GridSearchCV was:

* `n_estimators = 200`
* `learning_rate = 0.1`
* `max_depth = 2`

Test performance:

* Accuracy: 80.0%
* Precision: 66.0%
* Recall: 52.0%
* F1-Score: 58.0%

### Key Findings

* Logistic Regression achieved the highest test F1-score among the five baseline models, reaching **60.4%**.
* AdaBoost delivered comparable performance, with a test F1-score of 59.4%.
* Tuned Gradient Boosting achieved 80% test accuracy but did not outperform the Logistic Regression baseline in F1-score.
* Decision Tree and Random Forest showed a substantial gap between training and testing performance, indicating potential overfitting.
* Detecting churn remains challenging because the dataset contains more customers who stay than customers who leave.

## Business Insights

The model results highlight several practical considerations:

* Customers predicted to churn could be prioritized for retention campaigns.
* Recall should be considered carefully because missed churn cases represent customers the company may lose without intervention.
* Precision is also important because contacting customers who would have stayed may increase campaign costs.
* The final decision threshold should depend on the cost of losing customers and the effectiveness of retention offers.

## Limitations

* The dataset has an imbalanced target distribution.
* The models do not identify every customer who churns.
* Hyperparameter tuning does not guarantee better performance on unseen data.
* Further validation is needed before using predictions in a real business environment.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for customer churn prediction, from data preparation and preprocessing to model comparison and hyperparameter tuning.

Logistic Regression provided the strongest baseline result based on test F1-score, while AdaBoost and Gradient Boosting achieved comparable performance.

Future improvements could focus on increasing churn recall, investigating classification thresholds, and evaluating model performance through additional validation experiments.

## How to Run the Project

1. Clone the repository:

   ```bash
   git clone <your-repository-url>
   ```

2. Navigate to the project directory:

   ```bash
   cd <your-project-folder>
   ```

3. Install the required libraries:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

4. Open Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open the project notebook and run the cells in order.

**Note:** The notebook currently loads the dataset using a Google Drive file ID. Make sure the dataset is accessible before running the notebook.

## Project Structure

```text
Customer-Churn-Prediction/
│
├── Untitled8.ipynb
├── README.md
└── requirements.txt
```

*The structure above is a suggested repository layout. Update the filenames to match the files you upload to GitHub.*

# German Credit Risk Prediction

### **1. Objective: Predicting Credit Risk Using Machine Learning Models**
**Topic:** Credit risk classification

The objective of this project is to build machine learning models that **predict the credit risk** of a bank borrower based on their personal and financial characteristics. The target variable is the `Risk` column, which has two values:

* `good`: the borrower is likely to repay the loan,
* `bad`: the borrower is at risk of defaulting on the loan.

### **2. Dataset Introduction**

Source: [German Credit Risk - With Target](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk) on Kaggle
The original dataset contains **1,000 records** compiled by **Professor Hofmann**, where **each record represents a bank borrower**. Each borrower is **classified as having either "good" or "bad" credit risk** based on a set of relevant attributes.

#### 📋 **Contents**

* The original dataset has 20 **categorical/symbolic** attributes.
* Because the original dataset is relatively difficult to understand due to its numerous symbols and complex categories, the author wrote a Python script to **convert it into a more readable CSV format**.
* Some columns were removed because they were considered unnecessary or difficult to understand.

The updated attribute table is shown below:

---
#### 📑 **Dataset Attributes**

| **Attribute**      | **Data Type**                                                                                                                   | **Meaning**                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `Age`              | numeric                                                                                                                         | Borrower's age                                                     |
| `Sex`              | text (`male`, `female`)                                                                                                         | Gender                                                             |
| `Job`              | numeric:<br>0 - unskilled and non-resident<br>1 - unskilled and resident<br>2 - skilled<br>3 - highly skilled                   | Employment skill level and residency status                       |
| `Housing`          | text (`own`, `rent`, `free`)                                                                                                    | Housing type (owned, rented, or free)                              |
| `Saving accounts`  | text (`little`, `moderate`, `quite rich`, `rich`)                                                                               | Savings level                                                      |
| `Checking account` | numeric (Unit: Deutsche Mark - DM)                                                                                               | Checking account balance                                           |
| `Credit amount`    | numeric (DM)                                                                                                                    | Loan amount                                                        |
| `Duration`         | numeric (months)                                                                                                                | Loan duration in months                                            |
| `Purpose`          | text (`car`, `furniture/equipment`, `radio/TV`, `domestic appliances`, `repairs`, `education`, `business`, `vacation/others`)   | Purpose of the loan                                                |
| `Risk`             | text (`good`, `bad`)                                                                                                            | **Target variable**: whether the borrower has good or bad credit risk |

### **3. Models Used** 🤖

To solve this classification problem, several common machine learning algorithms are applied:

1. **Logistic Regression**

   * A simple linear model that is effective for binary classification problems.
   * Easy to interpret and suitable when the relationship between the independent variables and the target variable is linear.

2. **K-Nearest Neighbors (KNN)**

   * An algorithm based on the distance between data points.
   * Makes no assumptions about the data distribution and is suitable for nonlinear data.

3. **Decision Tree Classifier**

   * A simple, interpretable decision-tree model capable of handling both categorical and continuous data.
   * Able to represent complex relationships between features.

4. **Stacking Classifier**

   * An ensemble model that combines multiple base models (in this case, potentially Logistic Regression, KNN, and Decision Tree).
   * Its goal is to leverage the strengths of each individual model and improve prediction accuracy through an aggregated model (meta-model).

### **4. Workflow** 📌

* Preprocess the data by handling missing values, encoding labels (label encoding / one-hot encoding), and scaling features when necessary.
* Train and evaluate the models using metrics such as accuracy, precision, recall, and F1-score.
* Compare model performance and select the best model for the credit risk prediction system.

### 5. Data Preprocessing

#### 5.1. Initial Exploration and Cleaning

This is the step where we become familiar with the data.

* **Remove unnecessary columns**: The code removes the `'Unnamed: 0'` column. This column is typically an index created when saving a CSV file and has no predictive value, so it is treated as noise and removed.
* **Review basic statistics**: The `.info()`, `.describe()`, and `.value_counts()` methods are used to understand the dataset size, column contents, data types, and value distributions.

***

#### 5.2. Handling Missing Data

Real-world data is often incomplete. The code identifies missing values (`NaN`) in the `Saving accounts` and `Checking account` columns.

* **Method**: The code fills the missing values using the **mode**.
* **Reason**: The **mode** is the most frequently occurring value in a column. For categorical columns such as the savings account category, filling missing values with the most common value is a reasonable and safe choice compared with guessing or deleting rows, which would result in information loss.

***

#### 5.3. Handling Outliers

Outliers are data points that differ significantly from the rest of the data—for example, a borrower requesting an exceptionally large loan compared with the average. These values can bias the model and lead to inaccurate predictions.

* **Method**: The code uses the **IQR (Interquartile Range)** method.
* **How it works**: This method calculates a reasonable value range for each numerical column (`Age`, `Credit amount`, and `Duration`). Any value outside this range is considered an outlier and **removed from the dataset**.

***

#### 5.4. Data Encoding and Standardization ⚙️

This is a crucial step that transforms the data into a numerical format the models can process. The code uses `ColumnTransformer` to apply different transformations to each type of column:

* **For categorical columns** such as `Sex`, `Job`, `Housing`, and `Purpose`:
    * **Action**: Apply **One-Hot Encoding**.
    * **Explanation**: Models cannot directly interpret text values such as `'male'`, `'female'`, `'own'`, and `'rent'`. One-Hot Encoding converts each categorical value into a new binary (0/1) column. For example, the `Housing` column, with the three values `'own'`, `'rent'`, and `'free'`, is converted into three new columns: `Housing_own`, `Housing_rent`, and `Housing_free`. If a person owns their home, `Housing_own` is set to `1`, while the other two columns are set to `0`.

* **For numerical columns** such as `Age`, `Credit amount`, and `Duration`:
    * **Action**: Apply **StandardScaler**.
    * **Explanation**: These columns use very different scales (age values are usually in the tens, while loan amounts can reach the thousands). This may cause a model to incorrectly assume that `Credit amount` is more important than `Age`. `StandardScaler` **standardizes** all numerical columns to the same scale, with a mean of 0 and a standard deviation of 1, helping the model treat all attributes fairly.

***

### 5. Data Splitting (Train-Test Split)

After the data has been cleaned and standardized, the final step is to split it.

* **Action**: Use `train_test_split` to divide the data into two subsets:
    * **Training set**: Contains 80% of the data and is used to teach the model to learn patterns.
    * **Test set**: Contains 20% of the data that the model has never seen. It is used to evaluate how well the model performs on real-world data.
* **Important parameter—`stratify=y_encoded`**: This parameter ensures that the proportions of "good" and "bad" records in both the training and test sets are the same as in the original dataset. This is extremely important for an objective and accurate model evaluation.


### **6. How to Run the Project**

- Clone the repository. Open a terminal in your working directory and run:

```cmd
git clone https://github.com/Lam-Hung-ai/machine_learning_project.git
```

- Install the required libraries. Open a terminal in the project directory and run:

```cmd
pip install -r requirements.txt
```

- Run all cells in `main.ipynb`.

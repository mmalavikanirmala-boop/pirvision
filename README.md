# Pirvision – Machine Learning Classification

## 📌 Project Overview

Pirvision is a machine learning classification project focused on preparing and analyzing a dataset for classification tasks. The project uses Python libraries to perform Exploratory Data Analysis (EDA), handle outliers, balance target classes, select important features, transform skewed data, and scale numerical features.

The main objective is to prepare clean, balanced, and suitable data for building a classification model.

## 🎯 Objectives

- Perform Exploratory Data Analysis (EDA).
- Understand the dataset structure and statistical summary.
- Check for missing values and duplicate records.
- Analyze target class distribution.
- Identify correlations between numerical features.
- Detect and handle outliers using the IQR method.
- Balance target classes using SMOTE.
- Select important features using SelectKBest.
- Reduce feature skewness using the Yeo-Johnson transformation.
- Standardize features using StandardScaler.

## 🛠️ Technologies Used

- **Python** – Programming language
- **NumPy** – Numerical operations
- **Pandas** – Data manipulation and analysis
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical visualization
- **Scikit-learn** – Feature selection and preprocessing
- **Imbalanced-learn** – Class balancing using SMOTE
- **Jupyter Notebook** – Development environment

## 📂 Project Structure

```text
Pirvision-Classification/
│
├── malavika_ml_project__2_.ipynb
├── pirvision.csv
├── README.md
└── requirements.txt
```

*Note: Add `pirvision.csv` to the repository if you are permitted to share the dataset.*

## 📊 Project Workflow

### 1. Data Loading

Load the dataset using Pandas and create a DataFrame for analysis.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("pirvision.csv")
df = pd.DataFrame(data)
```

### 2. Exploratory Data Analysis (EDA)

Analyze the dataset using:

- `df.head()` – View the first five records.
- `df.tail()` – View the last five records.
- `df.info()` – Check column data types and non-null counts.
- `df.describe()` – View statistical summaries.
- `df.shape` – Check the number of rows and columns.
- `df.columns` – Display column names.

### 3. Missing Values and Duplicates

Check missing values and duplicate records.

```python
df.isnull().sum()
df.duplicated().sum()
```

### 4. Target Variable Analysis

The original target column is named `Label`. It is renamed to `target` for further processing.

```python
df = df.rename(columns={"Label": "target"})
```

A count plot is used to visualize the target class distribution.

### 5. Correlation Analysis

Calculate correlations between numerical features and visualize them using a heatmap.

```python
c = df.corr(numeric_only=True)
sns.heatmap(c, cmap="coolwarm")
plt.show()
```

### 6. Outlier Handling

Use the Interquartile Range (IQR) method to identify extreme values and cap them within calculated lower and upper bounds.

### 7. Class Balancing Using SMOTE

Apply Synthetic Minority Over-sampling Technique (SMOTE) to balance the target classes.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)
```

### 8. Feature Selection

Use SelectKBest with the ANOVA F-test to select 25 features based on their scores.

```python
from sklearn.feature_selection import SelectKBest, f_classif

skb = SelectKBest(score_func=f_classif, k=25)
```

### 9. Data Transformation

Use the Yeo-Johnson PowerTransformer to reduce feature skewness.

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method="yeo-johnson")
```

### 10. Feature Scaling

Apply StandardScaler to standardize the selected and transformed features.

```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_scaled = ss.fit_transform(X_transformed)
```

## 📈 Expected Outcome

The preprocessing workflow produces a feature matrix with selected features, transformed distributions, and standardized values. SMOTE is used to address class imbalance.

The notebook currently covers data exploration and preprocessing. Classification model training and evaluation should be added as separate steps before reporting accuracy or other performance metrics.

## 🚀 How to Run the Project

1. Clone the repository.

   ```bash
   git clone https://github.com/YOUR-USERNAME/Pirvision-Classification.git
   ```

2. Open the project folder.

   ```bash
   cd Pirvision-Classification
   ```

3. Install the required libraries.

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure `pirvision.csv` is in the project folder.

5. Open the notebook.

   ```bash
   jupyter notebook malavika_ml_project__2_.ipynb
   ```

6. Run the notebook cells in order.

## 📦 Requirements

Create a `requirements.txt` file with:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
jupyter
```

## 🔮 Future Enhancements

- Split the dataset into training and testing sets.
- Build classification models such as Logistic Regression, Decision Tree, Random Forest, or Support Vector Machine.
- Evaluate models using accuracy, precision, recall, F1-score, and a confusion matrix.
- Compare model performance and select a suitable model.
- Save the trained model for future predictions.

## 👩‍💻 Author

**Malavika M.**

Machine Learning Classification Project

## 📄 License

This project is intended for educational and learning purposes. Add a suitable open-source license if you want others to reuse or modify your code.

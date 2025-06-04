# 🏡 House Price Prediction Using Ensemble Machine Learning

This project focuses on the accurate prediction of housing prices using a combination of classical and ensemble machine learning algorithms. It covers two datasets — **California Housing (Ames)** and **King County, WA** — and aims to assist both individuals and business stakeholders in real estate decision-making by providing transparent and data-driven pricing predictions.

---

## 📌 Motivation

The real estate sector is rapidly expanding but plagued with inconsistencies, non-transparency, and misinformation from intermediaries. Accurate price prediction models can empower homeowners, investors, and analysts to make better financial decisions and avoid being misled by arbitrary market rates.

In a society driven by evolving housing demands, this project seeks to:

- Simplify price estimation with minimal fieldwork
- Avoid overfitting and inefficiencies found in traditional models
- Improve forecast accuracy using **ensemble learning**
- Serve both personal financial planning and large-scale business analysis


---

## 🔁 Methodology

### 🔧 Preprocessing
- Dropped irrelevant columns (e.g., `id`, `date`)
- Handled missing data using:
  - Mean/median for numerical
  - Mode/imputation for categorical
- Removed features with >75% missing data

### 📊 Exploratory Data Analysis
- Outlier detection and distribution analysis
- Correlation heatmaps
- Skewness correction with log/Box-Cox transformations

### 🧠 Feature Engineering
- Added `age`, `garage_age`, `total_area`, etc.
- One-hot and label encoding for categorical variables
- Feature scaling using `StandardScaler`

### 🔍 Models Evaluated

#### Machine Learning Models:
- Linear Regression
- K-Nearest Neighbors (KNN)
- Support Vector Regressor (SVR)

#### Ensemble Learning Models:
- Random Forest Regressor
- AdaBoost Regressor
- Gradient Boosting Regressor
- XGBoost
- CatBoost

### 📐 Evaluation Metrics
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score**

---

## 📈 Performance Summary

### 📍 California Dataset

| Model     | RMSE  | MSE   | MAE   | R²    |
|-----------|-------|-------|-------|-------|
| Linear    | 0.071 | 0.005 | 0.039 | 0.788 |
| KNN       | 0.082 | 0.007 | 0.057 | 0.722 |
| SVM       | 0.070 | 0.005 | 0.052 | 0.798 |
| XGBoost   | 0.051 | 0.003 | 0.033 | 0.891 |
| AdaBoost  | 0.069 | 0.005 | 0.052 | 0.804 |
| Gradient  | 0.050 | 0.002 | 0.034 | 0.896 |
| RandomFor | 0.054 | 0.003 | 0.037 | 0.880 |
| **CatBoost** | **0.048** | **0.002** | **0.033** | **0.903** |

### 📍 King County Dataset

| Model     | RMSE   | MSE    | MAE   | R²     |
|-----------|--------|--------|-------|--------|
| Linear    | 0.5037 | 0.2537 | 0.387 | 0.7463 |
| KNN       | 0.4182 | 0.1749 | 0.300 | 0.8251 |
| SVM       | 0.3826 | 0.1464 | 0.271 | 0.8536 |
| XGBoost   | 0.3443 | 0.1186 | 0.236 | 0.8814 |
| AdaBoost  | 0.4901 | 0.2402 | 0.375 | 0.7598 |
| Gradient  | 0.3323 | 0.1105 | 0.231 | 0.8895 |
| RandomFor | 0.3499 | 0.1225 | 0.241 | 0.8775 |
| **CatBoost** | **0.3237** | **0.1048** | **0.222** | **0.8952** |

---

## 🔬 Insights & Contributions

- **Ensemble models significantly outperform basic models** by reducing variance and avoiding overfitting.
- **CatBoost consistently achieves the highest prediction accuracy**, making it suitable for deployment in production-grade real estate platforms.
- The project illustrates the **importance of preprocessing and transformation**, especially when dealing with skewed or incomplete data.

---

## 🛠️ Tools & Technologies

- **Languages**: Python 3.9+
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`, `xgboost`, `catboost`
  - `scipy`, `statsmodels`

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/karthikmaddirala/House-Price-Prediction-using-Ensemble-learning.git
cd House-Price-Prediction-using-Ensemble-learning

Open in Jupiter Notebook or Google Colab
Install dependencies

# 🛒 Walmart Weekly Sales Prediction

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7+-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.3+-green.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Machine Learning regression model to predict weekly sales and understand economic indicators' impact on retail performance**

## 📋 Table of Contents
- [Context](#-context)
- [Project Objective](#-project-objective)
- [Goals](#-goals)
- [Data](#-data)
- [Technologies](#-technologies)
- [Methodology](#-methodology)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Author](#-author)

---

## 🎯 Context

### About Walmart
**Walmart Inc.** is an American multinational retail corporation that operates a chain of:
- 🏪 Hypermarkets
- 🛍️ Discount department stores
- 🥫 Grocery stores

**Founded**: 1962 by Sam Walton  
**Headquarters**: Bentonville, Arkansas, USA  
**Presence**: United States and international markets  
**Position**: World's largest retailer by revenue

### Business Challenge
Walmart's marketing service needs to understand the factors influencing weekly sales across their stores to:
- 📊 Better understand sales patterns
- 📈 Identify key economic indicators
- 🎯 Plan more effective marketing campaigns
- 💰 Optimize inventory and staffing
- 🔮 Forecast future sales accurately

---

## 🚀 Project Objective

Build a **machine learning regression model** to estimate weekly sales in Walmart stores with the highest possible precision.

### Key Questions
1. What are the main drivers of weekly sales?
2. How do economic indicators affect sales performance?
3. Can we predict sales accurately to support business decisions?
4. Which stores perform best and why?
5. What seasonal patterns exist in the data?

### Success Criteria
- **High prediction accuracy** (low RMSE, high R²)
- **Model interpretability** (understand feature importance)
- **Generalization** (avoid overfitting)
- **Actionable insights** for marketing strategies

---

## 🎯 Goals

The project is divided into three main parts:

### Part 1: Exploratory Data Analysis (EDA) & Preprocessing

### Part 2: Baseline Model - Linear Regression

### Part 3: Regularized Regression Models

### Bonus Challenge


---

## 📊 Data

### Dataset
**File**: `Walmart_Store_Sales.csv`


---

## 🛠️ Technologies

### Core Libraries
```python
pandas                  # Data manipulation
numpy                   # Numerical computing
scikit-learn            # Machine Learning
scipy                   # Statistical functions
```

### Machine Learning Models
- **Linear Regression**: Baseline model
- **Ridge Regression**: L2 regularization
- **Lasso Regression**: L1 regularization (feature selection)
- **ElasticNet**: Combination of L1 and L2 (bonus)

### Optimization & Validation
```python
GridSearchCV            # Hyperparameter tuning
cross_val_score         # Cross-validation
train_test_split        # Data splitting
```

### Visualization
```python
matplotlib              # Static plots
seaborn                 # Statistical visualizations
plotly                  # Interactive charts
```

---

## 🔬 Methodology

### Part 1: EDA & Preprocessing

---

### Part 2: Linear Regression (Baseline)


**Expected Issues:**
- ⚠️ Potential overfitting (high variance)
- ⚠️ Sensitivity to outliers
- ⚠️ Multicollinearity among features

---

### Part 3: Regularized Regression

#### 3.1 Ridge Regression (L2 Regularization)

**Ridge Benefits:**
- ✅ Reduces overfitting
- ✅ Handles multicollinearity
- ✅ Keeps all features (shrinks coefficients)

#### 3.2 Lasso Regression (L1 Regularization)

**Lasso Benefits:**
- ✅ Automatic feature selection
- ✅ Reduces model complexity
- ✅ Improves interpretability

---

### Part 4: Hyperparameter Optimization (Bonus)

---

## 📁 Project Structure

```
walmart-sales-prediction/
│
├── 📓 analysis.ipynb                    # Main analysis notebook
├── 📊 Walmart_Store_Sales.csv           # Dataset
├── 📝 README.md                         # This file
├── 📄 LICENSE                           # MIT License
└── 📦 requirements.txt                  # Python dependencies

```

---

## 💻 Installation

### Prerequisites
- Python 3.11 or higher
- Jupyter Notebook

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Ter0rra/02_Jedha_cdsd_bloc_3_Walmart
cd 02_Jedha_cdsd_bloc_3_Walmart
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook**
```bash
jupyter notebook analysis.ipynb
```

---

## 📚 Mathematical Background

### Ridge Regression (L2)
Minimizes: `RSS + α × Σ(βi²)`

- Shrinks coefficients towards zero
- Keeps all features
- Good when features are correlated

### Lasso Regression (L1)
Minimizes: `RSS + α × Σ|βi|`

- Can set coefficients exactly to zero
- Performs feature selection
- Produces sparse models

### Hyperparameter α
- **α = 0**: Standard linear regression
- **Small α**: Weak regularization
- **Large α**: Strong regularization (coefficients → 0)

---

## 👤 Author

**Romano Albert**
- 🔗 [LinkedIn](www.linkedin.com/in/albert-romano-ter0rra)
- 🐙 [GitHub](https://github.com/Ter0rra)


---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Jedha** for the online training
- **Walmart** for the business case and data
- **Scikit-learn** community for excellent ML tools
- **Pandas** and **NumPy** for data manipulation capabilities

---

## 📞 Support

Questions or suggestions?
- Open an issue on GitHub
- Contact via email
- Connect on LinkedIn

---

<div align="center">
  <strong>⭐ If this project helped you understand regularized regression, please star it! ⭐</strong>
  <br><br>
  <em>Happy predicting! 📊🛒</em>
</div>

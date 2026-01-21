# 📊 Conversion Rate Challenge - Newsletter Subscription Prediction

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7+-orange.svg)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Optimization-lightblue.svg)](https://optuna.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Machine Learning competition to predict newsletter subscription conversions using advanced model optimization techniques**

## 📋 Table of Contents
- [Context](#-context)
- [Project Objective](#-project-objective)
- [Challenge Description](#-challenge-description)
- [Data](#-data)
- [Technologies](#-technologies)
- [Methodology](#-methodology)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Author](#-author)

---

## 🎯 Context

### About the Competition
This project simulates a **Kaggle-style machine learning competition** where participants submit predictions to be evaluated independently on a leaderboard 🏅. The goal is to build the best performing model while preventing overfitting and ensuring generalization.

### Company: Data Science Weekly
[**www.datascienceweekly.org**](http://www.datascienceweekly.org) is a renowned newsletter curated by independent data scientists. The platform allows anyone to register their email address to receive weekly news about:
- 📊 Data Science developments
- 🤖 Machine Learning applications
- 🧮 Statistical innovations
- 💼 Industry use cases

---

## 🚀 Project Objective

The data scientists behind the newsletter want to:
1. **Understand user behavior** on their website
2. **Predict subscription likelihood** using minimal user information
3. **Identify key features** that drive conversions
4. **Discover actionable insights** to improve conversion rates

### Business Goals
- 📈 Increase newsletter subscription rate
- 🎯 Optimize marketing strategies
- 💡 Understand user decision-making process
- 🔍 Identify high-potential visitor segments

### Technical Goals
- Build a **classification model** to predict conversions
- Maximize **F1-Score** (competition metric)
- Handle **imbalanced data** effectively
- Perform **feature engineering** and selection
- Compare **multiple ML algorithms**
- Apply **hyperparameter optimization**

---

## 🏆 Challenge Description

### Competition Format

#### Dataset Structure
The dataset is split into two files (standard ML competition format):

1. **`conversion_data_train.csv`** (Labeled)
   - Contains **features (X)** and **target (Y)**
   - Used for model training and validation
   - Perform train/test split for local evaluation

2. **`conversion_data_test.csv`** (Unlabeled)
   - Contains **features (X) only**
   - Target variable has been removed
   - Used for final predictions submission

#### Evaluation Metric
**F1-Score** (harmonic mean of Precision and Recall)

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

This metric is ideal for imbalanced datasets as it balances:
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positive cases

#### Submission Format
Predictions must be submitted as a CSV file with:
- User ID
- Predicted conversion probability (0 or 1)

---

## 📊 Data

### Data Sources

| File | Description | Labels |
|------|-------------|--------|
| `conversion_data_train.csv` | Training data with features and target | ✅ Yes |
| `conversion_data_test.csv` | Test data for final predictions | ❌ No |


## 🛠️ Technologies

### Core Libraries
```python
scikit-learn              # Machine Learning framework
xgboost                   # Gradient Boosting algorithm
optuna                    # Bayesian hyperparameter optimization
pandas                    # Data manipulation
numpy                     # Numerical computing
```

### Modeling Tools
- **Scikit-learn**: LogisticRegression, RandomForest, SVM, etc.
- **XGBoost**: Gradient boosting for structured data
- **Optuna**: Advanced hyperparameter tuning
- **GridSearchCV**: Exhaustive parameter search

### Visualization
```python
matplotlib==3.10.6       # Static plots
seaborn==0.13.2          # Statistical visualizations
plotly==6.3.0            # Interactive charts
```

### Other Tools
```python
imbalanced-learn         # Handling class imbalance (SMOTE)
joblib                   # Model serialization
scipy                    # Statistical functions
```

---

## 🔬 Methodology

### Comprehensive Model Selection Pipeline

#### **Phase 1: Initial Screening (10 Models)**
Test a diverse set of classification algorithms:

```python
models = {
    'LR': LogisticRegression(max_iter=10_000),
    'Ridge' : RidgeClassifier(),
    'RFC' : RandomForestClassifier(),
    'BagClass': BaggingClassifier(),
    'Adaboost': AdaBoostClassifier(),
    'GBC': GradientBoostingClassifier(), 
    'Extra': ExtraTreesClassifier(),
    'HGBC' : HistGradientBoostingClassifier(),
    'DTC' : DecisionTreeClassifier(),
    'SVC' : SVC(),
    'XGBC' : XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor')
}
```

**Evaluation:**
- Cross-validation (5-fold)
- F1-Score on validation set
- Training time
- Model complexity

**Output:** Select **top 4 performers**

---

#### **Phase 2: Optuna Optimization (Top 4 Models)**
Apply **Bayesian hyperparameter optimization** using Optuna:

```python
import optuna

def tune_with_optuna(model, param_func, x_train, y_train):
    def objective(trial):
        params = param_func(trial)
        clf = model.__class__(**params)
        score = cross_val_score(clf, x_train, y_train, cv=3
        scoring="f1_weighted").mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, n_jobs=1, show_progress_bar=True)
    return study.best_params, study.best_value
```

**Features:**
- Intelligent search space exploration
- Parallel trial execution
- Early stopping
- Visualization of parameter importance

**Output:** Select **top 2 models**

---

#### **Phase 3: GridSearchCV Fine-tuning (Top 2 Models)**
Exhaustive search around Optuna-discovered optimal regions:

```python
from sklearn.model_selection import GridSearchCV

best_models = {}

for name in top_2:
    params = optuna_results[name]["params"]
    
    # grille plus fine autour des meilleurs params
    grid_params = {k: [v*0.8, v, v*1.2] if isinstance(v, (int, float)) else [v] for k,v in params.items()}
    
    if name == "LR":
        model = LogisticRegression(max_iter=10_000)
    elif name == "XGBC":
        model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor')
    else:
        continue

    grid = GridSearchCV(model, grid_params, cv=3, scoring="f1_weighted", n_jobs=1)
    grid.fit(x_train, y_train)
    best_models[name] = grid.best_estimator_

    print(f"\n📊 Best params for {name} : {grid.best_params_}")
    print(f"✅ Best F1: {grid.best_score_}")
```

**Output:** Select **best overall model**

---

#### **Phase 4: Final Training & Prediction**

1. **Merge all training data** (no more train/test split)
2. **Retrain best model** on complete dataset
3. **Generate predictions** on `data_test.csv`
4. **Export results** to CSV

```python
# Final training
best_model.fit(X_train_full, y_train_full)

# Predictions on test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Export submission
submission = pd.DataFrame({
    'id': test_ids,
    'converted': y_pred
})
submission.to_csv('predictions.csv', index=False)
```

---

## 📁 Project Structure

```
conversion-rate-challenge/
│
├── 📓 analysis.ipynb                     # Main analysis notebook
├── 📝 README.md                          # This file
├── 📄 LICENSE                            # MIT License
├── 📦 requirements.txt                   # Dependencies
├──  conversion_data_train.csv             # Training data (labeled)
├──  conversion_data_test.csv              # Test data (unlabeled)
└──  predictions.csv                       # Final predictions (generated)

```

---

## 💻 Installation

### Prerequisites
- Python 3.11 or higher
- Jupyter Notebook

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Ter0rra/02_Jedha_cdsd_bloc_3_Conversion_rate
cd 02_Jedha_cdsd_bloc_3_Conversion_rate
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

## 🔮 Future Improvements

- [ ] Deep learning models (Neural Networks)
- [ ] More advanced feature engineering
- [ ] AutoML frameworks (H2O, Auto-sklearn)
- [ ] MLOps pipeline (MLflow, DVC)

---

## 📚 References

### Documentation
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Optuna Tutorial](https://optuna.readthedocs.io/en/stable/tutorial/index.html)

### Inspiration
- [Kaggle Competitions](https://www.kaggle.com/competitions)
- [Data Science Weekly](https://www.datascienceweekly.org/)

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

- **Data Science Weekly** for the business case
- **Jedha Teaching team** for online training and the challenge
- **Optuna developers** for the amazing optimization framework
- **Scikit-learn community** for comprehensive ML tools

---

## 📞 Contact

Questions about the methodology or results?
- Open an issue on GitHub
- Email me directly
- Connect on LinkedIn

---

<div align="center">
  <strong>🏆 Competing for the top of the leaderboard! 🏆</strong>
  <br><br>
  <em>May the best F1-Score win! 📊</em>
</div>

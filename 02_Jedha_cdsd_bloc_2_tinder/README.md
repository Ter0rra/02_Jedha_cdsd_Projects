# 💘 Tinder Speed Dating Analysis

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Behavioral analysis of Tinder users to improve matching rates**

## 📋 Table of Contents
- [Context](#-context)
- [Project Objective](#-project-objective)
- [Data](#-data)
- [Technologies](#-technologies)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Methodology](#-methodology)
- [Key Results](#-key-results)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🎯 Context

**Tinder** is the world's leading dating application, launched in 2012 by Sean Rad during a hackathon in West Hollywood. With over **65 billion matches** recorded in 2021, the app allows users to:
- Swipe right ❤️ to like a profile
- Swipe left 👎 to pass
- Match when two people like each other

However, the marketing team is facing a **decline in match rates** and seeks to understand the factors that drive user interest and attraction.

---

## 🚀 Project Objective

Analyze data from a real **speed dating experiment** to identify:
- ✅ Criteria that influence mutual attraction
- 📊 User behavioral patterns
- 💡 Actionable insights to optimize Tinder profiles
- 🎯 Recommendations to improve matching rates

**Key Question**: What are the determining factors in the decision to accept a second date?

---

## 📊 Data

### Source
Controlled **speed dating experiment** organized by Tinder, where participants provided:
- Detailed profile information
- Preferences and selection criteria
- Their decision (yes/no) for a second date after each encounter

### Files
| File | Description |
|------|-------------|
| `Speed+Dating+Data.csv` | Main dataset containing speed dating results |
| `Speed+Dating+Data+Key.doc` | Variable documentation and metadata |

### Data Structure
Each row represents an **interaction between two people** during a speed date, including:
- Demographics (age, profession, interests)
- Mutual evaluations (attractiveness, sincerity, intelligence, ambition...)
- **Target variable**: Acceptance or rejection of a second date

---

## 🛠️ Technologies

### Language
- **Python 3.11+**

### Main Libraries
```python
pandas==2.3.2          # Data manipulation
numpy==2.3.3           # Numerical computing
matplotlib==3.10.6     # Visualizations
seaborn==0.13.2        # Statistical visualizations
scikit-learn==1.7.2    # Machine Learning
plotly==6.3.0          # Interactive visualizations
missingno==0.5.2       # Missing data analysis
```

### Environment
- **Jupyter Notebook** for interactive analysis
- **VS Code** as main editor
- Conda environment for dependency management

---

## 📁 Project Structure

```
tinder-speed-dating-analysis/
│
├── 📓 notebook.ipynb                    # Main analysis
├── 📊 Speed+Dating+Data.csv             # Dataset
├── 📄 Speed+Dating+Data+Key.doc         # Variable documentation
├── 📝 README.md                         # This file
├── 📦 requirements.txt                  # Python dependencies
└── 🖼️ figures/                          # (optional) Exported visualizations
```

---

## 💻 Installation

### Prerequisites
- Python 3.11 or higher
- Conda (recommended) or pip

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/your-username/tinder-speed-dating-analysis.git
cd tinder-speed-dating-analysis
```

2. **Create a virtual environment**
```bash
# With Conda
conda create -n tinder-analysis python=3.11
conda activate tinder-analysis

# Or with venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

5. **Open `notebook.ipynb` and run the cells**

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Descriptive analysis of variables
- Missing value detection and treatment
- Distribution visualization
- Correlation identification

### 2. Data Preprocessing
- Data cleaning
- Outlier handling
- Categorical variable encoding
- Feature engineering

### 3. Statistical Analysis
- Hypothesis testing
- Correlation analysis
- User segmentation

### 4. Insights & Recommendations
- Key success factors extraction
- Profile optimization suggestions

---

## 📈 Key Results

> ⚠️ *Section to be completed*

**Expected results examples:**
- 🎯 Top 5 criteria influencing matching
- 📊 Match rate by demographic category
- 💡 High-potential matching profiles

---

## 🔮 Future Improvements

- [ ] Integration of textual data (bios)
- [ ] Sentiment analysis on descriptions
- [ ] Personalized recommendation system
- [ ] Prediction API deployment
- [ ] Interactive dashboard (Streamlit/Dash)

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

- Tinder for providing the dataset
- The Data Science community for open-source tools

---

<div align="center">
  <strong>⭐ If this project was helpful, feel free to star it! ⭐</strong>
</div>

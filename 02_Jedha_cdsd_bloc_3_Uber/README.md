# 🚗 Uber Pickup Hotspot Analysis

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7+-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.3+-green.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Unsupervised Machine Learning to identify high-demand pickup zones and reduce user wait times**

## 📋 Table of Contents
- [Context](#-context)
- [Project Objective](#-project-objective)
- [Problem Statement](#-problem-statement)
- [Data](#-data)
- [Technologies](#-technologies)
- [Methodology](#-methodology)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Author](#-author)

---

## 🎯 Context

### About Uber
**Uber** is one of the most famous startups in the world. What started as a ride-sharing app for people who couldn't afford a taxi has expanded into:
- 🚗 **Uber Rides**: Traditional ride-sharing service
- 🍔 **Uber Eats**: Food delivery
- 📦 **Package delivery**: Parcel transportation
- 🚚 **Freight**: Cargo transportation
- 🚲 **Urban mobility**: Jump Bike and Lime (company-funded services)

**Global Presence:**
- ~70 countries
- ~900 cities
- $14+ billion in annual revenue

**Mission**: Revolutionize transportation across the world 🌍

---

## 🚀 Project Objective

### The Problem
Uber's data team has identified a critical issue: **drivers are not always available when and where users need them.**

### Real-World Example
A user in San Francisco's **Financial District** requests a ride, but Uber drivers are concentrated in the **Castro District**. Despite the proximity (~3 miles), users must wait **10-15 minutes** for pickup.

**User Behavior Study Results:**
- ✅ Acceptable wait time: **5-7 minutes**
- ❌ Above 7 minutes: **Users cancel rides**

### Solution
Build a recommendation system that identifies **high-demand zones** (hotspots) in major cities at different times of day, enabling:
- 📍 Better driver positioning
- ⏱️ Reduced wait times
- 💰 Increased ride completion rate
- 😊 Improved user satisfaction

---

## 🎯 Problem Statement

**Goal**: Use **unsupervised machine learning** (clustering algorithms) to:
1. Identify geographical clusters of high pickup demand
2. Recommend optimal zones for drivers to wait
3. Predict busy areas at different day of week
4. Reduce average user wait time

**Key Questions:**
- Where are the busiest pickup locations?
- How do demand patterns change throughout the day?
- Which clustering algorithm works best for this problem?

---

## 📊 Data

### Data Source
**Uber Trip Data** for New York City
- **URL**: [Download Dataset](https://full-stack-bigdata-datasets.s3.eu-west-3.amazonaws.com/Machine+Learning+non+Supervis%C3%A9/Projects/uber-trip-data.zip)
- **Format**: 8 CSV files (monthly data)
- **Period**: Multiple months of ride data

### Dataset Structure
Most CSV file contains:

| Column | Description | Type |
|--------|-------------|------|
| `Date/Time` | Pickup datetime | DateTime |
| `Lat` | Pickup latitude | Float |
| `Lon` | Pickup longitude | Float |
| `Base` | TLC base company code | String |

### Data Characteristics
- **8 separate CSV files** 
- **Geographic coverage**: New York City area
- **Coordinates**: GPS latitude/longitude

### Data Challenges
- 🔄 Multiple files to merge
- 📅 DateTime parsing required
- 🗺️ Geographic coordinate handling
- ❓ Missing values (NaN)
- 📊 Large dataset size (memory management)

---

## 🛠️ Technologies

### Core Libraries
```python
pandas                  # Data manipulation
numpy                   # Numerical computing
scikit-learn            # Machine Learning & Clustering
scipy                   # Statistical functions
```

### Clustering Algorithms
- **Mini-Batch K-Means**: Fast clustering for large datasets
- **K-Means**: Traditional centroid-based clustering
- **DBSCAN**: Density-based spatial clustering

### Visualization
```python
matplotlib              # Static plots
seaborn                 # Statistical visualizations
plotly                  # Interactive maps
folium                  # Geographic visualizations
```

### Data Processing
```python
datetime                # Temporal feature extraction
collections             # Dictionary management
warnings                # Suppress unnecessary warnings
```

---

## 🔬 Methodology

### Step 1: Import Libraries

```python
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from datetime import datetime
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

import warnings
warnings.filterwarnings("ignore")
```

### Step 2: Read Data (Dictionary Approach)

**Challenge**: 8 CSV files to manage efficiently

**Solution**: Store DataFrames in a dictionary

```python
# définit la fonction d'extraction de nom 
def get_name(filename):
    # Enlève l'extension .csv et récupère la dernière partie après le dernier '-'
    name_without_ext = filename.replace('.csv', '')
    name = name_without_ext.split('-')[-1]
    return name

uber_dic = {}

# Liste les fichiers du dossier
for file in os.listdir('./uber-trip-data'):
    if file.endswith('.csv'):
        key = get_name(file)
        uber_dic[key] = pd.read_csv(f'./uber-trip-data/{file}')
        print(f"Ajouté: {key}")
    else:
        print(f'{file} is not a csv!')

print(f"\nNombre de fichiers CSV chargés: {len(uber_dic)}")

# afficher les infos des df 
for key in uber_dic:
    print('\n' + '**'*50)
    print('='*50)
    print(f"df : {key}")
    print('='*50)
    print(uber_dic[key].head())  # Affiche les 5 premières lignes
    print('='*50)
    uber_dic[key].info()

```

**Benefits:**
- ✅ Easy access: `data_dict['apr']`
- ✅ Simplified iteration
- ✅ Memory efficient
- ✅ Clean code structure


### Step 3: Data Preprocessing

#### 3.1 DateTime Conversion

#### 3.2 Location Validation

#### 3.3 Handle Missing Values

#### 3.4 Merge All DataFrames

### Step 4: Clustering Analysis

#### Prepare Features for Clustering

#### 4.1 Mini-Batch K-Means

#### 4.2 K-Means (Standard)

#### 4.3 DBSCAN (Density-Based)

### Step 5: Clustering Comparison & Visualization

### Step 6: Conclusions

---

## 📁 Project Structure

```
uber-pickup-hotspot-analysis/
│
├── 📓 analysis.ipynb              # Main analysis notebook
├── 📝 README.md                   # This file
├── 📄 LICENSE                     # MIT License
│
└── 📂 data/                       # (Downloaded separately)
    ├── uber-raw-data-apr14.csv
    ├── uber-raw-data-may14.csv
    ├── uber-raw-data-jun14.csv
    ├── uber-raw-data-jul14.csv
    ├── uber-raw-data-aug14.csv
    ├── uber-raw-data-sep14.csv
    ├── uber-raw-data-janjune-15.csv
    └── uber-raw-data-jul-sep-15.csv
```

---

## 💻 Installation

### Prerequisites
- Python 3.11 or higher
- Jupyter Notebook
- Minimum 8GB RAM (for full dataset)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/uber-pickup-hotspot-analysis.git
cd uber-pickup-hotspot-analysis
```

2. **Download dataset**
```bash
# Download and extract data
wget https://full-stack-bigdata-datasets.s3.eu-west-3.amazonaws.com/Machine+Learning+non+Supervis%C3%A9/Projects/uber-trip-data.zip
unzip uber-trip-data.zip -d data/
```

3. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

4. **Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib plotly 
```

5. **Launch Jupyter Notebook**
```bash
jupyter notebook uber_projet.ipynb
```

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

- **Jedha** for online-training
- **Uber** for open-sourcing trip data
- **NYC TLC** for transportation data
- **Scikit-learn** community for clustering algorithms
- **Folium** for geographic visualizations

---

## 📞 Support

Questions about the clustering methodology or results?
- Open an issue on GitHub
- Connect on LinkedIn

---

<div align="center">
  <strong>🚗 Optimizing urban transportation, one cluster at a time! 🗺️</strong>
  <br><br>
  <em>Reducing wait times through data science! ⏱️📊</em>
</div> 



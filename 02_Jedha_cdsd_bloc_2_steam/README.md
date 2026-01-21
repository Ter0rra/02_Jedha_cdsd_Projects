# 🎮 Steam Marketplace Analysis

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Spark](https://img.shields.io/badge/Apache%20Spark-3.0+-orange.svg)](https://spark.apache.org/)
[![Databricks](https://img.shields.io/badge/Databricks-free-red.svg)](https://databricks.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Global analysis of the Steam marketplace to understand video game ecosystem trends**

## 📋 Table of Contents
- [Context](#-context)
- [Project Objective](#-project-objective)
- [Data](#-data)
- [Technologies](#-technologies)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Methodology](#-methodology)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🎯 Context

**Steam** is a digital distribution service and online store developed by Valve Corporation. Launched in September 2003, Steam has become the world's largest PC gaming platform, offering:
- 🎮 Automated game updates
- 🔐 Digital Rights Management (DRM)
- 🌐 Game server matchmaking and Valve Anti-Cheat measures
- 👥 Social networking and game streaming
- ☁️ Cloud storage for game progress
- 🛒 Virtual marketplace for collectible items

With **thousands of games** available and **millions of active users**, Steam represents a critical ecosystem for understanding the video game industry.

---

## 🚀 Project Objective

**Client**: Ubisoft, a leading French video game publisher

**Mission**: Conduct a comprehensive analysis of games available on the Steam marketplace to support the launch of a revolutionary new game.

### Key Questions
- 📊 What are the current trends in the video game industry?
- 🎯 Which game genres are most popular?
- 💰 What are the pricing strategies?
- ⭐ What factors influence game ratings and success?
- 🌍 How does the gaming market vary globally?
- 🕹️ What features do successful games have in common?

### Deliverables
- Comprehensive data analysis of Steam marketplace
- Actionable insights for game development strategy
- Market positioning recommendations
- Trend identification and forecasting

---

## 📊 Data

### Source
**Steam Marketplace** dataset containing comprehensive information about games available on the platform.

### Data Features
The dataset includes:
- **Game metadata**: Title, developer, publisher, release date
- **Pricing**: initial price, discounts, price
- **Ratings**: User reviews
- **Categories**: Genres, tags, features
- **Technical specs**: Supported platforms
- **Engagement metrics**: Number of reviews, Number of owners

---

## 🛠️ Technologies

### Platform
- **Databricks Community Edition** (Cloud-based analytics platform)
- **Databricks Free Edition** (Cloud-based analytics platform)
- **Apache Spark 3.0+** (Distributed computing framework)

### Languages & Libraries
```python
pyspark              # Distributed data processing
pandas               # Data manipulation
matplotlib           # Static visualizations
seaborn              # Statistical visualizations
plotly               # Interactive visualizations
numpy                # Numerical computing
```

### Big Data Stack
- **Spark SQL**: Structured data queries
- **Spark DataFrame API**: Data transformations
- **Databricks notebooks**: Interactive analysis environment

---

## 📁 Project Structure

```
steam-marketplace-analysis/
│
├── 🖼️ img/                                  # Exported charts and graphs
│   ├── 00_best_publisher.png
│   ├── 01_10top_of_top_100_games.png
│   ├── 02_10top_of_top_100_games_ratio.png
│   ├── 03_year_game_release_bar.png
│   └── etc ...
│
├── 📝 README.md                             # This file
├── 📓 Steam_databricks.ipynb                # Main analysis notebook (Databricks)
├── 📓 Steam_databricks_with_extra.ipynb     # analysis notebook with extra/open analysis (Databricks)
├── 📓 Steam_databricks.lnk                  # Main analysis notebook published (Databricks community)
└── 📄 structure.txt                         # data structure

```

---

## 💻 Installation

### Option 1: Databricks Free Edition (Recommended)

1. **Create a Databricks account**
   - Go to [Databricks Free Edition](https://www.databricks.com/fr/learn/free-edition)
   - Sign up for free

2. **Import the notebook**
   - Click "Import" in Databricks workspace
   - Upload `notebook.ipynb`

3. **Run the notebook**
   - Attach notebook to cluster
   - Execute cells sequentially

### Option 2: Databricks community Edition (Alternative)   

- Click on the link to open the notebook in Databricks Community Edition
   - 1 st way =>  Steam_databricks.lnk
   - 2 nd way => [Steam_datarick commnunity edition](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2654712914772717/3542083400711813/6640604564150917/latest.html)   

### Option 3: Local Setup (Alternative)

```bash
# Clone repository
git clone https://github.com/your-username/steam-marketplace-analysis.git
cd steam-marketplace-analysis

# Install dependencies
pip install pyspark pandas matplotlib seaborn plotly numpy

# Launch Jupyter with PySpark
jupyter notebook
```

---

## 🔬 Methodology

### 0. Data Ingestion & Exploration
- Load JSON data into Spark DataFrame
- Schema analysis and data profiling

### 1. Analysis at the "macro" level
- Which publisher has released the most games on Steam?
- What are the best rated games?
- Are there years with more releases?
- How are the prizes distributed?
- What are the most represented languages?
- Are there many games prohibited for children under 16/18?
- Conclusion

### 2. Genre Analysis
- What are the most represented genres?
- Are there any genres that have a better positive/negative review ratio?
- Do some publishers have favorite genres?
- What are the most lucrative genres?
- Conclusion

### 3. Platform analysis
- Are most games available on Windows/Mac/Linux instead?
- Do certain genres tend to be preferentially available on certain platforms?
- Conclusion

### 4. Conclusion of analysis

### Extra. open analysis
- *only available on Steam_Databricks_with_extra.ipynb*
- free analysis on tags and category

### *all part => Visualization & Insights*
- Create compelling visualizations
   - *All visualizations available in `/img` folder*

---

## 🔮 Future Improvements

- [ ] Real-time data pipeline from Steam API
- [ ] Sentiment analysis on user reviews
- [ ] Predictive modeling for game success
- [ ] Competitor benchmarking analysis
- [ ] Machine learning for price optimization
- [ ] Interactive dashboard (Dash/Streamlit)
- [ ] Time series forecasting for trends

---

## 🛠️ Technical Challenges Addressed

- **Big Data Processing**: Efficient handling of large JSON dataset with Spark
- **Nested Data Structures**: Parsing complex JSON hierarchies
- **Data Quality**: Handling missing values and inconsistencies
- **Scalability**: Leveraging distributed computing for performance
- **Visualization**: Creating informative charts from large datasets

---

## 📚 Key Technologies Explained

### Why Databricks?
- Cloud-based collaborative environment
- Built-in Spark optimization
- Free & community edition available
- Notebook version control

### Why Spark?
- Handles large-scale data processing
- In-memory computing for speed
- Scalable architecture
- Industry-standard for big data

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

- **Jedha** for the online trainings
- **Valve Corporation** for the Steam platform
- **Databricks Community** for free cloud platform
- **Apache Spark** for distributed computing framework
- **Ubisoft** for the project opportunity

---

## 📞 Contact

For questions or collaboration opportunities, feel free to reach out!

---

<div align="center">
  <strong>⭐ If this analysis was helpful, don't forget to star the repository! ⭐</strong>
</div>

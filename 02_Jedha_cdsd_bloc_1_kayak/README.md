# ✈️ Kayak Destination Recommender

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![AWS S3](https://img.shields.io/badge/AWS-S3-orange.svg)](https://aws.amazon.com/s3/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-NeonDB-blue.svg)](https://neon.tech/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **AI-powered travel destination recommendation system based on real-time weather and hotel data**

## 📋 Table of Contents
- [Context](#-context)
- [Project Objective](#-project-objective)
- [Data Sources](#-data-sources)
- [Technologies](#-technologies)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data Pipeline](#-data-pipeline)
- [API Documentation](#-api-documentation)
- [Features](#-features)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🎯 Context

**Kayak** is a leading travel search engine that helps users plan their next trip at the best price. Founded in 2004 by Steve Hafner and Paul M. English, Kayak is now part of **Booking Holdings**, which also owns:
- 🏨 Booking.com
- 💰 Priceline
- 🌏 Agoda
- 🚗 Rentalcars.com
- 🍽️ OpenTable

With over **$300 million in annual revenue**, Kayak operates in nearly every country and language, helping users book travel worldwide.

### User Research Insights
After conducting user studies, the marketing team discovered:
- **70% of users** planning a trip want more information about their destination
- Users tend to **distrust information** when they don't know the brand producing the content
- There's a strong demand for **data-driven recommendations**

---

## 🚀 Project Objective

Build a **destination recommendation application** that suggests the best vacation spots and hotels based on:
- 🌤️ **weather data by destination**
- 🏨 **Hotel availability and information**
- 📍 **Geographic location data**

### Key Features
- ✅ Recommend optimal destinations at any given time
- ✅ Provide hotel suggestions with accurate location data
- ✅ Leverage trustworthy, real-world data sources
- ✅ Enable data-driven decision making for travelers

---

## 📊 Data Sources

### 1. Weather Data
- **Source**: Météo France API
- **Data**: Temperature, precipitation, forecasts
- **Coverage**: Target cities in France

### 2. Hotel Data
- **Source**: Booking.com (Web Scraping)
- **Scripts**: 
  - `booking_url_hotel.py` - Extract hotel URLs
  - `booking_info_hotel.py` - Scrape detailed hotel information
- **Data**: ratings, desciptions

### 3. Geolocation Data
- **Source**: HERE API
- **Purpose**: Convert addresses to precise GPS coordinates
- **Use case**: Accurate hotel mapping and distance calculations

### Target Cities
Predefined list of popular travel destinations analyzed for recommendations.

---

## 🛠️ Technologies

### Core Stack
```python
Python 3.11+           # Main programming language
SQLAlchemy             # ORM and database toolkit
Pandas                 # Data manipulation
NumPy                  # Numerical computing
Requests               # HTTP library for API calls
```

### Cloud & Database
- **AWS S3**: Cloud storage for scraped data and raw files
- **Neon DB**: Serverless PostgreSQL database for structured data
- **SQLAlchemy**: ETL pipeline and database operations

### APIs
- **Météo France API**: Weather forecasting data
- **HERE API**: Geocoding and location services
- **Booking.com**: Hotel data (via web scraping)

### Development Tools
- **Jupyter Notebook**: Exploratory analysis
- **VS Code**: Primary IDE
- **Git**: Version control

---

## 🏗️ Project Architecture

```
┌─────────────────┐
│  Data Sources   │
│  (APIs + Web)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Collection│
│  (Scrapers +    │
│   API Calls)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   AWS S3        │
│ (Raw Storage)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ETL Pipeline   │
│  (SQLAlchemy)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Neon DB       │
│  (PostgreSQL)   │
└────────┬────────┘
```

---

## 📁 Project Structure

```
kayak-destination-recommender/
│
├── 📓 Kayak_Project.ipynb               # analysis nb
├── 📝 README.md                         # This file
├── 📦 environment.yml                   # Dependencies
├── 📄 LICENSE                           # MIT License
├── 🔑 .env.example                      # Environment 
│                    
├── 📂 src/                             # Source code
│   ├── booking_url_hotel.py            # Spyder hotel URLs
│   └── booking_info_hotel.py           # Spyder for details
│
├── 📂 data/                            # Data files
│   ├── all_cities_url_hotels.json
│   ├── all_hotels_details_insee.json
│   ├── hotels_details.json
│   ├── weather.csv
│   ├── cities_weather.csv
│   └── hotels_info.csv
│
└── 📂 img/                             # plot & screenshot of s3+db
```

---

## 💻 Installation

### Prerequisites
- Python 3.11 or higher
- AWS Account (for S3)
- Neon DB Account (free tier available)
- API Keys:
  - Météo France API
  - HERE API

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/Ter0rra/02_Jedha_cdsd_bloc_1_kayak/
cd 02_Jedha_cdsd_bloc_1_kayak
```

2. **Create virtual environment & Install dependencies**
```bash
conda env create -f environment.yml # env basefile  
```

3. **Configure environment variables**
```bash
cp config/.env.example .env
# Edit .env with your API keys and credentials
```

**Required environment variables:**
```env
# AWS S3
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET_NAME=your_bucket_name

# Neon DB
DATABASE_URL=postgresql://user:password@host/database

# APIs
HERE_API_KEY=your_api_key
HERE_api_URL=here_api_url
```

---

## 🚀 Usage

### 1. Collect Weather Data

### 2. Scrape Hotel Data
```bash
# Step 1: Get hotel URLs
python src/booking_url_hotel.py 

# Step 2: Scrape hotel details
python src/booking_info_hotel.py 
```

### 3. Geocode Hotel Addresses

### 4. Upload to S3

### 5. Run ETL Pipeline

### 6. Analyze Data

---

## 🔄 Data Pipeline

### Stage 1: Data Collection
1. **Weather API Call**: Fetch current and forecast data
2. **Web Scraping**: Extract hotel information from Booking.com
3. **Geocoding**: Convert addresses to GPS coordinates

### Stage 2: Storage
- Raw data stored in **AWS S3** (CSV format)

### Stage 3: ETL (Extract, Transform, Load)
Using **SQLAlchemy**:
- **Extract**: Load data from S3
- **Load**: Insert into Neon DB

### Stage 4: Database Schema

---

## 📚 API Documentation

### Météo France API
- **Service**: French meteo API
- **Rate Limit**: 1000 calls/day

### HERE API
- **Service**: Geocoding & Search API
- **Rate Limit**: 30k transactions/month (free tier)

### Booking.com Scraping
- **Compliance**: Respect robots.txt
- **Rate Limiting**: Implemented (1 by 1 requests)
- **User Agent**: Randomized to avoid blocking

---

## 🔮 Future Improvements

- [ ] Real-time recommendation API (FastAPI/Flask)
- [ ] Machine learning for personalized suggestions
- [ ] User preference profiling
- [ ] Mobile application integration
- [ ] Additional data sources (flights, activities)
- [ ] Sentiment analysis on hotel reviews
- [ ] Interactive dashboard (Streamlit)
- [ ] Automated daily data refresh
- [ ] Cost optimization alerts
- [ ] Multi-language support

---

## ⚠️ Important Notes

### Web Scraping Ethics
- Implement rate limiting
- Don't overload servers
- Consider using official APIs when available

### Data Privacy
- No personal user data stored
- Publicly available information only
- GDPR compliant

### API Rate Limits
Monitor your usage to avoid exceeding free tier limits.

---

## 🐛 Troubleshooting

### Common Issues

**S3 Upload Fails**
```bash
# Check AWS credentials
aws configure list
```

**Scraping Blocked**
- Increase delay between requests
- Rotate user agents
- Use proxy if necessary

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
- **Kayak** for the project inspiration
- **Météo France** for weather data
- **HERE Technologies** for geocoding services
- **Neon DB** for serverless PostgreSQL
- **AWS** for cloud infrastructure

---

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact via email

---

<div align="center">
  <strong>⭐ If this project helped you, please star the repository! ⭐</strong>
  <br><br>
  <em>Happy traveling! ✈️🌍</em>
</div>


# 🚗 GetAround: Rental Time Optimization and Pricing

This project, carried out as part of a partnership with GetAround (the "Airbnb for cars"), is divided into two parts:

1.  **Data Analysis and Visualization**: Creation of a dashboard to help product management optimize rental logistics.
2.  **Machine Learning and API**: Development of a daily price prediction endpoint for car owners.

-----

## 1\. 📊 Data Analysis & Visualization (Streamlit Web Dashboard)

The main objective is to provide factual insights to determine the best strategy for the **minimum delay between two rentals** to reduce customer dissatisfaction caused by late returns, while minimizing potential revenue loss.

The interactive dashboard is built with **Streamlit** and addresses key product management questions regarding the **threshold** (minimum delay duration) and the **scope** (all cars vs. Connect cars only).

### Dashboard Features

  * **Time Delta Analysis**: Visualization of the distribution of time elapsed between two consecutive rentals, with filters to simulate the impact of a minimum delay.
  * **Previous Delay Impact**: Study of the correlation between the delay of the previous rental and the status of the following rental (`successful` or `failed`), segmented by check-in type (`mobile` or `connect`).
  * **Key Metrics**: Display of the percentages of potentially affected rentals to help find the right balance between improving user experience and optimizing revenue.

### 🔗 Production Links

| Service | URL |
| :--- | :--- |
| **Streamlit Dashboard** | `[https://terorra-gar-cdsd-analysis.hf.space/]` |

### Technologies Used

  * **Framework**: Streamlit
  * **Data Analysis/Manipulation**: Pandas, NumPy
  * **Visualization**: Plotly Express

-----

## 2\. 🤖 Machine Learning & API

This section focuses on optimizing daily pricing for owners, using a Machine Learning model trained on vehicle pricing data.

A notebook is provided to show the preparation of the model used in the API.

The model is exposed via an API to allow its integration into production systems.

### /predict Endpoint

The API is hosted online and provides a `/predict` endpoint that allows submitting a vehicle's characteristics (model, mileage, power, options, etc.) and receiving an estimated daily rental price.

### 🔗 Production Links

| Service | URL |
| :--- | :--- |
| **API Documentation** | `[https://terorra-gar-cdsd-pred.hf.space/docs]` |

#### Input Example (JSON)

```json
{
  "input": [
    {
      "model_key": "Peugeot", 
      "mileage": 100000.0, 
      "engine_power": 135.0, 
      "fuel": "diesel", 
      "paint_color": "black", 
      "car_type": "sedan", 
      "private_parking_available": true, 
      "has_gps": true, 
      "has_air_conditioning": false, 
      "automatic_car": false, 
      "has_getaround_connect": true, 
      "has_speed_regulator": false, 
      "winter_tires": true
    }
  ]
}
```

#### Output Example (JSON)

```json
{
  "prediction": [60.50]
}
```

### Technologies Used

  * **API Framework**: FastAPI (recommended for performance and automatic docs)
  * **ML Model**: Scikit-Learn Pipeline (saved via `joblib`)
  * **Deployment**: Hugging Face Spaces (or other cloud service)

-----

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
- **GetAround** for the business case and data
- **HuggingFace Space** for space to deploied all projects 
- **Streamlit** & **FastAPI** for the community  

---

## 📞 Support

Questions or suggestions?
- Open an issue on GitHub
- Contact via email
- Connect on LinkedIn

---
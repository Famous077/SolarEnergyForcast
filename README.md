# Solar Power Forecasting

Forecast solar PV power output from historical generation and meteorological data (irradiance, temperature, cloud cover). Use proper time-series splits, train LightGBM/Prophet and optional LSTM, beat persistence and seasonal-naive baselines, and deliver an alert-capable dashboard for grid operators

**Project page:** https://famous077.github.io/OJT_Project/

**Live dashboard:** https://solarpowerpredictions.streamlit.app/

(It's on Streamlit's free tier, so if nobody has opened it for a few days it takes about a minute to wake up.)

## Overview

This project focuses on **short-term solar power forecasting** using time-series modeling and machine learning techniques. The goal is to build a **reliable, leakage-free forecasting pipeline** that can predict photovoltaic (PV) power generation at **30-minute intervals** using historical and engineered features.

Unlike naive approaches, this system emphasizes:
- Proper **time-series validation**
- Strong **baseline comparisons**
- Robust **data preprocessing**
- Model interpretability and deployment readiness

## Objectives

- Forecast solar PV output for short-term horizons (30-min intervals)
- Compare traditional baselines with machine learning and deep learning models
- Avoid **data leakage** using time-aware splits
- Evaluate performance across multiple metrics
- Build a system usable for **grid-level decision support**


## Problem Formulation

Given:
- Historical solar power generation data  
- Optional meteorological variables  

Predict:
- Future solar power output for the next time horizon

This is a **supervised time-series regression problem** with temporal dependencies and seasonal patterns.

## System Architecture

```
                ┌────────────────────────────┐
                │   Raw Data Sources         │
                │ (PV + Weather Data)       │
                └────────────┬──────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │   Data Preprocessing       │
                │ - Missing value handling   │
                │ - Time alignment           │
                │ - Daylight filtering       │
                │ - Normalization            │
                └────────────┬──────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │   Feature Engineering      │
                │ - Lag features (t-1, t-24) │
                │ - Rolling statistics       │
                │ - Time features (hour/day) │
                │ - Optional weather inputs  │
                └────────────┬──────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │   Model Training           │
                │                            │
                │ Baselines:                 │
                │ - Persistence Model        │
                │ - Seasonal Naive           │
                │                            │
                │ ML Models:                 │
                │ - LightGBM                 │
                │                            │
                │ DL Models:                 │
                │ - LSTM                     │
                │ - GRU (optional)           │
                └────────────┬──────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │   Evaluation               │
                │ - MAE                      │
                │ - RMSE                     │
                │ - MAPE                     │
                │ - R² Score                 │
                │ - Rolling validation       │
                └────────────┬──────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │   Prediction Pipeline      │
                │ - Forecast generation      │
                │ - Uncertainty estimation   │
                │ - Alert system             │
                └────────────┬──────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │   Dashboard (Streamlit)    │
                │ - Forecast visualization   │
                │ - Model comparison         │
                │ - SHAP explanations        │
                └────────────────────────────┘
```

---

## Tech Stack

- **Python**
- **Pandas, NumPy** (data processing)
- **Scikit-learn** (ML models & preprocessing)
- **LightGBM** (gradient boosting)
- **TensorFlow / PyTorch** (LSTM/GRU)
- **Streamlit** (dashboard)
- **SHAP** (model explainability)

##  Models Used

### Baselines
- Persistence Model
- Seasonal Naive Model  

### Machine Learning
- LightGBM (handles tabular + engineered features well)

### Deep Learning
- LSTM (captures temporal dependencies)
- GRU (optional faster alternative)

---

## Evaluation Strategy

To ensure realistic performance:

- **Rolling-origin validation** (time-aware split)
- No random shuffling
- Multiple evaluation metrics:
  - MAE
  - RMSE
  - MAPE
  - R² Score

##  Key Features

- Leakage-free time-series pipeline  
- Multi-model comparison framework  
- Feature engineering for temporal data  
- Real-time forecasting dashboard  
- Explainable predictions using SHAP  


## Project Structure

```
solar-forecasting/
│
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models/
│   ├── evaluation.py
│   └── pipeline.py
│
├── app/
│   └── streamlit_app.py
│
├── models/
├── requirements.txt
└── README.md
```

## How to run

```
pip install -r requirements.txt
python src/pipeline.py          # trains the saved forecaster, ~1 min -> models/forecaster.pkl
streamlit run app/app.py
```

The trained models live in `models/forecaster.pkl`. It is a binary file, so read `models/MODEL_CARD.md` to see what is inside and how to rebuild and check it. `src/pipeline.py` reads the tuned settings in `reports/lgbm_params.json`, so notebook 05 has to have been run once. Notebooks 01 to 08 run in order. 06 (Prophet) is run on Google Colab because Windows Smart App Control blocks Prophet's Stan binary on our machine.

The dashboard is written for a grid operator, not a data scientist. Five pages (Overview, Forecast, Alerts, Weather and drivers, System health) use plain words and kW. The Analyst page has the model comparison, SHAP and the PDF export.

Our data ends in 2026, so the app runs on a replay clock. You pick "now" in the sidebar (or step it with the -1h/+1h buttons), the model only sees data up to that hour, and the real output afterwards is drawn as a dashed line. A link such as `?site=site_a&t=2026-03-22T09:00Z` opens a given plant and hour.

## Future Improvements

- Add **weather API integration**
- Implement **Transformer-based models**
- Improve **uncertainty estimation**
- Multi-location generalization
- Real-time deployment pipeline


## Conclusion

This project demonstrates that:
- Time-series-aware modeling is critical for solar forecasting  
- Baselines are strong and must not be ignored  
- Deep learning models like LSTM improve performance when used correctly  
- Data quality and preprocessing significantly impact results  

## Contributions

Open to contributions, ideas, and improvements.

## License

MIT License
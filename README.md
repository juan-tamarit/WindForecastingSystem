# Wind Forecasting System

Backend-oriented wind prediction system developed as a Bachelor's Thesis.
The project builds an end-to-end pipeline from raw climate data ingestion to
time-series forecasting using Python.

##  Problem Statement
Accurate wind prediction is critical for energy planning and weather-related
decision making. Public datasets such as ERA5 provide large-scale climate data
that require preprocessing, storage and modeling before being usable.

##  Solution Overview
The system:
- Retrieves wind-related variables from the ERA5 dataset
- Stores and manages time-series data locally using MongoDB
- Preprocesses and transforms the data for forecasting
- Trains and evaluates time-series prediction models in Python

##  Architecture
ERA5 Dataset → Data Ingestion → MongoDB → Data Processing → Forecasting Model

##  Tech Stack
- Python
- MongoDB
- pandas, numpy
- time-series forecasting libraries
- ERA5 climate dataset

##  Features
- Local persistence of large climate datasets
- Modular data ingestion and preprocessing pipeline
- Time-series forecasting focused on wind prediction
- Reproducible experiments

##  Results
- Successfully trained forecasting models on historical wind data
- Evaluated prediction accuracy using standard time-series metrics

##  How to Run
```bash
git clone https://github.com/juan-tamarit/wind-forecasting-system.git
cd wind-forecasting-system
pip install -r requirements.txt
python main.py


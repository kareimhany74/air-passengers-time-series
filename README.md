# Air Passengers Time Series Forecasting Using Deep Learning

An end-to-end Time Series Analysis and Forecasting project designed to predict future trends using the classic Air Passengers dataset. This project implements specialized preprocessing for sequential datasets and leverages powerful deep learning architectures to capture long-term temporal dependencies.

---

## Project Overview
Time series forecasting is crucial for demand planning, financial analysis, and resource optimization. This project focuses on building a robust pipeline that transforms the historical Air Passengers time-stamped data into structured sequences, analyzes seasonal trends, and trains high-performance models to predict future passenger volume with minimal error.

## Technical Skills & Tools
* **Language:** Python
* **Environment:** Jupyter Notebook
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, TensorFlow / Keras, Scikit-Learn

---

## Data Pipeline & Methodology

### 1. Time Series Preprocessing & EDA
* Set and sorted historical timestamps as the dataframe index using the Air Passengers data.
* Handled missing sequential values using interpolation.
* Analyzed data components: Trend, Seasonality, and Noise.
* Checked for stationarity to ensure data stability for the models.

### 2. Feature Engineering for Sequences
* Transformed static data into supervised learning format using Sliding Window (Look-back) techniques.
* Applied MinMaxScaler to normalize data ranges, preventing exploding gradients during training.
* Split data chronologically to avoid data leakage.

### 3. Deep Learning & Predictive Modeling
Implemented advanced architectures optimized for sequential and temporal data:
* **Deep Neural Networks (DNN):** As a dense baseline for linear and non-linear patterns.
* **Long Short-Term Memory (LSTM):** A specialized Recurrent Neural Network (RNN) used to remember long-term dependencies.

### 4. Evaluation Metrics
Evaluated forecasting accuracy using standard regression metrics for time series:
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

---

## Repository Structure
```text
├── Time series Project.ipynb     # Main Jupyter Notebook with full analysis and DL models
├── AirPassengers.csv             # Historical air passengers dataset
└── README.md                     # Project documentation (This file)

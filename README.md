# Weather Prediction API Using Pre-Trained Regression Models

## Project Overview

This project implements a **Dockerized REST API for temperature prediction** using two pre-trained regression models.

The models are trained using the provided weather dataset to predict:

* **Maximum temperature (`max_t`)**
* **Minimum temperature (`min_t`)**

The project includes a Jupyter Notebook named **`web_scrapping.ipynb`**, which is used for data collection/preparation, exploratory analysis, preprocessing, and training the regression models.

The trained models are saved and included in the Docker image. Therefore, the models are **not retrained every time the API is started or a prediction request is received**.

---

## Project Structure

```text
temperature-prediction/
│
├── web_scrapping.ipynb
├── app.py
├── requirements.txt
├── Dockerfile
├── models/
│   ├── max_temperature_model.pkl
│   └── min_temperature_model.pkl
└── README.md
```

### Main Files

| File                        | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| `web_scrapping.ipynb`       | Data collection, preprocessing, analysis, and model training |
| `app.py`                    | REST API implementation                                      |
| `max_temperature_model.pkl` | Pre-trained regression model for maximum temperature         |
| `min_temperature_model.pkl` | Pre-trained regression model for minimum temperature         |
| `requirements.txt`          | Python dependencies                                          |
| `Dockerfile`                | Instructions for building the Docker image                   |
| `README.md`                 | Project documentation                                        |

---

## Methodology

The project follows these steps:

```text
Weather Dataset
      ↓
Data Cleaning & Preprocessing
      ↓
Feature Engineering
      ↓
Train Regression Models
      ↓
Save Pre-trained Models
      ↓
Build Docker Image
      ↓
Start REST API
      ↓
Temperature Prediction
```

Two separate regression models are trained:

1. **Maximum Temperature Regression Model**
2. **Minimum Temperature Regression Model**

The trained models are serialized and stored in the `models/` directory.

---

# 1. Data Preparation and Model Training

The notebook **`web_scrapping.ipynb`** contains the data preparation and model-training workflow.

Typical essential commands include:

```python
import pandas as pd
import numpy as np
```

Load the provided dataset:

```python
df = pd.read_csv("weather_data.csv")
```

Inspect the dataset:

```python
df.head()
df.info()
df.describe()
```

Check missing values:

```python
df.isnull().sum()
```

Remove or handle missing values as appropriate:

```python
df = df.dropna()
```

Convert the date column:

```python
df["date"] = pd.to_datetime(df["date"])
```

Create useful date-related features:

```python
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
```

---

# 2. Train/Test Split

The dataset is divided into training and testing data.

Example:

```python
from sklearn.model_selection import train_test_split

X = df[features]
y_max = df["max_t"]
y_min = df["min_t"]

X_train, X_test, y_max_train, y_max_test = train_test_split(
    X, y_max, test_size=0.2, random_state=42
)

X_train_min, X_test_min, y_min_train, y_min_test = train_test_split(
    X, y_min, test_size=0.2, random_state=42
)
```

The exact feature columns should correspond to the columns available in the supplied dataset.

---

# 3. Regression Models

The project trains two regression models.

For example:

```python
from sklearn.ensemble import RandomForestRegressor

max_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

min_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

Train the models:

```python
max_model.fit(X_train, y_max_train)
min_model.fit(X_train_min, y_min_train)
```

Evaluate the models:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

max_predictions = max_model.predict(X_test)

print("Max Temperature MAE:",
      mean_absolute_error(y_max_test, max_predictions))

print("Max Temperature R2:",
      r2_score(y_max_test, max_predictions))
```

Similarly, evaluate the minimum-temperature model:

```python
min_predictions = min_model.predict(X_test_min)

print("Min Temperature MAE:",
      mean_absolute_error(y_min_test, min_predictions))

print("Min Temperature R2:",
      r2_score(y_min_test, min_predictions))
```

---

# 4. Save the Pre-Trained Models

The models must be saved after training so that the API does **not train the models again**.

Use `joblib`:

```python
import joblib

joblib.dump(max_model, "models/max_temperature_model.pkl")
joblib.dump(min_model, "models/min_temperature_model.pkl")
```

The resulting files are:

```text
models/
├── max_temperature_model.pkl
└── min_temperature_model.pkl
```

These model files are copied into the Docker image.

---

# 5. REST API

The API can be implemented using **FastAPI**.

Example imports:

```python
from fastapi import FastAPI
import joblib
```

Create the application:

```python
app = FastAPI(
    title="Temperature Prediction API",
    description="API for predicting maximum and minimum temperatures",
    version="1.0.0"
)
```

Load the models when the application starts:

```python
max_model = joblib.load("models/max_temperature_model.pkl")
min_model = joblib.load("models/min_temperature_model.pkl")
```

This is important because the models are loaded from disk rather than being trained for every request.

---

# 6. Maximum Temperature Endpoint

### Endpoint

```text
POST /predict/max_t
```

### Request

```json
{
    "num_days": 7
}
```

The `num_days` parameter specifies how many future days should be predicted.

### Example Response

```json
{
    "predictions": [
        {
            "date": "2023-09-24",
            "max_temperature": 30.5
        },
        {
            "date": "2023-09-25",
            "max_temperature": 31.2
        }
    ]
}
```

The API generates predictions for the requested number of future days.

---

# 7. Minimum Temperature Endpoint

A similar endpoint is implemented for minimum temperature.

### Endpoint

```text
POST /predict/min_t
```

### Request

```json
{
    "num_days": 7
}
```

### Example Response

```json
{
    "predictions": [
        {
            "date": "2023-09-24",
            "min_temperature": 21.4
        },
        {
            "date": "2023-09-25",
            "min_temperature": 22.1
        }
    ]
}
```

---

# 8. API Endpoint Summary

| Method | Endpoint         | Purpose                                                      |
| ------ | ---------------- | ------------------------------------------------------------ |
| `POST` | `/predict/max_t` | Predict maximum temperature for the requested number of days |
| `POST` | `/predict/min_t` | Predict minimum temperature for the requested number of days |

---

# 9. Requirements

The project requires Python libraries such as:

```text
fastapi
uvicorn
pandas
numpy
scikit-learn
joblib
```

These dependencies should be placed in `requirements.txt`.

Example:

```text
fastapi
uvicorn
pandas
numpy
scikit-learn
joblib
```

---

# 10. Dockerfile

The Dockerfile creates an image containing the API, dependencies, and pre-trained models.

Example:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

The important point is that the trained model files are copied into the image:

```dockerfile
COPY models ./models
```

Therefore, the container already contains the trained models when it starts.

---

# 11. Build the Docker Image

From the project directory, run:

```bash
docker build -t temperature-prediction-api .
```

Check that the image was created:

```bash
docker images
```

---

# 12. Run the Docker Container

Run the container using:

```bash
docker run -p 8000:8000 temperature-prediction-api
```

The API will then be available at:

```text
http://localhost:8000
```

---

# 13. Test the API

### Maximum Temperature

Using `curl`:

```bash
curl -X POST "http://localhost:8000/predict/max_t" \
     -H "Content-Type: application/json" \
     -d '{"num_days": 7}'
```

### Minimum Temperature

```bash
curl -X POST "http://localhost:8000/predict/min_t" \
     -H "Content-Type: application/json" \
     -d '{"num_days": 7}'
```

---

# 14. Interactive API Documentation

FastAPI automatically provides interactive API documentation.

After starting the container, open:

```text
http://localhost:8000/docs
```

The Swagger interface can be used to test:

```text
POST /predict/max_t
POST /predict/min_t
```

without requiring a separate API client.

---

# 15. Important Requirement: Pre-Trained Models

The models are trained **before** the Docker container is started.

The workflow is:

```text
Dataset
   ↓
web_scrapping.ipynb
   ↓
Train models
   ↓
Save .pkl files
   ↓
Docker build
   ↓
Docker image contains trained models
   ↓
Container starts
   ↓
API loads models
   ↓
Prediction requests
```

The API does **not** execute:

```python
model.fit(...)
```

when a prediction request is received.

Instead, it loads the already-trained models:

```python
max_model = joblib.load("models/max_temperature_model.pkl")
min_model = joblib.load("models/min_temperature_model.pkl")
```

This satisfies the requirement that the Docker container uses **pre-trained regression models** rather than retraining the models for every prediction.

---

# 16. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* FastAPI
* Uvicorn
* Docker
* Jupyter Notebook

---

# 17. Project Objective

The objective of this project is to develop a reproducible machine-learning application that:

1. Uses the provided weather dataset.
2. Trains regression models for maximum and minimum temperature.
3. Saves the trained models.
4. Packages the models inside a Docker image.
5. Provides REST API endpoints for temperature prediction.
6. Allows users to specify the number of future days for prediction.
7. Avoids model retraining during API requests.

---

## How to Run

The complete workflow is:

```bash
# 1. Build the Docker image
docker build -t temperature-prediction-api .

# 2. Run the container
docker run -p 8000:8000 temperature-prediction-api

# 3. Open API documentation
# http://localhost:8000/docs
```

Then use either:

```text
POST /predict/max_t
```

or:

```text
POST /predict/min_t
```

with:

```json
{
    "num_days": 7
}
```

to obtain predictions for the requested number of future days.

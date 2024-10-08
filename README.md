# Credit Scoring Model for Bati Bank

## Overview

This project implements a credit scoring system using machine learning techniques to classify users as high-risk or low-risk based on their transaction data. The system utilizes a Random Forest model, enhanced by feature engineering and Weight of Evidence (WoE) binning, to provide accurate predictions.

## Table of Contents

- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Deployment](#deployment)
- [API Usage](#api-usage)
- [Requirements](#requirements)
- [License](#license)

## Dataset

The dataset used for this project contains transaction records with the following key columns:

- **TransactionId**: Unique identifier for each transaction.
- **CustomerId**: Identifier for each customer.
- **Amount**: Transaction amount.
- **FraudResult**: Indicator of whether the transaction was fraudulent (target variable).

The dataset consists of **95,662 entries** and **16 columns**, with no missing values.

## Feature Engineering

Feature engineering was performed to create meaningful metrics that enhance the model's predictive power. Key features derived include:

1. **Aggregate Features**:

   - Total Transaction Amount: Sum of all transaction amounts per customer.
   - Average Transaction Amount: Average transaction amount per customer.
   - Transaction Count: Total number of transactions per customer.
   - Standard Deviation of Transaction Amounts: Variability in transaction amounts.

2. **RFMS Features**:
   - Recency, Frequency, Monetary, and Stability metrics were calculated and transformed using WoE binning.

## Modeling

The following models were implemented and evaluated:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**

### Model Training

- The data was split into training (80%) and testing (20%) sets.
- Hyperparameter tuning was performed using Grid Search for each model to optimize performance.

### Evaluation Metrics

The models were evaluated based on:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

The Random Forest model achieved an accuracy of approximately 99.84% with a ROC-AUC score of 99.73%.

## Deployment

The trained Random Forest model was deployed using FastAPI and hosted on Render. This allows users to send transaction data and receive credit risk predictions via a RESTful API.

### FastAPI Application

The FastAPI application exposes an endpoint for predictions:

```python
@app.post("/predict")
def predict(data: dict):
    input_data = pd.DataFrame(data)
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}
```

## API Usage

To use the API, make a POST request to the `/predict` endpoint with the following JSON structure:

```json
{
  "Recency_WoE": 12.39,
  "Frequency_WoE": 0.0,
  "Monetary_WoE": 13.49,
  "Stability_WoE": -1.88
}
```

### Example Request

```bash
curl -X POST "https://your-api-url/predict" -H "Content-Type: application/json" -d '{"Recency_WoE": 12.39, "Frequency_WoE": 0.00, "Monetary_WoE": 13.49, "Stability_WoE": -1.88}'
```

### Example Response

```json
{
  "prediction": 0
}
```

Where 0 indicates a low-risk classification.

### Requirements

To run this project, ensure you have the following dependencies installed:

```text
fastapi
uvicorn
scikit-learn
pandas
numpy
matplotlib
joblib
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.

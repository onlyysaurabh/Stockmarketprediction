# modify the code to add date range to control how much data is used for training and prediction to make it less compute heavy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")
import pymongo
from pymongo import MongoClient
import pickle
import os
from tqdm import tqdm  # Import tqdm for progress bar
from datetime import datetime, timezone

# --- 1. Load Stock Price Data from MongoDB ---
def load_data_from_mongodb(mongo_uri, db_name, collection_name, stock_symbol, price_field='Close', start_date=None, end_date=None):
    """Loads stock price data from MongoDB, with optional date range."""
    client = None  # Initialize client outside the try block
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        query = {"Symbol": stock_symbol}
        if start_date and end_date:
            # Convert datetime.date to datetime.datetime with UTC timezone for MongoDB compatibility
            start_datetime = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
            end_datetime = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)
            query["Date"] = {"$gte": start_datetime, "$lte": end_datetime} # Add date range to query

        projection = {"Date": 1, price_field: 1, "_id": 0}
        sort = [("Date", pymongo.ASCENDING)]

        cursor = collection.find(query, projection=projection).sort(sort)
        data_list = list(cursor)

        if not data_list:
            raise ValueError(f"No data found for symbol '{stock_symbol}' in MongoDB collection '{collection_name}' within the specified date range.")

        df = pd.DataFrame(data_list)

        if 'Date' not in df.columns or price_field not in df.columns:
            raise ValueError(f"Required fields 'Date' and '{price_field}' not found in MongoDB data.")

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        price_series = df[price_field].squeeze()

        return price_series

    except pymongo.errors.ConnectionFailure as e:
        raise Exception(f"Could not connect to MongoDB: {e}")
    except Exception as e:
        raise Exception(f"Error loading data from MongoDB: {e}")
    finally:
        if client: # Check if client is defined before trying to close
            client.close()

# --- 2. Check for Stationarity using ADF Test ---
def check_stationarity(series):
    """Performs Augmented Dickey-Fuller test to check for stationarity."""
    dftest = adfuller(series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    if dfoutput['p-value'] <= 0.05:
        return True
    else:
        return False

# --- 3. Make Series Stationary (if needed) using Differencing ---
def make_stationary(series, d=1):
    """Makes a time series stationary by differencing."""
    if not check_stationarity(series):
        diff_series = series.diff(d).dropna()
        if check_stationarity(diff_series):
            return diff_series, d
        else:
            return None, d
    else:
        return series, 0

# --- 4. Split Data into Training and Testing Sets ---
def train_test_split(data, test_size=0.2):
    """Splits time series data into training and testing sets."""
    if not isinstance(data, pd.Series):
        raise ValueError("Input data must be a pandas Series.")
    split_index = int(len(data) * (1 - test_size))
    train_data, test_data = data[:split_index], data[split_index:]
    return train_data, test_data

# --- 5. Manually Define ARIMA (p, d, q) parameters (pmdarima is removed) ---
def get_manual_arima_params():
    """Manually defines ARIMA parameters (p, d, q)."""
    manual_order = (1, 1, 1)
    print(f"Using manual ARIMA order: ARIMA{manual_order}")
    return manual_order

# --- 6. Train ARIMA Model ---
def train_arima_model(train_series, order):
    """Trains an ARIMA model on the training data."""
    print("Starting ARIMA model training...")  # Progress message
    try:
        model = ARIMA(train_series, order=order)
        model_fit = model.fit()
        print("ARIMA model training finished.") # Progress message
        return model_fit
    except Exception as e:
        print(f"Error training ARIMA model: {e}")
        return None

# --- 7. Evaluate Model on Test Set (Walk-Forward Validation) ---
def evaluate_model(model_fit, train_series, test_series, original_series, diff_order, stock_symbol, order):
    """Evaluates the trained ARIMA model using walk-forward validation and displays
        evaluation metrics including confusion matrix, accuracy, precision, recall,
        sensitivity, specificity, and F1-score. Stores the trained model.
    """
    history = list(train_series)
    predictions = []
    for t in tqdm(range(len(test_series)), desc="Walk-Forward Validation"): # Added tqdm progress bar
        model = ARIMA(history, order=order)
        model_fit_wf = model.fit()
        output = model_fit_wf.forecast()
        yhat = output[0]
        predictions.append(yhat)
        history.append(test_series[t])

    if diff_order > 0:
        predictions_original_scale = []
        history_original_scale = list(original_series[:len(train_series)])

        for i in range(len(predictions)):
            yhat_original_scale = predictions[i] + history_original_scale[-1]
            predictions_original_scale.append(yhat_original_scale)
            history_original_scale.append(original_series[len(train_series) + i])
    else:
        predictions_original_scale = predictions
        history_original_scale = list(original_series)

    # Correct the start index for actual values in original scale
    start_index = len(train_series) + diff_order
    actuals_original_scale = original_series[start_index:].values

    # Ensure the lengths match the test series length
    actuals_original_scale = actuals_original_scale[:len(test_series)]
    predictions_original_scale = predictions_original_scale[:len(test_series)]

    # Calculate regression metrics
    rmse_val = np.sqrt(mean_squared_error(actuals_original_scale, predictions_original_scale))
    mae_val = mean_absolute_error(actuals_original_scale, predictions_original_scale)
    mape_val = np.mean(np.abs((actuals_original_scale - predictions_original_scale) / actuals_original_scale)) * 100

    print(f'\nModel Evaluation (Original Scale - Regression Metrics):')
    print(f'RMSE: {rmse_val:.2f}')
    print(f'MAE: {mae_val:.2f}')
    print(f'MAPE: {mape_val:.2f}%')

    # Calculate classification metrics
    actual_directions = np.diff(actuals_original_scale) > 0
    predicted_directions = np.diff(predictions_original_scale) > 0

    # Calculate confusion matrix and other metrics
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
    cm = confusion_matrix(actual_directions, predicted_directions)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(actual_directions, predicted_directions)
    recall = recall_score(actual_directions, predicted_directions)
    f1 = f1_score(actual_directions, predicted_directions)

    print(f'\nModel Evaluation (Direction Prediction - Classification Metrics):')
    print("Confusion Matrix:")
    print(cm)
    print(f'Accuracy: {accuracy_score(actual_directions, predicted_directions):.2f}')
    print(f'Sensitivity (Recall or True Positive Rate): {sensitivity:.2f}')
    print(f'Specificity (True Negative Rate): {specificity:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'F1-Score: {f1:.2f}')


    # Plotting Predictions vs Actual in Original Scale
    plt.figure(figsize=(12, 6))
    plt.plot(actuals_original_scale, label='Actual Prices', color='blue')
    plt.plot(predictions_original_scale, label='Predicted Prices', color='red')
    plt.title('ARIMA Model - Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    # --- Store Trained Model ---
    model_dir = f"models/{stock_symbol}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "arima.pkl")

    try:
        with open(model_path, 'wb') as file:
            pickle.dump(model_fit, file)
        print(f"\nTrained ARIMA model saved to: {model_path}")
    except Exception as e:
        print(f"Error saving model to {model_path}: {e}")


# --- Main Program ---
if __name__ == "__main__":
    # --- MongoDB Connection Details ---
    mongo_uri = "mongodb://localhost:27017/"
    db_name = "stock_market_db"
    collection_name = "stock_data"

    stock_symbol_to_predict = "AAPL"
    price_field_to_use = 'Close'

    # --- Date Range for Training and Prediction ---
    start_date_str = "2020-02-25"  # Example start date for data loading
    end_date_str = "2025-02-25"    # Example end date for data loading
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()


    try:
        # 1. Load Data from MongoDB with Date Range
        stock_series = load_data_from_mongodb(mongo_uri, db_name, collection_name, stock_symbol_to_predict, price_field_to_use, start_date, end_date)

        # 2. Handle Stationarity
        stationary_series, diff_order = make_stationary(stock_series.copy())
        if stationary_series is None:
            print("Failed to make series stationary. Exiting.")
            exit()

        # 3. Split Data
        train_data, test_data = train_test_split(stationary_series)
        original_train, original_test = train_test_split(stock_series) # Keep original_train, original_test although not directly used.

        # 4. Manually Set ARIMA Parameters (No pmdarima)
        best_order = get_manual_arima_params()

        # 5. Train ARIMA Model
        arima_model_fit = train_arima_model(train_data, best_order)
        if arima_model_fit is None:
            exit()

        # 6. Evaluate Model (includes classification metrics and model saving)
        evaluate_model(arima_model_fit, train_data, test_data, stock_series, diff_order, stock_symbol_to_predict, best_order)


    except FileNotFoundError as e:
        print(f"File Error: {e}")
    except ValueError as e:
        print(f"Value Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
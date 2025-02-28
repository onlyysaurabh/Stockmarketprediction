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

# --- Feature Engineering Functions ---
def calculate_moving_average(series, window=10):
    """Calculates the simple moving average."""
    return series.rolling(window=window).mean()

def calculate_rsi(series, period=14):
    """Calculates the Relative Strength Index."""
    delta = series.diff().dropna()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = up.ewm(span=period).mean()
    roll_down1 = np.abs(down.ewm(span=period).mean())
    RS = roll_up1 / roll_down1
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI

def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    """Calculates Moving Average Convergence Divergence."""
    ema_fast = series.ewm(span=fast_period).mean()
    ema_slow = series.ewm(span=slow_period).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def calculate_atr(df, period=14):
    """Calculates Average True Range."""
    high_low = df['High'] - df['Low']
    high_close_prev = np.abs(df['High'] - df['Close'].shift(1))
    low_close_prev = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# --- 1. Load Stock Price Data from MongoDB ---
def load_data_from_mongodb(mongo_uri, db_name, collection_name, stock_symbol, price_field='Close', start_date=None, end_date=None, feature_engineering=True):
    """Loads stock price data from MongoDB, with optional date range and feature engineering."""
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

        projection = {"Date": 1, "Open": 1, "High": 1, "Low": 1, "Close": 1, "Volume": 1, "_id": 0} # Include OHLCV for feature engineering
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

        if feature_engineering:
            df['MA_10'] = calculate_moving_average(df['Close'], window=10)
            df['RSI'] = calculate_rsi(df['Close'])
            macd_line, signal_line, macd_histogram = calculate_macd(df['Close'])
            df['MACD_Line'] = macd_line
            df['MACD_Signal'] = signal_line
            df['ATR'] = calculate_atr(df)
            df['Price_Range'] = df['High'] - df['Low']
            df['Volume_Change'] = df['Volume'].diff()

            df.dropna(inplace=True) # Important: Drop rows with NaN after feature engineering

        price_series = df[price_field].squeeze() # Still predicting 'Close' price

        return price_series, df # Return both price_series and the feature-rich DataFrame

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
    if not isinstance(data, pd.Series): # Modified to accept Series or DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas Series or DataFrame.")

    if isinstance(data, pd.Series):
        split_index = int(len(data) * (1 - test_size))
        train_data, test_data = data[:split_index], data[split_index:]
    elif isinstance(data, pd.DataFrame):
        split_index = int(len(data) * (1 - test_size))
        train_data, test_data = data[:split_index], data[split_index:]
    return train_data, test_data

# --- 5. Automatically Select ARIMA (p, d, q) parameters using AIC ---
def get_auto_arima_params(train_series, max_p=3, max_q=3):
    """Automatically defines ARIMA parameters (p, d, q) using AIC minimization."""
    best_aic = float("inf")
    best_order = None
    best_model_fit = None

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            order = (p, 0, q) # d is already determined by stationarity check
            try:
                model = ARIMA(train_series, order=order)
                model_fit = model.fit()
                aic = model_fit.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = order
                    best_model_fit = model_fit
            except Exception as e: # Catch any potential errors during model fitting
                continue # Just continue to the next combination if there's an issue

    if best_order:
        print(f"Automated ARIMA parameter selection - Best Order: ARIMA{best_order} with AIC={best_aic:.2f}")
        return best_order
    else:
        print("Automated ARIMA parameter selection failed to find a suitable order. Using default (1,1,1).")
        return (1, 1, 1) # Return default order if auto selection fails

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
def evaluate_model(model_fit, train_series, test_series, original_series, diff_order, stock_symbol, order, evaluation_results_collection, start_date, end_date):
    """Evaluates the trained ARIMA model using walk-forward validation and stores results in MongoDB."""
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
    plt.title(f'ARIMA Model for {stock_symbol} - Actual vs Predicted Stock Prices') # Include stock symbol in title
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

    # --- Store Evaluation Results in MongoDB ---
    evaluation_data = {
        "stock_symbol": stock_symbol,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "arima_order": str(order),
        "rmse": rmse_val,
        "mae": mae_val,
        "mape": mape_val,
        "accuracy": accuracy_score(actual_directions, predicted_directions),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "confusion_matrix": cm.tolist() # Store confusion matrix as list for MongoDB
    }
    try:
        evaluation_results_collection.insert_one(evaluation_data)
        print(f"Evaluation results for {stock_symbol} stored in MongoDB.")
    except Exception as e:
        print(f"Error storing evaluation results in MongoDB: {e}")


# --- Main Program ---
if __name__ == "__main__":
    # --- MongoDB Connection Details ---
    mongo_uri = "mongodb://localhost:27017/"
    db_name = "stock_market_db"
    collection_name = "stock_data"
    evaluation_collection_name = "arima_evaluation_results" # Collection to store evaluation results

    price_field_to_use = 'Close'

    # --- Date Range for Training and Prediction ---
    start_date_str = "2020-02-28"  # Original start date
    end_date_str = "2025-02-28"   # Modified end date to reduce compute

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    # --- Load Stock Symbols from CSV ---
    stocks_file = "stocks.csv" # Ensure stocks.csv is in the same directory or provide full path
    try:
        stocks_df = pd.read_csv(stocks_file)
        stock_symbols = stocks_df['Symbol'].tolist()
    except FileNotFoundError:
        print(f"Error: {stocks_file} not found. Please make sure it exists in the same directory.")
        exit()
    except KeyError:
        print(f"Error: 'Symbol' column not found in {stocks_file}. Please check the CSV file format.")
        exit()

    client = None # Initialize client outside the loop
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        evaluation_results_collection = db[evaluation_collection_name]

        for stock_symbol_to_predict in stock_symbols:
            print(f"\n--- Processing Stock: {stock_symbol_to_predict} ---")

            try:
                # 1. Load Data from MongoDB with Date Range and Feature Engineering
                stock_series, feature_df = load_data_from_mongodb(mongo_uri, db_name, collection_name, stock_symbol_to_predict, price_field_to_use, start_date, end_date, feature_engineering=True)

                # 2. Handle Stationarity (using 'Close' price series)
                stationary_series, diff_order = make_stationary(stock_series.copy())
                if stationary_series is None:
                    print("Failed to make series stationary. Skipping stock.")
                    continue # Skip to the next stock symbol

                # 3. Split Data (split the feature DataFrame as well to keep features aligned with target 'Close' price)
                train_data, test_data = train_test_split(stationary_series)
                original_train, original_test = train_test_split(stock_series) # Keep original_train, original_test although not directly used.
                feature_train_df, feature_test_df = train_test_split(feature_df) # Split feature-rich DataFrame

                # 4. Automatically Set ARIMA Parameters (using AIC for stationary series)
                best_order = get_auto_arima_params(train_data, max_p=3, max_q=3) # You can adjust max_p and max_q

                # 5. Train ARIMA Model
                arima_model_fit = train_arima_model(train_data, best_order)
                if arima_model_fit is None:
                    print("ARIMA model training failed. Skipping stock.")
                    continue # Skip to the next stock symbol

                # 6. Evaluate Model (includes classification metrics and model saving & result storing)
                evaluate_model(arima_model_fit, train_data, test_data, stock_series, diff_order, stock_symbol_to_predict, best_order, evaluation_results_collection, start_date, end_date)

            except ValueError as ve:
                print(f"Value Error for {stock_symbol_to_predict}: {ve}. Skipping stock.")
                continue # Skip to the next stock symbol
            except Exception as e:
                print(f"An error occurred for {stock_symbol_to_predict}: {e}. Skipping stock.")
                continue # Skip to the next stock symbol


    except pymongo.errors.ConnectionFailure as e:
        print(f"Could not connect to MongoDB: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in main program: {e}")
    finally:
        if client:
            client.close()
    print("\n--- Stock Price Prediction and Evaluation Completed for all symbols. ---")
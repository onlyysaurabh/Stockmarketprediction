import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pymongo
from pymongo import MongoClient
import pickle
import os
import numpy as np
from datetime import datetime

# --- 1. Load Stock Price Data from MongoDB ---
def load_data_from_mongodb(mongo_uri, db_name, collection_name, stock_symbol, start_date=None, end_date=None):
    """Loads stock price data from MongoDB, with optional date range."""
    client = None
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        query = {"Symbol": stock_symbol}
        if start_date and end_date:
            # Use datetime objects for MongoDB query, not date objects
            query["Date"] = {"$gte": datetime.combine(start_date, datetime.min.time()),
                             "$lte": datetime.combine(end_date, datetime.max.time())}

        projection = {"Date": 1, "Close": 1, "Volume": 1, "_id": 0}
        sort = [("Date", pymongo.ASCENDING)]

        cursor = collection.find(query, projection=projection).sort(sort)
        data_list = list(cursor)

        if not data_list:
            raise ValueError(f"No data found for symbol '{stock_symbol}' in MongoDB collection '{collection_name}' within the specified date range.")

        df = pd.DataFrame(data_list)

        if 'Date' not in df.columns or 'Close' not in df.columns:
            raise ValueError(f"Required fields 'Date' and 'Close' not found in MongoDB data.")

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        return df

    except pymongo.errors.ConnectionFailure as e:
        raise Exception(f"Could not connect to MongoDB: {e}")
    except Exception as e:
        raise Exception(f"Error loading data from MongoDB: {e}")
    finally:
        if client:
            client.close()

# --- 2. Prepare Data for SVM ---
def prepare_data_for_svm(df, look_back=60):
    """Prepares data for SVM by creating lagged features and scaling."""
    if 'Close' not in df.columns:
        raise ValueError("Required field 'Close' not found in DataFrame.")

    # Scale only the 'Close' price
    close_scaler = StandardScaler()
    df['Close_scaled'] = close_scaler.fit_transform(df[['Close']])

    df['Target'] = df['Close_scaled'].shift(-1)  # Shift scaled 'Close' to create target variable
    df.dropna(inplace=True)

    X = df[['Close_scaled']] # Use scaled close price as initial feature
    y = df['Target']

    # Create lagged features from scaled 'Close'
    for i in range(1, look_back + 1):
        X[f'Close_Lag_{i}'] = df['Close_scaled'].shift(i)
    X.dropna(inplace=True)

    X_scaled = X.drop('Close_scaled', axis=1).values # Drop the initial scaled close, keep only lags
    # No need to scale X_scaled again as Close_scaled is already scaled, and lags are from scaled data.

    return X_scaled, y.values[look_back:], close_scaler

# --- 3. Train SVM Model ---
def train_svm_model(X_train, y_train, kernel='rbf', C=1.0, epsilon=0.1):
    """Trains an SVM model with the specified parameters."""
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(X_train, y_train)
    return model

# --- 4. Evaluate Model ---
def evaluate_model(model, X_test, y_test, close_scaler, stock_symbol, start_date, end_date, evaluation_results_collection):
    """Evaluates the trained SVM model and stores results in MongoDB."""
    y_pred = model.predict(X_test)

    # Inverse transform to get original scale - using close_scaler
    y_pred_orig = close_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_orig = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Calculate regression metrics
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100

    print("\nRegression Metrics:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R-squared (R2): {r2:.4f}")

    # Calculate classification metrics - using original scale prices for direction
    y_test_class = np.where(np.diff(y_test_orig) > 0, 1, 0)
    y_pred_class = np.where(np.diff(y_pred_orig) > 0, 1, 0)

    # Check if there are enough predictions to compute diff
    if len(y_test_orig) > 1 and len(y_pred_orig) > 1:
        # Ensure lengths are compatible after diff operation
        min_len = min(len(y_test_class), len(y_pred_class))
        y_test_class = y_test_class[:min_len]
        y_pred_class = y_pred_class[:min_len]

        accuracy = accuracy_score(y_test_class, y_pred_class)
        conf_matrix = confusion_matrix(y_test_class, y_pred_class)
        class_report = classification_report(y_test_class, y_pred_class, zero_division=0)

        print("\nClassification Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print("  Confusion Matrix:\n", conf_matrix)
        print("  Classification Report:\n", class_report)
    else:
        accuracy = 0
        conf_matrix = np.array([])
        class_report = "Not enough data for classification metrics."
        print("\nNot enough data to compute classification metrics (requires more than one data point).")


    # Plotting Predictions vs Actual in Original Scale
    #plt.figure(figsize=(12, 6))
    #plt.plot(y_test_orig, label='Actual Prices', color='blue')
    #plt.plot(y_pred_orig, label='Predicted Prices', color='red')
    #plt.title(f'SVM Model for {stock_symbol} - Actual vs Predicted Stock Prices') # Include stock symbol in title
    #plt.xlabel('Time')
    #plt.ylabel('Stock Price')
    #plt.legend()
    #plt.show()

    # --- Store Evaluation Results in MongoDB ---
    evaluation_data = {
        "stock_symbol": stock_symbol,
        "start_date": start_date.isoformat(),  # Store as ISO string
        "end_date": end_date.isoformat(),      # Store as ISO string
        "svr_kernel": model.kernel,
        "svr_c": model.C,
        "svr_epsilon": model.epsilon,
        "regression_mse": mse,
        "regression_rmse": rmse,
        "regression_mae": mae,
        "regression_mape": mape,
        "regression_r2": r2,
        "classification_accuracy": accuracy,
        "classification_confusion_matrix": conf_matrix.tolist() if conf_matrix.size > 0 else [], # Store confusion matrix as list
        "classification_report": class_report # Store classification report as string
    }
    try:
        evaluation_results_collection.insert_one(evaluation_data)
        print(f"\nEvaluation results for {stock_symbol} stored in MongoDB.")
    except Exception as e:
        print(f"Error storing evaluation results in MongoDB: {e}")

    return mse, rmse, mae, r2, mape, accuracy, conf_matrix, class_report

# --- 5. Save Trained Model ---
def save_model(model, stock_symbol, close_scaler, model_name="svm_model.pkl"):
    """Saves the trained model and the scaler to a pickle file."""
    model_dir = f"models/{stock_symbol}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    scaler_path = os.path.join(model_dir, "close_scaler.pkl") # Save scaler

    try:
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"\nTrained SVM model saved to: {model_path}")
    except Exception as e:
        print(f"Error saving model to {model_path}: {e}")

    try:
        with open(scaler_path, 'wb') as file: # Save scaler
            pickle.dump(close_scaler, file)
        print(f"Scaler saved to: {scaler_path}")
    except Exception as e:
        print(f"Error saving scaler to {scaler_path}: {e}")


# --- Main Program ---
if __name__ == "__main__":
    # --- MongoDB Connection Details ---
    mongo_uri = "mongodb://localhost:27017/"
    db_name = "stock_market_db"
    collection_name = "stock_data"
    evaluation_collection_name = "svm_evaluation_results" # Collection for SVM results

    # --- Date Range for Training and Prediction ---
    start_date_str = "2020-02-25" # Original start date
    end_date_str = "2025-02-25"  # Modified end date to reduce data

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()


    # --- Load Stock Symbols from CSV ---
    stocks_file = "stocks.csv" # Ensure stocks.csv is in the same directory
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
                # 1. Load Data from MongoDB with Date Range
                df = load_data_from_mongodb(mongo_uri, db_name, collection_name, stock_symbol_to_predict, start_date, end_date)

                # 2. Prepare Data for SVM
                X_scaled, y, close_scaler = prepare_data_for_svm(df, look_back=60)

                # 3. Split Data
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

                # 4. Train SVM Model
                svm_model = train_svm_model(X_train, y_train)

                # 5. Evaluate Model and Store Results
                evaluate_model(svm_model, X_test, y_test, close_scaler, stock_symbol_to_predict, start_date, end_date, evaluation_results_collection)

                # 6. Save Trained Model and Scaler
                save_model(svm_model, stock_symbol_to_predict, close_scaler)

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

    print("\n--- SVM Stock Price Prediction and Evaluation Completed for all symbols. ---")
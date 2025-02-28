import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, accuracy_score, confusion_matrix,
                             classification_report)
from sklearn.model_selection import train_test_split, GridSearchCV
import pymongo
from pymongo import MongoClient
import pickle
import os
import numpy as np
from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_regression


# --- 1. Load Stock Price Data from MongoDB ---
def load_data_from_mongodb(mongo_uri, db_name, collection_name, stock_symbol,
                           start_date=None, end_date=None):
    """Loads stock data from MongoDB, with optional date range."""
    client = None
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        query = {"Symbol": stock_symbol}
        if start_date and end_date:
            query["Date"] = {
                "$gte": datetime.combine(start_date, datetime.min.time()),
                "$lte": datetime.combine(end_date, datetime.max.time())
            }

        projection = {"Date": 1, "Close": 1, "Volume": 1, "Open": 1, "High": 1,
                      "Low": 1, "_id": 0}  # Include OHLC
        sort = [("Date", pymongo.ASCENDING)]

        cursor = collection.find(query, projection=projection).sort(sort)
        data_list = list(cursor)

        if not data_list:
            raise ValueError(
                f"No data found for symbol '{stock_symbol}' in MongoDB "
                f"collection '{collection_name}' within the specified date "
                f"range."
            )

        df = pd.DataFrame(data_list)

        if 'Date' not in df.columns or 'Close' not in df.columns:
            raise ValueError(
                f"Required fields 'Date' and 'Close' not found in MongoDB "
                f"data."
            )

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
def prepare_data_for_svm(df, look_back=60, num_features_to_select=30):
    """Prepares data: creates features, scales, selects, and splits."""

    if 'Close' not in df.columns:
        raise ValueError("Required field 'Close' not found in DataFrame.")

    # 1. Feature Engineering (on a *copy* to avoid SettingWithCopyWarning)
    df = df.copy()
    df['SMA'] = df['Close'].rolling(window=look_back).mean()
    df['EMA'] = df['Close'].ewm(span=look_back, adjust=False).mean()
    df['Price_Range'] = df['High'] - df['Low']
    df['Volume_Change'] = df['Volume'].diff()
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)  # Drop NaNs *after* feature engineering

    # 2. Create Lagged Features (do this *before* scaling and target creation)
    for i in range(1, look_back + 1):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)

    # 3. Define Target Variable (shift *after* lag creation)
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    # 4. Separate Features (X) and Target (y)
    X = df.drop('Target', axis=1)
    y = df['Target']

    # 5. Feature Scaling (Scale *after* feature engineering, *before* selection)
    close_features = ['Close'] + [f'Close_Lag_{i}' for i in range(1, look_back + 1)]
    other_features = [col for col in X.columns if col not in close_features]

    close_scaler = StandardScaler()
    X.loc[:, close_features] = close_scaler.fit_transform(X[close_features])  # Scale close features

    other_scaler = StandardScaler()
    if other_features:  # Handle the case where there are no "other" features
        X.loc[:, other_features] = other_scaler.fit_transform(X[other_features])

    # 6. Feature Selection (Select *after* scaling, *before* splitting)
    if num_features_to_select < X.shape[1]:
        selector = SelectKBest(score_func=f_regression,
                               k=num_features_to_select)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        X = pd.DataFrame(X_selected, index=X.index,
                         columns=selected_features) # Keep index & names!

    else:
      selected_features = X.columns.to_list()

    # 7. Train-Test Split (*after* feature engineering, scaling, and selection)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 8. Scale Target (important for inverse transform later)
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1,1)).flatten()



    return (X_train, X_test, y_train_scaled, y_test_scaled, close_scaler,
            other_scaler if other_features else None, target_scaler, selected_features)


# --- 3. Train SVM Model ---
def train_svm_model(X_train, y_train, kernel='rbf'):
    """Trains SVM model with hyperparameter tuning (GridSearchCV)."""
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1, 10],
        'epsilon': [0.01, 0.1, 1],
        'kernel': [kernel]  # Keep kernel option
    }

    model = SVR()
    grid_search = GridSearchCV(model, param_grid, cv=3,
                               scoring='neg_mean_squared_error', verbose=0,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


# --- 4. Evaluate Model ---
def evaluate_model(model, X_test, y_test, close_scaler, other_scaler, target_scaler, stock_symbol, start_date, end_date, evaluation_results_collection):
    """Evaluates the model, inverse transforms, and stores results."""
    y_pred = model.predict(X_test)

    # Inverse transform predictions and actual values to original scale
    y_pred_orig = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

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

    # Calculate classification metrics
    y_test_class = np.where(np.diff(y_test_orig) > 0, 1, 0)
    y_pred_class = np.where(np.diff(y_pred_orig) > 0, 1, 0)

    if len(y_test_orig) > 1 and len(y_pred_orig) > 1:
        min_len = min(len(y_test_class), len(y_pred_class))
        y_test_class = y_test_class[:min_len]
        y_pred_class = y_pred_class[:min_len]

        accuracy = accuracy_score(y_test_class, y_pred_class)
        conf_matrix = confusion_matrix(y_test_class, y_pred_class)
        class_report = classification_report(y_test_class, y_pred_class,
                                             zero_division=0)

        print("\nClassification Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print("  Confusion Matrix:\n", conf_matrix)
        print("  Classification Report:\n", class_report)
    else:
        accuracy = 0
        conf_matrix = np.array([])
        class_report = "Not enough data for classification metrics."
        print("\nNot enough data for classification metrics.")

    # Store evaluation results
    best_params = model.get_params()
    evaluation_data = {
        "stock_symbol": stock_symbol,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "svr_kernel": best_params.get('kernel', 'rbf'),
        "svr_c": best_params.get('C', 1.0),
        "svr_epsilon": best_params.get('epsilon', 0.1),
        "svr_gamma": best_params.get('gamma', 'scale'),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix.tolist() if conf_matrix.size > 0 else [],
        "report": class_report
    }
    try:
        evaluation_results_collection.insert_one(evaluation_data)
        print(f"\nEvaluation results for {stock_symbol} stored in MongoDB.")
    except Exception as e:
        print(f"Error storing evaluation results in MongoDB: {e}")

    return mse, rmse, mae, r2, mape, accuracy, conf_matrix, class_report


# --- 5. Save Trained Model ---
def save_model(model, stock_symbol, close_scaler, other_scaler, target_scaler, selected_features, model_name="svm_model.pkl"):
    """Saves the model and scalers."""
    model_dir = f"models/{stock_symbol}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    close_scaler_path = os.path.join(model_dir, "close_scaler.pkl")
    other_scaler_path = os.path.join(model_dir, "other_scaler.pkl")
    target_scaler_path = os.path.join(model_dir, "target_scaler.pkl")  # Target scaler
    features_path = os.path.join(model_dir, "selected_features.pkl")


    try:
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"\nTrained SVM model saved to: {model_path}")
    except Exception as e:
        print(f"Error saving model to: {model_path}: {e}")

    try:
        with open(close_scaler_path, 'wb') as file:
            pickle.dump(close_scaler, file)
        print(f"Close scaler saved to: {close_scaler_path}")
    except Exception as e:
        print(f"Error saving close scaler to: {close_scaler_path}: {e}")

    if other_scaler is not None:
        try:
            with open(other_scaler_path, 'wb') as file:
                pickle.dump(other_scaler, file)
            print(f"Other features scaler saved to: {other_scaler_path}")
        except Exception as e:
            print(f"Error saving other features scaler to: {other_scaler_path}: {e}")

    try:
        with open(target_scaler_path, 'wb') as file:
            pickle.dump(target_scaler, file)  # Save target scaler
        print(f"Target scaler saved to: {target_scaler_path}")
    except Exception as e:
        print(f"Error saving target scaler to: {target_scaler_path}: {e}")

    try:
        with open(features_path, 'wb') as file:
            pickle.dump(selected_features, file)
        print(f"Selected features saved to: {features_path}")
    except Exception as e:
        print(f"Error saving selected features to: {features_path}: {e}")



# --- Main Program ---
if __name__ == "__main__":
    # --- MongoDB Connection Details ---
    mongo_uri = "mongodb://localhost:27017/"
    db_name = "stock_market_db"
    collection_name = "stock_data"
    evaluation_collection_name = "svm_evaluation_results"

    # --- Date Range ---
    start_date_str = "2020-02-25"
    end_date_str = "2025-02-25"
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    # --- Load Stock Symbols ---
    stocks_file = "stocks.csv"
    try:
        stocks_df = pd.read_csv(stocks_file)
        stock_symbols = stocks_df['Symbol'].tolist()
    except FileNotFoundError:
        print(f"Error: {stocks_file} not found.")
        exit()
    except KeyError:
        print(f"Error: 'Symbol' column not found in {stocks_file}.")
        exit()

    client = None
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        evaluation_results_collection = db[evaluation_collection_name]

        for stock_symbol_to_predict in stock_symbols:
            print(f"\n--- Processing Stock: {stock_symbol_to_predict} ---")
            try:
                # 1. Load Data
                df = load_data_from_mongodb(
                    mongo_uri, db_name, collection_name,
                    stock_symbol_to_predict, start_date, end_date
                )

                # 2. Prepare Data
                look_back = 60
                num_features = 30
                (X_train, X_test, y_train_scaled, y_test_scaled,
                 close_scaler, other_scaler, target_scaler, selected_features) = prepare_data_for_svm(
                    df, look_back=look_back,
                    num_features_to_select=num_features
                )


                # 3. Train Model
                svm_model = train_svm_model(X_train, y_train_scaled)

                # 4. Evaluate and Store Results
                evaluate_model(svm_model, X_test, y_test_scaled, close_scaler,
                               other_scaler, target_scaler,
                               stock_symbol_to_predict, start_date, end_date,
                               evaluation_results_collection)

                # 5. Save Model and Scalers
                save_model(svm_model, stock_symbol_to_predict, close_scaler,
                           other_scaler, target_scaler, selected_features)

            except ValueError as ve:
                print(f"Value Error for {stock_symbol_to_predict}: {ve}. "
                      f"Skipping stock.")
                continue
            except Exception as e:
                print(f"An error occurred for {stock_symbol_to_predict}: {e}. "
                      f"Skipping stock.")
                continue

    except pymongo.errors.ConnectionFailure as e:
        print(f"Could not connect to MongoDB: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in main program: {e}")
    finally:
        if client:
            client.close()

    print("\n--- SVM Stock Price Prediction and Evaluation Completed. ---")
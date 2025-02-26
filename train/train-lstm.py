import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix, classification_report
import os
import datetime

def calculate_technical_indicators(df, short_window=20, long_window=50, lag_days=1):
    """
    Calculates technical indicators: Moving Averages, and Lagged Returns.

    Args:
        df: DataFrame with 'Close' prices.
        short_window: Window size for the short moving average.
        long_window: Window size for the long moving average.
        lag_days: Number of days to lag returns.

    Returns:
        DataFrame with added indicator columns.  Returns an empty DataFrame on error.
    """
    if 'Close' not in df.columns:
        print("Error: 'Close' column not found in DataFrame.")
        return pd.DataFrame()

    df = df.copy()  # Work on a copy to avoid modifying the original

    # Simple Moving Averages
    df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()

    # Exponential Moving Averages (EMA)
    df['EMA_Short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_Long'] = df['Close'].ewm(span=long_window, adjust=False).mean()

    # Lagged Returns (for creating target variable and features)
    df[f'Lagged_Return_{lag_days}'] = df['Close'].shift(lag_days)

    # Drop rows with NaN values introduced by indicators
    df.dropna(inplace=True)
    return df



def create_sequences(data, seq_length):
    """
    Creates sequences for LSTM input.

    Args:
        data: NumPy array of features.
        seq_length: Length of the input sequences.

    Returns:
        X: Input sequences.
        y: Corresponding target values (next day's close).
    """
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 0])  # Assuming 'Close' is the first column after scaling
    return np.array(X), np.array(y)


def create_classification_labels(y_regression, y_lagged):
    """
    Creates classification labels (Up/Down) based on regression targets.
    """
    return np.where(y_regression > y_lagged, 1, 0)  # 1 for Up, 0 for Down


def evaluate_regression(y_true, y_pred):
    """Evaluates regression performance."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("Regression Metrics:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  R-squared (R2): {r2:.4f}")
    return mse, rmse, mae, r2



def evaluate_classification(y_true, y_pred):
    """Evaluates classification performance."""
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)

    print("\nClassification Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print("  Confusion Matrix:\n", conf_matrix)
    print("  Classification Report:\n", class_report)
    return accuracy, conf_matrix, class_report


def train_lstm_model(symbol, start_date, end_date, seq_length=60, short_window=20, long_window=50, lag_days=1, lstm_units=50, dropout_rate=0.2, epochs=50, batch_size=32, patience=10):
    """
    Trains and evaluates LSTM models for regression and classification.

    Args:
        symbol: Stock symbol (e.g., "AAPL").
        start_date: Start date for data fetching (YYYY-MM-DD).
        end_date: End date for data fetching (YYYY-MM-DD).
        seq_length: Sequence length for LSTM input.
        short_window:  Short moving average window.
        long_window: Long moving average window.
        lag_days: Number of days to lag returns.
        lstm_units: Number of LSTM units.
        dropout_rate: Dropout rate.
        epochs: Number of training epochs.
        batch_size: Batch size.
        patience:  Patience for EarlyStopping

    Returns:
        None (saves models and prints evaluation results)
    """

    # --- 1. Data Fetching and Preprocessing ---
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for {symbol} between {start_date} and {end_date}.")
            return
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        return

    data = calculate_technical_indicators(data, short_window, long_window, lag_days)
    if data.empty: # check if indicators were calculated correctly
      return

    # Feature Selection and Scaling
    features = ['Close', 'SMA_Short', 'SMA_Long', 'EMA_Short', 'EMA_Long', f'Lagged_Return_{lag_days}']  # Include indicators
    data_filtered = data[features]


    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_filtered)


    # --- 2. Create Sequences and Labels ---
    X, y_regression = create_sequences(scaled_data, seq_length)
    y_classification = create_classification_labels(y_regression, data_filtered[f'Lagged_Return_{lag_days}'].values[seq_length:])

    # --- 3. Train/Test Split ---
    # Split based on time to preserve chronological order
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train_reg, y_test_reg = y_regression[:train_size], y_regression[train_size:]
    y_train_class, y_test_class = y_classification[:train_size], y_classification[train_size:]


    # --- 4. Build LSTM Models ---

    # Regression Model
    model_reg = Sequential([
        LSTM(lstm_units, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(dropout_rate),
        LSTM(lstm_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)  # Output layer for regression
    ])
    model_reg.compile(optimizer='adam', loss='mse')

    # Classification Model
    model_class = Sequential([
        LSTM(lstm_units, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(dropout_rate),
        LSTM(lstm_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model_class.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



    # --- 5. Train Models with Callbacks ---
    model_dir = f"./data/{symbol}"
    os.makedirs(model_dir, exist_ok=True)

    # Callbacks for early stopping and saving best models
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    checkpoint_reg = ModelCheckpoint(os.path.join(model_dir, f"lstm_regression_{symbol}.h5"), monitor='val_loss', save_best_only=True)
    checkpoint_class = ModelCheckpoint(os.path.join(model_dir, f"lstm_classification_{symbol}.h5"), monitor='val_accuracy', save_best_only=True)


    print(f"Training Regression Model for {symbol}...")
    history_reg = model_reg.fit(X_train, y_train_reg, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping, checkpoint_reg], verbose=1)

    print(f"\nTraining Classification Model for {symbol}...")
    history_class = model_class.fit(X_train, y_train_class, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping, checkpoint_class], verbose=1)



    # --- 6. Evaluate Models ---
    # Regression Evaluation
    y_pred_reg = model_reg.predict(X_test)
    # Inverse transform to get original scale
    y_pred_reg_orig = scaler.inverse_transform(np.hstack([y_pred_reg, np.zeros((y_pred_reg.shape[0], scaled_data.shape[1]-1))]))[:, 0]
    y_test_reg_orig = scaler.inverse_transform(np.hstack([y_test_reg.reshape(-1,1), np.zeros((y_test_reg.shape[0], scaled_data.shape[1]-1))]))[:, 0]
    evaluate_regression(y_test_reg_orig, y_pred_reg_orig)


    # Classification Evaluation
    y_pred_class = (model_class.predict(X_test) > 0.5).astype(int)  # Convert probabilities to 0 or 1
    evaluate_classification(y_test_class, y_pred_class)


if __name__ == '__main__':
    symbol = "AAPL"
    start_date = "2010-01-01"
    end_date = "2023-12-31"
    train_lstm_model(symbol, start_date, end_date)

    symbol = "MSFT"
    start_date = "2015-01-01"
    # Get today's date
    today = datetime.date.today()
    # Format the date as YYYY-MM-DD
    end_date = today.strftime("%Y-%m-%d")
    train_lstm_model(symbol, start_date, end_date, epochs=75, lstm_units = 60, patience=15) # example of more specific training
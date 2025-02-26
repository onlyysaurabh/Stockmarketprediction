import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX
#from sklearn.model_selection import train_test_split # remove train_test_split
from sklearn.metrics import mean_squared_error
from pymongo import MongoClient
import datetime
import os

# --- MongoDB Connection ---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "stock_market_db"
COMMODITY_COLLECTION_NAME = "commodity_data"

def fetch_commodity_data_from_mongodb(db_name, collection_name, symbols, start_date, end_date):
    """
    Fetches commodity data from MongoDB for specified symbols and date range.

    Args:
        db_name (str): Name of the MongoDB database.
        collection_name (str): Name of the MongoDB collection.
        symbols (list): List of commodity symbols to fetch.
        start_date (datetime.date): Start date for data retrieval.
        end_date (datetime.date): End date for data retrieval.

    Returns:
        pandas.DataFrame: DataFrame containing combined commodity data, or None if error.
    """
    client = MongoClient(MONGO_URI)
    db = client[db_name]
    collection = db[collection_name]

    combined_data = {}
    for symbol in symbols:
        data = list(collection.find({
            "Symbol": symbol,
            "Date": {"$gte": datetime.datetime.combine(start_date, datetime.datetime.min.time()), # Convert date to datetime
                     "$lte": datetime.datetime.combine(end_date, datetime.datetime.max.time())}  # Convert date to datetime
        }, projection={'_id': False}).sort("Date", 1))

        if data:
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['Date']).dt.date # Convert to date only
            df.set_index('Date', inplace=True)
            combined_data[symbol] = df['Close']
        else:
            print(f"No data found in MongoDB for symbol: {symbol}")
            client.close()
            return None

    client.close()
    return pd.DataFrame(combined_data)


def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetches stock data using yfinance.

    Args:
        symbol (str): Stock symbol.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).

    Returns:
        pandas.DataFrame: Stock data DataFrame or None if error.
    """
    try:
        # Corrected 'end_date' to 'end' in yf.download
        data = yf.download(symbol, start=start_date, end=end_date)
        if not data.empty:
            return data['Close']
        else:
            print(f"No data found for symbol: {symbol}")
            return None
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {e}")
        return None

def prepare_varmax_data(stock_data, commodity_data, lag_diff=1):
    """
    Prepares data for VARMAX model, combining stock and commodity data and differencing.

    Args:
        stock_data (pandas.Series): Stock price data.
        commodity_data (pandas.DataFrame): Commodity price data.
        lag_diff (int): Lag for differencing.

    Returns:
        pandas.DataFrame: Combined and differenced DataFrame.
    """
    if stock_data is None or commodity_data is None:
        return None

    combined_df = pd.concat([stock_data, commodity_data], axis=1).dropna()
    combined_df.columns = ['Stock_Close'] + list(commodity_data.columns)

    # Differencing - apply to all columns
    df_diff = combined_df.diff(periods=lag_diff).dropna()
    return df_diff


def evaluate_forecast(y_true, y_predicted):
    """
    Evaluates forecast using RMSE.

    Args:
        y_true (pandas.Series or numpy.ndarray): True values.
        y_predicted (pandas.Series or numpy.ndarray): Predicted values.

    Returns:
        float: RMSE value.
    """
    rmse_val = np.sqrt(mean_squared_error(y_true, y_predicted))
    print(f"RMSE: {rmse_val:.3f}")
    return rmse_val

def train_varmax(data_diff, p=1, q=0, maxiter=20):
    """
    Trains VARMAX model.

    Args:
        data_diff (pandas.DataFrame): Differenced data.
        p (int): Order of the VAR component.
        q (int): Order of the MA component.
        maxiter (int): Maximum iterations for fitting.

    Returns:
        statsmodels.tsa.statespace.varmax.VARMAX: Fitted VARMAX model.
    """
    model = VARMAX(data_diff, order=(p, q), trend='c') # Including constant trend
    try:
        result = model.fit(maxiter=maxiter, disp=False) # disp=False to suppress convergence output
        return result
    except Exception as e:
        print(f"Error fitting VARMAX model: {e}")
        return None


def walk_forward_validation_varmax(df_diff, df_original, n_test, p, q, symbol_stock):
    """
    Performs walk-forward validation for VARMAX model.

    Args:
        df_diff (pandas.DataFrame): Differenced data.
        df_original (pandas.DataFrame): Original data for undifferencing.
        n_test (int): Number of test steps.
        p (int): Order of VAR component.
        q (int): Order of MA component.
        symbol_stock (str): Stock symbol for referencing original price.

    Returns:
        pandas.DataFrame: DataFrame with predicted values.
    """
    #train, test_diff = train_test_split(df_diff, n_test, shuffle=False) # Removed train_test_split
    train_size = len(df_diff) - n_test
    train = df_diff.iloc[:train_size]
    test_diff = df_diff.iloc[train_size:]

    #_, test_original = train_test_split(df_original[['Stock_Close']], n_test, shuffle=False) # Removed train_test_split
    test_original = df_original[['Stock_Close']].iloc[train_size:]


    history = train
    pred_stock = np.array([])

    for _ in range(len(test_diff)):
        model_fit = train_varmax(history, p, q)
        if model_fit is None: # Handle model fitting failure
            return None

        yhat = model_fit.predict(start=len(history), end=len(history))
        pred_stock = np.append(pred_stock, yhat.values.flatten()[0]) # Assuming stock price is the first variable

        history = pd.concat([history, test_diff.iloc[[0]]]) # Append current step's test data to history
        test_diff = test_diff.iloc[1:] # Move to the next step

    test_pred_df = pd.DataFrame(index=test_original.index)
    test_pred_df['Stock_Pred_Diff'] = pred_stock
    test_pred_df['Stock_Pred_Undiff'] = np.zeros(len(test_original))

    # Undifferencing the predictions
    test_pred_df['Stock_Pred_Undiff'].iloc[0] = df_original['Stock_Close'].iloc[len(train)] + test_pred_df['Stock_Pred_Diff'].iloc[0]
    for i in range(1, len(test_original)):
        test_pred_df['Stock_Pred_Undiff'].iloc[i] = test_pred_df['Stock_Pred_Undiff'].iloc[i-1] + test_pred_df['Stock_Pred_Diff'].iloc[i]

    return test_pred_df


def main_varmax(symbol_stock="AAPL", commodity_symbols=["CL=F", "GC=F"], start_date="2020-02-25", end_date="2025-02-25", n_test=60, varmax_p=1, varmax_q=0):
    """
    Main function to train and evaluate VARMAX model for stock price prediction.

    Args:
        symbol_stock (str): Stock symbol to predict.
        commodity_symbols (list): List of commodity symbols to use as exogenous factors.
        start_date (str): Start date for data fetching (YYYY-MM-DD).
        end_date (str): End date for data fetching (YYYY-MM-DD).
        n_test (int): Number of days for test set in walk-forward validation.
        varmax_p (int): p order for VARMAX.
        varmax_q (int): q order for VARMAX.
    """
    # --- 1. Fetch Data ---
    stock_close_data = fetch_stock_data(symbol_stock, start_date, end_date)
    commodity_data = fetch_commodity_data_from_mongodb(
        DB_NAME, COMMODITY_COLLECTION_NAME, commodity_symbols,
        datetime.datetime.strptime(start_date, '%Y-%m-%d').date(),
        datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
    )

    if stock_close_data is None or commodity_data is None:
        print("Error: Could not fetch necessary data. Exiting.")
        return

    # Align indices to handle potential date mismatches after fetching from MongoDB
    start_date_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
    date_range = pd.date_range(start=start_date_dt, end=end_date_dt).date

    stock_close_data = stock_close_data[stock_close_data.index.isin(date_range)]
    commodity_data = commodity_data[commodity_data.index.isin(date_range)]


    # --- 2. Prepare Data ---
    df_diff = prepare_varmax_data(stock_close_data, commodity_data)
    if df_diff is None:
        print("Error: Could not prepare VARMAX data. Exiting.")
        return


    # --- 3. Walk-Forward Validation ---
    test_predictions_df = walk_forward_validation_varmax(
        df_diff,
        prepare_varmax_data(stock_close_data, commodity_data, lag_diff=0), # Send original data for undifferencing
        n_test, varmax_p, varmax_q, symbol_stock
    )

    if test_predictions_df is None: # Handle case where walk-forward validation failed (e.g., model fitting issues)
        print("Error during VARMAX walk-forward validation. Exiting.")
        return

    # --- 4. Evaluate ---
    if test_predictions_df is not None:
        original_stock_prices = fetch_stock_data(symbol_stock, start_date, end_date).loc[test_predictions_df.index] # Fetch original prices for test period
        if original_stock_prices is not None:
            rmse_varmax = evaluate_forecast(original_stock_prices.values, test_predictions_df['Stock_Pred_Undiff'].values)


if __name__ == "__main__":
    stock_symbol = "AAPL"
    commodity_symbols_list = ["CL=F", "GC=F"] # Oil and Gold as exogenous
    start_date_str = "2020-02-25"
    end_date_str = "2025-02-25"
    n_test_steps = 60
    p_order = 1
    q_order = 0


    main_varmax(
        symbol_stock=stock_symbol,
        commodity_symbols=commodity_symbols_list,
        start_date=start_date_str,
        end_date=end_date_str,
        n_test=n_test_steps,
        varmax_p=p_order,
        varmax_q=q_order
    )
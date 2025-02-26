import yfinance as yf
import pymongo
from pymongo import MongoClient, errors
import datetime
import time
import pandas as pd

# --- MongoDB Connection ---
MONGO_URI = "mongodb://localhost:27017/"  # Make sure MongoDB is running
DB_NAME = "stock_market_db"
COLLECTION_NAME = "commodity_data"  # More general collection name

# --- Commodity Symbols and Names ---
symbols_and_names = [
    ("CL=F", "Oil (WTI Crude)"),
    ("GC=F", "Gold"),
    ("BTC-USD", "Bitcoin"),
    ("^GSPC", "S&P 500"),
    ("DX-Y.NYB", "US Dollar Index"),
    ("^IXIC", "NASDAQ Composite")
]


def get_historical_data(symbol, name, start_date, end_date):
    """
    Fetches historical data for a given symbol from Yahoo Finance.

    Args:
        symbol (str): The Yahoo Finance symbol.
        name (str):  The name of the commodity/index/cryptocurrency.
        start_date (datetime.date): Start date.
        end_date (datetime.date): End date.

    Returns:
        pandas.DataFrame: Historical data, or None if an error occurs.
    """
    try:
        ticker = yf.Ticker(symbol)
        # Fetch data.  Pass datetime.date objects directly.
        data = ticker.history(start=start_date, end=end_date)

        # Add the name and symbol as columns.
        data['Name'] = name
        data['Symbol'] = symbol
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol} ({name}): {e}")
        return None


def connect_to_mongodb(uri, db_name, max_pool_size=10):
    """
    Connects to MongoDB and returns the database object. Uses a connection pool.

    Args:
        uri (str): MongoDB connection URI.
        db_name (str): Name of the database.
        max_pool_size (int): Maximum number of connections in the pool.

    Returns:
        pymongo.database.Database: The MongoDB database object, or None on error.
    """
    try:
        client = MongoClient(uri, maxPoolSize=max_pool_size)
        db = client[db_name]
        return db
    except errors.ConnectionFailure as e:
        print(f"Could not connect to MongoDB: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def create_indexes(db, collection_name):
    """Creates indexes on the collection to improve query performance."""
    collection = db[collection_name]
    # Create indexes (if they don't already exist)
    collection.create_index([("Symbol", 1), ("Date", 1)], unique=True)  # Composite unique index
    collection.create_index("Symbol")  # Index on Symbol for faster lookups
    collection.create_index("Date")      # Index for date range queries.
    print("Indexes created successfully.")

def store_data_in_mongodb(db, collection_name, data, symbol, name):
    """
    Stores the fetched historical data in MongoDB using update_one with upsert=True.

    Args:
        db (pymongo.database.Database): The MongoDB database object.
        collection_name (str): The name of the collection.
        data (pandas.DataFrame): The historical data to store.
        symbol (str):  The Yahoo Finance symbol (for logging).
        name (str): The name of the commodity/index.
    """
    if data is None or data.empty:
        print(f"No data to store for {symbol} ({name}).")
        return

    collection = db[collection_name]
    data_dict = data.reset_index().to_dict(orient='records')

    # Convert Timestamp objects to timezone-aware datetime objects (for MongoDB compatibility)
    for record in data_dict:
        for key, value in record.items():
            if isinstance(value, pd.Timestamp):
                record[key] = value.to_pydatetime().replace(tzinfo=datetime.timezone.utc)
            # Ensure numeric types
            elif key in ("Open", "High", "Low", "Close", "Dividends", "Stock Splits"):
                try:
                    record[key] = float(value)
                except (ValueError, TypeError):
                    record[key] = 0.0  # Or handle as appropriate (e.g., NaN, None)
            elif key == "Volume":
                try:
                    record[key] = int(value)
                except (ValueError, TypeError):
                    record[key] = 0 # Or handle missing/invalid volumes

    try:
        for record in data_dict:
            # Use update_one with upsert=True for each record
            result = collection.update_one(
                {"Symbol": record["Symbol"], "Date": record["Date"]},  # Filter criteria
                {"$set": record},  # Update operation: set all fields
                upsert=True  # Insert if not found
            )
            if result.upserted_id:
                print(f"Inserted new data for {symbol} ({name}) on {record['Date']}")
            elif result.modified_count > 0:
                 print(f"Updated existing data for {symbol} ({name}) on {record['Date']}")

    except errors.DuplicateKeyError:
        print(f"Duplicate key error for {symbol} on {record['Date']}. Skipping.")
    except Exception as e:
        print(f"An unexpected error occurred while storing data for {symbol} ({name}): {e}")



def main():
    """
    Main function to fetch and store historical commodity data.
    """
    db = connect_to_mongodb(MONGO_URI, DB_NAME)
    if db is None:
        return  # Exit if database connection failed
    
    create_indexes(db, COLLECTION_NAME) #create the indexes

    # Get today's date and calculate the start date (e.g., one year ago)
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365)  # Fetch data for the past year

    for symbol, name in symbols_and_names:
        print(f"Fetching data for {symbol} ({name})...")
        data = get_historical_data(symbol, name, start_date, end_date)
        if data is not None:
            store_data_in_mongodb(db, COLLECTION_NAME, data, symbol, name)
        time.sleep(1)  # Add a small delay to avoid hitting rate limits


if __name__ == "__main__":
    main()
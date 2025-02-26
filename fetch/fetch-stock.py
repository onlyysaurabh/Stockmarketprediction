import yfinance as yf
from pymongo import MongoClient, errors
import pandas as pd
from datetime import datetime, timezone

# --- MongoDB Connection ---
MONGO_URI = "mongodb://localhost:27017/"  # Replace with your MongoDB connection string
# Use a connection pool for better performance and resource management
client = MongoClient(MONGO_URI, maxPoolSize=10)  # Adjust maxPoolSize as needed
db = client["stock_market_db"]  # Replace with your database name
stock_data_collection = db["stock_data"]

# Create indexes to improve query performance
stock_data_collection.create_index("Symbol")  # Index on Symbol for faster lookups
stock_data_collection.create_index([("Symbol", 1), ("Date", 1)], unique=True)  # Composite index for upserts

def fetch_and_store_stock_data(stock_symbol):
    """Fetches stock data from yfinance and stores it in MongoDB using upsert."""
    try:
        stock = yf.Ticker(stock_symbol)
        data = stock.history(period="max")

        # Check if data is empty
        if data.empty:
            print(f"No data found for {stock_symbol}")
            return

        # Format data for MongoDB
        formatted_data = []
        for index, row in data.iterrows():
            # Convert Timestamp to datetime and make it timezone-aware
            date_obj = index.to_pydatetime().replace(tzinfo=timezone.utc)
            
            #Check to make sure the types are correct before storing to avoid a db error
            try:
                formatted_data.append({
                    "Date": date_obj,
                    "Open": float(row['Open']),
                    "High": float(row['High']),
                    "Low": float(row['Low']),
                    "Close": float(row['Close']),
                    "Volume": int(row['Volume']),
                    "Dividends": float(row['Dividends']) if 'Dividends' in row else 0.0,
                    "Stock Splits": float(row['Stock Splits']) if 'Stock Splits' in row else 0.0,
                })
            except KeyError as e:
                print(f"Missing key in data for {stock_symbol}: {e}.  Row: {row}")
                continue  # Skip this row, or handle the missing data appropriately
            except (ValueError, TypeError) as e:
                print(f"Invalid data type for {stock_symbol}: {e}. Row: {row}")
                continue # Or handle differently


        # Use update_one with upsert=True to handle both insert and update
        for record in formatted_data:
            try:
                result = stock_data_collection.update_one(
                    {"Symbol": stock_symbol, "Date": record["Date"]},  # Filter by symbol AND date
                    {"$set": record},  # Update or set the entire record
                    upsert=True,  # Insert if not found
                )
                if result.upserted_id:
                    print(f"Inserted new data for {stock_symbol} on {record['Date']}")
                elif result.modified_count > 0:
                    print(f"Updated existing data for {stock_symbol} on {record['Date']}")
            except errors.DuplicateKeyError:
                print(f"Duplicate key error for {stock_symbol} on {record['Date']}. Skipping.")
                continue # Important to continue, so other entries can process.
            except Exception as e:
                print(f"Error during upsert for {stock_symbol}: {e}")
                continue

    except Exception as e:
        print(f"Error fetching/storing stock data for {stock_symbol}: {e}")



def process_stock_symbols(file_path='stocks.csv'):
    """Reads stock symbols from a CSV file and processes each one."""
    try:
        stocks_df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    if 'Symbol' not in stocks_df.columns:
        print("Error: 'Symbol' column not found in the CSV file.")
        return

    for index, row in stocks_df.iterrows():
        stock_symbol = row['Symbol'].strip()
        if not stock_symbol:  # Check for empty symbols
            print("Warning: Found an empty stock symbol in the CSV. Skipping.")
            continue
        fetch_and_store_stock_data(stock_symbol)

if __name__ == "__main__":
    process_stock_symbols()  # Call the processing function
    client.close()  # Close the MongoDB connection
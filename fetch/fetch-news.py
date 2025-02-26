import datetime
import json
import time
from GoogleNews import GoogleNews
from pymongo import MongoClient
from transformers import pipeline
from dateutil.relativedelta import relativedelta
import pandas as pd

# --- MongoDB Connection ---
MONGO_URI = "mongodb://localhost:27017/"  # Replace with your MongoDB connection string
client = MongoClient(MONGO_URI)
db = client["stock_market_db"]  # Replace with your database name
news_data_collection = db["news_data"]

# --- FinBERT Pipeline ---
classifier = pipeline('sentiment-analysis', model='ProsusAI/finbert')

# --- JSON file for tracking progress ---
PROGRESS_FILE = "news_fetch_progress.json"

def fetch_and_store_news(stock_symbol, stock_name, start_date, end_date):
    """
    Fetches news from Google News within a date range,
    analyzes sentiment using FinBERT, and stores the data in MongoDB.
    """
    try:
        googlenews = GoogleNews(lang='en')
        googlenews.set_time_range(start_date, end_date)  # Set the date range

        search_query = f"{stock_symbol} {stock_name}"
        googlenews.search(search_query)
        news_results = googlenews.results()

        news_data = []
        for news in news_results:
            title = news['title']
            date_str = news['date']
            link = news['link']

            # Parse the date string and convert to datetime object
            date = parse_date(date_str)
            # Store as datetime object, not string
            formatted_date = date if date else None

            # Get sentiment score using FinBERT
            sentiment = get_sentiment(title)

            news_data.append({
                "symbol": stock_symbol,
                "date": formatted_date,  # Store the datetime object
                "title": title,
                "link": link,
                "sentiment": sentiment['label'],  # Store both label
                "sentiment_score": sentiment['score'] # and score
            })

        # Store/Update the news data in MongoDB using upsert
        if news_data:
            for item in news_data:
                # Check if a news item with the same link already exists
                existing_item = news_data_collection.find_one({"link": item["link"]})
                if existing_item:
                    # Update the existing document
                    news_data_collection.update_one(
                        {"link": item["link"]},
                        {"$set": item}  # Update all fields
                    )
                    print(f"Updated news item for {stock_symbol}: {item['title']}")
                else:
                    # Insert the new document
                    news_data_collection.insert_one(item)
                    print(f"Inserted news item for {stock_symbol}: {item['title']}")

            print(f"News data for {stock_symbol} processed and stored/updated in MongoDB.")
        else:
            print(f"No news found for {stock_symbol}.")

    except Exception as e:
        print(f"Error fetching/analyzing news for {stock_symbol}: {e}")

    finally:
        client.close()

def parse_date(date_str):
    """Parses the date string into a datetime object."""
    try:
        parts = date_str.split()

        if len(parts) >= 3 and parts[1] in ('hour', 'hours', 'minute', 'minutes', 'day', 'days', 'week', 'weeks', 'month', 'months'):
            num_units = int(parts[0])
            unit = parts[1].rstrip('s')  # Remove 's'

            if unit == 'hour':
                date = datetime.datetime.now() - relativedelta(hours=num_units)
            elif unit == 'minute':
                date = datetime.datetime.now() - relativedelta(minutes=num_units)
            elif unit == 'day':
                date = datetime.datetime.now() - relativedelta(days=num_units)
            elif unit == 'week':
                date = datetime.datetime.now() - relativedelta(weeks=num_units)
            elif unit == 'month':
                date = datetime.datetime.now() - relativedelta(months=num_units)
            else:
                date = None
        else:
            # Handle other date formats (e.g., "Oct 18, 2024")
             date = datetime.datetime.strptime(date_str, "%b %d, %Y")

        return date

    except ValueError:
        print(f"Error parsing date: {date_str}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while parsing: {e}")
        return None

def get_sentiment(title):
    """Gets the sentiment score for the given title using FinBERT."""
    try:
        result = classifier(title)[0]
        return result  # Return the entire dictionary
    except Exception as e:
        print(f"Error getting sentiment for '{title}': {e}")
        return {"label": "Neutral", "score": 0.0}  # Return a neutral sentiment

def load_progress():
    """Loads progress from the JSON file."""
    try:
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}  # Return an empty dictionary if the file doesn't exist

def save_progress(progress):
    """Saves progress to the JSON file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)

if __name__ == "__main__":
    try:
        stocks_df = pd.read_csv('stocks.csv')  # Read stock symbols and names from CSV
    except Exception as e:
        print(f"Error reading stocks.csv: {e}")
        exit()

    # Check if 'Symbol' and 'Name' columns exist
    if not ('Symbol' in stocks_df.columns and 'Name' in stocks_df.columns):
        print("Error: 'Symbol' and/or 'Name' columns not found in stocks.csv")
        exit()

    # Specify the date range in mm/dd/yyyy format (for news data)
    start_date = "02/01/2024"  # Corrected date
    end_date = "02/07/2024"  # Corrected date

    progress = load_progress()  # Load progress from JSON

    # Iterate through all stocks (no skipping based on progress file)
    for index, row in stocks_df.iterrows():
        stock_symbol = row['Symbol']
        stock_name = row['Name']

        # Check if this stock has been processed *for this specific date range*.
        if progress.get(stock_symbol) == f"{start_date}-{end_date}":
            print(f"Skipping {stock_symbol} (already processed for this date range).")
            continue  # Skip to the next stock

        fetch_and_store_news(stock_symbol, stock_name, start_date, end_date)
        progress[stock_symbol] = f"{start_date}-{end_date}"  # Mark as done for this date range
        save_progress(progress)  # Save progress

        # Add a sleep timer to prevent IP blocking
        time.sleep(5)  # Sleep for 5 seconds (adjust as needed)
import yfinance as yf
from pymongo import MongoClient
import pandas as pd
import numpy as np  # Import numpy for NaN handling

# --- MongoDB Connection ---
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["stock_market_db"]
stock_info_collection = db["stock_info"]
financials_collection = db["stock_financials"]  # Separate collection for financials
cashflow_collection = db["stock_cashflow"]
balancesheet_collection = db["stock_balancesheet"]
recommendations_collection = db["stock_recommendations"] # New collection
sustainability_collection = db["stock_sustainability"] # New collection
institutional_holders_collection = db["stock_institutional_holders"] # New collection


def fetch_and_store_stock_info(stock_symbol):
    """Fetches detailed stock information and stores it in MongoDB."""
    try:
        stock = yf.Ticker(stock_symbol)

        # --- Basic Stock Info ---
        info = stock.info

        # Handle missing data gracefully using .get() and providing a default value (None)
        info_dict = {
            "symbol": stock_symbol,
            "companyName": info.get('longName'),
            "sector": info.get('sector'),
            "industry": info.get('industry'),
            "marketCap": info.get('marketCap'),
            "trailingPE": info.get('trailingPE'),
            "forwardPE": info.get('forwardPE'),
            "dividendYield": info.get('dividendYield'),
            "website": info.get('website'),
            "phone": info.get('phone'),
            "address": info.get('address1'),  # Assuming you want the main address
            "city": info.get('city'),
            "state": info.get('state'),
            "zip": info.get('zip'),
            "country": info.get('country'),
            "fullTimeEmployees": info.get('fullTimeEmployees'),
            "longBusinessSummary": info.get('longBusinessSummary'),
            "institutionalOwnershipPct": info.get('heldPercentInstitutions'),  # Institutional Ownership %
            "ceoName": "",  # Placeholder , will update below
            "recommendationKey": info.get('recommendationKey'), # Overall recommendation
            "targetPrice": info.get("targetMeanPrice") # Average target price
        }

        # ---  Get CEO Name (more reliable method) ---
        try:
            major_holders = stock.get_major_holders()
            # The following lines of code finds the name of the ceo.
            # It searches for the row that has "Name" and get's the value for that row.
            if major_holders is not None and 0 in major_holders.index and 1 in major_holders.columns:
                ceo_row = major_holders[major_holders[0].str.contains("Name", na=False)]
                if not ceo_row.empty:
                    info_dict["ceoName"] = ceo_row[1].iloc[0]  # gets the value of the first row

        except Exception as e:
            print(f"  Could not fetch CEO name reliably for {stock_symbol}: {e}")

        # Store basic info
        stock_info_collection.update_one({"symbol": stock_symbol}, {"$set": info_dict}, upsert=True)
        print(f"Basic stock info for {stock_symbol} stored/updated.")

        # --- Financial Statements (Income Statement, Balance Sheet, Cash Flow) ---

        # Income Statement
        try:
            income_stmt = stock.income_stmt
            if not income_stmt.empty:
                for index, row in income_stmt.iterrows():
                    # --- FIX: Check if index is already a string ---
                    date_str = index.strftime("%Y-%m-%d") if isinstance(index, pd.Timestamp) else str(index)
                    financial_data = {
                        "symbol": stock_symbol,
                        "date": date_str,
                        "type": "income_statement",
                        "Revenue": row.get('Total Revenue'),  # Use .get() for safety
                        "CostOfRevenue": row.get('Cost Of Revenue'),
                        "GrossProfit": row.get('Gross Profit'),
                        "OperatingExpenses": row.get('Operating Expense'),
                        "OperatingIncome": row.get('Operating Income'),
                        "NetIncome": row.get("Net Income"),
                        "BasicEPS": row.get("Basic EPS"),
                        "EBITDA": row.get("EBITDA")
                    }
                    # Store/update income statement data
                    financials_collection.update_one(
                        {"symbol": stock_symbol, "date": date_str, "type": "income_statement"},
                        {"$set": financial_data},
                        upsert=True,
                    )
                print(f"  Income statement data for {stock_symbol} stored/updated.")
        except Exception as e:
            print(f"  Could not fetch/store income statement for {stock_symbol}: {e}")

        # Balance Sheet
        try:
            balance_sheet = stock.balance_sheet
            if not balance_sheet.empty:
                for index, row in balance_sheet.iterrows():
                    # --- FIX: Check if index is already a string ---
                    date_str = index.strftime("%Y-%m-%d") if isinstance(index, pd.Timestamp) else str(index)
                    balance_sheet_data = {
                        "symbol": stock_symbol,
                        "date": date_str,
                        "type": "balance_sheet",
                        "TotalAssets": row.get('Total Assets'),
                        "TotalLiabilities": row.get('Total Liabilities Net Minority Interest'),
                        "StockholdersEquity": row.get('Stockholders Equity'),
                        "Inventory": row.get('Inventory'),
                        "AccountReceivables": row.get("Accounts Receivable")

                    }
                    balancesheet_collection.update_one(
                        {"symbol": stock_symbol, "date": date_str, "type": "balance_sheet"},
                        {"$set": balance_sheet_data},
                        upsert=True,
                    )
                print(f" Balance sheet data for {stock_symbol} stored/updated.")
        except Exception as e:
            print(f" could not fetch/store balancesheet for {stock_symbol}:{e}")

        # Cash Flow Statement
        try:
            cash_flow = stock.cashflow
            if not cash_flow.empty:
                for index, row in cash_flow.iterrows():
                    # --- FIX: Check if index is already a string ---
                    date_str = index.strftime("%Y-%m-%d") if isinstance(index, pd.Timestamp) else str(index)
                    cash_flow_data = {
                        "symbol": stock_symbol,
                        "date": date_str,
                        "type": "cash_flow",
                        "CashFromOperations": row.get('Operating Cash Flow'),
                        "CapitalExpenditures": row.get('Capital Expenditure'),
                        "CashFromInvesting": row.get('Investing Cash Flow'),
                        "FreeCashFlow": row.get("Free Cash Flow")
                    }
                    cashflow_collection.update_one(
                        {"symbol": stock_symbol, "date": date_str, "type": "cash_flow"},
                        {"$set": cash_flow_data},
                        upsert=True
                    )
                print(f" Cash Flow Statement data for {stock_symbol} stored/updated.")
        except Exception as e:
            print(f" Could not fetch/store cash flow statement for {stock_symbol}: {e}")

        # ---  Price Ratios (calculated where possible) ---
        try:
            # Some ratios can be calculated if we have the necessary data:
            financials_cursor = financials_collection.find({"symbol": stock_symbol, "type": "income_statement"}).sort(
                "date", -1).limit(1)  # Most recent
            financials = list(financials_cursor)

            cashflow_cursor = cashflow_collection.find({"symbol": stock_symbol, "type": "cash_flow"}).sort("date",
                                                                                                       -1).limit(1)
            cashflow = list(cashflow_cursor)

            balancesheet_cursor = balancesheet_collection.find({"symbol": stock_symbol, "type": "balance_sheet"}).sort(
                "date", -1).limit(1)
            balancesheet = list(balancesheet_cursor)

            # Make sure that financials, cashflow and balance_sheet have values.
            if financials and cashflow and balancesheet:
                financials = financials[0]  # unpacks the array
                cashflow = cashflow[0]
                balancesheet = balancesheet[0]

                # Use .get() to avoid errors
                price_ratios = {
                    "symbol": stock_symbol,
                    "PriceToEarnings": info.get("trailingPE"),  # From basic info if available
                    "ForwardPriceToEarnings": info.get("forwardPE"),  # From basic info
                    "PriceToFreeCashFlow": (info.get("marketCap") / cashflow.get("FreeCashFlow")) if cashflow.get(
                        "FreeCashFlow") else None,  # Calculate if possible
                    "PriceToBook": info.get("priceToBook"),
                    "PriceToSales": (info.get("marketCap") / financials.get("Revenue")) if financials.get(
                        "Revenue") else None,
                    "EVEBITDA": info.get("enterpriseToEbitda"),
                    "GrossMargin": (financials.get("GrossProfit") / financials.get("Revenue")) if financials.get(
                        "GrossProfit") and financials.get("Revenue") else None,
                    "ProfitMargin": (financials.get("NetIncome") / financials.get("Revenue")) if financials.get(
                        "NetIncome") and financials.get("Revenue") else None,
                    "BasicEPSGrowth": ((financials.get("BasicEPS") - financials_collection.find_one(
                        {"symbol": stock_symbol, "type": "income_statement"}, sort=[("date", 1)]).get("BasicEPS",
                                                                                                   0)) / abs(
                        financials_collection.find_one({"symbol": stock_symbol, "type": "income_statement"},
                                                      sort=[("date", 1)]).get("BasicEPS", 1))) if financials.get(
                        "BasicEPS") else None,
                    "EquityReturn": (financials.get("NetIncome") / balancesheet.get("StockholdersEquity")) if financials.get(
                        "NetIncome") and balancesheet.get("StockholdersEquity") else None,
                    "ReturnOnAssets": (financials.get("NetIncome") / balancesheet.get("TotalAssets")) if financials.get(
                        "NetIncome") and balancesheet.get("TotalAssets") else None,
                    "ReturnOnCapital": (financials.get("OperatingIncome") / balancesheet.get(
                        "TotalAssets")) if financials.get("OperatingIncome") and balancesheet.get(
                        "TotalAssets") else None,  # operating income is commonly used as proxy of invested capital
                    "AccountReceivablesTurnover": (financials.get("Revenue") / balancesheet.get(
                        "AccountReceivables")) if financials.get("Revenue") and balancesheet.get(
                        "AccountReceivables") else None,
                }

                # Clean values
                for key, value in price_ratios.items():
                    if isinstance(value, float) and (np.isinf(value) or np.isnan(value)):
                        price_ratios[key] = None

                stock_info_collection.update_one({"symbol": stock_symbol}, {"$set": price_ratios}, upsert=True)
                print(f"  Price ratios for {stock_symbol} calculated and stored/updated.")


        except Exception as e:
            print(f"  Error calculating/storing price ratios for {stock_symbol}: {e}")


        # --- Analyst Recommendations ---
        try:
            recommendations = stock.recommendations
            if recommendations is not None and not recommendations.empty:
                # Convert dates and ensure the DataFrame has the expected columns.
                recommendations = recommendations.reset_index()
                if 'Date' in recommendations.columns:
                    recommendations['Date'] = pd.to_datetime(recommendations['Date'])
                    recommendations_data = []
                    for _, row in recommendations.iterrows():
                         # Fill missing columns safely
                        rec_data = {
                            "symbol": stock_symbol,
                            "date": row['Date'].strftime("%Y-%m-%d"), #Consistent date
                            "firm": row.get('Firm', None),  # Use .get()
                            "toGrade": row.get('To Grade', None),
                            "fromGrade": row.get('From Grade', None),
                            "action": row.get('Action', None),
                        }
                        recommendations_data.append(rec_data)


                    # Bulk insert/update.  Much more efficient.
                    if recommendations_data: # Only insert if there's data
                        for rec in recommendations_data:
                            recommendations_collection.update_one(
                                {"symbol": rec["symbol"], "date": rec["date"], "firm": rec["firm"]},
                                {"$set": rec},
                                upsert=True
                            )

                    print(f"  Analyst recommendations for {stock_symbol} stored/updated.")
                else:
                    print(f"  'Date' column missing in recommendations for {stock_symbol}.")

        except Exception as e:
            print(f"  Error fetching/storing recommendations for {stock_symbol}: {e}")


        # --- Sustainability Data (ESG) ---
        try:
            sustainability = stock.sustainability
            if sustainability is not None and not sustainability.empty :
                sustainability = sustainability.reset_index() # Reset index to iterate correctly

                sustainability_data = []
                for _, row in sustainability.iterrows():
                    sust_data = {
                        "symbol": stock_symbol,
                        "type": row.get("index",None), # Use get to prevent Key error
                        "value": row.get("Value",None)
                    }
                    sustainability_data.append(sust_data)

                # Store in MongoDB (upsert based on symbol and type)
                if sustainability_data:
                    for item in sustainability_data:
                         sustainability_collection.update_one(
                            {"symbol": item["symbol"], "type": item["type"]},
                            {"$set": item},
                            upsert=True  # Important: creates if not exists, updates if exists
                        )
                print(f"  Sustainability data for {stock_symbol} stored/updated.")

        except Exception as e:
            print(f"  Error fetching/storing sustainability data for {stock_symbol}: {e}")

        # --- Major Holders ---
        try:
            major_holders = stock.get_major_holders()
            if major_holders is not None and not major_holders.empty:
                major_holders_data = []
                for index, row in major_holders.iterrows():
                    holder_data = {
                        "symbol": stock_symbol,
                        "holder_type": row.get(0, None),
                        "percentage": row.get(1, None)

                    }
                    major_holders_data.append(holder_data)
                if major_holders_data:
                    for holder in major_holders_data:
                        institutional_holders_collection.update_one(
                            {"symbol": holder["symbol"], "holder_type": holder["holder_type"]},
                            {"$set": holder},
                            upsert = True
                        )
                print(f" Major holders data for {stock_symbol} stored/updated")
        except Exception as e:
            print(f"  Error fetching/storing major holders data for {stock_symbol}: {e}")



    except Exception as e:
        print(f"Error fetching/storing data for {stock_symbol}: {e}")


if __name__ == "__main__":
    try:
        stocks_df = pd.read_csv('stocks.csv')
    except Exception as e:
        print(f"Error reading stocks.csv: {e}")
        exit()
    # Ensure Symbol column
    if 'Symbol' not in stocks_df.columns:
        print("Error: 'Symbol' column not found in stocks.csv")
        exit()

    for index, row in stocks_df.iterrows():
        stock_symbol = row['Symbol']
        fetch_and_store_stock_info(stock_symbol)

    client.close()
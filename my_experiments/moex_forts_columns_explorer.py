#!/usr/bin/env python3
"""
MOEX FORTS History Columns Explorer

This script demonstrates how to fetch and display all available columns from the MOEX FORTS history endpoint.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path so we can import the tvr_service modules
src_path = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(src_path))

import pandas as pd
import requests
import time
from typing import Any, Dict, Optional
from datetime import date as _date

# Define constants and helper functions
BASE_URL = "https://iss.moex.com"
HIST_ENDPOINT = "/iss/history/engines/futures/markets/forts/securities.json"
MAX_RETRIES = 5
RETRY_DELAY = 1.0

def _iss_get(url_path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Perform GET request to MOEX ISS with simple retry logic."""
    url = BASE_URL + url_path
    headers = {
        "User-Agent": "tvr-portfolio-agent",
        "Accept": "application/json",
    }
    last_exc: Optional[Exception] = None
    for _ in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_exc = exc
            time.sleep(RETRY_DELAY)
    raise RuntimeError(f"MOEX ISS request failed after retries: {url} params={params}\n{last_exc}")

def main():
    print("Fetching data from MOEX FORTS history endpoint...")
    
    # Get data for a specific date (you can change this to any date you want to explore)
    sample_date = "2023-10-01"
    
    try:
        payload = _iss_get(HIST_ENDPOINT, params={"date": sample_date, "limit": 1})
        print(f"Successfully fetched data for {sample_date}")
    except Exception as e:
        print(f"Error fetching data: {e}")
        # Try with today's date if the specific date fails
        try:
            sample_date = _date.today().isoformat()
            payload = _iss_get(HIST_ENDPOINT, params={"date": sample_date, "limit": 1})
            print(f"Successfully fetched data for today ({sample_date})")
        except Exception as e2:
            print(f"Error fetching data for today: {e2}")
            payload = None
    
    # Extract and display the available columns
    if payload:
        history_section = payload.get("history", {})
        columns = history_section.get("columns", [])
        
        print(f"\nAvailable columns in MOEX FORTS history data:\n")
        for i, column in enumerate(columns, 1):
            print(f"{i:2d}. {column}")
        
        print(f"\n\nTotal number of columns: {len(columns)}")
        
        # Let's also look at the actual data structure
        data_rows = history_section.get("data", [])
        if data_rows:
            print(f"\n\nSample data row (first row):\n")
            sample_row = data_rows[0]
            for i, (column, value) in enumerate(zip(columns, sample_row), 1):
                print(f"{i:2d}. {column}: {value}")
    else:
        print("No data to display")
    
    # Let's also create a DataFrame to better visualize the data structure
    if payload and 'history' in payload:
        history_section = payload.get("history", {})
        columns = history_section.get("columns", [])
        data_rows = history_section.get("data", [])
        
        if columns and data_rows:
            df = pd.DataFrame(data_rows, columns=columns)
            print("\nDataFrame with sample data:")
            print(df.head())
            print(f"\nDataFrame shape: {df.shape}")
            print(f"\nColumn data types:")
            print(df.dtypes)
        else:
            print("No data to create DataFrame")
    else:
        print("No payload data available")

if __name__ == "__main__":
    main()
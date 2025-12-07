import requests
import pandas as pd
import os
from datetime import datetime

# Configuration
LATITUDE = 13.75
LONGITUDE = 100.50
START_DATE = "2021-08-01"  # Matching start of Traffy data
END_DATE = "2025-01-31"    # Matching end of Traffy data
OUTPUT_PATH = "data/raw/weather_bangkok.csv"

def fetch_weather_data():
    """
    Fetches historical weather data from Open-Meteo API for Bangkok.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "Asia/Bangkok"
    }

    print(f"Fetching weather data for Bangkok ({START_DATE} to {END_DATE})...")
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract daily data
        daily_data = data.get("daily", {})
        
        df = pd.DataFrame({
            "date": daily_data.get("time"),
            "weather_code": daily_data.get("weather_code"),
            "temperature_max": daily_data.get("temperature_2m_max"),
            "temperature_min": daily_data.get("temperature_2m_min"),
            "rainfall_mm": daily_data.get("precipitation_sum")
        })
        
        # Calculate average temp for convenience
        df["temperature_avg"] = (df["temperature_max"] + df["temperature_min"]) / 2
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        
        # Save to CSV
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ Weather data saved to {OUTPUT_PATH}")
        print(f"   Total records: {len(df)}")
        print(df.head())
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data: {e}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    fetch_weather_data()

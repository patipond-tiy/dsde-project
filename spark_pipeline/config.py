import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Input Paths
TRAFFY_DATA_PATH = os.path.join(BASE_DIR, "bangkok_traffy.csv")
WEATHER_DATA_PATH = os.path.join(DATA_DIR, "raw", "weather_bangkok.csv")

# Output Paths
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "traffy_features.parquet")

# Spark Config
SPARK_APP_NAME = "TraffyFonduePrediction"
SPARK_MASTER = "local[*]"  # Run locally with all cores

# Schema Constants
TRAFFY_COLUMNS = [
    "ticket_id", "type", "organization", "comment", "coords", 
    "photo", "after_photo", "address", "subdistrict", "district", 
    "province", "timestamp", "state", "star", "count_reopen", "last_activity"
]

# We need to ensure these match the actual CSV structure, 
# but Spark can infer schema or we can enforce it.
# Based on previous analysis, we know some column names might differ slightly 
# (e.g. 'ticket_id' vs 'ticket_id', etc). 
# We will use header=True in Spark read.

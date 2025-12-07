import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spark_pipeline import config, feature_engineering

def main():
    # Initialize Spark
    print(f"Initializing Spark Session: {config.SPARK_APP_NAME}")
    spark = SparkSession.builder \
        .appName(config.SPARK_APP_NAME) \
        .master(config.SPARK_MASTER) \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")

    # 1. Load Data
    print(f"Loading Traffy data from {config.TRAFFY_DATA_PATH}...")
    traffy_df = spark.read.csv(config.TRAFFY_DATA_PATH, header=True, inferSchema=True, quote='"', escape='"', multiLine=True)
    
    # 2. Clean Data
    print("Cleaning Traffy data...")
    cleaned_df = feature_engineering.clean_data(traffy_df)
    
    # 3. Feature Engineering
    print("Extracting features...")
    features_df = feature_engineering.extract_temporal_features(cleaned_df)
    final_df = feature_engineering.extract_text_features(features_df)
    
    # 5. Select Columns for Parquet
    # We select columns relevant for ML and Visualization
    columns_to_keep = [
        "ticket_id", "type", "comment", "organization", "district", "subdistrict", "province",
        "timestamp", "last_activity", "state",
        "resolution_time_days", "longitude", "latitude",
        "hour", "day_of_week", "day_of_month", "month", "year", "is_weekend",
        "comment_length",
        "is_rainy_season"
    ]
    
    # Check if columns exist before selecting (weather might be missing if join failed completely, though it shouldn't)
    existing_cols = [c for c in columns_to_keep if c in final_df.columns]
    final_df = final_df.select(existing_cols)

    # 6. Save to Parquet
    print(f"Saving processed data to {config.PROCESSED_DATA_PATH}...")
    # Coalesce to 1 to have a single file (optional, good for small-medium data)
    final_df.coalesce(1).write.mode("overwrite").parquet(config.PROCESSED_DATA_PATH)
    
    print("✅ Pipeline completed successfully.")
    
    # Show sample
    final_df.show(5)
    print(f"Total processed records: {final_df.count()}")

    spark.stop()

if __name__ == "__main__":
    main()

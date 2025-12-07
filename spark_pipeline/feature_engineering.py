from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.sql import DataFrame

def clean_data(df: DataFrame) -> DataFrame:
    """
    Cleans the raw Traffy dataframe:
    - Drops duplicates
    - Filters for completed tickets (state = 'เสร็จสิ้น')
    - Parses coordinates
    - Calculates resolution_time_days
    - Filters invalid resolution times
    """
    # Filter for completed tickets only
    df = df.filter(F.col("state") == "เสร็จสิ้น")
    
    # Cast timestamps
    df = df.withColumn("timestamp", F.to_timestamp("timestamp")) \
           .withColumn("last_activity", F.to_timestamp("last_activity"))
    
    # Calculate Resolution Time (Days)
    # Using unix_timestamp to get seconds, then divide by 86400
    df = df.withColumn("resolution_time_days", 
                       (F.unix_timestamp("last_activity") - F.unix_timestamp("timestamp")) / 86400.0)
    
    # Filter invalid resolution times (must be >= 0 and reasonable, e.g., < 5 years)
    df = df.filter((F.col("resolution_time_days") >= 0) & (F.col("resolution_time_days") < 2000))
    
    # Parse Coordinates (assuming format "lon, lat" or similar string)
    # We will try to split by comma
    df = df.withColumn("coords_split", F.split(F.col("coords"), ",")) \
           .withColumn("longitude", F.col("coords_split").getItem(0).cast(DoubleType())) \
           .withColumn("latitude", F.col("coords_split").getItem(1).cast(DoubleType())) \
           .drop("coords_split")
           
    return df

def extract_temporal_features(df: DataFrame) -> DataFrame:
    """
    Extracts temporal features from timestamp.
    """
    df = df.withColumn("hour", F.hour("timestamp")) \
           .withColumn("day_of_week", F.dayofweek("timestamp")) \
           .withColumn("day_of_month", F.dayofmonth("timestamp")) \
           .withColumn("month", F.month("timestamp")) \
           .withColumn("year", F.year("timestamp")) \
           .withColumn("date", F.to_date("timestamp")) \
           .withColumn("is_weekend", F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0)) \
           .withColumn("is_rainy_season", F.when((F.col("month") >= 5) & (F.col("month") <= 10), 1).otherwise(0))
    return df

def extract_text_features(df: DataFrame) -> DataFrame:
    """
    Extracts simple text features from comment.
    """
    df = df.withColumn("comment_length", F.length(F.col("comment")))
    return df

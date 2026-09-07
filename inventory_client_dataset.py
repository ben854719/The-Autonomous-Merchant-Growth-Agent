!pip install polars

import polars as pl

# Load source data if missing and prepare cleaned_df
if 'order_id_order_date_36' not in globals():
    order_id_order_date_36 = pl.read_csv('order_id_order_date_36.csv')

cleaned_df = order_id_order_date_36.with_columns([
    pl.col("customer_name").str.to_lowercase(),
    pl.col("channel").str.to_lowercase(),
    pl.col("status").str.to_lowercase()
]).unique()

# Extract time-series features using flexible parsing to handle mixed formats
cleaned_df = cleaned_df.with_columns([
    pl.col("order_date").str.to_datetime(strict=False).alias("order_datetime")
])

# If some dates failed (returned null), try a secondary parse for the DD-MM-YYYY format
cleaned_df = cleaned_df.with_columns(
    pl.when(pl.col("order_datetime").is_null())
    .then(pl.col("order_date").str.strptime(pl.Datetime, format="%d-%m-%Y", strict=False))
    .otherwise(pl.col("order_datetime"))
    .alias("order_datetime")
)

# Derive temporal features from the successfully parsed datetime
cleaned_df = cleaned_df.with_columns([
    pl.col("order_datetime").dt.year().alias("order_year"),
    pl.col("order_datetime").dt.month().alias("order_month"),
    pl.col("order_datetime").dt.day().alias("order_day"),
    pl.col("order_datetime").dt.weekday().alias("order_weekday"),
    pl.col("order_datetime").dt.week().alias("order_week"),
    pl.col("order_datetime").dt.quarter().alias("order_quarter")
])

display(cleaned_df.head())

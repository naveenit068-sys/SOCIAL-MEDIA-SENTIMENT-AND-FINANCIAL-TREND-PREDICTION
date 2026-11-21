import pandas as pd

def combine_sentiment_and_stock(sentiment_df, stock_df, company="TCS"):
    # The available stock-price file represents TCS.NS, so only use TCS posts.
    selected = sentiment_df[sentiment_df["Company"] == company.upper()].copy()
    if selected.empty:
        raise ValueError(
            f"No posts found for {company}. The stock file is for {company}; "
            "check the Company values in social_media_posts.csv."
        )

    daily = selected.groupby("Date").agg(
        daily_sentiment=("sentiment_score", "mean"),
        post_count=("Post", "count"),
        positive_ratio=("sentiment", lambda x: (x == "Positive").mean()),
        negative_ratio=("sentiment", lambda x: (x == "Negative").mean()),
        neutral_ratio=("sentiment", lambda x: (x == "Neutral").mean())
    ).reset_index()

    stock = stock_df.sort_values("Date").copy()
    daily = daily.sort_values("Date").copy()
    combined = pd.merge_asof(stock, daily, on="Date", direction="backward")

    # Retain only dates up to the last date for which sentiment data exists.
    combined = combined[combined["Date"] <= daily["Date"].max()].copy()
    combined[["daily_sentiment", "post_count", "positive_ratio",
              "negative_ratio", "neutral_ratio"]] = combined[
        ["daily_sentiment", "post_count", "positive_ratio",
         "negative_ratio", "neutral_ratio"]
    ].fillna(0)

    return combined.reset_index(drop=True)

def create_financial_features(combined):
    df = combined.sort_values("Date").copy()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Previous_Return"] = df["Daily_Return"].shift(1)
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["Volatility"] = df["Daily_Return"].rolling(5).std()
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Next_Close"] = df["Close"].shift(-1)
    df["Target"] = (df["Next_Close"] > df["Close"]).astype(int)
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.dropna().reset_index(drop=True)
    if len(df) < 20:
        raise ValueError(
            "Not enough combined daily records to train and test the model. "
            "Use a larger dataset with more TCS posts and stock dates."
        )
    return df

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import load_social_posts, load_stock_data, OUTPUT_DIR
from src.sentiment_analysis import analyse_sentiment, get_trending_topics
from src.financial_analysis import combine_sentiment_and_stock, create_financial_features
from src.model_training import train_and_compare
from src.evaluation import print_model_evaluation
from src.visualization import save_all_visualizations

def main():
    print("=== Social Media Sentiment and Financial Trend Prediction ===")

    print("\n1. Loading and preprocessing datasets...")
    posts = load_social_posts()
    stock = load_stock_data()
    print(f"Social media posts: {len(posts)}")
    print(f"Stock records: {len(stock)}")

    print("\n2. Performing sentiment analysis...")
    sentiment = analyse_sentiment(posts)
    sentiment.to_csv(OUTPUT_DIR / "sentiment_analysis_results.csv", index=False)
    print(sentiment["sentiment"].value_counts().to_string())

    print("\n3. Identifying trending topics...")
    topics = get_trending_topics(posts)
    topics.to_csv(OUTPUT_DIR / "trending_topics.csv", index=False)

    print("\n4. Combining TCS sentiment with historical stock data...")
    combined = combine_sentiment_and_stock(sentiment, stock, company="TCS")
    combined.to_csv(OUTPUT_DIR / "combined_financial_sentiment_data.csv", index=False)

    print("\n5. Creating financial features and training models...")
    model_data = create_financial_features(combined)
    results, models, split = train_and_compare(model_data)
    results.to_csv(OUTPUT_DIR / "model_comparison_results.csv", index=False)

    latest = model_data.iloc[[-1]]
    enhanced_model = models["Prices + Social Sentiment"]
    prediction = int(enhanced_model.predict(latest[
        ["Open", "High", "Low", "Close", "Volume", "Daily_Return",
         "Previous_Return", "MA_5", "MA_10", "Volatility", "Volume_Change",
         "daily_sentiment", "positive_ratio", "negative_ratio",
         "neutral_ratio", "post_count"]
    ])[0])
    prediction_text = "UP" if prediction == 1 else "DOWN"
    pd.DataFrame([{
        "Date": latest["Date"].iloc[0],
        "Predicted_Next_Day_Direction": prediction_text
    }]).to_csv(OUTPUT_DIR / "financial_trend_prediction.csv", index=False)

    print("\n6. Evaluating models...")
    print_model_evaluation(results)

    print("\n7. Saving visualizations...")
    save_all_visualizations(sentiment, topics, combined, results)

    print("\nLatest model prediction:", prediction_text)
    print("\nCompleted. Check the outputs/ and visualizations/ folders.")
    print("Note: This is a synthetic-data academic demonstration, not investment advice.")

if __name__ == "__main__":
    main()

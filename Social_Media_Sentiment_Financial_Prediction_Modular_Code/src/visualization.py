from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
VIS_DIR = ROOT / "visualizations"
VIS_DIR.mkdir(exist_ok=True)

def save_all_visualizations(sentiment_df, topics_df, combined_df, results):
    # 1. Sentiment distribution
    counts = sentiment_df["sentiment"].value_counts()
    plt.figure(figsize=(7, 5))
    counts.reindex(["Positive", "Negative", "Neutral"], fill_value=0).plot(kind="bar")
    plt.title("Social Media Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Posts")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "sentiment_distribution.png", dpi=150)
    plt.close()

    # 2. Trending topics
    if not topics_df.empty:
        plt.figure(figsize=(9, 5))
        topics_df.sort_values("count").plot(
            x="topic", y="count", kind="barh", legend=False, figsize=(9, 5)
        )
        plt.title("Trending Topics in Social Media Posts")
        plt.xlabel("Frequency")
        plt.ylabel("Topic")
        plt.tight_layout()
        plt.savefig(VIS_DIR / "trending_topics.png", dpi=150)
        plt.close()

    # 3. Stock price trend
    plt.figure(figsize=(9, 5))
    plt.plot(combined_df["Date"], combined_df["Close"])
    plt.title("TCS Historical Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "stock_price_trend.png", dpi=150)
    plt.close()

    # 4. Model comparison
    plt.figure(figsize=(8, 5))
    results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1_Score"]].plot(
        kind="bar", figsize=(9, 5)
    )
    plt.title("Baseline vs Sentiment-Enhanced Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "model_comparison.png", dpi=150)
    plt.close()

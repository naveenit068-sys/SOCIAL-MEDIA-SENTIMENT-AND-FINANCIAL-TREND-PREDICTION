import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

def analyse_sentiment(posts_df):
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    sia = SentimentIntensityAnalyzer()
    result = posts_df.copy()
    result["sentiment_score"] = result["Post"].apply(
        lambda text: sia.polarity_scores(str(text))["compound"]
    )

    def label(score):
        if score >= 0.05:
            return "Positive"
        if score <= -0.05:
            return "Negative"
        return "Neutral"

    result["sentiment"] = result["sentiment_score"].apply(label)
    return result

def get_trending_topics(posts_df, top_n=15):
    texts = posts_df["Post"].fillna("").astype(str)
    if texts.str.strip().eq("").all():
        return pd.DataFrame(columns=["topic", "count"])
    vectorizer = CountVectorizer(stop_words="english", max_features=top_n)
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return pd.DataFrame(columns=["topic", "count"])
    counts = matrix.sum(axis=0).A1
    return (
        pd.DataFrame({"topic": vectorizer.get_feature_names_out(), "count": counts})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

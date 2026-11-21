from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

BASELINE_FEATURES = [
    "Open", "High", "Low", "Close", "Volume", "Daily_Return",
    "Previous_Return", "MA_5", "MA_10", "Volatility", "Volume_Change"
]
SENTIMENT_FEATURES = [
    "daily_sentiment", "positive_ratio", "negative_ratio",
    "neutral_ratio", "post_count"
]

def train_and_compare(df):
    results = []
    trained = {}
    split = int(len(df) * 0.8)
    if split <= 0 or split >= len(df):
        raise ValueError("Not enough rows for an 80/20 chronological train-test split.")

    y_train, y_test = df["Target"].iloc[:split], df["Target"].iloc[split:]

    for name, features in [
        ("Historical Prices Only", BASELINE_FEATURES),
        ("Prices + Social Sentiment", BASELINE_FEATURES + SENTIMENT_FEATURES)
    ]:
        X_train = df[features].iloc[:split]
        X_test = df[features].iloc[split:]
        model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42,
            class_weight="balanced"
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction, zero_division=0),
            "Recall": recall_score(y_test, prediction, zero_division=0),
            "F1_Score": f1_score(y_test, prediction, zero_division=0)
        }
        results.append(metrics)
        trained[name] = model

        filename = "baseline_model.joblib" if name == "Historical Prices Only" else "sentiment_enhanced_model.joblib"
        joblib.dump(model, MODEL_DIR / filename)

    return pd.DataFrame(results), trained, split

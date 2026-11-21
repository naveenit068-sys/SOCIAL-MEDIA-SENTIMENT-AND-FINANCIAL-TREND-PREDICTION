# Social Media Sentiment and Financial Trend Prediction

## Project overview
This academic proof-of-concept studies whether social-media sentiment can be combined with historical stock-price information to classify the next trading day's price direction.

## Project structure
- `app.py` — runs the complete workflow.
- `src/preprocessing.py` — loads and cleans the CSV datasets.
- `src/sentiment_analysis.py` — performs VADER sentiment analysis and extracts frequent topics.
- `src/financial_analysis.py` — combines TCS sentiment with stock data and creates financial features.
- `src/model_training.py` — trains and saves baseline and sentiment-enhanced Random Forest models.
- `src/evaluation.py` — reports evaluation metrics.
- `src/visualization.py` — creates project charts.
- `dataset/` — place `social_media_posts.csv` and `stock_data.csv` here.
- `models/` — saved trained model files.
- `outputs/` — generated CSV reports.
- `visualizations/` — generated charts.

## Dataset files required
Place these files inside `dataset/`:
1. `social_media_posts.csv` with date, company, and post-text columns. Accepted column names include `Date`, `Company`, `Post` or `Text`.
2. `stock_data.csv` with date, Open, High, Low, Close, and Volume columns.

The stock dataset should correspond to TCS (for example, TCS.NS). The program filters social posts to company `TCS` before combining them with TCS prices.

## Run locally
Install Python 3.10 or newer, then run:

```bash
pip install -r requirements.txt
python app.py
```

The first sentiment-analysis run may download the NLTK VADER lexicon.

## Method
- VADER compound sentiment score: Positive >= 0.05, Negative <= -0.05, otherwise Neutral.
- Daily sentiment features: average score, positive/negative/neutral ratios, and post count.
- Financial features: daily return, previous return, 5-day and 10-day moving averages, volatility, and volume change.
- Models: Random Forest baseline using historical prices and Random Forest enhanced with sentiment features.
- Evaluation: chronological 80% training / 20% testing split; accuracy, precision, recall, and F1-score.

## Important limitations
The sample social-media data used in the project may be synthetic. Repeated or synthetic posts do not represent real investor opinion. Date-only posts do not establish the exact time at which a post became available relative to a market close. Therefore, results are a demonstration of the workflow and must not be treated as evidence of a reliable trading signal or as investment advice. A stronger study should use timestamped, real posts and a strictly defined prediction cutoff.

## Outputs
The run creates CSV reports in `outputs/`, model files in `models/`, and charts in `visualizations/`.

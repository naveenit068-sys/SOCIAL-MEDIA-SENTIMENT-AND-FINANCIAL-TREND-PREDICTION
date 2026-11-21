def print_model_evaluation(results):
    print("\nMODEL EVALUATION (chronological 80/20 split)")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    baseline = results.loc[results["Model"] == "Historical Prices Only", "Accuracy"].iloc[0]
    enhanced = results.loc[results["Model"] == "Prices + Social Sentiment", "Accuracy"].iloc[0]
    print("\nAccuracy comparison:")
    print(f"Historical-prices-only baseline: {baseline:.4f}")
    print(f"Sentiment-enhanced model:        {enhanced:.4f}")
    if enhanced > baseline:
        print("The sentiment-enhanced model performed better on this test split.")
    elif enhanced < baseline:
        print("The historical-prices-only baseline performed better on this test split.")
    else:
        print("Both models achieved the same accuracy on this test split.")
    print("This is an academic proof-of-concept, not investment advice.")

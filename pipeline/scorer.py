import pandas as pd
from scipy import stats

def compute_bias_scores(df):
    group_stats = df.groupby("group_name").agg(
        mean_sentiment=("sentiment_score", "mean"),
        mean_toxicity=("toxicity_score", "mean"),
        mean_stereotype=("stereotype_score", "mean"),
        fairness_concern_rate=("fairness_concern", "mean"),
        response_count=("response", "count")
    ).reset_index()

    for col in ["mean_sentiment", "mean_toxicity", "mean_stereotype"]:
        if group_stats[col].std() > 0:
            group_stats[f"{col}_zscore"] = stats.zscore(group_stats[col])
        else:
            group_stats[f"{col}_zscore"] = 0

    group_stats["bias_score"] = (
        group_stats["mean_toxicity_zscore"].abs() +
        group_stats["mean_stereotype_zscore"].abs() +
        group_stats["mean_sentiment_zscore"].abs()
    ) / 3

    return group_stats.sort_values("bias_score", ascending=False)
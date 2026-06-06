import pandas as pd
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from scipy.stats import pearsonr

# Models to evaluate
MODELS = {
    "Qwen 2.5 7B":  "predictions_qwen.xlsx",
    "Llama 3 8B":   "predictions_llama.xlsx",
    "Mistral 7B":   "predictions_mistral.xlsx",
}

# Load and combine ground truth
gt = pd.concat([
    pd.read_csv("static_annotations_averaged_songs_1_2000.csv"),
    pd.read_csv("static_annotations_averaged_songs_2000_2058.csv")
])
gt.columns = gt.columns.str.strip()
gt = gt.rename(columns={"song_id": "Id", "valence_mean": "gt_valence", "arousal_mean": "gt_arousal"})

# Assign each song to an emotional quadrant (1–4)
def get_quadrant(valence, arousal):
    if valence >= 5 and arousal >= 5: return 1  # Happy
    if valence <  5 and arousal >= 5: return 2  # Angry
    if valence <  5 and arousal <  5: return 3  # Sad
    if valence >= 5 and arousal <  5: return 4  # Calm

# Evaluate each model
rows = []
for model, path in MODELS.items():
    df = pd.read_excel(path).merge(gt, on="Id").dropna(subset=["predicted_valence", "predicted_arousal"])

    q_pred = df.apply(lambda r: get_quadrant(r["predicted_valence"], r["predicted_arousal"]), axis=1)
    q_gt   = df.apply(lambda r: get_quadrant(r["gt_valence"],        r["gt_arousal"]),        axis=1)

    rows.append({
        "Model":             model,
        "MAE Valence":       round(mean_absolute_error(df["gt_valence"], df["predicted_valence"]), 3),
        "MAE Arousal":       round(mean_absolute_error(df["gt_arousal"], df["predicted_arousal"]), 3),
        "Pearson r Valence": round(pearsonr(df["gt_valence"], df["predicted_valence"])[0], 3),
        "Pearson r Arousal": round(pearsonr(df["gt_arousal"], df["predicted_arousal"])[0], 3),
        "Weighted Kappa":    round(cohen_kappa_score(q_gt, q_pred, weights="linear"), 3),
    })

# Print and save results
results = pd.DataFrame(rows).set_index("Model")
print(results.to_string())
results.to_excel("evaluation_results.xlsx")
